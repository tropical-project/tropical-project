import asyncio

import enum

import os
import time

from collections import deque

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence as GenericSequence,
    Type,
    Union,
)

import ray

import torch
import zmq
import zmq.asyncio
from prometheus_client import start_http_server
from vllm import AsyncEngineArgs, RequestOutput, SamplingParams
from vllm.config import (
    CacheConfig,
    DecodingConfig,
    DeviceConfig,
    LoadConfig,
    LoRAConfig,
    ModelConfig,
    MultiModalConfig,
    ObservabilityConfig,
    ParallelConfig,
    PromptAdapterConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm.core.scheduler import ScheduledSequenceGroup, SchedulerOutputs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream, RequestTracker
from vllm.engine.llm_engine import _LOCAL_LOGGING_INTERVAL_SEC, LLMEngine
from vllm.engine.metrics import StatLoggerBase, Stats
from vllm.engine.output_processor.interfaces import SequenceGroupOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.engine.output_processor.util import create_output_by_sequence_group
from vllm.executor.executor_base import ExecutorBase
from vllm.inputs.data import LLMInputs, PromptInputs
from vllm.logger import init_logger
from vllm.outputs import EmbeddingRequestOutput, RequestOutput, RequestOutputFactory
from vllm.sequence import (
    CompletionSequenceGroupOutput,
    EmbeddingSequenceGroupOutput,
    ExecuteModelRequest,
    Logprob,
    LoRARequest,
    PoolerOutput,
    PoolingParams,
    PromptAdapterRequest,
    SamplerOutput,
    Sequence,
    SequenceGroup,
    SequenceGroupMetadata,
    SequenceGroupOutput,
    SequenceOutput,
    SequenceStage,
    SequenceStatus,
)
from vllm.tracing import extract_trace_context, init_tracer, SpanAttributes, SpanKind
from vllm.usage.usage_lib import UsageContext

from sgir_distserve.config import EngineKind

from sgir_distserve.core.pd_scheduler import PDScheduler, PDSchedulerOutputs

from sgir_distserve.engine.distserve_llm_engine import DistserveLLMEngine
from sgir_distserve.engine.distserve_metrics import (
    DistserveLoggingStatLogger,
    DistservePrometheusStatLogger,
    DistserveStats,
    SGIR_DISTSERVE_LABEL,
)
from sgir_distserve.executor.distserve_ray_gpu_executor import PDRayGPUExecutorAsync
from sgir_distserve.sequence import (
    DistserveExecuteModelRequest,
    DistserveSequenceGroup,
    DistserveSequenceStage,
    MigrateReason,
    MigrateSnapshot,
)
from sgir_distserve.tracing import DistserveSpanAttributes


class PDEngine(DistserveLLMEngine):
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        multimodal_config: Optional[MultiModalConfig],
        speculative_config: Optional[SpeculativeConfig],
        decoding_config: Optional[DecodingConfig],
        observability_config: Optional[ObservabilityConfig],
        prompt_adapter_config: Optional[PromptAdapterConfig],
        engine_id: int,
        state_callback,
        dummy_prefill=False,
        executor_class: Type[ExecutorBase] = PDRayGPUExecutorAsync,
        log_stats: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> None:
        super().__init__(
            model_config,
            cache_config,
            parallel_config,
            scheduler_config,
            device_config,
            load_config,
            lora_config,
            multimodal_config,
            speculative_config,
            decoding_config,
            observability_config,
            prompt_adapter_config,
            executor_class,
            log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

        self.engine_id = engine_id
        self.scheduler: List[PDScheduler] = [
            PDScheduler(
                model_config.model,
                scheduler_config,
                cache_config,
                lora_config,
                parallel_config.pipeline_parallel_size,
            )
            for _ in range(parallel_config.pipeline_parallel_size)
        ]
        self.tracer = None
        if self.observability_config.otlp_traces_endpoint:
            self.tracer = init_tracer(
                "distserve.pd_engine",
                self.observability_config.otlp_traces_endpoint,
            )

        # Create sequence output processor, e.g. for beam search or
        # speculative decoding.
        self.output_processor = SequenceGroupOutputProcessor.create_output_processor(
            self.scheduler_config,
            self.detokenizer,
            self.scheduler,
            self.seq_counter,
            self.get_tokenizer_for_seq,
            stop_checker=StopChecker(
                self.scheduler_config.max_model_len,
                self.get_tokenizer_for_seq,
            ),
        )
        self.dummy_prefill = dummy_prefill
        self.state_callback = state_callback
        print(self.model_config.served_model_name)
        print(self.tokenizer.get_lora_tokenizer().eos_token_id)

    def _process_model_outputs(
        self,
        output: GenericSequence[Union[SamplerOutput, PoolerOutput]],
        scheduled_seq_groups: List[ScheduledSequenceGroup],
        ignored_seq_groups: List[SequenceGroup],
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        """Apply the model output to the sequences in the scheduled seq groups.

        Returns RequestOutputs that can be returned to the client.
        """

        now = time.time()

        # Organize outputs by [sequence group][step] instead of
        # [step][sequence group].
        output_by_sequence_group = create_output_by_sequence_group(
            output, num_seq_groups=len(scheduled_seq_groups)
        )

        # Update the scheduled sequence groups with the model outputs.
        for scheduled_seq_group, outputs, seq_group_meta in zip(
            scheduled_seq_groups, output_by_sequence_group, seq_group_metadata_list
        ):
            seq_group = scheduled_seq_group.seq_group
            seq_group.update_num_computed_tokens(scheduled_seq_group.token_chunk_size)
            if self.model_config.embedding_mode:
                self._process_sequence_group_outputs(seq_group, outputs)
                continue

            self.output_processor.process_prompt_logprob(seq_group, outputs)
            if seq_group_meta.do_sample:
                self.output_processor.process_outputs(seq_group, outputs)

        # Free the finished sequence groups.
        for scheduler in self.scheduler:
            scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[Union[RequestOutput, EmbeddingRequestOutput]] = []
        for scheduled_seq_group in scheduled_seq_groups:
            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            if seq_group.metrics.first_token_time:
                request_output = RequestOutputFactory.create(seq_group)
                request_outputs.append(request_output)
        for seq_group in ignored_seq_groups:
            request_output = RequestOutputFactory.create(seq_group)
            request_outputs.append(request_output)
        return request_outputs

    def step(self) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        seq_group_metadata_list, scheduler_outputs = self.scheduler[0].schedule()

        if not scheduler_outputs.is_empty():
            finished_requests_ids = self.scheduler[
                0
            ].get_and_reset_finished_requests_ids()
            # dummy prefill
            if self.dummy_prefill:
                prefill_outputs = [
                    CompletionSequenceGroupOutput(
                        samples=[
                            SequenceOutput(
                                parent_seq_id=seq.seq_group.get_seqs()[0].seq_id,
                                output_token=323,
                                logprobs={
                                    323: Logprob(
                                        logprob=-2.158910036087036,
                                        rank=1,
                                        decoded_token=None,
                                    )
                                },
                            )
                        ],
                        prompt_logprobs=None,
                    )
                    for seq in scheduler_outputs.scheduled_seq_groups[
                        : scheduler_outputs.num_prefill_groups
                    ]
                ]
                new_metadata_list = seq_group_metadata_list[
                    scheduler_outputs.num_prefill_groups :
                ]
            exe_begin = time.time()
            execute_model_req = DistserveExecuteModelRequest(
                seq_group_metadata_list=(
                    seq_group_metadata_list
                    if not self.dummy_prefill
                    else new_metadata_list
                ),
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                virtual_engine=0,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids,
                blocks_to_migration=scheduler_outputs.blocks_to_migration,
            )
            if scheduler_outputs.num_prefill_groups:
                self.state_callback(scheduler_outputs)
            output = ray.get(
                self.model_executor.execute_model(execute_model_req=execute_model_req)
            )
            exe_end = time.time()
            if self.dummy_prefill:
                if output:
                    outputs = deque(output[0].outputs)
                    outputs.extendleft(prefill_outputs)
                    output[0].outputs = list(outputs)
                else:
                    output = [
                        SamplerOutput(
                            outputs=prefill_outputs,
                            sampled_token_probs=None,
                            sampled_token_ids=None,
                            spec_decode_worker_metrics=None,
                        )
                    ]
        else:
            output = []

        request_outputs = self._process_model_outputs(
            output,
            scheduler_outputs.scheduled_seq_groups,
            scheduler_outputs.ignored_seq_groups,
            seq_group_metadata_list,
        )
        now = time.time()
        for sched_seq_group in scheduler_outputs.scheduled_seq_groups[
            0 : scheduler_outputs.num_prefill_groups
        ]:
            seq_group = sched_seq_group.seq_group
            seq_group.set_chunk_by_chunk_time(now)

        if (
            scheduler_outputs.num_prefill_groups > 0
            or len(scheduler_outputs.migration_seq_groups) > 0
        ):
            for sched_seq_group in scheduler_outputs.scheduled_seq_groups[
                scheduler_outputs.num_prefill_groups :
            ]:
                seq_group = sched_seq_group.seq_group
                seq_group.set_interference_time(exe_end - exe_begin)

            for sched_seq_group in scheduler_outputs.migration_seq_groups:
                seq_group = sched_seq_group.seq_group
                seq_group.set_interference_others_time((exe_end - exe_begin))
            if self.engine_id < 2:
                for sched_seq_group in scheduler_outputs.scheduled_seq_groups[
                    : scheduler_outputs.num_prefill_groups
                ]:
                    seq_group = sched_seq_group.seq_group
                    seq_group.set_interference_others_time((exe_end - exe_begin))

        # Log stats.
        self.do_log_stats(scheduler_outputs, output)

        # Tracing
        self.do_tracing(scheduler_outputs)

        migration_to_seq_groups = []
        for sched_seq_group in scheduler_outputs.scheduled_seq_groups[
            0 : scheduler_outputs.num_prefill_groups
        ]:
            seq_group = sched_seq_group.seq_group
            if (
                not seq_group.is_finished()
                and seq_group.get_seqs()[0].get_output_len() == 1
            ):
                self.scheduler[0].running.remove(seq_group)
                block_index = self.scheduler[0].block_manager.get_block_table(
                    seq_group.get_seqs()[0]
                )
                seq_group.migration_snapshots.append(
                    MigrateSnapshot(
                        seq_id=seq_group.get_seqs()[0].seq_id,
                        engine_id=self.engine_id,
                        virtual_engine_id=0,
                        block_index=block_index,
                        reason=MigrateReason.Disaggregation,
                    )
                )
                seq_group.get_seqs()[0].data._stage = DistserveSequenceStage.MIGRATE
                seq_group.get_seqs()[0].status = SequenceStatus.WAITING
                migration_to_seq_groups.append(seq_group)
        total_seq_length = 0
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups[
            scheduler_outputs.num_prefill_groups :
        ]:
            seq_group = scheduled_seq_group.seq_group
            total_seq_length += seq_group.get_seqs()[0].get_len()

        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups[
            scheduler_outputs.num_prefill_groups :
        ]:
            seq_group: DistserveSequenceGroup = scheduled_seq_group.seq_group
            seq_group.set_token_by_token_time(time.time())
            seq_group.set_decoding_batch_size(
                len(scheduler_outputs.scheduled_seq_groups)
            )
            seq_group.set_decoding_batched_seq_length(total_seq_length)

        # add migration done callback
        migration_done_seq_groups = []
        now = time.time()
        for sched_migration_seq_group in scheduler_outputs.migration_seq_groups:
            seq_group: DistserveSequenceGroup = sched_migration_seq_group.seq_group
            seq_group.metrics.migration_done_time = now
            seq_group.get_seqs()[0].data._stage = SequenceStage.DECODE
            seq_group.get_seqs()[0].status = SequenceStatus.RUNNING
            self.scheduler[0].running.extend([seq_group])
            migration_done_seq_groups.append(seq_group)

        return request_outputs, migration_to_seq_groups, migration_done_seq_groups

    def create_trace_span(self, seq_group: DistserveSequenceGroup) -> None:
        if self.tracer is None or seq_group.sampling_params is None:
            return
        arrival_time_nano_seconds = int(seq_group.metrics.arrival_time * 1e9)

        trace_context = extract_trace_context(seq_group.trace_headers)

        with self.tracer.start_as_current_span(
            "llm_request",
            kind=SpanKind.SERVER,
            context=trace_context,
            start_time=arrival_time_nano_seconds,
        ) as seq_span:
            metrics = seq_group.metrics
            ttft = metrics.first_token_time - metrics.arrival_time
            e2e_time = metrics.finished_time - metrics.arrival_time
            migration_time = metrics.migration_done_time - metrics.first_token_time
            # attribute names are based on
            # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/llm-spans.md
            seq_span.set_attribute(
                SpanAttributes.LLM_RESPONSE_MODEL, self.model_config.model
            )
            seq_span.set_attribute(SpanAttributes.LLM_REQUEST_ID, seq_group.request_id)
            seq_span.set_attribute(
                SpanAttributes.LLM_REQUEST_TEMPERATURE,
                seq_group.sampling_params.temperature,
            )
            seq_span.set_attribute(
                SpanAttributes.LLM_REQUEST_TOP_P, seq_group.sampling_params.top_p
            )
            seq_span.set_attribute(
                SpanAttributes.LLM_REQUEST_MAX_TOKENS,
                seq_group.sampling_params.max_tokens,
            )
            seq_span.set_attribute(
                SpanAttributes.LLM_REQUEST_BEST_OF, seq_group.sampling_params.best_of
            )
            seq_span.set_attribute(
                SpanAttributes.LLM_REQUEST_N, seq_group.sampling_params.n
            )
            seq_span.set_attribute(
                SpanAttributes.LLM_USAGE_NUM_SEQUENCES, seq_group.num_seqs()
            )
            seq_span.set_attribute(
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS, len(seq_group.prompt_token_ids)
            )
            seq_span.set_attribute(
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                sum([seq.get_output_len() for seq in seq_group.get_finished_seqs()]),
            )
            seq_span.set_attribute(
                SpanAttributes.LLM_LATENCY_TIME_IN_QUEUE, metrics.time_in_queue
            )
            seq_span.set_attribute(
                DistserveSpanAttributes.LLM_LATENCY_TIME_TO_PREFILL,
                ttft - metrics.time_in_queue,
            )
            seq_span.set_attribute(SpanAttributes.LLM_LATENCY_TIME_TO_FIRST_TOKEN, ttft)
            seq_span.set_attribute(SpanAttributes.LLM_LATENCY_E2E, e2e_time)
            seq_span.set_attribute(
                DistserveSpanAttributes.LLM_LATENCY_MIGRATION, migration_time
            )

            chunk_by_chunk_time = []
            chunk_by_chunk_time.append(metrics.first_scheduled_time)
            chunk_by_chunk_time.extend(metrics.chunk_by_chunk_time)
            chunk_by_chunk_time = [
                chunk_by_chunk_time[i] - chunk_by_chunk_time[i - 1]
                for i in range(1, len(chunk_by_chunk_time))
            ]
            seq_span.set_attribute(
                DistserveSpanAttributes.LLM_LATENCY_CHUNK_BY_CHUNK, chunk_by_chunk_time
            )

            seq_span.set_attribute(
                DistserveSpanAttributes.LLM_LATENCY_INTERFERENCE_TIME,
                metrics.interference_time,
            )

            seq_span.set_attribute(
                DistserveSpanAttributes.LLM_LATENCY_INTERFERENCE_OTHERS_TIME,
                metrics.interference_others_time,
            )

            if metrics.token_by_token_time:
                seq_span.set_attribute(
                    DistserveSpanAttributes.LLM_LATENCY_TOKEN_BY_TOKEN,
                    [metrics.token_by_token_time[0] - metrics.first_token_time]
                    + [
                        metrics.token_by_token_time[i]
                        - metrics.token_by_token_time[i - 1]
                        for i in range(1, len(metrics.token_by_token_time))
                    ],
                )
            else:
                seq_span.set_attribute(
                    DistserveSpanAttributes.LLM_LATENCY_TOKEN_BY_TOKEN, []
                )

            seq_span.set_attribute(
                DistserveSpanAttributes.LLM_PREFILL_BATCH_SIZE,
                metrics.prefill_batch_size,
            )

            seq_span.set_attribute(
                DistserveSpanAttributes.LLM_PREFILL_BATCHED_SEQ_LENGTH,
                metrics.prefill_batched_seq_length,
            )

            seq_span.set_attribute(
                DistserveSpanAttributes.LLM_DECODING_BATCH_SIZE,
                metrics.decoding_batch_size,
            )

            seq_span.set_attribute(
                DistserveSpanAttributes.LLM_DECODING_BATCHED_SEQ_LENGTH,
                metrics.decoding_batched_seq_length,
            )

            seq_span.add_event(
                "waiting done",
                timestamp=int((metrics.arrival_time + metrics.time_in_queue) * 1e9),
            )
            seq_span.add_event(
                "first token", timestamp=int(metrics.first_token_time * 1e9)
            )
            seq_span.add_event(
                "migration done", timestamp=int(metrics.migration_done_time * 1e9)
            )

    def _get_stats(
        self,
        scheduler_outputs: SchedulerOutputs | None,
        model_output: List[SamplerOutput] | None = None,
    ) -> DistserveStats:
        stats = super()._get_stats(scheduler_outputs, model_output)
        now = stats.now

        num_waiting_for_migrating = sum(
            len(scheduler.waiting_for_migration) for scheduler in self.scheduler
        )
        num_running_for_migrating = sum(
            len(scheduler.running_for_migration) for scheduler in self.scheduler
        )

        scheduler = self.scheduler[0]

        num_batched_tokens = 0
        for seq_group in scheduler.running:
            # seq_group = running.seq_group
            num_batched_tokens += seq_group.get_seqs()[0].get_len()

        time_to_migration = []
        if scheduler_outputs is not None:
            for migration_seq_group in scheduler_outputs.migration_seq_groups:
                seq_group = migration_seq_group.seq_group
                # print("time config begin")
                # print("first_token_time", seq_group.metrics.first_token_time)
                # print("decode_begin_time", seq_group.metrics.decode_begin_time)
                # print("migration_begin_time", seq_group.metrics.migration_begin_time)
                # print("last_token_time", seq_group.metrics.last_token_time)
                # print("now:", now)
                # print("time config done")
                time_to_migration.append(seq_group.get_last_latency(now))

        return DistserveStats.from_stats(
            stats=stats,
            instance_id=0,
            num_batched_tokens=num_batched_tokens,
            num_waiting_for_migrating=num_waiting_for_migrating,
            num_running_for_migrating=num_running_for_migrating,
            time_to_migration=time_to_migration,
        )
