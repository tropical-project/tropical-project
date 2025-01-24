import asyncio

import os
import time

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Type,
    Union,
)

import ray

import torch
import zmq
import zmq.asyncio
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
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream, RequestTracker
from vllm.engine.llm_engine import _LOCAL_LOGGING_INTERVAL_SEC, LLMEngine
from vllm.engine.metrics import StatLoggerBase, Stats
from vllm.engine.output_processor.interfaces import SequenceGroupOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.executor.executor_base import ExecutorBase
from vllm.inputs.data import LLMInputs, PromptInputs
from vllm.logger import init_logger
from vllm.outputs import EmbeddingRequestOutput
from vllm.sequence import (
    Logprob,
    LoRARequest,
    PoolingParams,
    PromptAdapterRequest,
    SamplerOutput,
    Sequence,
    SequenceGroupOutput,
    SequenceOutput,
)
from vllm.tracing import extract_trace_context, init_tracer, SpanAttributes, SpanKind
from vllm.usage.usage_lib import UsageContext

from sgir_distserve.core.pd_scheduler import PDScheduler

from sgir_distserve.engine.distserve_llm_engine import DistserveLLMEngine
from sgir_distserve.engine.distserve_metrics import (
    DistserveLoggingStatLogger,
    DistservePrometheusStatLogger,
    DistserveStats,
)
from sgir_distserve.sequence import DistserveExecuteModelRequest, DistserveSequenceGroup
from sgir_distserve.tracing import DistserveSpanAttributes


logger = init_logger(__name__)


ENGINE_ITERATION_TIMEOUT_S = int(
    os.environ.get("VLLM_ENGINE_ITERATION_TIMEOUT_S", "120")
)


class AsyncEngineDeadError(RuntimeError):
    pass


class _AsyncPDEngine(DistserveLLMEngine):
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
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        engine_id: int,
        prefill_done_callback: Callable[[int, int, DistserveSequenceGroup], None],
        migration_done_callback: Callable[[int, int, DistserveSequenceGroup], None],
        preemption_callback: Callable[[int, int, DistserveSequenceGroup], None],
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

        self.prefill_done_callback = prefill_done_callback
        self.migration_done_callback = migration_done_callback
        self.preemption_callback = preemption_callback

    async def add_request_async(
        self,
        request_id: str,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        ttft_slo: Optional[float] = None,
        tpot_slo: Optional[float] = None,
    ) -> None:
        if lora_request is not None and not self.lora_config:
            raise ValueError(
                f"Got lora_request {lora_request} but LoRA is " "not enabled!"
            )
        if arrival_time is None:
            arrival_time = time.time()

        processed_inputs = await self.process_model_inputs_async(
            request_id=request_id,
            inputs=inputs,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )

        self._add_processed_request(
            request_id=request_id,
            processed_inputs=processed_inputs,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            trace_headers=trace_headers,
            ttft_slo=ttft_slo,
            tpot_slo=tpot_slo,
        )

    def _add_processed_request(
        self,
        request_id: str,
        processed_inputs: LLMInputs,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: float,
        lora_request: Optional[LoRARequest],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        trace_headers: Optional[Mapping[str, str]] = None,
        ttft_slo: Optional[float] = None,
        tpot_slo: Optional[float] = None,
    ) -> None:
        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        eos_token_id = self._get_eos_token_id(lora_request)

        seq = Sequence(
            seq_id,
            processed_inputs,
            block_size,
            eos_token_id,
            lora_request,
            prompt_adapter_request,
        )

        # Create a SequenceGroup based on SamplingParams or PoolingParams
        if isinstance(params, SamplingParams):
            seq_group = self._create_sequence_group_with_sampling(
                request_id,
                seq,
                params,
                arrival_time=arrival_time,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
            )
        elif isinstance(params, PoolingParams):
            seq_group = self._create_sequence_group_with_pooling(
                request_id,
                seq,
                params,
                arrival_time=arrival_time,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
            )
        else:
            raise ValueError("Either SamplingParams or PoolingParams must be provided.")

        # Add the sequence group to the scheduler with least unfinished seqs.
        costs = [
            scheduler.get_num_unfinished_seq_groups() for scheduler in self.scheduler
        ]
        min_cost_scheduler = self.scheduler[costs.index(min(costs))]
        min_cost_scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        for scheduler in self.scheduler:
            scheduler.abort_seq_group(request_id)
            scheduler.abort_seq_group_migration(request_id)

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
            instance_id=self.engine_id,
            num_batched_tokens=num_batched_tokens,
            num_waiting_for_migrating=num_waiting_for_migrating,
            num_running_for_migrating=num_running_for_migrating,
            time_to_migration=time_to_migration,
        )

    async def step_async(
        self, engine_id: int, virtual_engine_id: int
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler[
            virtual_engine_id
        ].schedule()

        # add migration done callback
        for sched_migration_seq_group in scheduler_outputs.migration_seq_groups:
            seq_group = sched_migration_seq_group.seq_group
            seq_group.decode_virtual_engine = virtual_engine_id
            seq_group.decode_engine = engine_id
            seq_group.metrics.migration_begin_time = time.time()

        if not scheduler_outputs.is_empty():
            # Execute the model.
            finished_requests_ids = self.scheduler[
                virtual_engine_id
            ].get_and_reset_finished_requests_ids()
            execute_model_req = DistserveExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                virtual_engine=virtual_engine_id,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids,
                blocks_to_migration=scheduler_outputs.blocks_to_migration,
            )
            output = await self.model_executor.execute_model_async(execute_model_req)
        else:
            output = []

        request_outputs = self._process_model_outputs(
            output,
            scheduler_outputs.scheduled_seq_groups,
            scheduler_outputs.ignored_seq_groups,
            seq_group_metadata_list,
        )

        # Log stats.
        self.do_log_stats(scheduler_outputs, output)

        # Tracing
        self.do_tracing(scheduler_outputs)

        total_seq_length = 0
        for sched_seq_group in scheduler_outputs.scheduled_seq_groups[
            0 : scheduler_outputs.num_prefill_groups
        ]:
            seq_group = sched_seq_group.seq_group
            if (
                not seq_group.is_finished()
                # prefill has a free token
                # evaluate if this seq group should be migrated after TTFT
                and seq_group.get_seqs()[0].get_output_len() == 1
            ):
                self.prefill_done_callback(
                    self.engine_id,
                    virtual_engine_id,
                    seq_group,
                )

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
        for sched_migration_seq_group in scheduler_outputs.migration_seq_groups:
            seq_group = sched_migration_seq_group.seq_group
            self.migration_done_callback(engine_id, virtual_engine_id, seq_group)

        for seq_group in scheduler_outputs.preempted_seq_groups:
            # seq_group = sched_preempted_seq_group.seq_group
            self.preemption_callback(engine_id, virtual_engine_id, seq_group)

        return request_outputs, scheduler_outputs.migration_seq_groups

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
            seq_span.set_attribute(SpanAttributes.LLM_LATENCY_TIME_TO_FIRST_TOKEN, ttft)
            seq_span.set_attribute(SpanAttributes.LLM_LATENCY_E2E, e2e_time)
            seq_span.set_attribute(
                DistserveSpanAttributes.LLM_LATENCY_MIGRATION, migration_time
            )
            seq_span.set_attribute(
                DistserveSpanAttributes.LLM_LATENCY_TIME_TO_PREFILL,
                ttft - metrics.time_in_queue,
            )
            if len(metrics.token_by_token_time) > 0:
                seq_span.set_attribute(
                    DistserveSpanAttributes.LLM_LATENCY_TOKEN_BY_TOKEN,
                    [metrics.token_by_token_time[0] - metrics.migration_done_time]
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
            for i, t in enumerate(metrics.token_by_token_time):
                seq_span.add_event(f"token_{i}", timestamp=int(t * 1e9))
