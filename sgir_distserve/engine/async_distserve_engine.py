import argparse

import asyncio
import os
import time

from collections import deque
from concurrent.futures import ThreadPoolExecutor

from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np

import ray

import torch
import zmq
import zmq.asyncio
from transformers import PreTrainedTokenizer

from vllm import AsyncEngineArgs, RequestOutput, SamplingParams
from vllm.config import (
    CacheConfig,
    DecodingConfig,
    DeviceConfig,
    EngineConfig,
    LoadConfig,
    LoRAConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm.engine.async_llm_engine import (
    _log_task_completion,
    AsyncEngineDeadError,
    AsyncLLMEngine,
    AsyncStream,
    ENGINE_ITERATION_TIMEOUT_S,
    RequestTracker,
)
from vllm.engine.async_timeout import asyncio_timeout
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.metrics import StatLoggerBase
from vllm.engine.output_processor.util import create_output_by_sequence_group

from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.executor.executor_base import ExecutorBase
from vllm.executor.ray_utils import initialize_ray_cluster
from vllm.inputs import INPUT_REGISTRY, LLMInputs, PromptInputs
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput
from vllm.sequence import (
    EmbeddingSequenceGroupOutput,
    PoolingParams,
    PromptAdapterRequest,
    Sequence,
    SequenceGroup,
    SequenceStage,
    SequenceStatus,
)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter

from sgir_distserve import _C

from sgir_distserve.config import EngineKind, SGIRDistserveConfig
from sgir_distserve.core.global_scheduler import DispatchPolicy

from sgir_distserve.engine.arg_utils import SGIRDistserveArgs
from sgir_distserve.engine.async_distserve_request_tracker import (
    AsyncDistserveRequestTracker as DistserveRequestTracker,
)
from sgir_distserve.engine.async_pd_engine import _AsyncPDEngine

from sgir_distserve.engine.distserve_metrics import (
    DistserveLoggingStatLogger,
    DistservePrometheusStatLogger,
    SGIR_DISTSERVE_LABEL,
)
from sgir_distserve.engine.pd_profiler import PDProfiler

# ray
from sgir_distserve.executor.distserve_ray_gpu_executor import PDRayGPUExecutorAsync

from sgir_distserve.sequence import (
    DistserveSequenceGroup,
    DistserveSequenceStage,
    MigrateReason,
    MigrateSnapshot,
)


logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 1


class AsyncDistserveEngine(AsyncLLMEngine):
    _pd_engine_class: Type[_AsyncPDEngine] = _AsyncPDEngine

    def __init__(
        self,
        sgir_distserve_config: SGIRDistserveConfig,
        *args,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        **kwargs,
    ) -> None:
        self.async_distserve_engine_config = sgir_distserve_config
        # use ray in worker by default
        self.engine_use_ray = False
        self.worker_use_ray = True

        self.log_requests = log_requests

        labels = {}
        for engine_id, config in enumerate(sgir_distserve_config.replica_config):
            label = {
                SGIR_DISTSERVE_LABEL.MODEL_NAME: config.engine_config.model_config.served_model_name,
                SGIR_DISTSERVE_LABEL.INSTANCE_ID_LABEL: engine_id,
            }
            labels[engine_id] = label

        stat_loggers = {
            "logging": DistserveLoggingStatLogger(
                local_interval=_LOCAL_LOGGING_INTERVAL_SEC
            ),
            "prometheus": DistservePrometheusStatLogger(
                local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                labels=labels,
                max_model_len=sgir_distserve_config.replica_config[
                    0
                ].engine_config.model_config.max_model_len,
            ),
        }

        # prefill engine configuration
        # engine args
        pd_kwargs = []
        for engine_id, config in enumerate(sgir_distserve_config.replica_config):
            engine_kwargs = config.engine_config.to_dict()
            engine_kwargs.update(kwargs)
            engine_kwargs.update(
                {
                    "engine_id": engine_id,
                    "prefill_done_callback": self.prefill_done_callback,
                    "migration_done_callback": self.migration_done_callback,
                    "preemption_callback": self.preemption_callback,
                    "executor_class": PDRayGPUExecutorAsync,
                    "stat_loggers": stat_loggers,
                }
            )
            pd_kwargs.append(engine_kwargs)

        pd = []
        with ThreadPoolExecutor(len(pd_kwargs)) as executor:
            for engine_kwargs in pd_kwargs:
                pd.append(
                    executor.submit(
                        self._init_engine,
                        *args,
                        **engine_kwargs,
                    )
                )

        self.pd_engine: List[_AsyncPDEngine] = [p.result() for p in pd]

        logger.info("Decode engine initialized")
        logger.info(f"Engine initialized")

        self.background_loop: List[Optional[asyncio.Future]] = [
            None
        ] * self.num_replicas

        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded: List[Optional[asyncio.Task[Any]]] = [
            None
        ] * self.num_replicas
        self.start_engine_loop = start_engine_loop
        self._errored_with: Optional[BaseException] = None

        # Lazy initialized fields
        self._request_tracker: DistserveRequestTracker

        for engine_id, engine in enumerate(self.pd_engine):
            engine._initialize_kv_caches_handlers()

        handlers = [engine.get_handlers() for engine in self.pd_engine]

        for engine_id, engine in enumerate(self.pd_engine):
            engine.register_handlers(engine_id, handlers)

        stat_loggers["prometheus"].metrics.sgir_distserve_config.info(
            {"num_replicas": str(self.num_replicas)}
        )

        self.input_processor = INPUT_REGISTRY.create_input_processor(
            self.pd_engine[0].model_config
        )

        self.partition_id = sgir_distserve_config.partition_id
        self.p_set = set(i for i in range(0, sgir_distserve_config.partition_id))
        self.d_set = set(
            i for i in range(sgir_distserve_config.partition_id, self.num_replicas)
        )

        self.prefill_batch_config = sgir_distserve_config.prefill_engine_batch_config
        self.decode_batch_config = sgir_distserve_config.decode_engine_batch_config

        # SLO QUEUE
        self.seq_counter = Counter()
        self.waiting_for_dispatch: Deque[DistserveSequenceGroup] = deque([])
        self.waiting_for_migration: Deque[DistserveSequenceGroup] = deque([])
        driver_config = self.async_distserve_engine_config.replica_config[
            0
        ].engine_config
        self.pd_profiler = PDProfiler(
            model=self.async_distserve_engine_config.replica_config[
                0
            ].engine_config.model_config.model,
            chunk=(
                driver_config.scheduler_config.max_num_batched_tokens
                if driver_config.scheduler_config.chunked_prefill_enabled
                else -1
            ),
        )

        self.tokenizer = None

        self.global_stat_loggers = {
            "logging": DistserveLoggingStatLogger(
                local_interval=_LOCAL_LOGGING_INTERVAL_SEC
            ),
            "prometheus": DistservePrometheusStatLogger(
                local_interval=0.5,
                labels={
                    SGIR_DISTSERVE_LABEL.MODEL_NAME: config.engine_config.model_config.served_model_name,
                    SGIR_DISTSERVE_LABEL.INSTANCE_ID_LABEL: -1,
                },
                max_model_len=sgir_distserve_config.replica_config[
                    0
                ].engine_config.model_config.max_model_len,
                metrics=stat_loggers["prometheus"].metrics,
            ),
        }

        # disable decode instance slo-aware-chunked-preemption
        for engine_id in self.d_set:
            self.pd_engine[engine_id].scheduler[0].slo_aware_chunked_preemption = False

    @property
    def num_replicas(self):
        return len(self.pd_engine)

    @property
    def engine(self):
        return self.pd_engine[0]

    def _error_callback(
        self,
        exc: Exception,
        engine_id: int,
    ) -> None:
        self.set_errored(exc)
        self._request_tracker.propagate_exception(
            exc,
            engine_id=engine_id,
        )

    async def get_seq_group_from_config(
        self,
        request_id: str,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> DistserveSequenceGroup:
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

        # Create the sequences.
        block_size = self.async_distserve_engine_config.block_size
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
        seq_group = self.pd_engine[0]._create_sequence_group_with_sampling(
            request_id,
            seq,
            params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
        )

        distserve_seq_group = DistserveSequenceGroup.from_seq_group(
            seq_group, params.ttft_slo, params.tpot_slo
        )

        return distserve_seq_group

    def _get_eos_token_id(self, lora_request: Optional[LoRARequest]) -> Optional[int]:
        if self.tokenizer is None:
            logger.warning(
                "Using None for EOS token id because tokenizer " "is not initialized"
            )
            return None

        return self.tokenizer.get_lora_tokenizer(lora_request).eos_token_id

    async def process_model_inputs_async(
        self,
        request_id: str,
        inputs: PromptInputs,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> LLMInputs:
        if isinstance(inputs, str):
            inputs = {"prompt": inputs}

        if "prompt_token_ids" not in inputs:
            tokenizer = self.get_tokenizer_group(
                "prompts must be None if " "skip_tokenizer_init is True"
            )

            prompt_token_ids = await tokenizer.encode_async(
                request_id=request_id,
                prompt=inputs["prompt"],
                lora_request=lora_request,
            )
        else:
            prompt_token_ids = inputs["prompt_token_ids"]

        if prompt_adapter_request:
            prompt_token_ids = (
                [0] * prompt_adapter_request.prompt_adapter_num_virtual_tokens
                + prompt_token_ids
            )

        llm_inputs = LLMInputs(
            prompt_token_ids=prompt_token_ids,
            prompt=inputs.get("prompt"),
            multi_modal_data=inputs.get("multi_modal_data"),
        )

        return self.input_processor(llm_inputs)

    def get_idle_p_engine(self) -> int:
        if len(self.p_set) == self.num_replicas:
            costs = [
                (idx, self.pd_engine[idx].scheduler[0].get_num_unfinished_seq_groups())
                for idx in self.p_set
            ]
            return costs[costs.index(min(costs, key=lambda x: x[1]))][0]

        idle_set = set()
        for idx in self.p_set:
            engine = self.pd_engine[idx]
            unfinished_jobs = engine.scheduler[0].get_num_unfinished_seq_groups()
            max_num_seqs = self.prefill_batch_config.max_num_seqs
            max_num_batched_tokens = (
                self.prefill_batch_config.max_num_batched_tokens
                or self.engine.get_model_config().max_model_len
            )
            engine_waiting = engine.scheduler[0].waiting
            engine_running = engine.scheduler[0].running
            waiting_tokens = sum(
                [
                    seq_group.get_seqs()[0].data.get_prompt_len()
                    for seq_group in engine_waiting
                ]
            )

            if unfinished_jobs == 0 or (
                waiting_tokens < 2048 and len(engine_waiting) < 4
            ):
                idle_set.add(idx)
        if idle_set:
            costs = [
                (idx, self.pd_engine[idx].scheduler[0].get_num_unfinished_seq_groups())
                for idx in idle_set
            ]
            return costs[costs.index(min(costs, key=lambda x: x[1]))][0]
        return None

    def get_inlinable_d_engine(self, prefill_seq_group) -> int:
        for idx in self.d_set:
            engine = self.pd_engine[idx]
            scheduler = engine.scheduler[0]
            has_prefill = any(
                [seq_group.is_prefill() for seq_group in scheduler.running]
            )
            if scheduler.waiting or has_prefill:
                continue
            prefill_prediction = self.pd_profiler.get_prefill_seq_prediction(
                prefill_seq_group
            )
            # stall_free = True
            now = time.time()
            decode_slack = 0
            stall_free = True
            for seq_group in scheduler.running:
                # decode_slack += self.pd_profiler.get_decode_slack(seq_group, now)
                # print(self.pd_profiler.get_decode_slack(seq_group, now))
                if (
                    self.pd_profiler.get_decode_slack(seq_group, now)
                    - prefill_prediction
                    < seq_group.tpot_slo * 1
                ):
                    stall_free = False
                    break
            if stall_free:
                return idx
        return None

    async def collect_request(self):
        total_prompts_cnt = 0
        time_to_last_event = time.time()
        record_duration = 0.5  # 5s
        while True:
            await self._request_tracker.wait_for_requests_in_pool()
            print(f"get new requests!!!")
            new_requests_configs = self._request_tracker.get_new_requests_from_pool()
            for kwargs in new_requests_configs:
                seq_group: DistserveSequenceGroup = (
                    await self.get_seq_group_from_config(**kwargs)
                )
                self.waiting_for_dispatch.append(seq_group)
                total_prompts_cnt += seq_group.get_num_uncomputed_tokens()
            now = time.time()
            if now - time_to_last_event > record_duration:
                time_to_last_event = now
                logger: DistservePrometheusStatLogger = self.global_stat_loggers[
                    "prometheus"
                ]
                logger._log_gauge(
                    logger.metrics.gauge_global_scheduler_num_arrived_prompt_tokens,
                    total_prompts_cnt,
                )
                total_prompts_cnt = 0
            self.waiting_for_dispatch = deque(
                sorted(
                    self.waiting_for_dispatch,
                    key=lambda x: x.get_num_uncomputed_tokens(),
                )
            )

    async def dispatch_request(self):
        while True:
            # Migrate request
            # TODO: Step 3: 把 decode 整理整理腾腾地方

            # dispatch request
            # if self.edf:
            #     now = time.time()
            #     self.waiting_for_dispatch = deque(
            #         sorted(
            #             self.waiting_for_dispatch,
            #             key=partial(self.pd_profiler.get_prefill_slack, now=now),
            #         )
            #     )

            if self.waiting_for_dispatch:
                seq_group = self.waiting_for_dispatch[0]
                # TODO: Step 1: 判断是否有空余的 Engine
                if (
                    self.async_distserve_engine_config.dispatch_policy
                    == DispatchPolicy.Slack
                    and self.pd_profiler
                ):
                    inlinable_idx: int = self.get_inlinable_d_engine(seq_group)
                    if inlinable_idx != None:
                        engine = self.pd_engine[inlinable_idx]
                        # print("preemptable engine idx: ", preemptable_engine_idx)
                        engine.scheduler[0].add_seq_group(seq_group)
                        seq_group.get_seqs()[0].seq_id = next(engine.seq_counter)
                        self._request_tracker.requests_event[inlinable_idx].set()
                        self.waiting_for_dispatch.popleft()
                        await asyncio.sleep(0.001)
                        continue
                idle_idx = self.get_idle_p_engine()
                if idle_idx != None:
                    seq_group.get_seqs()[0].seq_id = next(
                        self.pd_engine[idle_idx].seq_counter
                    )
                    self.pd_engine[idle_idx].scheduler[0].add_seq_group(seq_group)
                    self._request_tracker.requests_event[idle_idx].set()
                    self.waiting_for_dispatch.popleft()
                    await asyncio.sleep(0.001)
                    continue

            await asyncio.sleep(0.001)

    async def add_request(
        self,
        request_id: str,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> AsyncStream:
        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError)."
                )

        if arrival_time is None:
            arrival_time = time.time()

        stream = self._request_tracker.add_request_pool(
            request_id,
            verbose=self.log_requests,
            inputs=inputs,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
        )

        return stream

    def profile_step(self):
        for pd_engine in self.pd_engine:
            pd_engine.model_executor.profile_step()

    def scale_down(self):
        raise NotImplementedError

    def scale_up(self):
        raise NotImplementedError

    def get_engine_id_by_cost(self, engine_set):
        costs = [
            (idx, self.pd_engine[idx].scheduler[0].get_num_unfinished_seq_groups())
            for idx in engine_set
        ]
        engine_id = min(costs, key=lambda x: x[1])[0]
        return engine_id, 0

    def prefill_done_callback(
        self,
        src_engine_id,
        src_virtual_engine_id,
        seq_group: DistserveSequenceGroup,
    ):
        if not self.d_set:
            # if all engine is prefill engine, then don not perform migration
            return
        des_engine_id, des_virtual_engine_id = self.get_engine_id_by_cost(self.d_set)

        if des_engine_id != src_engine_id:
            logger.debug("Migration Happened")
            seq_group.metrics.migration_begin_time = time.time()
            seq_group.metrics.decode_begin_time = time.time()
            src_scheduler = self.pd_engine[src_engine_id].scheduler[
                src_virtual_engine_id
            ]
            src_scheduler.running.remove(seq_group)
            block_index = src_scheduler.block_manager.get_block_table(
                seq_group.get_seqs()[0]
            )
            seq_group.migration_snapshots.append(
                MigrateSnapshot(
                    seq_id=seq_group.get_seqs()[0].seq_id,
                    engine_id=src_engine_id,
                    virtual_engine_id=src_virtual_engine_id,
                    block_index=block_index,
                    reason=MigrateReason.Disaggregation,
                )
            )

            seq_group.get_seqs()[0].data._stage = DistserveSequenceStage.MIGRATE
            seq_group.get_seqs()[0].status = SequenceStatus.WAITING

            des_scheduler = self.pd_engine[des_engine_id].scheduler[
                des_virtual_engine_id
            ]
            des_scheduler.add_seq_group_to_waiting_for_migration(seq_group)

            seq_group.get_seqs()[0].seq_id = next(
                self.pd_engine[des_engine_id].seq_counter
            )

            self._request_tracker.migrate_request(
                des_engine_id, seq_group.request_id, seq_group
            )

    def migration_done_callback(
        self,
        des_engine_id,
        des_virtual_engine_id,
        seq_group: DistserveSequenceGroup,
    ):
        migration_snapshot = seq_group.migration_snapshots[-1]
        src_engine_id = migration_snapshot.engine_id
        src_virtual_engine_id = migration_snapshot.virtual_engine_id
        seq_group.get_seqs()[0].data._stage = SequenceStage.DECODE
        seq_group.get_seqs()[0].status = SequenceStatus.RUNNING
        seq_group.metrics.migration_done_time = time.time()
        # seq_group.metrics
        des_scheduler = self.pd_engine[des_engine_id].scheduler[des_virtual_engine_id]
        des_scheduler.running.extend([seq_group])
        src_scheduler = self.pd_engine[src_engine_id].scheduler[src_virtual_engine_id]

        src_seq = seq_group.get_seqs()[0].fork(migration_snapshot.seq_id)
        src_scheduler.free_seq(src_seq)

        # print(self.waiting_for_dispatch)

    def preemption_callback(
        self, des_engine_id, des_virtual_engine_id, seq_group: DistserveSequenceGroup
    ):
        des_scheduler = self.pd_engine[des_engine_id].scheduler[des_virtual_engine_id]
        des_scheduler.waiting.remove(seq_group)

        src_engine_id, src_virtual_engine_id = self.get_engine_id_by_cost(
            end_id=self.partition_id
        )
        src_scheduler = self.pd_engine[src_engine_id].scheduler[src_virtual_engine_id]
        src_scheduler.add_seq_group(seq_group)

        seq_group.migration_snapshots.append(
            MigrateSnapshot(
                seq_id=seq_group.get_seqs()[0].seq_id,
                engine_id=src_engine_id,
                virtual_engine_id=src_virtual_engine_id,
                block_index=[],
                reason=MigrateReason.RecomputePreemption,
            )
        )

        self._request_tracker.requests_event[src_engine_id].set()

    def _init_engine(self, *args, **kwargs) -> Union[_AsyncPDEngine, "ray.ObjectRef"]:
        self._engine_class = self._pd_engine_class
        if not self.engine_use_ray:
            engine_class = self._engine_class
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
        else:
            # FIXME(woosuk): This is a bit hacky. Be careful when changing the
            # order of the arguments.
            cache_config = kwargs["cache_config"]
            parallel_config = kwargs["parallel_config"]
            if parallel_config.tensor_parallel_size == 1:
                num_gpus = cache_config.gpu_memory_utilization
            else:
                num_gpus = 1
            engine_class = ray.remote(num_gpus=num_gpus)(self._engine_class).remote
        return engine_class(*args, **kwargs)

    async def pd_engine_step(self, engine_id: int, virtual_engine_id: int) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""

        _, _, finished_requests = (
            self._request_tracker.get_new_and_migrate_and_finished_requests(engine_id)
        )

        if finished_requests:
            await self._engine_abort(finished_requests, engine_id=engine_id)

        if self.engine_use_ray:
            request_outputs, migrations = await self.pd_engine[engine_id].step.remote()  # type: ignore
        else:
            request_outputs, migrations = await self.pd_engine[engine_id].step_async(
                engine_id=engine_id,
                virtual_engine_id=virtual_engine_id,
            )

        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                request_output,
                verbose=self.log_requests,
                engine_id=engine_id,
            )

        return len(request_outputs) > 0 or len(migrations) > 0

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.errored:
            raise AsyncEngineDeadError(
                "Background loop has errored already."
            ) from self._errored_with
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        # Initialize the RequestTracker here so it uses the right event loop.
        self._request_tracker = DistserveRequestTracker(self.num_replicas)

        # dist to prefill and decode
        asyncio.get_event_loop().create_task(self.dispatch_request())
        asyncio.get_event_loop().create_task(self.collect_request())
        for i in range(self.num_replicas):
            self._background_loop_unshielded[i] = asyncio.get_event_loop().create_task(
                self.run_pd_engine_loop(i)
            )
            self._background_loop_unshielded[i].add_done_callback(
                partial(
                    _log_task_completion,
                    error_callback=partial(
                        self._error_callback,
                        engine_id=i,
                    ),
                )
            )
            self.background_loop[i] = asyncio.shield(
                self._background_loop_unshielded[i]
            )

    @classmethod
    def from_engine_args(
        cls,
        sgir_distserve_args: SGIRDistserveArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "AsyncDistserveEngine":
        """Creates an async LLM engine from the engine arguments."""
        sgir_distserve_config = sgir_distserve_args.create_sgir_distserve_config()

        for replica_config in sgir_distserve_config.replica_config:
            initialize_ray_cluster(replica_config.engine_config.parallel_config)

        # Create the async LLM engine.
        engine = cls(
            sgir_distserve_config=sgir_distserve_config,
            log_requests=not sgir_distserve_args.disable_log_requests,
            log_stats=not sgir_distserve_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=None,
        )
        return engine

    async def _engine_abort(
        self,
        request_ids: Iterable[str],
        engine_id,
    ):
        engine = self.pd_engine[engine_id]
        if self.engine_use_ray:
            await engine.abort_request.remote(request_ids)  # type: ignore
        else:
            engine.abort_request(request_ids)

    async def run_pd_engine_loop(self, engine_id: int):
        if self.engine_use_ray:
            pipeline_parallel_size = 1  # type: ignore
        else:
            pipeline_parallel_size = self.pd_engine[
                engine_id
            ].parallel_config.pipeline_parallel_size
        has_requests_in_progress = [False] * pipeline_parallel_size
        while True:
            if not any(has_requests_in_progress):
                logger.debug("Waiting for new requests...")
                # Stop the execute model loop in parallel workers until there
                # are more requests to process. This avoids waiting
                # indefinitely in torch.distributed ops which may otherwise
                # timeout, and unblocks the RPC thread in the workers so that
                # they can process any other queued control plane messages,
                # such as add/remove lora adapters.
                if self.engine_use_ray:
                    await self.pd_engine[
                        engine_id
                    ].stop_remote_worker_execution_loop.remote()  # type: ignore
                else:
                    await self.pd_engine[
                        engine_id
                    ].stop_remote_worker_execution_loop_async()
                await self._request_tracker.wait_for_requests(engine_id=engine_id)
                logger.debug("Got new requests!")
                requests_in_progress = [
                    asyncio.create_task(
                        self.pd_engine_step(engine_id=engine_id, virtual_engine_id=ve)
                    )
                    for ve in range(pipeline_parallel_size)
                ]
                has_requests_in_progress = [True] * pipeline_parallel_size

            # Abort if iteration takes too long due to unrecoverable errors
            # (eg. NCCL timeouts).
            try:
                async with asyncio_timeout(ENGINE_ITERATION_TIMEOUT_S):
                    done, _ = await asyncio.wait(
                        requests_in_progress, return_when=asyncio.FIRST_COMPLETED
                    )
                    for _ in range(pipeline_parallel_size):
                        await asyncio.sleep(0)
                for task in done:
                    result = task.result()
                    virtual_engine = requests_in_progress.index(task)
                    if self.engine_use_ray:
                        has_unfinished_requests = await self.pd_engine[
                            engine_id
                        ].has_unfinished_requests_for_virtual_engine.remote(
                            virtual_engine
                        )
                    else:
                        has_unfinished_requests = self.pd_engine[
                            engine_id
                        ].has_unfinished_requests_for_virtual_engine(virtual_engine)
                    if result or has_unfinished_requests:
                        requests_in_progress[virtual_engine] = asyncio.create_task(
                            self.pd_engine_step(
                                engine_id=engine_id,
                                virtual_engine_id=virtual_engine,
                            )
                        )
                        has_requests_in_progress[virtual_engine] = True
                    else:
                        has_requests_in_progress[virtual_engine] = False
            except asyncio.TimeoutError as exc:
                logger.error("Engine iteration timed out. This should never happen!")
                self.set_errored(exc)
                raise
            await asyncio.sleep(0)

    async def get_model_config(self, engine_id=0) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.pd_engine[engine_id].get_model_config.remote()
        else:
            return self.pd_engine[engine_id].get_model_config()

    async def do_log_stats(self) -> None:
        if self.engine_use_ray:
            await asyncio.gather(
                *([pd_engine.do_log_stats.remote() for pd_engine in self.pd_engine])
            )  # type: ignore
        else:
            for pd_engine in self.pd_engine:
                pd_engine.do_log_stats()

    async def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        t = time.perf_counter()
        logger.debug("Starting health check...")
        if self.is_stopped:
            raise AsyncEngineDeadError("Background loop is stopped.")

        if self.engine_use_ray:
            try:
                await asyncio.gather(
                    [pd_engine.check_health.remote() for pd_engine in self.pd_engine]
                )
            except ray.exceptions.RayActorError as e:
                raise RuntimeError("Engine is dead.") from e
        else:
            await asyncio.gather(
                [pd_engine.check_health_async() for pd_engine in self.pd_engine]
            )
        logger.debug(f"Health check took {time.perf_counter()-t}s")

    async def get_decoding_config(self, engine_id=0) -> DecodingConfig:
        """Get the decoding configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.pd_engine[engine_id].get_decoding_config.remote()
        else:
            return self.pd_engine[engine_id].get_decoding_config()

    async def is_tracing_enabled(self, engine_id=0) -> bool:
        if self.engine_use_ray:
            return await self.pd_engine[engine_id].is_tracing_enabled.remote()
        else:
            return self.pd_engine[engine_id].is_tracing_enabled()

    @property
    def is_running(self) -> bool:
        return (
            self.background_loop is not None
            and all([x is not None for x in self._background_loop_unshielded])
            and not all([x.done() for x in self._background_loop_unshielded])
        )

    @property
    def is_stopped(self) -> bool:
        return self.errored or (
            (
                self.background_loop is not None
                and self._background_loop_unshielded is not None
                and self._background_loop_unshielded.done()
            )
        )

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> "PreTrainedTokenizer":
        if self.engine_use_ray:
            return await self.pd_engine[0].get_tokenizer.remote(  # type: ignore
                lora_request
            )

        return (
            await self.pd_engine[0]
            .get_tokenizer_group()
            .get_lora_tokenizer_async(lora_request)
        )
