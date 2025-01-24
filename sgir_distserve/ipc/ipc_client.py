import asyncio
import sys
import time
from typing import Dict, List

import zmq
import zmq.asyncio

from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoTokenizer,
    BatchEncoding,
)
from vllm.config import DecodingConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine, RequestTracker
from vllm.engine.llm_engine import _load_generation_config_dict

# from vllm.engine.llm_engine import _LOCAL_LOGGING_INTERVAL_SEC
from vllm.inputs.data import LLMInputs
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sequence import Sequence, SequenceGroup

from vllm.tracing import init_tracer
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import Counter

from sgir_distserve import _C

from sgir_distserve.core.global_scheduler import GlobalScheduler
from sgir_distserve.engine.arg_utils import SGIRDistserveArgs
from sgir_distserve.engine.distserve_metrics import (
    DistserveLoggingStatLogger,
    DistservePrometheusStatLogger,
    SGIR_DISTSERVE_LABEL,
)
from sgir_distserve.engine.distserve_request_tracker import DistserveRequestTracker
from sgir_distserve.ipc import (
    DispatchRequest,
    FeedbackRequest,
    IPC_INPUT_EXT,
    IPC_OUTPUT_EXT,
    MessageKind,
    ProfileKind,
    ProfileRequest,
)

from sgir_distserve.sampling_params import DistserveSamplingParams
from sgir_distserve.sequence import DistserveSequenceGroup

_LOCAL_LOGGING_INTERVAL_SEC = 1
logger = init_logger("vllm.engine.ipc_distserve_engine")


class DispatchTracker:
    def __init__(self, scheduler: GlobalScheduler):
        self.global_scheduler = scheduler
        self.exec_request_event = asyncio.Event()
        self.migration_request_event = asyncio.Event()
        self.free_request_event = asyncio.Event()

    async def wait_for_exec_request(self):
        if not self.global_scheduler.waiting:
            await self.exec_request_event.wait()
        self.exec_request_event.clear()

    async def wait_for_migration_request(self):
        if not self.global_scheduler.waiting_for_migration:
            await self.migration_request_event.wait()
        self.migration_request_event.clear()

    async def wait_for_free_request(self):
        if not self.global_scheduler.waiting_for_free:
            await self.free_request_event.wait()
        self.free_request_event.clear()


class IPCEngineClient(AsyncLLMEngine):
    def __init__(
        self,
        sgir_distserve_args: SGIRDistserveArgs,
        ipc_path_map: Dict[int, str],
        log_requests=True,
    ):
        self.log_requests = log_requests

        self.serving_config = sgir_distserve_args.create_sgir_distserve_config()

        engine_config = self.serving_config.replica_config[0].engine_config
        self.global_stat_loggers = {
            "logging": DistserveLoggingStatLogger(
                local_interval=_LOCAL_LOGGING_INTERVAL_SEC
            ),
            "prometheus": DistservePrometheusStatLogger(
                local_interval=0.5,
                labels={
                    SGIR_DISTSERVE_LABEL.MODEL_NAME: engine_config.model_config.served_model_name,
                    SGIR_DISTSERVE_LABEL.INSTANCE_ID_LABEL: -1,
                },
                max_model_len=engine_config.model_config.max_model_len,
            ),
        }
        self.global_stat_loggers["prometheus"].info(
            "cache_config",
            self.serving_config.replica_config[0].engine_config.cache_config,
        )

        # Tokenizer
        self.tokenizer = get_tokenizer(
            self.model_config.model,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        partition_id = self.serving_config.partition_id
        self.p_set = {i for i in range(partition_id)}
        self.d_set = {i for i in range(partition_id, self.num_replica)}

        self.ctx = zmq.asyncio.Context()
        self.engine_dispatch: Dict[int, str] = {}
        self.engine_feedback: Dict[int, str] = {}
        for idx, ipc_path in ipc_path_map.items():
            engine_dispatch = self.ctx.socket(zmq.PUSH)
            engine_dispatch.connect(f"{ipc_path}{IPC_INPUT_EXT}")
            self.engine_dispatch[idx] = engine_dispatch
            engine_feedback = self.ctx.socket(zmq.PULL)
            engine_feedback.bind(f"{ipc_path}{IPC_OUTPUT_EXT}")
            self.engine_feedback[idx] = engine_feedback

        self._request_tracker: DistserveRequestTracker
        self._dispatch_tracker: DispatchTracker

        # Global Scheduler
        self.global_scheduler = GlobalScheduler(
            self.p_set,
            self.d_set,
            model=self.driver_engine_config.model_config.model,
            chunk_size=(
                self.driver_engine_config.scheduler_config.max_num_batched_tokens
                if self.driver_engine_config.scheduler_config.chunked_prefill_enabled
                else None
            ),
            enable_preemption=self.driver_engine_config.scheduler_config.slo_aware_chunked_preemption,
            enable_multiplexing=self.serving_config.slo_aware_multiplexing,
            max_multiplexing_length=self.serving_config.max_multiplexing_length,
            dispatch_policy=self.serving_config.dispatch_policy,
            schedule_policy=self.serving_config.schedule_policy,
        )

        self.seq_counter = Counter()

        self.handler_collected = False
        self.background_loop_started = False

        self.tracer = None
        if self.driver_engine_config.observability_config.otlp_traces_endpoint:
            self.tracer = init_tracer(
                "vllm.llm_engine",
                self.driver_engine_config.observability_config.otlp_traces_endpoint,
            )

        self.unique_nccl_comm_id = _C.get_nccl_unique_id()

        self.start_background_loop()

        self.generation_config_fields = _load_generation_config_dict(
            self.driver_engine_config.model_config
        )

    @property
    def num_replica(self):
        return len(self.serving_config.replica_config)

    @property
    def is_running(self):
        return self.background_loop_started

    @property
    def driver_engine_config(self):
        return self.serving_config.replica_config[0].engine_config

    @property
    def engine_set(self):
        return self.p_set | self.d_set

    async def recv_feedback(self, i):
        while not self.handler_collected:
            await asyncio.sleep(0.5)
        g_sch = self.global_scheduler
        req_tracker = self._request_tracker
        while True:
            recv_obj: FeedbackRequest = await self.engine_feedback[i].recv_pyobj()
            if recv_obj.kind == MessageKind.Run:
                for req_output in recv_obj.feedback_message:
                    g_sch.update_dispatch(req_output)
                    req_tracker.update_stream(req_output)
            elif recv_obj.kind == MessageKind.Migration:
                for seq_group in recv_obj.feedback_message:
                    if not seq_group.is_finished():
                        g_sch.add_migration_seq_group(seq_group)
                        self._dispatch_tracker.migration_request_event.set()
            elif recv_obj.kind == MessageKind.Free:
                for seq_group in recv_obj.feedback_message:
                    g_sch.add_free_seq_group(seq_group)
                    self._dispatch_tracker.free_request_event.set()
            elif recv_obj.kind == MessageKind.State:
                self.global_scheduler.dispatch_snapshot_manager.engine_state[i] = (
                    recv_obj.feedback_message
                )

    async def collect_handler(self):
        handlers = []
        for idx in self.engine_set:
            handler = await self.engine_feedback[idx].recv_pyobj()
            handlers.append(handler)
        for idx in self.engine_set:
            self.engine_dispatch[idx].send_pyobj(handlers)
        self.handler_collected = True

    async def dispatch_exec_request(self):
        while not self.handler_collected:
            await asyncio.sleep(0.5)
        g_sch = self.global_scheduler
        while True:
            engine_id, seq_group = g_sch.dispatch()
            if engine_id != None:
                dispatch_request = DispatchRequest(
                    kind=MessageKind.Run, dispatch_message=seq_group
                )
                self.engine_dispatch[engine_id].send_pyobj(dispatch_request)
                g_sch.dispatch_snapshot_manager.engine_state[
                    engine_id
                ].waiting_for_dispatch.append(seq_group)
            await asyncio.sleep(0.001)

    async def dispatch_migration_request(self):
        while not self.handler_collected:
            await asyncio.sleep(0.5)
        g_sch = self.global_scheduler
        while True:
            await self._dispatch_tracker.wait_for_migration_request()
            engine_id, seq_group = g_sch.dispatch_migration()
            if engine_id != None:
                dispatch_request = DispatchRequest(
                    kind=MessageKind.Migration, dispatch_message=seq_group
                )
                self.engine_dispatch[engine_id].send_pyobj(dispatch_request)

    async def dispatch_free_request(self):
        g_sch = self.global_scheduler
        while not self.handler_collected:
            await asyncio.sleep(0.5)
        while True:
            await self._dispatch_tracker.wait_for_free_request()
            engine_id, seq_group = g_sch.dispatch_free()
            if engine_id != None:
                dispatch_request = DispatchRequest(
                    kind=MessageKind.Free, dispatch_message=seq_group
                )
                self.engine_dispatch[engine_id].send_pyobj(dispatch_request)

    async def collect_request(self):
        g_sch = self.global_scheduler
        total_prompts_cnt = 0
        time_to_last_event = time.time()
        record_duration = 0.5  # 5s
        while True:
            await self._request_tracker.wait_for_new_requests()
            logger.debug("get new request")
            configs, _ = self._request_tracker.get_new_and_finished_requests()
            for cfg in configs:
                seq = Sequence(
                    seq_id=next(self.seq_counter),
                    inputs=LLMInputs(**cfg["inputs"]),
                    block_size=self.driver_engine_config.cache_config.block_size,
                    eos_token_id=self.tokenizer.eos_token_id,
                    lora_request=None,
                    prompt_adapter_request=cfg["prompt_adapter_request"],
                )
                seq_group = SequenceGroup(
                    request_id=cfg["request_id"],
                    seqs=[seq],
                    arrival_time=time.time(),
                    sampling_params=cfg["params"],
                    trace_headers=cfg["trace_headers"],
                )
                params: DistserveSamplingParams = cfg["params"]
                params.update_from_generation_config(
                    self.generation_config_fields, self.tokenizer.eos_token_id
                )
                seq_group = DistserveSequenceGroup.from_seq_group(
                    seq_group, params.ttft_slo, params.tpot_slo
                )
                g_sch.add_seq_group(seq_group)
                self._dispatch_tracker.exec_request_event.set()
                total_prompts_cnt += seq_group.get_num_uncomputed_tokens()

            now = time.time()
            if now - time_to_last_event > record_duration:
                time_to_last_event = now
                prometheus_logger: DistservePrometheusStatLogger = (
                    self.global_stat_loggers["prometheus"]
                )
                prometheus_logger._log_gauge(
                    prometheus_logger.metrics.gauge_global_scheduler_num_arrived_prompt_tokens,
                    total_prompts_cnt,
                )
                print(total_prompts_cnt)
                total_prompts_cnt = 0

    def start_background_loop(self) -> None:
        self._request_tracker = DistserveRequestTracker()
        self._dispatch_tracker = DispatchTracker(self.global_scheduler)
        for i in range(self.num_replica):
            asyncio.get_event_loop().create_task(self.recv_feedback(i))
        asyncio.get_event_loop().create_task(self.collect_request())
        asyncio.get_event_loop().create_task(self.dispatch_exec_request())
        asyncio.get_event_loop().create_task(self.dispatch_migration_request())
        asyncio.get_event_loop().create_task(self.dispatch_free_request())
        asyncio.get_event_loop().create_task(self.collect_handler())
        self.background_loop_started = True

    @property
    def model_config(self):
        return self.driver_engine_config.model_config

    async def get_model_config(self):
        return self.model_config

    async def get_decoding_config(self) -> DecodingConfig:
        return DecodingConfig()

    async def get_tokenizer(self, lora_request=None):
        return self.tokenizer

    async def is_tracing_enabled(self) -> bool:
        return self.tracer is not None
