import os
import signal
import time
from typing import Dict, List, Optional, Union

import numpy as np

import zmq
from prometheus_client import start_http_server

from vllm.config import EngineConfig
from vllm.executor.ray_utils import initialize_ray_cluster
from vllm.outputs import RequestOutput

from vllm.sequence import SequenceStage, SequenceStatus
from vllm.usage.usage_lib import UsageContext

from sgir_distserve import _C
from sgir_distserve.core.global_scheduler import ScheduleState
from sgir_distserve.core.pd_scheduler import PDScheduler, PDSchedulerOutputs

from sgir_distserve.engine.arg_utils import SGIRDistserveArgs
from sgir_distserve.engine.distserve_metrics import (
    DistserveLoggingStatLogger,
    DistservePrometheusStatLogger,
    DistserveStats,
    SGIR_DISTSERVE_LABEL,
)
from sgir_distserve.engine.pd_engine import PDEngine
from sgir_distserve.engine.pd_profiler import PDProfiler
from sgir_distserve.ipc import (
    DispatchRequest,
    FeedbackRequest,
    IPC_INPUT_EXT,
    IPC_OUTPUT_EXT,
    MessageKind,
    ProfileKind,
    ProfileRequest,
)
from sgir_distserve.sequence import DistserveSequenceGroup, DistserveSequenceStage

_LOCAL_LOGGING_INTERVAL_SEC = 1


class DistIPCEngine:
    def __init__(
        self,
        engine_id: int,
        engine_config: EngineConfig,
        ipc_path: str,
        rnccl_comm_unique_id: List[int],
        dummy_prefill=False,
        prometheus_port: Optional[int] = None,
    ):
        # ipc config
        self.ctx = zmq.Context()
        if engine_id == 0:
            print(ipc_path)

        self.input_socket = self.ctx.socket(zmq.PULL)
        self.input_socket.bind(f"{ipc_path}{IPC_INPUT_EXT}")

        self.output_socket = self.ctx.socket(zmq.PUSH)
        self.output_socket.connect(f"{ipc_path}{IPC_OUTPUT_EXT}")

        self.rnccl_comm_unique_id = rnccl_comm_unique_id

        ray_address = os.getenv("RAY_ADDRESS", None)
        initialize_ray_cluster(engine_config.parallel_config, ray_address=ray_address)
        if prometheus_port:
            start_http_server(prometheus_port)

        self.engine_id = engine_id
        kwargs = engine_config.to_dict()

        self.stat_loggers = {
            "logging": DistserveLoggingStatLogger(
                local_interval=_LOCAL_LOGGING_INTERVAL_SEC
            ),
            "prometheus": DistservePrometheusStatLogger(
                local_interval=0.5,
                labels={
                    0: {
                        SGIR_DISTSERVE_LABEL.MODEL_NAME: engine_config.model_config.served_model_name,
                        SGIR_DISTSERVE_LABEL.INSTANCE_ID_LABEL: 0,
                    }
                },
                max_model_len=engine_config.model_config.max_model_len,
            ),
        }
        kwargs.update(
            {
                "engine_id": engine_id,
                "state_callback": self.send_schedule_state_callback,
                "dummy_prefill": dummy_prefill,
                "stat_loggers": self.stat_loggers,
            }
        )

        self.llm_engine = PDEngine(**kwargs)

        self.llm_engine._initialize_engine_id()
        self.llm_engine._initialize_kv_caches_handlers()

        self.submit_handlers()
        self.register_handlers()

        self.llm_engine._initialize_nccl_kv_cache_handlers(self.rnccl_comm_unique_id)

        self.pd_profiler = PDProfiler(
            model=self.llm_engine.model_config.model,
            chunk=(
                self.llm_engine.scheduler[0].scheduler_config.max_num_batched_tokens
                if self.llm_engine.scheduler[0].scheduler_config.chunked_prefill_enabled
                else -1
            ),
        )

    def submit_handlers(self):
        self.output_socket.send_pyobj(self.llm_engine.get_handlers())

    def register_handlers(self):
        handlers = self.input_socket.recv_pyobj()
        self.llm_engine.register_handlers(self.engine_id, handlers)

    def handle_new_input(self):
        while self.input_socket.poll(timeout=0) != 0:
            request: DispatchRequest = self.input_socket.recv_pyobj()
            request_kind = request.kind
            if isinstance(request_kind, MessageKind):
                seq_group = request.dispatch_message
                scheduler = self.llm_engine.scheduler[0]
            if request_kind == MessageKind.Run:
                seq_group.get_seqs()[0].seq_id = next(self.llm_engine.seq_counter)
                if not scheduler.slo_aware_chunked_preemption:
                    scheduler.add_seq_group(seq_group)
                else:
                    scheduler.waiting.appendleft(seq_group)
            elif request_kind == MessageKind.Migration:
                migration_snapshot = seq_group.migration_snapshots[-1]
                if migration_snapshot.engine_id == self.llm_engine.engine_id:
                    seq_group.get_seqs()[0].status = SequenceStatus.RUNNING
                    seq_group.get_seqs()[0].data._stage = SequenceStage.DECODE
                    scheduler.running.append(seq_group)
                else:
                    seq_group.get_seqs()[0].data._stage = DistserveSequenceStage.MIGRATE
                    seq_group.get_seqs()[0].status = SequenceStatus.WAITING
                    seq_group.get_seqs()[0].seq_id = next(self.llm_engine.seq_counter)
                    scheduler.add_seq_group_to_waiting_for_migration(seq_group)
            elif request_kind == MessageKind.Free:
                migration_snapshot = seq_group.migration_snapshots[-1]
                src_seq = seq_group.get_seqs()[0].fork(migration_snapshot.seq_id)
                scheduler.free_seq(src_seq)
            else:
                raise AttributeError

    def send_outputs(
        self,
        outputs: List[RequestOutput],
        migrations: List[DistserveSequenceGroup],
        frees: List[DistserveSequenceGroup],
    ):
        if outputs:
            request = FeedbackRequest(kind=MessageKind.Run, feedback_message=outputs)
            self.output_socket.send_pyobj(request)
            if not self.llm_engine.scheduler[0].get_num_unfinished_seq_groups() > 0:
                schedule_state = ScheduleState(isRunning=False, waiting_for_dispatch=[])
                request = FeedbackRequest(
                    kind=MessageKind.State, feedback_message=schedule_state
                )
                self.output_socket.send_pyobj(request)
        if migrations:
            request = FeedbackRequest(
                kind=MessageKind.Migration, feedback_message=migrations
            )
            self.output_socket.send_pyobj(request)
        if frees:
            request = FeedbackRequest(kind=MessageKind.Free, feedback_message=frees)
            self.output_socket.send_pyobj(request)

    def send_schedule_state_callback(self, scheduler_output: PDSchedulerOutputs):
        if self.pd_profiler.indb:
            seq_group = scheduler_output.scheduled_seq_groups[0].seq_group
            now = time.time()
            chunk_time = self.pd_profiler.get_chunk_seq_prediction(seq_group)
            num_computed_tokens = seq_group.get_seqs()[0].data.get_num_computed_tokens()
            chunk_idx = np.floor(
                num_computed_tokens
                / self.llm_engine.scheduler_config.max_num_batched_tokens
            )
            schedule_state = ScheduleState(
                isRunning=True,
                begin_time=now,
                predicted_end_time=now + chunk_time,
                chunk_idx=chunk_idx,
                duration=chunk_time,
                seq_group_pred=[
                    self.pd_profiler.get_prefill_seq_prediction(seq_group)
                    for seq_group in (
                        self.llm_engine.scheduler[0].running
                        + self.llm_engine.scheduler[0].waiting
                    )
                    if seq_group.is_prefill()
                ],
                seq_group_slack=[
                    self.pd_profiler.get_prefill_slack(seq_group, now=now)
                    for seq_group in (
                        self.llm_engine.scheduler[0].running
                        + self.llm_engine.scheduler[0].waiting
                    )
                    if seq_group.is_prefill()
                ],
                waiting_for_dispatch=[],
            )
            request = FeedbackRequest(
                kind=MessageKind.State, feedback_message=schedule_state
            )
            self.output_socket.send_pyobj(request)

    def run_engine_step_loop(self):
        while True:
            self.handle_new_input()
            outputs, migration_to_seq_groups, migration_done_seq_groups = (
                self.llm_engine.step()
            )
            self.send_outputs(
                outputs, migration_to_seq_groups, migration_done_seq_groups
            )


def run_mp_engine(
    engine_id: int,
    engine_args: SGIRDistserveArgs,
    usage_context: UsageContext,
    ipc_path: str,
    rnccl_comm_unique_id: List[int],
    prometheus_port: Optional[int] = None,
):
    if engine_id >= engine_args.partition_id:
        engine_args.slo_aware_chunked_preemption = False
    sgir_distserve_config = engine_args.create_sgir_distserve_config()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm
        raise KeyboardInterrupt("MQLLMEngine terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    ipc_engine = DistIPCEngine(
        engine_id,
        sgir_distserve_config.replica_config[engine_id].engine_config,
        ipc_path,
        rnccl_comm_unique_id,
        dummy_prefill=sgir_distserve_config.dummy_prefill,
        prometheus_port=prometheus_port,
    )
    ipc_engine.run_engine_step_loop()


if __name__ == "__main__":
    SGIRDistserveArgs(model="test")
