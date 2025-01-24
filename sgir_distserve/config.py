import dataclasses
import enum
from typing import List, Optional

from vllm.config import EngineConfig, ModelConfig, SchedulerConfig

from sgir_distserve.core.global_scheduler import (
    DispatchPolicy,
    EngineKind,
    SchedulePolicy,
)


@dataclasses.dataclass
class ReplicaConfig:
    kind: EngineKind
    engine_config: EngineConfig
    prometheus_port: Optional[int] = None


@dataclasses.dataclass
class BatchConfig:
    max_num_seqs: int
    max_num_batched_tokens: int
    enable_chunked_prefill: bool = False


@dataclasses.dataclass(frozen=True)
class SGIRDistserveConfig:
    partition_id: int

    dispatch_policy: DispatchPolicy
    schedule_policy: SchedulePolicy

    block_size: int
    max_model_len: int

    dummy_prefill: bool

    slo_aware_multiplexing: bool
    max_multiplexing_length: int
    multiplexing_decode_batch_latency_watermark: float
    multiplexing_decode_batched_tokens_watermark: int

    prefill_engine_batch_config: Optional[BatchConfig]
    decode_engine_batch_config: Optional[BatchConfig]

    replica_config: List[ReplicaConfig]


class SGIRDistserveSchedulerConfig(SchedulerConfig):
    def __init__(
        self,
        max_num_batched_tokens: int | None,
        max_num_seqs: int,
        max_model_len: int,
        use_v2_block_manager: bool = False,
        num_lookahead_slots: int = 0,
        delay_factor: float = 0,
        enable_chunked_prefill: bool = False,
        embedding_mode: bool | None = False,
        preemption_mode: str | None = None,
        slo_aware_chunked_preemption: bool = False,
        schedule_policy: SchedulePolicy = SchedulePolicy.SJF,
    ) -> None:
        super().__init__(
            max_num_batched_tokens,
            max_num_seqs,
            max_model_len,
            use_v2_block_manager,
            num_lookahead_slots,
            delay_factor,
            enable_chunked_prefill,
            embedding_mode,
            preemption_mode,
        )
        self.slo_aware_chunked_preemption = slo_aware_chunked_preemption
        self.schedule_policy = schedule_policy
