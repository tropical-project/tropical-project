import copy
import enum
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Set, Tuple, TYPE_CHECKING, Union

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams

# vllm 5.3.0

from vllm.sequence import (
    ExecuteModelRequest,
    HiddenStates,
    PoolingParams,
    PromptAdapterRequest,
    RequestMetrics,
    Sequence,
    SequenceGroup,
    SequenceGroupMetadata,
    SequenceGroupState,
    SequenceStage,
    SequenceStatus,
)

from sgir_distserve.sampling_params import DistserveSamplingParams


class DistserveSequenceStatus(enum.Enum):
    MIGRATING = enum.auto()


class DistserveSequenceStage(enum.Enum):
    MIGRATE = enum.auto()


@dataclass
class DistserveRequestMetrics:
    arrival_time: float
    last_token_time: float
    first_scheduled_time: Optional[float]
    first_token_time: Optional[float]
    time_in_queue: Optional[float]
    decode_begin_time: Optional[float]
    migration_begin_time: Optional[float]
    migration_done_time: Optional[float]
    token_by_token_time: List[float]
    chunk_by_chunk_time: List[float]
    interference_time: List[float]
    interference_others_time: List[float]
    prefill_batch_size: int
    prefill_batched_seq_length: int
    decoding_batch_size: List[int]
    decoding_batched_seq_length: List[int]
    finished_time: Optional[float] = None


class MigrateReason(enum.Enum):
    Disaggregation = enum.auto()
    Balance = enum.auto()
    RecomputePreemption = enum.auto()


@dataclass
class MigrateSnapshot:
    seq_id: int
    engine_id: int
    virtual_engine_id: int
    block_index: int
    reason: MigrateReason


class DistserveSequenceGroup(SequenceGroup):
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
        seqs: The list of sequences.
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
        lora_request: LoRA request.
        embeddings: The embeddings vectors of the prompt of the sequence group
            for an embedding model.
        pooling_params: The pooling parameters used to generate the pooling
            for an embedding model.
        encoder_seq: Optional, the single encoder sequence. Should be None
                     unless you are working with an encoder/decoder model.
        trace_headers: OpenTelemetry trace headers.
        prompt_adapter_request: Prompt Adapter request.
    """

    metrics: DistserveRequestMetrics

    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        arrival_time: float,
        sampling_params: Optional[DistserveSamplingParams] = None,
        lora_request: Optional[LoRARequest] = None,
        embeddings: Optional[List[float]] = None,
        pooling_params: Optional[PoolingParams] = None,
        encoder_seq: Optional[Sequence] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        ttft_slo: Optional[float] = None,
        tpot_slo: Optional[float] = None,
    ) -> None:
        super().__init__(
            request_id=request_id,
            seqs=seqs,
            arrival_time=arrival_time,
            sampling_params=sampling_params,
            lora_request=lora_request,
            embeddings=embeddings,
            pooling_params=pooling_params,
            encoder_seq=encoder_seq,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
        )
        self.migration_snapshots: List[MigrateSnapshot] = []
        self.ttft_slo = ttft_slo or 10000
        self.tpot_slo = tpot_slo or 10000

    def set_token_by_token_time(self, t):
        self.metrics.token_by_token_time.append(t)

    def set_chunk_by_chunk_time(self, t):
        self.metrics.chunk_by_chunk_time.append(t)

    def set_interference_time(self, t):
        self.metrics.interference_time.append(t)

    def set_interference_others_time(self, t):
        self.metrics.interference_others_time.append(t)

    def set_prefill_batch_size(self, n):
        self.metrics.prefill_batch_size = n

    def set_prefill_batch_size(self, n):
        self.metrics.prefill_batched_seq_length = n

    def set_decoding_batch_size(self, n):
        self.metrics.decoding_batch_size.append(n)

    def set_prefill_batched_seq_length(self, n):
        self.metrics.prefill_batched_seq_length = n

    def set_decoding_batched_seq_length(self, n):
        self.metrics.decoding_batched_seq_length.append(n)

    @classmethod
    def from_seq_group(cls, seq_group: SequenceGroup, ttft_slo, tpot_slo):
        distserve_seq_group = cls(
            request_id=seq_group.request_id,
            seqs=seq_group.get_seqs(),
            arrival_time=time.time(),
            sampling_params=seq_group.sampling_params,
            lora_request=seq_group.lora_request,
            embeddings=seq_group.embeddings,
            pooling_params=seq_group.pooling_params,
            encoder_seq=seq_group.encoder_seq,
            trace_headers=seq_group.trace_headers,
            prompt_adapter_request=seq_group.prompt_adapter_request,
            ttft_slo=ttft_slo,
            tpot_slo=tpot_slo,
        )
        seq_group_metrics = seq_group.metrics
        distserve_seq_group.metrics = DistserveRequestMetrics(
            arrival_time=seq_group_metrics.arrival_time,
            last_token_time=seq_group_metrics.last_token_time,
            first_scheduled_time=seq_group_metrics.first_scheduled_time,
            first_token_time=seq_group_metrics.first_token_time,
            decode_begin_time=time.time(),
            migration_begin_time=time.time(),
            migration_done_time=time.time(),
            time_in_queue=seq_group_metrics.time_in_queue,
            token_by_token_time=[],
            chunk_by_chunk_time=[],
            interference_time=[],
            interference_others_time=[],
            prefill_batched_seq_length=0,
            prefill_batch_size=0,
            decoding_batch_size=[],
            decoding_batched_seq_length=[],
            finished_time=seq_group_metrics.finished_time,
        )
        return distserve_seq_group


@dataclass
class DistserveExecuteModelRequest:
    """The model execution request, containing CPU metadata only. The LLM
    engine should create an instance of this class for each request batch."""

    # The sequence group metadata list.
    seq_group_metadata_list: List[SequenceGroupMetadata]
    # Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: List[Tuple[int, int]] = field(default_factory=list)
    # Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: List[Tuple[int, int]] = field(default_factory=list)
    # Blocks to copy. Source to dest block.
    blocks_to_copy: List[Tuple[int, int]] = field(default_factory=list)

    # patch >>>
    # Blocks to migration. prefill instance to decode instance
    blocks_to_migration: List[Tuple[int, int, int]] = field(default_factory=list)
    # <<< patch

    # Virtual engine ID for pipeline parallel.
    virtual_engine: int = 0
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int = 0
    # The number of requests in the running queue.
    running_queue_size: int = 0
    # Optional hidden states from prior step.
    previous_hidden_states: Optional[HiddenStates] = None
    # The number of forward steps to run.
    num_steps: int = 1
    # Finished request ids since last step.
    finished_requests_ids: List[str] = field(default_factory=list)

    def clone(
        self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> "ExecuteModelRequest":
        """Clone the request with a new sequence group metadata list."""
        return DistserveExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=self.blocks_to_swap_in.copy(),
            blocks_to_swap_out=self.blocks_to_swap_out.copy(),
            blocks_to_copy=self.blocks_to_copy.copy(),
            blocks_to_migration=self.blocks_to_migration.copy(),
            virtual_engine=self.virtual_engine,
            num_lookahead_slots=self.num_lookahead_slots,
            running_queue_size=self.running_queue_size,
            previous_hidden_states=self.previous_hidden_states,
            num_steps=self.num_steps,
            finished_requests_ids=self.finished_requests_ids,
        )
