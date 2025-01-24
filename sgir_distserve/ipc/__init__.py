import enum
from dataclasses import dataclass
from typing import List, Mapping, Optional, overload, Union

from vllm import PoolingParams
from vllm.inputs import LLMInputs
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams

from sgir_distserve.core.global_scheduler import ScheduleState

from sgir_distserve.sequence import DistserveSequenceGroup

IPC_INPUT_EXT = "_input_socket"
IPC_OUTPUT_EXT = "_output_socket"


@dataclass
class RPCProcessRequest:
    prompt: LLMInputs
    params: Union[SamplingParams, PoolingParams]
    request_id: str
    lora_request: Optional[LoRARequest] = None
    trace_headers: Optional[Mapping[str, str]] = None
    prompt_adapter_request: Optional[PromptAdapterRequest] = None


class MessageKind(enum.Enum):
    # Execution Kind
    Run = enum.auto()
    Migration = enum.auto()
    Free = enum.auto()

    # State Kind
    State = enum.auto()


class ProfileKind(enum.Enum):
    Start = enum.auto()
    Step = enum.auto()
    Stop = enum.auto()


@dataclass
class ProfileRequest:
    kind: ProfileKind


@dataclass
class DispatchRequest:
    kind: MessageKind
    dispatch_message: Union[DistserveSequenceGroup,]


@dataclass
class FeedbackRequest:
    kind: MessageKind
    feedback_message: Union[
        List[RequestOutput], List[DistserveSequenceGroup], ScheduleState
    ]
