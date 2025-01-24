"""A GPU worker class."""

import dataclasses
import functools
import gc
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.distributed
import torch.profiler

from sgir_distserve import _C
from sgir_distserve.env import SGIR_DISTSERVE_RESULT_PATH
from sgir_distserve.migration_request import CudaHandlerConfig, KVCacheHandlerConfig
from sgir_distserve.sequence import DistserveExecuteModelRequest
from sgir_distserve.worker.cache_engine import DistserveCacheEngine

from vllm.attention import AttentionBackend

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    LoRAConfig,
    ModelConfig,
    MultiModalConfig,
    ParallelConfig,
    PromptAdapterConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm.distributed import (
    broadcast_tensor_dict,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.parallel_state import get_pp_group

from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.sequence import IntermediateTensors, SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import GPUModelRunnerBase
from vllm.worker.worker import Worker
from vllm.worker.worker_base import ModelRunnerInputBase


logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class DistserveWorkerInput:
    """Local inputs to each worker. May contain device-specific data. These
    fields should be broadcastable to other workers.
    """

    num_seq_groups: Optional[int] = None
    blocks_to_swap_in: Optional[torch.Tensor] = None
    blocks_to_swap_out: Optional[torch.Tensor] = None
    blocks_to_copy: Optional[torch.Tensor] = None
    blocks_to_migration: Optional[torch.Tensor] = None
    virtual_engine: int = 0

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type["DistserveWorkerInput"],
        tensor_dict: Dict[str, Any],
    ) -> "DistserveWorkerInput":
        """
        Pop fields from the given tensor_dict and populate a new instance of
        WorkerInput.
        """
        return cls(
            num_seq_groups=tensor_dict.pop("num_seq_groups"),
            blocks_to_swap_in=tensor_dict.pop("blocks_to_swap_in"),
            blocks_to_swap_out=tensor_dict.pop("blocks_to_swap_out"),
            blocks_to_copy=tensor_dict.pop("blocks_to_copy"),
            blocks_to_migration=tensor_dict.pop("blocks_to_migration"),
            virtual_engine=tensor_dict["virtual_engine"],
        )

    def as_broadcastable_tensor_dict(self) -> Dict[str, Union[int, torch.Tensor]]:
        """
        Extract broadcastable fields.
        """
        tensor_dict = {
            "num_seq_groups": self.num_seq_groups,
            "blocks_to_swap_in": self.blocks_to_swap_in,
            "blocks_to_swap_out": self.blocks_to_swap_out,
            "blocks_to_copy": self.blocks_to_copy,
            "blocks_to_migration": self.blocks_to_migration,
            "virtual_engine": self.virtual_engine,
        }

        return tensor_dict


def _init_attn_metadata_from_tensor_dict(
    attn_backend: "AttentionBackend",
    tensor_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Helper method to initialize AttentionMetadata based on an
    AttentionBackend and broadcastable AttentionMetadata fields.
    """
    # Extract the fields used to create AttentionMetadata.
    valid_attn_kwargs = {}
    for field in dataclasses.fields(attn_backend.get_metadata_cls()):
        val = tensor_dict.pop(field.name, None)
        if val is not None:
            valid_attn_kwargs[field.name] = val

    attn_metadata = attn_backend.make_metadata(**valid_attn_kwargs)
    tensor_dict["attn_metadata"] = attn_metadata
    return tensor_dict


def trace_handler(prof, engine_id=0, rank=0):
    now = time.time()
    print("dumping trace file [Start]")
    prof.export_chrome_trace(
        f"{SGIR_DISTSERVE_RESULT_PATH}/profile/engine_{engine_id}_rank_{rank}_{now}.json"
    )
    print("dumping trace file [Done]")


class PDWorker(Worker):
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        multimodal_config: Optional[MultiModalConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
    ) -> None:
        super().__init__(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config,
            local_rank,
            rank,
            distributed_init_method,
            lora_config,
            multimodal_config,
            speculative_config,
            prompt_adapter_config,
            is_driver_worker,
            model_runner_cls,
        )

        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.engine_id: int = 0
        self._manager_ptr: int
        self.cache_engine: List[DistserveCacheEngine]
        self.gpu_cache: Optional[List[List[torch.Tensor]]]
        self.gpu_cache_handlers = []
        self.prof: torch.profiler.profile = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=5, active=75),
            on_trace_ready=functools.partial(
                trace_handler, engine_id=self.engine_id, rank=self.rank
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

        self.rnccl_comm_unique_id: List[int]
        self.rnccl_comm: _C.RNCCLComm
        self.handler_config: Dict[int, KVCacheHandlerConfig]

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = [
            DistserveCacheEngine(
                self.cache_config,
                self.model_config,
                self.parallel_config,
                self.device_config,
                rank=self.rank,
            )
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.gpu_cache = [
            self.cache_engine[ve].gpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]

    def _init_cache_handler(self):
        for engine in self.cache_engine:
            engine._init_cache_handler()

    def _init_nccl_cache_handler(self, unique_id):
        self.rnccl_comm_unique_id = unique_id

        world_size = 0
        for config in self.handler_config.values():
            world_size += config.pp_size * config.tp_size

        rank_base = 0
        for idx in range(self.engine_id):
            config = self.handler_config[idx]
            rank_base += config.tp_size * config.pp_size
        self.rnccl_comm = _C.RNCCLComm(unique_id, world_size, rank_base + self.rank)

    def _init_engine_id(self, engine_id: int):
        self.engine_id = engine_id

    def register_handlers(self, engine_id: int, handlers: List[KVCacheHandlerConfig]):
        self.handler_config = {
            engine_id: handler for engine_id, handler in enumerate(handlers)
        }
        self.engine_id = engine_id
        _C_handlers = []
        for idx, handle in enumerate(handlers):
            if idx == engine_id:
                _C_handle = _C.KVCacheHandlerConfig(
                    handle.tp_size,
                    handle.pp_size,
                    handle.num_gpu_blocks,
                    handle.block_size,
                )
            else:
                _C_handle = _C.KVCacheHandlerConfig(
                    handle.tp_size,
                    handle.pp_size,
                    handle.num_gpu_blocks,
                    handle.block_size,
                    handle.kv_cache_handlers,
                    handle.kv_cache_offsets,
                )

            _C_handlers.append(_C_handle)

        model_config = self.model_config
        parallel_config = self.parallel_config
        num_bytes_per_elem = model_config.dtype.itemsize
        num_layers = (
            model_config.get_num_layers(parallel_config)
            * parallel_config.pipeline_parallel_size
        )
        head_size = model_config.get_head_size()
        total_num_kv_heads = model_config.get_total_num_kv_heads()
        migrate_ptr = _C.ops.init_migration_manager(
            self.engine_id,
            num_bytes_per_elem,
            num_layers,
            total_num_kv_heads,
            head_size,
            _C_handlers,
        )
        for engine in self.cache_engine:
            engine.register_handlers(self.engine_id, migrate_ptr)

    def get_migration_handlers(self):
        return [engine.get_migration_handlers() for engine in self.cache_engine]

    def get_migration_offsets(self):
        return [engine.get_migration_offsets() for engine in self.cache_engine]

    def get_nccl_handlers(self):
        return self.pd_rank

    @torch.inference_mode()
    def prepare_worker_input(
        self, execute_model_req: DistserveExecuteModelRequest
    ) -> DistserveWorkerInput:
        virtual_engine = execute_model_req.virtual_engine
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(
            execute_model_req.blocks_to_swap_in, device="cpu", dtype=torch.int64
        ).view(-1, 2)
        blocks_to_swap_out = torch.tensor(
            execute_model_req.blocks_to_swap_out, device="cpu", dtype=torch.int64
        ).view(-1, 2)
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(
            execute_model_req.blocks_to_copy, device=self.device, dtype=torch.int64
        ).view(-1, 2)

        blocks_to_migration = torch.tensor(
            execute_model_req.blocks_to_migration, device=self.device, dtype=torch.int64
        ).view(-1, 4)

        return DistserveWorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
            blocks_to_migration=blocks_to_migration,
        )

    @torch.inference_mode()
    def execute_worker(self, worker_input: DistserveWorkerInput) -> None:
        virtual_engine = worker_input.virtual_engine
        # Issue cache operations.
        if (
            worker_input.blocks_to_swap_in is not None
            and worker_input.blocks_to_swap_in.numel() > 0
        ):
            self.cache_engine[virtual_engine].swap_in(worker_input.blocks_to_swap_in)
        if (
            worker_input.blocks_to_swap_out is not None
            and worker_input.blocks_to_swap_out.numel() > 0
        ):
            self.cache_engine[virtual_engine].swap_out(worker_input.blocks_to_swap_out)
        if (
            worker_input.blocks_to_copy is not None
            and worker_input.blocks_to_copy.numel() > 0
        ):
            self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)
        if (
            worker_input.blocks_to_migration is not None
            and worker_input.blocks_to_migration.numel() > 0
        ):
            # num_blocks_to_migrate = worker_input.blocks_to_migration.size(0)
            # begin = time.time()
            self.cache_engine[virtual_engine].migrate(worker_input.blocks_to_migration)
            # end = time.time()
            # num_layers = (
            #     self.model_config.get_num_layers(self.parallel_config)
            #     * self.parallel_config.pipeline_parallel_size
            # )
            # total_bytes = (
            #     (2)
            #     * num_layers
            #     * num_blocks_to_migrate
            #     * self.cache_config.block_size
            #     * self.model_config.get_total_num_kv_heads()
            #     * self.model_config.get_head_size()
            #     * self.model_config.dtype.itemsize
            # )
            # exe_time = end - begin
            # print(f"num_blocks_to_migrate: {num_blocks_to_migrate}")
            # print(f"block_size: {self.cache_config.block_size}")
            # print(f"num_layers: {num_layers}")
            # print(f"total_bytes: {total_bytes}")
            # print(f"Bandwidth: {round(total_bytes / exe_time / 1e9, 2)}GBps")

    def execute_model(
        self, execute_model_req: Optional[DistserveExecuteModelRequest] = None
    ) -> Optional[List[SamplerOutput]]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""
        if self.is_driver_worker:
            if execute_model_req is None:
                if self.do_metadata_broadcast:
                    # This signals that there's no more requests to process for
                    # now. All workers are running infinite loop with
                    # broadcast_tensor_dict, and it stops the loop when the
                    # driver broadcasts an empty input. Send an empty input to
                    # notify all other workers to stop their execution loop.
                    broadcast_tensor_dict({}, src=0)
                return None

            worker_input: DistserveWorkerInput = self.prepare_worker_input(
                execute_model_req=execute_model_req
            )
            # patch: only migration happend
            if worker_input.num_seq_groups > 0:
                model_input: ModelRunnerInputBase = (
                    self.model_runner.prepare_model_input(
                        execute_model_req.seq_group_metadata_list,
                        execute_model_req.virtual_engine,
                        execute_model_req.finished_requests_ids,
                    )
                )
            num_steps = execute_model_req.num_steps

            if self.do_metadata_broadcast:
                broadcast_data = worker_input.as_broadcastable_tensor_dict()
                if worker_input.num_seq_groups > 0:
                    broadcast_data.update(model_input.as_broadcastable_tensor_dict())
                broadcast_data["num_steps"] = num_steps
                broadcast_tensor_dict(broadcast_data, src=0)
        else:
            assert self.do_metadata_broadcast
            broadcast_data = broadcast_tensor_dict(src=0)
            if not broadcast_data:
                return None

            num_steps = broadcast_data.pop("num_steps")
            worker_input = DistserveWorkerInput.from_broadcasted_tensor_dict(
                broadcast_data
            )
            if worker_input.num_seq_groups > 0:
                model_input = (
                    self.model_runner.make_model_input_from_broadcasted_tensor_dict(
                        broadcast_data
                    )
                )

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict()
            )

        output = self.model_runner.execute_model(
            model_input,
            (
                self.kv_cache[worker_input.virtual_engine]
                if self.kv_cache is not None
                else None
            ),
            intermediate_tensors,
            num_steps,
        )

        if not get_pp_group().is_last_rank:
            # output is IntermediateTensors
            get_pp_group().send_tensor_dict(output.tensors)
            return [None]

        torch.cuda.synchronize()
        # self.prof.step()

        # output is List[SamplerOutput]
        return output
