from typing import Dict, List

import torch
from sgir_distserve import _C
from sgir_distserve.migration_request import CudaHandlerConfig
from torch._tensor import Tensor

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import is_pin_memory_available, STR_DTYPE_TO_TORCH_DTYPE
from vllm.worker.cache_engine import CacheEngine

logger = init_logger(__name__)


class DistserveCacheEngine(CacheEngine):
    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
        rank: int,
    ) -> None:
        super().__init__(
            cache_config=cache_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device_config=device_config,
        )

        self.rank = rank

        self.engine_id: int
        self.migration_ptr: int
        self.gpu_cache_handlers: List[CudaHandlerConfig] = []

    def _init_cache_handler(self):
        for gpu_cache in self.gpu_cache:
            (
                device,
                handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ) = gpu_cache.untyped_storage()._share_cuda_()
            self.gpu_cache_handlers.append([handle, storage_offset_bytes])

    def get_migration_handlers(self):
        return [handler[0] for handler in self.gpu_cache_handlers]

    def get_migration_offsets(self):
        return [handler[1] for handler in self.gpu_cache_handlers]

    def register_handlers(self, engine_id: int, migration_manager_ptr: int):
        self.engine_id = engine_id
        self._manager_ptr = migration_manager_ptr

    def migrate(self, blocks_to_migration: torch.Tensor):
        _C.ops.migrate(
            self._manager_ptr, self.gpu_cache, blocks_to_migration, self.rank
        )

    def _allocate_kv_cache(self, num_blocks: int, device: str) -> List[Tensor]:
        """Allocates KV cache on the specified device."""
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        if device == "cuda":
            kv_cache_shape = (
                self.num_attention_layers,
            ) + self.attn_backend.get_kv_cache_shape(
                num_blocks, self.block_size, self.num_kv_heads, self.head_size
            )
            kv_cache = torch.empty(
                kv_cache_shape, dtype=self.dtype, pin_memory=pin_memory, device=device
            )
            kv_cache = [kv_cache[i] for i in range(self.num_attention_layers)]
        else:
            kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                num_blocks, self.block_size, self.num_kv_heads, self.head_size
            )
            kv_cache: List[torch.Tensor] = []
            for _ in range(self.num_attention_layers):
                kv_cache.append(
                    torch.empty(
                        kv_cache_shape,
                        dtype=self.dtype,
                        pin_memory=pin_memory,
                        device=device,
                    )
                )
        return kv_cache
