from typing import List

from pydantic import BaseModel


class CudaHandlerConfig(BaseModel):
    device: int
    handle: bytes
    storage_size_bytes: int
    storage_offset_bytes: int
    ref_counter_handle: bytes
    ref_counter_offset: int
    event_handle: bytes
    event_sync_required: bool
    shape: List[int]
    stride: List[int]


class KVCacheHandlerConfig(BaseModel):
    engine_id: int
    tp_size: int
    pp_size: int
    num_gpu_blocks: int
    block_size: int
    kv_cache_handlers: List[List[List[bytes]]]
    kv_cache_offsets: List[List[List[int]]]


class KVCacheHandlerRegisterConfig(BaseModel):
    handlers: List[KVCacheHandlerConfig]
