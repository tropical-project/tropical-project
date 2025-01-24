from typing import Iterable, List, Union

from vllm.engine.async_llm_engine import _AsyncLLMEngine

from vllm.logger import init_logger

from sgir_distserve import _C
from sgir_distserve.migration_request import KVCacheHandlerConfig


logger = init_logger(__name__)


class DistserveLLMEngine(_AsyncLLMEngine):
    engine_id: int

    def _initialize_kv_caches_handlers(self) -> None:
        self.model_executor._init_cache_handler()

    def _initialize_nccl_kv_cache_handlers(self, unique_id) -> None:
        self.model_executor._init_nccl_cache_handler(unique_id)

    def _initialize_engine_id(self) -> None:
        self.model_executor._init_engine_id(self.engine_id)

    def profile_start(self):
        self.model_executor.profile_start()

    def profile_step(self):
        self.model_executor.profile_step()

    def profile_stop(self):
        self.model_executor.profile_stop()

    def get_handlers(self):
        kv_cache_handlers = self.model_executor.get_migration_handlers()
        kv_cache_offsets = self.model_executor.get_migration_offsets()
        parallel_config = self.parallel_config
        cache_config = self.cache_config

        return KVCacheHandlerConfig(
            engine_id=self.engine_id,
            tp_size=parallel_config.tensor_parallel_size,
            pp_size=parallel_config.pipeline_parallel_size,
            num_gpu_blocks=cache_config.num_gpu_blocks
            // parallel_config.pipeline_parallel_size,
            block_size=cache_config.block_size,
            kv_cache_handlers=kv_cache_handlers,
            kv_cache_offsets=kv_cache_offsets,
        )

    def register_handlers(self, engine_id: int, handlers):
        self.model_executor.register_handlers(engine_id, handlers)
