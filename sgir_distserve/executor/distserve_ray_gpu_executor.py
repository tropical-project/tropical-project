from typing import Any, Dict, List, Optional, Tuple

import ray

from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.ray_gpu_executor import RayGPUExecutor, RayGPUExecutorAsync
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, SamplerOutput


logger = init_logger(__name__)


class PDRayGPUExecutor(RayGPUExecutor):
    def _get_worker_wrapper_args(self) -> Dict[str, Any]:
        if self.speculative_config is not None:
            worker_module_name = "sgir_distserve.worker.distserve_worker"
            worker_class_name = "create_spec_worker"
        else:
            worker_module_name = "sgir_distserve.worker.distserve_worker"
            worker_class_name = f"PDWorker"

        return dict(
            worker_module_name=worker_module_name,
            worker_class_name=worker_class_name,
            trust_remote_code=self.model_config.trust_remote_code,
        )

    def _init_cache_handler(self) -> None:
        self._run_workers("_init_cache_handler")

    def _init_nccl_cache_handler(self, unique_id) -> None:
        self._run_workers("_init_nccl_cache_handler", unique_id)

    def register_handlers(self, engine_id: int, handlers):
        self._run_workers("register_handlers", engine_id, handlers)

    def _init_engine_id(self, engine_id) -> None:
        self._run_workers("_init_engine_id", engine_id)

    def register_nccl_handlers(self, engine_id: int, handlers):
        self._run_workers("register_nccl_handlers", engine_id, handlers)

    def profile_start(self):
        return self._run_workers("profile_start")

    def profile_step(self):
        return self._run_workers("profile_step")

    def profile_stop(self):
        return self._run_workers("profile_stop")

    def get_migration_handlers(self):
        return self._run_workers("get_migration_handlers")

    def get_migration_offsets(self):
        return self._run_workers("get_migration_offsets")

    def _run_workers(
        self,
        method: str,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        use_dummy_driver: bool = True,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        # patch use dummy worker default to initialize both prefill and decode instance
        # TODO: use mp mode
        return super()._run_workers(
            method,
            *args,
            async_run_tensor_parallel_workers_only=async_run_tensor_parallel_workers_only,
            all_args=all_args,
            all_kwargs=all_kwargs,
            use_dummy_driver=use_dummy_driver,
            max_concurrent_workers=max_concurrent_workers,
            **kwargs,
        )


class PDRayGPUExecutorAsync(PDRayGPUExecutor, RayGPUExecutorAsync, ExecutorAsyncBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.use_ray_compiled_dag:
            self.driver_exec_method = self.driver_dummy_worker.execute_method.remote

    def _driver_execute_model(
        self, execute_model_req: Optional[ExecuteModelRequest]
    ) -> Optional[List[SamplerOutput]]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        assert (
            not self.use_ray_spmd_worker
        ), "driver_worker does not exist for VLLM_USE_RAY_SPMD_WORKER=1"
        return self.driver_dummy_worker.execute_method.remote(
            "execute_model", execute_model_req
        )

    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> List[SamplerOutput]:
        if not self.use_ray_spmd_worker:
            return super().execute_model(execute_model_req)

        if self.forward_dag is None:
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)

        outputs = ray.get(self.forward_dag.execute(execute_model_req))
        return outputs[0]
