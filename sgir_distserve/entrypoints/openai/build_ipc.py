import multiprocessing
from typing import Optional

from sgir_distserve._C import get_nccl_unique_id

from sgir_distserve.engine.arg_utils import SGIRDistserveArgs
from sgir_distserve.ipc.ipc_engine import run_mp_engine
from sgir_distserve.utils.ipc_op import get_open_zmq_ipc_path

from vllm.usage.usage_lib import UsageContext


def build_async_engine_client_from_engine_args(
    engine_args: SGIRDistserveArgs,
    ray_address: Optional[str] = None,
):
    sgir_distserve_config = engine_args.create_sgir_distserve_config()

    ipc_path_map = {}
    rnccl_comm_unique_id = get_nccl_unique_id()
    for engine_id, replica_config in enumerate(sgir_distserve_config.replica_config):
        ipc_path = get_open_zmq_ipc_path()
        context = multiprocessing.get_context("spawn")

        engine_process = context.Process(
            target=run_mp_engine,
            args=(
                engine_id,
                engine_args,
                UsageContext.OPENAI_API_SERVER,
                ipc_path,
                rnccl_comm_unique_id,
                replica_config.prometheus_port,
            ),
        )
        ipc_path_map[engine_id] = ipc_path
        engine_process.start()
    return ipc_path_map
