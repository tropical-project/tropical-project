import os
import tempfile
from uuid import uuid4


def get_open_zmq_ipc_path() -> str:
    base_rpc_path = os.getenv("SGIR_DISTSERVE_RPC_BASE_PATH", tempfile.gettempdir())
    return f"ipc://{base_rpc_path}/{uuid4()}"
