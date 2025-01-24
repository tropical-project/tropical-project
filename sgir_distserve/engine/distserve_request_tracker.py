import asyncio
from typing import Dict, List, Optional, Set, Tuple

from vllm.engine.async_llm_engine import AsyncStream, RequestTracker
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sequence import SequenceGroup

from sgir_distserve.sequence import DistserveSequenceGroup

logger = init_logger(__name__)


class DistserveRequestTracker(RequestTracker):
    """Synchronous abstraction for tracking requests."""

    def update_stream(self, output: RequestOutput):
        rid = output.request_id
        if rid in self._request_streams:
            self._request_streams[rid].put(output)
            if output.finished:
                self._request_streams[rid].finish()
                self._request_streams.pop(rid, None)
