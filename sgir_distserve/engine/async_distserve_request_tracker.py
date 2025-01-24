import asyncio
from typing import Dict, List, Optional, Set, Tuple

from vllm.engine.async_llm_engine import AsyncStream, RequestTracker
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sequence import SequenceGroup

from sgir_distserve.sequence import DistserveSequenceGroup

logger = init_logger(__name__)


class AsyncDistserveRequestTracker(RequestTracker):
    """Synchronous abstraction for tracking requests."""

    def __init__(self, num_replicas) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}

        self._finished_requests: List[asyncio.Queue[str]] = [
            asyncio.Queue() for _ in range(num_replicas)
        ]

        self._new_requests_in_pool: asyncio.Queue[Tuple[AsyncStream, dict]] = (
            asyncio.Queue()
        )

        self._new_requests: List[
            asyncio.Queue[Tuple[AsyncStream, DistserveSequenceGroup]]
        ] = [asyncio.Queue() for _ in range(num_replicas)]

        self._migrate_requests: List[
            asyncio.Queue[Tuple[AsyncStream, DistserveSequenceGroup]]
        ] = [asyncio.Queue() for _ in range(num_replicas)]

        self.requests_event = [asyncio.Event() for _ in range(num_replicas)]
        self.new_requests_in_pool_event = asyncio.Event()

    def __contains__(self, item):
        return item in self._request_streams

    def __len__(self) -> int:
        return len(self._request_streams)

    def propagate_exception(
        self,
        exc: Exception,
        request_id: Optional[str] = None,
        engine_id: int = 0,
    ) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
            self.abort_request(
                request_id,
                engine_id=engine_id,
            )
        else:
            for rid, stream in self._request_streams.items():
                stream.put(exc)
                self.abort_request(
                    rid,
                    engine_id=engine_id,
                )

    def process_request_output(
        self,
        request_output: RequestOutput,
        *,
        verbose: bool = False,
        engine_id: int = 0,
    ) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            if verbose:
                logger.info(f"Finished request {request_id}.")
            self.abort_request(
                request_id,
                engine_id=engine_id,
            )

    def process_exception(
        self,
        request_id: str,
        exception: Exception,
        *,
        verbose: bool = False,
        engine_id: int = 0,
    ) -> None:
        """Propagate an exception from the engine."""
        self._request_streams[request_id].put(exception)
        if verbose:
            logger.info(f"Finished request {request_id}.")
        self.abort_request(
            request_id,
            engine_id=engine_id,
        )

    def add_request_to_pool():
        raise NotImplementedError

    def add_request_to_engine():
        raise NotImplementedError

    def add_request(
        self,
        engine_id,
        request_id: str,
        *,
        verbose: bool = False,
        **engine_add_request_kwargs,
    ) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._new_requests[engine_id].put_nowait(
            (
                stream,
                {
                    "request_id": request_id,
                    **engine_add_request_kwargs,
                },
            )
        )

        self.requests_event[engine_id].set()

        if verbose:
            logger.info("Added request %s.", request_id, "Instance id: %s")
        return stream

    def add_request_pool(
        self,
        request_id: str,
        *,
        verbose: bool = False,
        **engine_add_request_kwargs,
    ) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)

        if verbose:
            logger.info("Added request %s.", request_id, "Instance id: %s")

        self._new_requests_in_pool.put_nowait(
            (
                stream,
                {
                    "request_id": request_id,
                    **engine_add_request_kwargs,
                },
            )
        )
        self.new_requests_in_pool_event.set()

        return stream

    def migrate_request(
        self, engine_id, request_id: str, seq_group: SequenceGroup
    ) -> AsyncStream:
        stream = self._request_streams[request_id]
        self._migrate_requests[engine_id].put_nowait((stream, seq_group))
        self.requests_event[engine_id].set()
        return stream

    def abort_request(
        self,
        request_id: str,
        *,
        engine_id=0,
        verbose: bool = False,
    ) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f"Aborted request {request_id}.")

        self._finished_requests[engine_id].put_nowait(request_id)

        if (
            request_id not in self._request_streams
            or self._request_streams[request_id].finished
        ):
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_new_requests_from_pool(self) -> Tuple[List[Dict]]:
        new_requests: List[Dict] = []
        while not self._new_requests_in_pool.empty():
            stream, new_request = self._new_requests_in_pool.get_nowait()
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)
        return new_requests

    def get_new_and_migrate_and_finished_requests(
        self, engine_id: int
    ) -> Tuple[List[Dict], List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        migrate_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests[engine_id].empty():
            request_id = self._finished_requests[engine_id].get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests[engine_id].empty():
            stream, new_request = self._new_requests[engine_id].get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            # self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        while not self._migrate_requests[engine_id].empty():
            stream, migrate_request = self._migrate_requests[engine_id].get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            migrate_requests.append(migrate_request)

        return new_requests, migrate_requests, finished_requests

    async def wait_for_requests_in_pool(self):
        if not self.has_new_requests_in_pool():
            await self.new_requests_in_pool_event.wait()
        self.new_requests_in_pool_event.clear()

    def has_new_requests_in_pool(self):
        return not self._new_requests_in_pool.empty()

    async def wait_for_requests(self, engine_id):
        if not self.has_requests(engine_id):
            await self.requests_event[engine_id].wait()
        self.requests_event[engine_id].clear()

    def has_requests(self, engine_id):
        return (not self._new_requests[engine_id].empty()) and (
            not self._migrate_requests[engine_id].empty()
        )
