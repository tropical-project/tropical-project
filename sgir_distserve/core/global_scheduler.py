import enum
import time
from collections import deque

from copy import copy
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
from vllm.logger import init_logger

from vllm.outputs import RequestOutput

from sgir_distserve.engine.pd_profiler import PDProfiler

from sgir_distserve.sequence import DistserveSequenceGroup, DistserveSequenceStage


logger = init_logger("vllm.core.global_scheduler")


class EngineKind(enum.Enum):
    Prefill = enum.auto()
    Decode = enum.auto()

    # TODO: reserve for extending
    Hybrid = enum.auto()


class SchedulePolicy(enum.Enum):
    FCFS = enum.auto()
    SJF = enum.auto()
    LJF = enum.auto()
    EDF = enum.auto()
    # Least Laxity First
    LLF = enum.auto()


class DispatchPolicy(enum.Enum):
    RoundRobin = enum.auto()
    LeastUnfinishedJobs = enum.auto()
    LeastUnfinishedPrefillTokens = enum.auto()
    Slack = enum.auto()


def get_schedule_policy_by_name(policy_name: str) -> SchedulePolicy:
    policy_name = policy_name.lower()
    policy_name_to_policy = {
        "fcfs": SchedulePolicy.FCFS,
        "sjf": SchedulePolicy.SJF,
        "ljf": SchedulePolicy.LJF,
        "edf": SchedulePolicy.EDF,
        "llf": SchedulePolicy.LLF,
    }

    if policy_name not in policy_name_to_policy:
        raise KeyError(f"No schedule policy named {policy_name}")

    return policy_name_to_policy[policy_name]


def get_dispatch_policy_by_name(policy_name: str) -> SchedulePolicy:
    policy_name = policy_name.lower()
    policy_name_to_policy = {
        "round-robin": DispatchPolicy.RoundRobin,
        "least-num-unfinished-jobs": DispatchPolicy.LeastUnfinishedJobs,
        "least-num-unfinished-tokens": DispatchPolicy.LeastUnfinishedPrefillTokens,
        "slack": DispatchPolicy.Slack,
    }

    if policy_name not in policy_name_to_policy:
        raise KeyError(f"No schedule policy named {policy_name}")

    return policy_name_to_policy[policy_name]


class DispatchType(enum.Enum):
    Prefill = enum.auto()
    Decode = enum.auto()
    Migration = enum.auto()


@dataclass
class DispatchContext:
    begin: float
    dispatch_type: DispatchType
    seq_group: DistserveSequenceGroup


@dataclass
class ScheduleState:
    isRunning: bool
    begin_time: float = None
    predicted_end_time: float = None
    duration: float = None
    chunk_idx: int = None
    seq_group_slack: List[float] = None
    seq_group_pred: List[float] = None
    waiting_for_dispatch: List[DistserveSequenceGroup] = None


class DispatchSnapshot:
    def __init__(self):
        self.context: Dict[str, DispatchContext] = {}

    def get_context(
        self, dispatch_type: Optional[DispatchType] = None
    ) -> List[DispatchContext]:
        if dispatch_type is None:
            return list(self.context.values())
        iters = filter(
            lambda ctx: ctx.dispatch_type == dispatch_type,
            self.context.values(),
        )
        return list(iters)

    def add_dispatch(
        self,
        seq_group: DistserveSequenceGroup,
        dispatch_type=DispatchType.Prefill,
    ) -> None:
        now = time.time()
        dispatch_request = DispatchContext(
            begin=now, seq_group=seq_group, dispatch_type=dispatch_type
        )
        self.context[seq_group.request_id] = dispatch_request

    def remove_dispatch(self, request_id: str) -> None:
        if request_id in self.context:
            self.context.pop(request_id)

    def update_dispatch(self, output: RequestOutput) -> None:
        if output.request_id not in self.context:
            return
        data = self.context[output.request_id].seq_group.get_seqs()[0].data
        for output_token in output.outputs:
            data._output_token_ids.append(output_token.token_ids[-1])


class DispatchSnapshotManager:
    def __init__(self, engine_set: Set):
        self.engine_state: Dict[int, ScheduleState]
        self.engine_state = {
            idx: ScheduleState(isRunning=False, waiting_for_dispatch=[])
            for idx in engine_set
        }
        self.cnt = {idx: 0 for idx in engine_set}
        self.dispatch_snapshot: Dict[int, DispatchSnapshot]
        self.dispatch_snapshot = {idx: DispatchSnapshot() for idx in engine_set}

    def get_context(self, engine_id, dispatch_type: Optional[DispatchType] = None):
        return self.dispatch_snapshot[engine_id].get_context(dispatch_type)

    def add_dispatch(
        self,
        engine_id: int,
        seq_group: DistserveSequenceGroup,
        dispatch_type: DispatchType,
    ) -> None:
        self.dispatch_snapshot[engine_id].add_dispatch(seq_group, dispatch_type)

    def remove_dispatch(
        self,
        request_id: str,
        engine_id: Optional[int] = None,
    ) -> None:
        if engine_id is None:
            engine_ids = list(self.dispatch_snapshot.keys())
        else:
            engine_ids = [engine_id]
        for idx in engine_ids:
            self.dispatch_snapshot[idx].remove_dispatch(request_id)

    def update_dispatch(
        self, request_output: RequestOutput, engine_id: Optional[int] = None
    ) -> None:
        if engine_id is None:
            engine_ids = list(self.dispatch_snapshot.keys())
        else:
            engine_ids = [engine_id]
        for idx in engine_ids:
            self.dispatch_snapshot[idx].update_dispatch(request_output)


class GlobalScheduler:
    def __init__(
        self,
        p_set: Set,
        d_set: Set,
        model,
        enable_multiplexing=False,
        max_multiplexing_length=4096,
        multiplexing_decode_batch_latency_watermark=0.85,
        multiplexing_decode_batched_tokens_watermark=400000,
        enable_preemption=False,
        chunk_size=None,
        dispatch_policy: DispatchPolicy = DispatchPolicy.Slack,
        schedule_policy: SchedulePolicy = SchedulePolicy.SJF,
    ):
        self.p_set = p_set
        self.d_set = d_set
        self.pd_profiler = PDProfiler(
            model=model, chunk=chunk_size if chunk_size != None else -1
        )

        self.dispatch_snapshot_manager = DispatchSnapshotManager(self.engine_set)

        self.waiting: Deque[DistserveSequenceGroup] = deque()
        self.waiting_for_migration: Deque[DistserveSequenceGroup] = deque()
        self.waiting_for_free: Deque[DistserveSequenceGroup] = deque()
        self.gen_dispatch = self.get_dispatch_generator(dispatch_policy=dispatch_policy)
        self.gen_schedule = self.get_schedule_generator(schedule_policy=schedule_policy)

        self.enable_preemption = enable_preemption

        self.enable_multiplexing = enable_multiplexing
        self.max_multiplexing_length = max_multiplexing_length
        self.multiplexing_decode_batch_latency_watermark = (
            multiplexing_decode_batch_latency_watermark
        )
        self.multiplexing_decode_batched_tokens_watermark = (
            multiplexing_decode_batched_tokens_watermark
        )

    @property
    def engine_set(self):
        return self.p_set | self.d_set

    @property
    def num_replicas(self):
        return len(self.engine_set)

    def get_num_unfinished_jobs(
        self, engine_id: int, dispatch_type: Optional[DispatchType] = None
    ):
        return len(self.dispatch_snapshot_manager.get_context(engine_id, dispatch_type))

    def get_num_unfinished_tokens(self, idx: int, dispatch_type=DispatchType.Prefill):
        context = self.dispatch_snapshot_manager.get_context(idx, dispatch_type)
        seq_groups = [context.seq_group for context in context]
        return sum(seq_group.get_seqs()[0].get_prompt_len() for seq_group in seq_groups)

    def add_seq_group(self, seq_group: DistserveSequenceGroup):
        self.waiting.append(seq_group)

    def add_migration_seq_group(self, seq_group: DistserveSequenceGroup):
        self.waiting_for_migration.append(seq_group)

    def add_free_seq_group(self, seq_group: DistserveSequenceGroup):
        self.waiting_for_free.append(seq_group)

    def remove_dispatch(self, request_id: str) -> None:
        self.dispatch_snapshot_manager.remove_dispatch(request_id)

    def update_dispatch(self, output: RequestOutput) -> None:
        self.dispatch_snapshot_manager.update_dispatch(output)
        if output.finished:
            self.remove_dispatch(output.request_id)

    def dispatch(self):
        if not self.waiting:
            return None, None
        self.waiting = self.gen_schedule(self.waiting)
        seq_group = self.waiting[0]
        engine_id = self.gen_dispatch(seq_group)
        if engine_id != None:
            self.waiting.popleft()
            self.dispatch_snapshot_manager.add_dispatch(
                engine_id=engine_id,
                seq_group=seq_group,
                dispatch_type=DispatchType.Prefill,
            )
            return engine_id, seq_group
        return None, None

    def dispatch_migration(self):
        if not self.waiting_for_migration:
            return None, None
        seq_group = self.waiting_for_migration.popleft()
        if not self.d_set:
            return seq_group.migration_snapshots[-1].engine_id, seq_group
        costs = [
            (
                idx,
                self.get_num_unfinished_tokens(idx, dispatch_type=DispatchType.Decode),
            )
            for idx in self.d_set
        ]
        des_engine_id = costs[costs.index(min(costs, key=lambda x: x[1]))][0]

        self.remove_dispatch(seq_group.request_id)
        self.dispatch_snapshot_manager.add_dispatch(
            des_engine_id, seq_group, dispatch_type=DispatchType.Decode
        )

        return des_engine_id, seq_group

    def dispatch_free(self):
        if not self.waiting_for_free:
            return None, None
        seq_group = self.waiting_for_free.popleft()
        return seq_group.migration_snapshots[-1].engine_id, seq_group

    def get_idle_p_engine_id(self, seq_group: DistserveSequenceGroup) -> int:
        # # (depracated) VLLM Scheduling
        # if len(self.p_set) == self.num_replicas:
        #     costs = [(idx, self.get_num_unfinished_jobs()) for idx in self.p_set]
        #     return costs[costs.index(min(costs, key=lambda x: x[1]))][0]

        # Clockwork Scheduling
        idle_set = set()
        for idx in self.p_set:
            engine_state = self.dispatch_snapshot_manager.engine_state[idx]
            if engine_state.waiting_for_dispatch:
                continue
            if engine_state.isRunning == False:
                idle_set.add(idx)
            elif not self.enable_preemption:
                # now = time.time()
                # seq_group_pred = engine_state.seq_group_pred
                # # TODO: Convert this to SJF
                # if sum(seq_group_pred) < self.pd_profiler.get_prefill_slack(
                #     seq_group, now
                # ):
                #     idle_set.add(idx)
                now = time.time()
                hol_seq_group_pred = self.pd_profiler.get_prefill_seq_prediction(
                    seq_group
                )
                pred_end = engine_state.predicted_end_time
                hol_slack = (
                    seq_group.metrics.arrival_time
                    + seq_group.ttft_slo
                    - pred_end
                    - hol_seq_group_pred
                )

                now = time.time()

                seq_group_pred = copy(engine_state.seq_group_pred)
                seq_group_pred.append(hol_seq_group_pred)
                seq_group_slack = copy(engine_state.seq_group_slack)
                seq_group_slack.append(hol_slack)

                pred_slack = sorted(zip(seq_group_pred, seq_group_slack))
                seq_group_pred = [p[0] for p in pred_slack]
                seq_group_slack = [p[1] for p in pred_slack]

                seq_group_pred = np.cumsum([0] + list(np.array(seq_group_pred) * 1.02))[
                    :-1
                ]
                seq_group_slack = (
                    np.array(seq_group_slack)
                    - seq_group_pred
                    - (now - engine_state.begin_time)
                )

                if np.all(seq_group_slack > 0):
                    idle_set.add(idx)
            else:
                now = time.time()
                hol_seq_group_pred = self.pd_profiler.get_prefill_seq_prediction(
                    seq_group
                )
                pred_end = engine_state.predicted_end_time
                hol_slack = (
                    seq_group.metrics.arrival_time
                    + seq_group.ttft_slo
                    - pred_end
                    - hol_seq_group_pred
                )

                now = time.time()

                seq_group_pred = copy(engine_state.seq_group_pred)
                seq_group_pred.append(hol_seq_group_pred)
                seq_group_slack = copy(engine_state.seq_group_slack)
                seq_group_slack.append(hol_slack)

                pred_slack = sorted(zip(seq_group_pred, seq_group_slack))
                seq_group_pred = [p[0] for p in pred_slack]
                seq_group_slack = [p[1] for p in pred_slack]

                seq_group_pred = np.cumsum([0] + list(np.array(seq_group_pred) * 1.02))[
                    :-1
                ]
                seq_group_slack = (
                    np.array(seq_group_slack)
                    - seq_group_pred
                    - (now - engine_state.begin_time)
                )

                if np.all(seq_group_slack > 0):
                    idle_set.add(idx)

        if idle_set:
            costs = [
                (
                    idx,
                    self.get_num_unfinished_tokens(idx),
                )
                for idx in idle_set
            ]
            idx = costs[costs.index(min(costs, key=lambda x: x[1:]))][0]
            self.dispatch_snapshot_manager.cnt[idx] += 1
            return idx
        return None

    def get_inlinable_d_engine_id(self, seq_group) -> int:
        """
        Core block: SLO-Aware Multiplexing !!!!!
        """
        inlinable_idx = []
        for idx in self.d_set:
            if seq_group.get_seqs()[0].get_prompt_len() < 512:
                inlinable_idx.append(idx)
                continue
            num_prefill = self.get_num_unfinished_jobs(idx, DispatchType.Prefill)
            if num_prefill > 0:
                continue
            prediction = self.pd_profiler.get_prefill_seq_prediction(seq_group)
            now = time.time()

            running_decode = [
                ctx.seq_group
                for ctx in self.dispatch_snapshot_manager.get_context(
                    idx, dispatch_type=DispatchType.Decode
                )
            ]
            # Conservative multiplexing
            if (
                self.pd_profiler.get_decode_seq_prediction(running_decode)
                > self.multiplexing_decode_batch_latency_watermark
            ) or (
                self.get_num_unfinished_tokens(idx, DispatchType.Decode)
                > self.multiplexing_decode_batched_tokens_watermark
            ):
                logger.warning(
                    "[warning] watermark overflow, prediction: ",
                    self.pd_profiler.get_decode_seq_prediction(running_decode),
                )
                continue
            stall_free = True
            # SLO Aware Multiplexing
            for seq_group in running_decode:
                if (
                    self.pd_profiler.get_decode_slack(seq_group, now) - prediction
                    < seq_group.tpot_slo * 1
                ):
                    stall_free = False
                    break
            if stall_free:
                inlinable_idx.append(idx)

            # Load Balance
            if inlinable_idx:
                costs = [
                    (
                        idx,
                        self.get_num_unfinished_tokens(idx, DispatchType.Decode),
                    )
                    for idx in inlinable_idx
                ]
                idx = costs[costs.index(min(costs, key=lambda x: x[1:]))][0]
                self.dispatch_snapshot_manager.cnt[idx] += 1
                return idx
        return None

    def get_schedule_generator(
        self, schedule_policy: SchedulePolicy
    ) -> Callable[[Deque[DistserveSequenceGroup]], Deque[DistserveSequenceGroup]]:
        match schedule_policy:
            case SchedulePolicy.FCFS:
                return lambda waiting: waiting
            case SchedulePolicy.LJF:
                return lambda waiting: deque(
                    sorted(
                        waiting,
                        key=lambda x: x.get_seqs()[0].get_prompt_len(),
                        reverse=True,
                    )
                )
            case SchedulePolicy.SJF:
                return lambda waiting: deque(
                    sorted(
                        waiting,
                        key=lambda x: x.get_seqs()[0].get_prompt_len(),
                        reverse=False,
                    )
                )
            case SchedulePolicy.EDF:
                return lambda waiting: deque(
                    sorted(
                        waiting,
                        key=lambda x: x.metrics.arrival_time + x.ttft_slo,
                        reverse=False,
                    )
                )
            case SchedulePolicy.LLF:
                return lambda waiting: deque(
                    sorted(
                        waiting,
                        key=lambda x: self.pd_profiler.get_prefill_slack(
                            x, now=time.time()
                        ),
                        reverse=False,
                    )
                )
            case _:
                raise AttributeError

    def round_robin_generator(self):
        cnt = -1

        def wrapper_fn(seq_group: DistserveSequenceGroup):
            nonlocal cnt
            cnt += 1
            return cnt % len(self.p_set)

        return wrapper_fn

    def least_unfinished_job_generator(self, seq_group):
        cost = [self.get_num_unfinished_jobs(i) for i in self.p_set]
        return cost.index(min(cost))

    def least_unfinished_prefill_token_generator(self, seq_group):
        cost = [self.get_num_unfinished_tokens(i) for i in self.p_set]
        return cost.index(min(cost))

    def slack_generator(self, seq_group: DistserveSequenceGroup):
        if self.enable_multiplexing:
            if seq_group.get_seqs()[0].get_prompt_len() < 8192:
                inline_idx: int = self.get_inlinable_d_engine_id(seq_group)
                if inline_idx != None:
                    decode_ctx = self.dispatch_snapshot_manager.get_context(
                        inline_idx, DispatchType.Decode
                    )
                    decode_seq_groups = [ctx.seq_group for ctx in decode_ctx]
                    iter_pred = self.pd_profiler.get_decode_seq_prediction(
                        decode_seq_groups
                    )
                    print(
                        f"inline {seq_group.request_id} "
                        f"({seq_group.get_seqs()[0].get_prompt_len()}) "
                        f"to engine {inline_idx}.",
                        f"decode iter prediction: {iter_pred}",
                    )
                    return inline_idx

        idle_idx = self.get_idle_p_engine_id(seq_group)
        if idle_idx != None:
            return idle_idx

        return None

    def get_dispatch_generator(
        self, dispatch_policy: DispatchPolicy
    ) -> Callable[[DistserveSequenceGroup], Optional[int]]:
        match dispatch_policy:
            case DispatchPolicy.RoundRobin:
                return self.round_robin_generator()
            case DispatchPolicy.LeastUnfinishedJobs:
                return self.least_unfinished_job_generator
            case DispatchPolicy.LeastUnfinishedPrefillTokens:
                return self.least_unfinished_prefill_token_generator
            case DispatchPolicy.Slack:
                if self.pd_profiler.indb:
                    return self.slack_generator
                else:
                    logger.warning(
                        f"[warning] {self.pd_profiler} not in profiler, "
                        f"fallback to  {self.least_unfinished_prefill_token_generator}",
                    )
                    return self.least_unfinished_prefill_token_generator
            case _:
                raise NotImplementedError
