import enum
import time
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from itertools import permutations
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.core.policy import Policy, PolicyFactory

from vllm.core.scheduler import (
    PreemptionMode,
    ScheduledSequenceGroup,
    Scheduler,
    SchedulerOutputs,
    SchedulerPrefillOutputs,
    SchedulerRunningOutputs,
    SchedulerSwappedInOutputs,
    SchedulingBudget,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import (
    Logprob,
    Sequence,
    SequenceData,
    SequenceGroup,
    SequenceGroupMetadata,
    SequenceGroupOutput,
    SequenceOutput,
    SequenceStatus,
)
from vllm.utils import merge_dicts

from sgir_distserve.core.global_scheduler import SchedulePolicy

from sgir_distserve.engine.pd_profiler import PDProfiler

from sgir_distserve.sequence import (
    DistserveSequenceGroup,
    DistserveSequenceStage,
    DistserveSequenceStatus,
)


logger = init_logger(__name__)


@dataclass
class SchedulerMigrateOutputs:
    """The requests that are scheduled from a waiting queue.

    Could contain a fresh prefill requests or preempted requests that need
    to be recomputed from scratch.
    """

    # Selected sequences for prefill.
    seq_groups: List[SequenceGroup]
    blocks_to_migration: List[Tuple[int, int, int, int]]
    # Ignored sequence groups.
    ignored_seq_groups: List[SequenceGroup]
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
            seq_groups=[],
            ignored_seq_groups=[],
            num_lookahead_slots=0,
        )


@dataclass
class PDSchedulerOutputs(SchedulerOutputs):
    migration_seq_groups: List[ScheduledSequenceGroup]
    preempted_seq_groups: List[SequenceGroup]
    blocks_to_migration: List[Tuple[int, int, int]]
    time_stamp: float

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (
            not self.scheduled_seq_groups
            and not self.blocks_to_swap_in
            and not self.blocks_to_swap_out
            and not self.blocks_to_copy
            and not self.blocks_to_migration
        )


class PDScheduler(Scheduler):
    def __init__(
        self,
        model,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
    ) -> None:
        super().__init__(
            scheduler_config,
            cache_config,
            lora_config,
            pipeline_parallel_size,
        )

        self.enforce_migrate = False

        self.slo_aware_chunked_preemption = (
            scheduler_config.slo_aware_chunked_preemption
        )
        self.schedule_policy = scheduler_config.schedule_policy
        self.waiting_for_migration: Deque[SequenceGroup] = deque()
        self.running_for_migration: Deque[SequenceGroup] = deque()

        if scheduler_config.chunked_prefill_enabled:
            chunk = scheduler_config.max_num_batched_tokens
        else:
            chunk = None
        self.pd_profiler = PDProfiler(model=model, chunk=chunk if chunk != None else -1)

    def add_seq_group(
        self,
        seq_group: DistserveSequenceGroup,
    ) -> None:
        self.waiting.append(seq_group)

    def get_num_unfinished_seq_groups(self) -> int:
        return (
            len(self.waiting)
            + len(self.running)
            + len(self.swapped)
            + len(self.waiting_for_migration)
            + len(self.running_for_migration)
        )

    def add_seq_group_to_waiting_for_migration(
        self,
        seq_group: DistserveSequenceGroup,
    ) -> None:
        self.waiting_for_migration.append(seq_group)

    def abort_seq_group_migration(self, request_id: str | Iterable[str]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        for state_queue in [self.waiting_for_migration, self.running_for_migration]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity.
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                self._finished_requests_ids.append(aborted_group.request_id)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

    def _schedule_migrates(
        self,
        waiting_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            waiting_queue: The queue that contains prefill requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            A tuple of remaining waiting_queue after scheduling and
            SchedulerSwappedInOutputs.
        """
        """
        addapted from _schedule_prefills
        """
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[SequenceGroup] = []
        # We don't sort waiting queue because we assume it is sorted.
        # Copy the queue so that the input queue is not modified.
        waiting_queue = deque([s for s in waiting_queue])

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt " "sequence."
            )
            num_new_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.WAITING, enable_chunking, budget
            )

            if not enable_chunking:
                assert num_new_tokens == 1

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long" " and exceeds limit of %d",
                    num_new_tokens,
                    prompt_limit,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of block_manager",
                    num_new_tokens,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (
                    self.lora_enabled
                    and lora_int_id > 0
                    and lora_int_id not in curr_loras
                    and len(curr_loras) >= self.lora_config.max_loras
                ):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens == 0 or not budget.can_schedule(
                num_new_tokens=num_new_tokens, num_new_seqs=num_new_seqs
            ):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)
            seq_groups.append(
                ScheduledSequenceGroup(
                    seq_group=seq_group, token_chunk_size=num_new_tokens
                )
            )

            # patch: migrating occupied 1 tokens
            budget.add_num_batched_tokens(seq_group.request_id, 1)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        blocks_to_migration = []
        for sched_seq_group in seq_groups:
            seq_group: DistserveSequenceGroup = sched_seq_group.seq_group
            seq: Sequence = sched_seq_group.seq_group.get_seqs()[0]
            seq.status = DistserveSequenceStatus.MIGRATING
            block_index_in_decode_engine = self.block_manager.get_block_table(seq)

            migration_snapshot = seq_group.migration_snapshots[-1]

            block_index_in_prefill_engine = migration_snapshot.block_index
            for prefill_engine_idx, decode_engine_idx in zip(
                block_index_in_prefill_engine, block_index_in_decode_engine
            ):
                src_engine_ids = migration_snapshot.engine_id
                src_virtual_engine_ids = migration_snapshot.virtual_engine_id
                blocks_to_migration.append(
                    (
                        src_engine_ids,
                        src_virtual_engine_ids,
                        decode_engine_idx,
                        prefill_engine_idx,
                    )
                )

        return waiting_queue, SchedulerMigrateOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True),
            blocks_to_migration=blocks_to_migration,
        )

    def _schedule_prefills(
        self,
        waiting_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            waiting_queue: The queue that contains prefill requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            A tuple of remaining waiting_queue after scheduling and
            SchedulerSwappedInOutputs.
        """

        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[SequenceGroup] = []
        # We don't sort waiting queue because we assume it is sorted.
        # Copy the queue so that the input queue is not modified.
        waiting_queue = deque([s for s in waiting_queue])

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt " "sequence."
            )
            num_new_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.WAITING, enable_chunking, budget
            )
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long" " and exceeds limit of %d",
                    num_new_tokens,
                    prompt_limit,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of block_manager",
                    num_new_tokens,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (
                    self.lora_enabled
                    and lora_int_id > 0
                    and lora_int_id not in curr_loras
                    and len(curr_loras) >= self.lora_config.max_loras
                ):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens == 0 or not budget.can_schedule(
                num_new_tokens=num_new_tokens, num_new_seqs=num_new_seqs
            ):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)
            seq_groups.append(
                ScheduledSequenceGroup(
                    seq_group=seq_group, token_chunk_size=num_new_tokens
                )
            )
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)
            # break

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return waiting_queue, SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True),
        )

    def _schedule_slo_aware_chunked_preemption(
        self,
        running_queue: deque,
        waiting_queue: deque,
        budget: SchedulingBudget,
    ):
        blocks_to_copy: List[Tuple[int, int]] = []
        enable_chunking = True
        prefill_seq_groups: List[ScheduledSequenceGroup] = []

        running_waiting_queue: List[SequenceGroup] = waiting_queue + running_queue
        now = time.time()

        running_waiting_queue = deque(
            sorted(
                running_waiting_queue,
                key=lambda x: self.pd_profiler.get_prefill_seq_prediction(x),
                reverse=False,
            )
        )

        # all_running_waiting_queue = list(
        #     permutations(running_waiting_queue, len(running_waiting_queue))
        # )
        # all_running_waiting_queue = list(
        #     filter(
        #         lambda x: all(
        #             (
        #                 self.pd_profiler.get_queue_slack_time(x, now)
        #                 - self.pd_profiler.get_queue_pred_end_time(x)
        #                 > 0
        #             )
        #         ),
        #         all_running_waiting_queue,
        #     )
        # )

        # if any(
        #     self.pd_profiler.get_queue_slack_time(running_waiting_queue, now)
        #     - self.pd_profiler.get_queue_pred_end_time(running_waiting_queue)
        #     < 0
        # ):
        # running_waiting_queue = deque(
        #     sorted(
        #         running_waiting_queue,
        #         key=lambda x: self.pd_profiler.get_prefill_slack(x, now),
        #         reverse=False,
        #     )
        # )
        # else:
        #     cost = [
        #         sum(self.pd_profiler.get_queue_pred_queue_time(rw_queue))
        #         for rw_queue in all_running_waiting_queue
        #     ]
        #     running_waiting_queue = deque(
        #         all_running_waiting_queue[cost.index(min(cost))]
        #     )

        for seq_group in running_waiting_queue:
            # print(seq_group.get_num_uncomputed_tokens())
            seq_group.get_seqs()[0].status = SequenceStatus.RUNNING

        if len(running_waiting_queue) > 0:
            print(
                "running_wait_queue: ",
                [
                    (
                        seq_group.get_seqs()[0].data.get_prompt_len(),
                        seq_group.get_seqs()[0].data.get_num_computed_tokens(),
                    )
                    for seq_group in running_waiting_queue
                ],
            )

        while running_waiting_queue:
            seq_group = running_waiting_queue[0]
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget
            )
            if (
                num_running_tokens == 0
                or budget.num_curr_seqs >= self.scheduler_config.max_num_seqs
            ):
                break

            running_waiting_queue.popleft()
            # first scheduling
            if (
                seq_group.get_num_uncomputed_tokens()
                == seq_group.get_seqs()[0].get_prompt_len()
            ):
                seq_group.get_seqs()[0].status = SequenceStatus.WAITING
                self._allocate_and_set_running(seq_group)
            else:
                self._append_slots(seq_group, blocks_to_copy)
            prefill_seq_groups.append(
                ScheduledSequenceGroup(
                    seq_group=seq_group, token_chunk_size=num_running_tokens
                )
            )
            budget.add_num_batched_tokens(seq_group.request_id, num_running_tokens)
            # OPTIMIZATION:  Note that get_max_num_running_seqs is
            # expensive. For the default scheduling chase where
            # enable_chunking is False, num_seqs are updated before running
            # this method, so we don't have to update it again here.
            if enable_chunking:
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.add_num_seqs(seq_group.request_id, num_running_seqs)

        return running_waiting_queue, SchedulerRunningOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=prefill_seq_groups,
            preempted=[],
            swapped_out=[],
            blocks_to_swap_out=[],
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=False),
        )

    def _schedule_chunked_prefill_head_of_line(self):
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        enable_chunking = True
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        remaining_waiting, prefills = (
            self.waiting,
            SchedulerPrefillOutputs.create_empty(),
        )

        running_waiting = self.running + self.waiting
        for seq_group in running_waiting:
            seq_group.get_seqs()[0].status = SequenceStatus.RUNNING
        if running_waiting:
            seq_group = running_waiting[0]
            running_waiting.popleft()
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget
            )
            self.running = deque([seq_group])
            if (
                seq_group.get_num_uncomputed_tokens()
                == seq_group.get_seqs()[0].get_prompt_len()
            ):
                seq_group.get_seqs()[0].status = SequenceStatus.WAITING
                self._allocate_and_set_running(seq_group)
            else:
                self._append_slots(seq_group, blocks_to_copy)
            prefill_seq_groups.append(
                ScheduledSequenceGroup(
                    seq_group=seq_group, token_chunk_size=num_running_tokens
                )
            )
        self.waiting = deque(running_waiting)

        blocks_to_copy: List[Tuple[int, int]] = []

        running_scheduled = SchedulerRunningOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=prefill_seq_groups,
            preempted=[],
            swapped_out=[],
            blocks_to_swap_out=[],
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=False),
        )

        return PDSchedulerOutputs(
            scheduled_seq_groups=(
                prefills.seq_groups
                + running_scheduled.prefill_seq_groups
                + running_scheduled.decode_seq_groups
            ),
            num_prefill_groups=(
                len(prefills.seq_groups) + len(running_scheduled.prefill_seq_groups)
            ),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_swap_in=[],
            blocks_to_copy=running_scheduled.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(
                len(running_scheduled.preempted) + len(running_scheduled.swapped_out)
            ),
            blocks_to_migration=[],
            migration_seq_groups=[],
            preempted_seq_groups=running_scheduled.preempted,
            time_stamp=time.time(),
        )

    def _schedule_chunked_prefill_with_slo_aware_preemption(self):
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )

        remaining_waiting, prefills = (
            self.waiting,
            SchedulerPrefillOutputs.create_empty(),
        )

        remaining_running, running_scheduled = (
            self.running,
            SchedulerRunningOutputs.create_empty(),
        )
        # Decoding should be always scheduled first by fcfs.
        remaining_waiting, running_scheduled = (
            self._schedule_slo_aware_chunked_preemption(
                self.running, self.waiting, budget
            )
        )

        for seq_group in remaining_waiting:
            seq_group.get_seqs()[0].status = SequenceStatus.WAITING

        assert budget.num_batched_tokens <= self.scheduler_config.max_num_batched_tokens
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting = remaining_waiting
        # Update new running requests.
        # self.running = remaining_running
        self.running = deque([])
        self.running.extend([s.seq_group for s in running_scheduled.prefill_seq_groups])
        # 此处调度仅包含没有 prefill 的阶段，所以没有 swapped in 和 swapped out
        # TODO: Handle Swapped out preemption in Decode Instance
        self.swapped = deque()
        return PDSchedulerOutputs(
            scheduled_seq_groups=(
                prefills.seq_groups
                + running_scheduled.prefill_seq_groups
                + running_scheduled.decode_seq_groups
            ),
            num_prefill_groups=(
                len(prefills.seq_groups) + len(running_scheduled.prefill_seq_groups)
            ),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_swap_in=[],
            blocks_to_copy=running_scheduled.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(
                len(running_scheduled.preempted) + len(running_scheduled.swapped_out)
            ),
            blocks_to_migration=[],
            migration_seq_groups=[],
            preempted_seq_groups=running_scheduled.preempted,
            time_stamp=time.time(),
        )

    def _schedule_default(self) -> SchedulerOutputs:
        """Schedule queued requests.

        The current policy is designed to optimize the throughput. First,
        it batches as many prefill requests as possible. And it schedules
        decodes. If there's a pressure on GPU memory, decode requests can
        be swapped or preempted.
        """
        # Include running requests to the budget.
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        # Make sure we include num running seqs before scheduling prefill,
        # so that we don't schedule beyond max_num_seqs for prefill.
        for seq_group in self.running:
            budget.add_num_seqs(
                seq_group.request_id, seq_group.get_max_num_running_seqs()
            )
        curr_loras = (
            set(
                seq_group.lora_int_id
                for seq_group in self.running
                if seq_group.lora_int_id > 0
            )
            if self.lora_enabled
            else None
        )

        remaining_waiting, prefills = (
            self.waiting,
            SchedulerPrefillOutputs.create_empty(),
        )

        remaining_migrating, migrations = (
            self.waiting_for_migration,
            SchedulerMigrateOutputs.create_empty(),
        )
        remaining_running, running_scheduled = (
            self.running,
            SchedulerRunningOutputs.create_empty(),
        )
        remaining_swapped, swapped_in = (
            self.swapped,
            SchedulerSwappedInOutputs.create_empty(),
        )

        # If any requests are swapped, prioritized swapped requests.
        if not self.swapped:
            remaining_migrating, migrations = self._schedule_migrates(
                self.waiting_for_migration, budget, curr_loras, enable_chunking=False
            )
            # patch: sort prefills by edf
            remaining_waiting, prefills = self._schedule_prefills(
                self.waiting, budget, curr_loras, enable_chunking=False
            )

        fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Don't schedule decodes if prefills are scheduled.
        # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
        # only contains decode requests, not chunked prefills.
        if len(prefills.seq_groups) == 0:
            remaining_running, running_scheduled = self._schedule_running(
                self.running, budget, curr_loras, fcfs_policy, enable_chunking=False
            )

            # If any sequence group is preempted, do not swap in any sequence
            # group. because it means there's no slot for new running requests.
            if (
                len(running_scheduled.preempted) + len(running_scheduled.swapped_out)
                == 0
            ):
                remaining_swapped, swapped_in = self._schedule_swapped(
                    self.swapped, budget, curr_loras, fcfs_policy
                )

        assert budget.num_batched_tokens <= self.scheduler_config.max_num_batched_tokens
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        self.waiting_for_migration = remaining_migrating
        self.running_for_migration = deque([s.seq_group for s in migrations.seq_groups])

        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend([s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend([s.seq_group for s in swapped_in.decode_seq_groups])

        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)
        preempted = len(running_scheduled.preempted) + len(
            running_scheduled.swapped_out
        )

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0
        return PDSchedulerOutputs(
            scheduled_seq_groups=(
                prefills.seq_groups
                + running_scheduled.decode_seq_groups
                + swapped_in.decode_seq_groups
            ),
            num_prefill_groups=len(prefills.seq_groups),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy + swapped_in.blocks_to_copy,
            ignored_seq_groups=migrations.ignored_seq_groups
            + swapped_in.infeasible_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
            blocks_to_migration=migrations.blocks_to_migration,
            migration_seq_groups=migrations.seq_groups,
            preempted_seq_groups=running_scheduled.preempted,
            time_stamp=time.time(),
        )

    def _schedule_chunked_prefill(self):
        """Schedule queued requests.

        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to blocked
        by prefill requests.
        """
        if self.schedule_policy in [SchedulePolicy.SJF]:
            self.waiting = deque(
                sorted(
                    self.waiting,
                    key=lambda x: x.get_seqs()[0].get_prompt_len(),
                    reverse=False,
                )
            )
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        remaining_waiting, prefills = (
            self.waiting,
            SchedulerPrefillOutputs.create_empty(),
        )
        remaining_running, running_scheduled = (
            self.running,
            SchedulerRunningOutputs.create_empty(),
        )
        remaining_swapped, swapped_in = (
            self.swapped,
            SchedulerSwappedInOutputs.create_empty(),
        )

        # Decoding should be always scheduled first by fcfs.
        fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")
        remaining_running, running_scheduled = self._schedule_running(
            self.running, budget, curr_loras, fcfs_policy, enable_chunking=True
        )

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(running_scheduled.swapped_out) == 0:
            remaining_swapped, swapped_in = self._schedule_swapped(
                self.swapped, budget, curr_loras, fcfs_policy
            )

        remaining_migrating, migrations = self._schedule_migrates(
            self.waiting_for_migration, budget, curr_loras, enable_chunking=False
        )

        if not migrations.seq_groups:
            # Schedule new prefills.
            remaining_waiting, prefills = self._schedule_prefills(
                self.waiting, budget, curr_loras, enable_chunking=True
            )

        assert budget.num_batched_tokens <= self.scheduler_config.max_num_batched_tokens
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend([s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend([s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend([s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend([s.seq_group for s in swapped_in.prefill_seq_groups])

        self.waiting_for_migration = remaining_migrating
        self.running_for_migration = deque([s.seq_group for s in migrations.seq_groups])

        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)

        preempted = len(running_scheduled.preempted) + len(
            running_scheduled.swapped_out
        )

        return PDSchedulerOutputs(
            scheduled_seq_groups=(
                prefills.seq_groups
                + running_scheduled.prefill_seq_groups
                + swapped_in.prefill_seq_groups
                + running_scheduled.decode_seq_groups
                + swapped_in.decode_seq_groups
            ),
            num_prefill_groups=(
                len(prefills.seq_groups)
                + len(swapped_in.prefill_seq_groups)
                + len(running_scheduled.prefill_seq_groups)
            ),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy + swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups
            + swapped_in.infeasible_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
            blocks_to_migration=migrations.blocks_to_migration,
            migration_seq_groups=migrations.seq_groups,
            preempted_seq_groups=running_scheduled.preempted,
            time_stamp=time.time(),
        )

    def _schedule(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        """
        patch: scheduler_policy
        """
        if self.slo_aware_chunked_preemption:
            return self._schedule_chunked_prefill_with_slo_aware_preemption()
        elif self.scheduler_config.chunked_prefill_enabled:
            return self._schedule_chunked_prefill()
        else:
            return self._schedule_default()
