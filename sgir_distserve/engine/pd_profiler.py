from collections import namedtuple
from typing import List, Union

import numpy as np

from sgir_distserve.sequence import DistserveSequenceGroup
from sgir_distserve.utils.db_op import (
    query_chunk_profiler,
    query_decode_profiler,
    query_prefill_profiler,
)


class PDProfiler:
    def __init__(
        self,
        model: str = "NousResearch/Meta-Llama-3-8B-Instruct",
        tp: int = 2,
        pp=1,
        chunk=4096,
    ):
        self.model = model
        self.tp = tp
        self.pp = pp
        self.chunk = chunk
        self.indb = True
        self.chunk_size = chunk
        prefill_config = query_prefill_profiler(model, tp, pp, chunk)
        decode_config = query_decode_profiler(model, tp, pp)
        chunk_config = query_chunk_profiler(model, tp, pp, chunk)

        if (not prefill_config) or (not decode_config):
            self.indb = False
            return

        self.prefill_coeff = np.array(
            [prefill_config[0][-3], prefill_config[0][-2], prefill_config[0][-1]]
        )
        self.decode_coeff = np.array([decode_config[0][-2], decode_config[0][-1]])
        if chunk != -1:
            self.chunk_coeff = np.array([chunk_config[0][-2], chunk_config[0][-1]])

    def __repr__(self):
        return f"PDProfiler, model={self.model}, tp={self.tp}, pp={self.pp}, chunk={self.chunk}"

    def get_chunk_seq_prediction(self, seq_group: DistserveSequenceGroup):
        num_computed_tokens = seq_group.get_seqs()[0].data.get_num_computed_tokens()
        if self.chunk_size < 0:
            return self.get_prefill_seq_prediction(seq_group)
        chunk_idx = np.floor(num_computed_tokens / self.chunk_size)
        chunk_param = np.array([chunk_idx, 1])
        return np.dot(chunk_param, self.chunk_coeff)

    def get_prefill_seq_prediction(
        self,
        seq_group: Union[DistserveSequenceGroup, int],
    ):
        # by now only supported Meta-Llama-3-8B-Insturct (from 4 ~ 8000)
        if isinstance(seq_group, DistserveSequenceGroup):
            seq_len = seq_group.get_seqs()[0].get_prompt_len()
        else:
            seq_len = seq_group

        seq_param = np.array([seq_len * seq_len, seq_len, 1])
        pred_duration = np.dot(seq_param, self.prefill_coeff)
        if isinstance(seq_group, DistserveSequenceGroup):
            num_computed = seq_group.get_seqs()[0].data.get_num_computed_tokens()
            if num_computed > 0:
                pred_duration -= self.get_prefill_seq_prediction(num_computed)
        return pred_duration

    def get_decode_seq_prediction(self, seq_groups: List[DistserveSequenceGroup]):
        seq_len = 0
        for seq_group in seq_groups:
            seq_len += (
                seq_group.get_seqs()[0].get_prompt_len()
                + seq_group.get_seqs()[0].get_output_len()
            )
        seq_param = np.array([seq_len, 1])
        return np.dot(seq_param, self.decode_coeff)

    def get_prefill_slack(self, seq_group: DistserveSequenceGroup, now: float):
        slack = (
            seq_group.metrics.arrival_time
            + seq_group.ttft_slo
            - now
            - self.get_prefill_seq_prediction(seq_group)
        )

        return slack

    def get_decode_slack(
        self,
        seq_group: DistserveSequenceGroup,
        now: float,
        spec_slack_tokens=1,
    ):
        # 忽略掉最新一次的 decode 对 slack 的影响
        return (
            seq_group.metrics.first_token_time
            + seq_group.tpot_slo
            * max(spec_slack_tokens, seq_group.get_seqs()[0].data.get_output_len())
            - now
        )

    def get_num_tokens_watermarks_by_slo(self, tpot_slo: float):
        return (tpot_slo - self.decode_coeff[1]) / self.decode_coeff[0]

    def get_queue_pred_end_time(self, seq_groups: List[DistserveSequenceGroup]):
        if not seq_groups:
            return np.array([])
        preds = [self.get_prefill_seq_prediction(seq_group) for seq_group in seq_groups]
        return np.cumsum(preds) - preds[0]

    def get_queue_pred_queue_time(self, seq_groups: List[DistserveSequenceGroup]):
        if not seq_groups:
            return np.array([])
        end_time = self.get_queue_pred_end_time(seq_groups)
        return end_time - end_time[0]

    def get_queue_slack_time(self, seq_groups: List[DistserveSequenceGroup], now):
        return np.array(
            [self.get_prefill_slack(seq_group, now) for seq_group in seq_groups]
        )
