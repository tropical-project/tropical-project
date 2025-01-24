from typing import Dict, Optional

from vllm.sampling_params import SamplingParams


class DistserveSamplingParams(SamplingParams):
    ttft_slo: Optional[float] = None
    tpot_slo: Optional[float] = None

    def set_distserve_params(self, kwargs: Dict):
        self.ttft_slo = kwargs.pop("ttft_slo", None)
        self.tpot_slo = kwargs.pop("tpot_slo", None)
        self.attributes = kwargs
