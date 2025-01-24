import os
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, TYPE_CHECKING, Union

import numpy as np
import prometheus_client

from vllm.executor.ray_utils import ray
from vllm.logger import init_logger

if ray is not None:
    from ray.util import metrics as ray_metrics
else:
    ray_metrics = None

if TYPE_CHECKING:
    from vllm.spec_decode.metrics import SpecDecodeWorkerMetrics

from vllm.engine.metrics import (
    get_throughput,
    local_interval_elapsed,
    LoggingStatLogger,
    Metrics,
    PrometheusStatLogger,
    Stats,
)
from vllm.utils import enable_trace_function_call

from sgir_distserve.config import EngineKind

logger = init_logger("vllm.engine.metrics")


class DistserveMetrics(Metrics):
    def __init__(self, labelnames: List[str], max_model_len: int):
        super().__init__(labelnames, max_model_len)
        self._create_sgir_distserve_config()

        self.gauge_global_scheduler_num_arrived_prompt_tokens = self._gauge_cls(
            name="sgir_distserve:num_arrived_prompt_tokens",
            documentation="sgir_distserve:num_arrived_prompt_tokens",
            labelnames=labelnames,
        )

        self.gauge_scheduler_running_for_migrating = self._gauge_cls(
            name=f"sgir_distserve:num_requests_running_for_migrating",
            documentation="Number of requests currently migrating on GPU.",
            labelnames=labelnames,
        )

        self.gauge_scheduler_batched_tokens = self._gauge_cls(
            name=f"sgir_distserve:num_batched_tokens",
            documentation="Number of requests currently migrating on GPU.",
            labelnames=labelnames,
        )

        self.gauge_scheduler_waiting_for_migrating = self._gauge_cls(
            name=f"sgir_distserve:num_requests_waiting_for_migrating",
            documentation="Number of requests currently waiting for migration on GPU.",
            labelnames=labelnames,
        )
        self.histogram_time_to_migration = self._histogram_cls(
            name=f"sgir_distserve:time_to_migration",
            documentation="Histogram of time per output token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.75,
                1.0,
                2.5,
            ],
        )

    def _create_sgir_distserve_config(self):
        self.sgir_distserve_config = prometheus_client.Info(
            name="sgir_distserve:instance_config", documentation="instance config"
        )


@dataclass
class DistserveStats(Stats):
    """Created by LLMEngine for use by StatLogger."""

    instance_id: Optional[int] = None
    num_batched_tokens: int = 0
    num_waiting_for_migrating: int = 0
    num_running_for_migrating: int = 0
    time_to_migration: Optional[List[float]] = None

    @classmethod
    def from_stats(
        cls,
        stats: Stats,
        instance_id: int,
        num_batched_tokens: int = 0,
        num_waiting_for_migrating: int = 0,
        num_running_for_migrating: int = 0,
        time_to_migration: int = None,
    ):
        return cls(
            instance_id=instance_id,
            num_waiting_for_migrating=num_waiting_for_migrating,
            num_running_for_migrating=num_running_for_migrating,
            time_to_migration=time_to_migration or [],
            num_batched_tokens=num_batched_tokens,
            **asdict(stats),
        )


@dataclass
class SGIR_DISTSERVE_LABEL:
    INSTANCE_ID_LABEL = "instance_id"
    MODEL_NAME = "model_name"


class DistservePrometheusStatLogger(PrometheusStatLogger):
    _metrics_cls = DistserveMetrics

    def __init__(
        self,
        local_interval: float,
        labels: Dict[str, str],
        max_model_len: int,
        metrics: DistserveMetrics = None,
    ) -> None:
        super(PrometheusStatLogger, self).__init__(local_interval)
        # Prometheus metrics
        self.labels = labels
        self.metrics: DistserveMetrics = metrics or self._metrics_cls(
            labelnames=[
                SGIR_DISTSERVE_LABEL.INSTANCE_ID_LABEL,
                SGIR_DISTSERVE_LABEL.MODEL_NAME,
            ],
            max_model_len=max_model_len,
        )
        self.num_batched_tokens = []

    def _log_prometheus(
        self,
        stats: DistserveStats,
    ) -> None:
        super()._log_prometheus(stats)
        self._log_gauge(
            self.metrics.gauge_scheduler_waiting_for_migrating,
            stats.num_waiting_for_migrating,
        )
        self._log_gauge(
            self.metrics.gauge_scheduler_running_for_migrating,
            stats.num_running_for_migrating,
        )

        self._log_histogram(
            self.metrics.histogram_time_to_migration, stats.time_to_migration
        )

    def log(self, stats: DistserveStats):
        labels_preserve = self.labels
        self.labels = self.labels[stats.instance_id]
        """Logs to prometheus and tracked stats every iteration."""
        # Log to prometheus.
        self._log_prometheus(stats)

        # Save tracked stats for token counters.
        self.num_prompt_tokens.append(stats.num_prompt_tokens_iter)
        self.num_generation_tokens.append(stats.num_generation_tokens_iter)

        # Update spec decode metrics
        self.maybe_update_spec_decode_metrics(stats)
        self.num_batched_tokens.append(stats.num_batched_tokens)

        # Log locally every local_interval seconds.
        if local_interval_elapsed(stats.now, self.last_local_log, self.local_interval):
            # Compute summary metrics for tracked stats (and log them
            # to promethus if applicable).
            prompt_throughput = get_throughput(
                self.num_prompt_tokens, now=stats.now, last_log=self.last_local_log
            )
            generation_throughput = get_throughput(
                self.num_generation_tokens, now=stats.now, last_log=self.last_local_log
            )

            self._log_prometheus_interval(
                prompt_throughput=prompt_throughput,
                generation_throughput=generation_throughput,
            )

            if self.spec_decode_metrics is not None:
                self._log_gauge(
                    self.metrics.gauge_spec_decode_draft_acceptance_rate,
                    self.spec_decode_metrics.draft_acceptance_rate,
                )
                self._log_gauge(
                    self.metrics.gauge_spec_decode_efficiency,
                    self.spec_decode_metrics.system_efficiency,
                )
                self._log_counter(
                    self.metrics.counter_spec_decode_num_accepted_tokens,
                    self.spec_decode_metrics.accepted_tokens,
                )
                self._log_counter(
                    self.metrics.counter_spec_decode_num_draft_tokens,
                    self.spec_decode_metrics.draft_tokens,
                )
                self._log_counter(
                    self.metrics.counter_spec_decode_num_emitted_tokens,
                    self.spec_decode_metrics.emitted_tokens,
                )

            self._log_gauge(
                self.metrics.gauge_scheduler_batched_tokens,
                int(sum(self.num_batched_tokens) / len(self.num_batched_tokens)),
            )

            # Reset tracked stats for next interval.
            self.num_prompt_tokens = []
            self.num_generation_tokens = []
            self.num_batched_tokens = []
            self.last_local_log = stats.now
            self.spec_decode_metrics = None
        self.labels = labels_preserve


class DistserveLoggingStatLogger(LoggingStatLogger):
    """StatLogger is used LLMEngine to log to Promethus and Stdout."""

    def __init__(self, local_interval: float) -> None:
        # Tracked stats over current local logging interval.
        self.num_prompt_tokens = defaultdict(list)
        self.num_generation_tokens = defaultdict(list)
        now = time.time()
        self.last_local_log = defaultdict(lambda: now)
        self.local_interval = local_interval
        self.spec_decode_metrics: Optional["SpecDecodeWorkerMetrics"] = None

    def log(self, stats: DistserveStats) -> None:
        """Called by LLMEngine.
        Logs to Stdout every self.local_interval seconds."""

        # Save tracked stats for token counters.
        self.num_prompt_tokens[stats.instance_id].append(stats.num_prompt_tokens_iter)
        self.num_generation_tokens[stats.instance_id].append(
            stats.num_generation_tokens_iter
        )

        # Update spec decode metrics
        self.maybe_update_spec_decode_metrics(stats)

        # Log locally every local_interval seconds.
        if local_interval_elapsed(
            stats.now, self.last_local_log[stats.instance_id], self.local_interval
        ):
            # Compute summary metrics for tracked stats (and log them
            # to promethus if applicable).
            prompt_throughput = get_throughput(
                self.num_prompt_tokens[stats.instance_id],
                now=stats.now,
                last_log=self.last_local_log[stats.instance_id],
            )
            generation_throughput = get_throughput(
                self.num_generation_tokens[stats.instance_id],
                now=stats.now,
                last_log=self.last_local_log[stats.instance_id],
            )

            # Log to stdout.
            logger.info(
                f"instance id: {stats.instance_id} "
                "Avg prompt throughput: %.1f tokens/s, "
                "Avg generation throughput: %.1f tokens/s, "
                "Running: %d reqs, Swapped: %d reqs, "
                "Migrating waiting: %d reqs, Migrating Running: %d reqs, "
                "Pending: %d reqs, GPU KV cache usage: %.1f%%, "
                "CPU KV cache usage: %.1f%%.",
                prompt_throughput,
                generation_throughput,
                stats.num_running_sys,
                stats.num_swapped_sys,
                stats.num_waiting_for_migrating,
                stats.num_running_for_migrating,
                stats.num_waiting_sys,
                stats.gpu_cache_usage_sys * 100,
                stats.cpu_cache_usage_sys * 100,
            )

            if self.spec_decode_metrics is not None:
                logger.info(
                    self._format_spec_decode_metrics_str(self.spec_decode_metrics)
                )

            # Reset tracked stats for next interval.
            self.num_prompt_tokens[stats.instance_id] = []
            self.num_generation_tokens[stats.instance_id] = []
            self.last_local_log[stats.instance_id] = stats.now
            self.spec_decode_metrics = None
