from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from vllm.config import EngineConfig
from vllm.engine.arg_utils import AsyncEngineArgs, FlexibleArgumentParser

from sgir_distserve.config import (
    BatchConfig,
    ReplicaConfig,
    SGIRDistserveConfig,
    SGIRDistserveSchedulerConfig,
)

from sgir_distserve.core.global_scheduler import (
    EngineKind,
    get_dispatch_policy_by_name,
    get_schedule_policy_by_name,
)


@dataclass
class SGIRDistserveArgs(AsyncEngineArgs):
    # for replica config
    partition_id: Optional[int] = None
    replica_spec: Tuple[str] = ()
    prometheus_port: Tuple[str] = ()

    # schedule config
    schedule_policy: str = "fcfs"
    dispatch_policy: str = "round-robin"

    # mode
    callback_mode: str = "ipc"

    # for prefill
    prefill_max_num_seqs: int = 256
    prefill_max_num_batched_tokens: Optional[int] = None
    dummy_prefill: bool = False
    slo_aware_chunked_preemption: bool = False
    slo_aware_multiplexing: bool = False
    max_multiplexing_length: int = 8192
    multiplexing_decode_batch_latency_watermark: int = 0.85
    multiplexing_decode_batched_tokens_watermark: int = 400000
    prefill_gpu_memory_utilization: float = 0.9

    # for decode
    decode_max_num_seqs: int = 256
    decode_max_num_batched_tokens: Optional[int] = None
    decode_gpu_memory_utilization: float = 0.9

    def create_engine_config(self) -> EngineConfig:
        engine_config = super().create_engine_config()
        scheduler_config = engine_config.scheduler_config
        schedule_policy = get_schedule_policy_by_name(self.schedule_policy)

        patched_scheduler_config = SGIRDistserveSchedulerConfig(
            max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
            max_num_seqs=scheduler_config.max_num_seqs,
            max_model_len=scheduler_config.max_model_len,
            use_v2_block_manager=scheduler_config.use_v2_block_manager,
            num_lookahead_slots=scheduler_config.num_lookahead_slots,
            delay_factor=scheduler_config.delay_factor,
            enable_chunked_prefill=scheduler_config.chunked_prefill_enabled,
            preemption_mode=scheduler_config.preemption_mode,
            slo_aware_chunked_preemption=self.slo_aware_chunked_preemption,
            schedule_policy=schedule_policy,
        )

        return EngineConfig(
            model_config=engine_config.model_config,
            cache_config=engine_config.cache_config,
            parallel_config=engine_config.parallel_config,
            scheduler_config=patched_scheduler_config,
            device_config=engine_config.device_config,
            load_config=engine_config.load_config,
            lora_config=engine_config.lora_config,
            multimodal_config=engine_config.multimodal_config,
            speculative_config=engine_config.speculative_config,
            decoding_config=engine_config.decoding_config,
            observability_config=engine_config.observability_config,
            prompt_adapter_config=engine_config.prompt_adapter_config,
        )

    def create_sgir_distserve_config(self) -> SGIRDistserveConfig:
        replica_spec = [eval(replica) for replica in self.replica_spec]
        engine_config = []
        partition_id = self.partition_id or int(np.ceil(len(replica_spec) / 2))
        self.schedule_policy = self.schedule_policy.lower()

        schedule_policy = get_schedule_policy_by_name(self.schedule_policy)
        dispatch_policy = get_dispatch_policy_by_name(self.dispatch_policy)

        prometheus_port = self.prometheus_port or [None] * len(replica_spec)

        for spec_id, (spec, prometheus_port) in enumerate(
            zip(replica_spec, prometheus_port)
        ):
            tp_size, pp_size = spec
            replica_args = deepcopy(self)
            replica_args.tensor_parallel_size = tp_size
            replica_args.pipeline_parallel_size = pp_size
            # configuration inject
            if spec_id < partition_id:
                replica_args.max_num_seqs = self.prefill_max_num_seqs
                replica_args.max_num_batched_tokens = (
                    self.prefill_max_num_batched_tokens
                )
                replica_args.gpu_memory_utilization = (
                    self.prefill_gpu_memory_utilization
                )
            else:
                replica_args.max_num_seqs = self.decode_max_num_seqs
                replica_args.max_num_batched_tokens = self.decode_max_num_batched_tokens
                replica_args.gpu_memory_utilization = self.decode_gpu_memory_utilization

            if partition_id == len(replica_spec):
                engine_kind = EngineKind.Hybrid
            elif spec_id < partition_id:
                engine_kind = EngineKind.Prefill
            else:
                engine_kind = EngineKind.Decode

            engine_config.append(
                ReplicaConfig(
                    engine_kind,
                    replica_args.create_engine_config(),
                    prometheus_port=prometheus_port,
                )
            )

        distserve_config = SGIRDistserveConfig(
            partition_id=partition_id,
            schedule_policy=schedule_policy,
            dispatch_policy=dispatch_policy,
            dummy_prefill=self.dummy_prefill,
            slo_aware_multiplexing=self.slo_aware_multiplexing,
            max_multiplexing_length=self.max_multiplexing_length,
            multiplexing_decode_batch_latency_watermark=self.multiplexing_decode_batch_latency_watermark,
            multiplexing_decode_batched_tokens_watermark=self.multiplexing_decode_batched_tokens_watermark,
            prefill_engine_batch_config=BatchConfig(
                max_num_seqs=replica_args.max_num_seqs,
                max_num_batched_tokens=replica_args.prefill_max_num_batched_tokens,
                enable_chunked_prefill=replica_args.enable_chunked_prefill,
            ),
            decode_engine_batch_config=BatchConfig(
                max_num_seqs=replica_args.max_num_seqs,
                max_num_batched_tokens=replica_args.decode_max_num_batched_tokens,
                enable_chunked_prefill=replica_args.enable_chunked_prefill,
            ),
            replica_config=engine_config,
            block_size=self.block_size,
            max_model_len=self.max_model_len,
        )

        return distserve_config

    @staticmethod
    def add_cli_args(
        parser: FlexibleArgumentParser, async_args_only: bool = False
    ) -> FlexibleArgumentParser:
        parser = AsyncEngineArgs.add_cli_args(parser, async_args_only)
        parser.add_argument(
            "--partition-id", type=int, help="partition id", required=False
        )
        parser.add_argument(
            "--replica-spec",
            nargs="+",
            type=str,
            help="--replica-spec tp_size_0,dp_size_0;tp_size_1,dp_size_1;...",
            default=("1,1",),
        )
        parser.add_argument("--prometheus-port", type=int, default=None, nargs="+")

        parser.add_argument(
            "--dispatch-policy",
            "--disp-policy",
            type=str,
            help="--dispatch-policy",
            required=False,
            default=SGIRDistserveArgs.dispatch_policy,
            choices=[
                "round-robin",
                "slack",
                "least-num-unfinished-tokens",
                "least-num-unfinished-jobs",
            ],
        )

        parser.add_argument(
            "--schedule-policy",
            "-sch-policy",
            type=str,
            help="--schedule-policy",
            required=False,
            default=SGIRDistserveArgs.schedule_policy,
            choices=["fcfs", "sjf", "ljf", "edf", "llf"],
        )

        parser.add_argument(
            "--slo-aware-chunked-preemption",
            "-chunked-preemption",
            help="--chunked-preemption",
            action="store_true",
        )

        parser.add_argument(
            "--slo-aware-multiplexing",
            "-multiplexing",
            help="--slo-aware-multiplexing",
            action="store_true",
        )

        parser.add_argument(
            "--max-multiplexing-length",
            "-multiplexing-length",
            help="--prefill-max-multiplexing-length",
            type=int,
            default=8192,
        )

        parser.add_argument(
            "--multiplexing-decode-batch-latency-watermark",
            "-multiplexing-latency-watermark",
            help="--multiplexing-decode-batch-latency-watermark",
            type=float,
            default=0.85,
        )

        parser.add_argument(
            "--multiplexing-decode-batched-tokens-watermark",
            "-multiplexing-token-watermark",
            help="--multiplexing-decode-batched-tokens-watermark",
            type=int,
            default=400000,
        )

        parser.add_argument(
            "--dummy-prefill",
            "-dummy-p",
            help="--dummy-prefill",
            action="store_true",
        )

        parser.add_argument(
            "--prefill-max-num-batched-tokens",
            type=int,
            required=False,
            default=SGIRDistserveArgs.prefill_max_num_batched_tokens,
            help="Maximum number of batched tokens per " "iteration.",
        )

        parser.add_argument(
            "--prefill-max-num-seqs",
            type=int,
            required=False,
            default=SGIRDistserveArgs.prefill_max_num_seqs,
            help="Maximum number of sequences per iteration.",
        )

        parser.add_argument(
            "--prefill-gpu-memory-utilization",
            type=float,
            default=SGIRDistserveArgs.prefill_gpu_memory_utilization,
        )

        parser.add_argument(
            "--decode-max-num-batched-tokens",
            type=int,
            default=SGIRDistserveArgs.decode_max_num_batched_tokens,
            help="Maximum number of batched tokens per " "iteration.",
        )

        parser.add_argument(
            "--decode-max-num-seqs",
            type=int,
            default=SGIRDistserveArgs.decode_max_num_seqs,
            help="Maximum number of sequences per iteration.",
        )

        parser.add_argument(
            "--decode-gpu-memory-utilization",
            type=float,
            default=SGIRDistserveArgs.decode_gpu_memory_utilization,
        )

        parser.add_argument(
            "--callback-mode",
            type=str,
            default=SGIRDistserveArgs.callback_mode,
            choices=["async", "ipc"],
        )

        return parser
