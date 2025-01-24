import dataclasses
import logging
import time

import matplotlib.pyplot as plt

import ray

logging.basicConfig(level=logging.INFO)

# 配置日志
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


import torch

from sgir_distserve import _C


# llama-3-8B as default
@dataclasses.dataclass
class ModelConfig:
    num_layers: int = 32
    num_total_kv_heads: int = 8
    head_size: int = 128
    dtype: torch.dtype = torch.half
    device: str = "cuda"


@dataclasses.dataclass
class CacheConfig:
    block_size: int = 16
    total_num_gpu_blocks: int = 23000


@dataclasses.dataclass
class ParallelConfig:
    tp_size: int = 1
    pp_size: int = 1


class Instance:
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ):
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config

        self.kv = torch.empty(
            [
                model_config.num_layers,
                2,
                cache_config.total_num_gpu_blocks,
                cache_config.block_size,
                model_config.num_total_kv_heads,
                model_config.head_size,
            ],
            dtype=model_config.dtype,
            device=model_config.device,
        )
        self.kv = [self.kv[i] for i in range(model_config.num_layers)]
        self.migration_manager_ptr = None

    def get_handlers(self):
        handles = []
        offsets = []

        for kv in self.kv:
            (
                _,
                handle,
                _,
                storage_offset_bytes,
                _,
                _,
                _,
                _,
            ) = kv.untyped_storage()._share_cuda_()
            handles.append(handle)
            offsets.append(storage_offset_bytes)

        return handles, offsets

    def init_migration_manager(
        self,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ):
        self.migration_manager_ptr = _C.ops.init_migration_manager(
            # rank
            0,
            self.model_config.dtype.itemsize,
            self.model_config.num_layers,
            self.cache_config.block_size,
            self.model_config.num_total_kv_heads,
            self.model_config.head_size,
            parallel_config.tp_size,
            parallel_config.pp_size,
            cache_config.total_num_gpu_blocks,
            self.parallel_config.tp_size,
            self.parallel_config.pp_size,
            self.cache_config.total_num_gpu_blocks,
        )

    def register_handler(self, handles, offsets):
        _C.ops.register_handler(self.migration_manager_ptr, handles, offsets)

    def migrate(self, block_mapping):
        begin = time.time()
        _C.ops.migrate(
            self.migration_manager_ptr,
            self.kv,
            block_mapping,
        )
        torch.cuda.synchronize()
        end = time.time()

        num_layers = self.model_config.num_layers
        num_heads = self.model_config.num_total_kv_heads
        head_size = self.model_config.head_size

        num_blocks = block_mapping.size(0)
        block_size = self.cache_config.block_size

        num_bytes_per_elems = self.model_config.dtype.itemsize

        total_bytes = (
            num_layers
            * 2
            * num_blocks
            * block_size
            * num_heads
            * head_size
            * num_bytes_per_elems
        )
        # print(f"latency: {end - begin}")
        # print(f"total bytes: {total_bytes}")
        print(f"bandwidth: {round(total_bytes / (end - begin) / 1e9, 2)}GBps")

        return round(total_bytes / (end - begin) / 1e9, 2)


ray.init(log_to_driver=True)
model_config = ModelConfig()

prefill_cache_config = CacheConfig()
decode_cache_config = CacheConfig()

prefill_parallel_config = ParallelConfig()
decode_parallel_config = ParallelConfig()


prefill_instance = ray.remote(num_gpus=1)(Instance).remote(
    model_config=model_config,
    cache_config=prefill_cache_config,
    parallel_config=prefill_parallel_config,
)
decode_instance = ray.remote(num_gpus=1)(Instance).remote(
    model_config=model_config,
    cache_config=decode_cache_config,
    parallel_config=decode_parallel_config,
)
handler, offset = ray.get(prefill_instance.get_handlers.remote())

ray.get(
    decode_instance.init_migration_manager.remote(
        prefill_cache_config, prefill_parallel_config
    )
)

ray.get(decode_instance.register_handler.remote([[handler]], [[offset]]))

idx = []
for i in range(128):
    idx.append([0, i, i])

# warm up
ray.get(decode_instance.migrate.remote(torch.tensor(idx)))
ray.get(decode_instance.migrate.remote(torch.tensor(idx)))
ray.get(decode_instance.migrate.remote(torch.tensor(idx)))
ray.get(decode_instance.migrate.remote(torch.tensor(idx)))
ray.get(decode_instance.migrate.remote(torch.tensor(idx)))
ray.get(decode_instance.migrate.remote(torch.tensor(idx)))
ray.get(decode_instance.migrate.remote(torch.tensor(idx)))
ray.get(decode_instance.migrate.remote(torch.tensor(idx)))
ray.get(decode_instance.migrate.remote(torch.tensor(idx)))
ray.get(decode_instance.migrate.remote(torch.tensor(idx)))
ray.shutdown()


def profile(
    num_layers,
    num_heads,
    head_size,
    block_size,
    num_total_blocks,
    num_idx,
):
    ray.init(log_to_driver=True)
    model_config = ModelConfig(
        num_layers=num_layers,
        num_total_kv_heads=num_heads,
        head_size=head_size,
    )

    prefill_cache_config = CacheConfig(
        block_size=block_size,
        total_num_gpu_blocks=num_total_blocks,
    )
    decode_cache_config = CacheConfig(
        block_size=block_size,
        total_num_gpu_blocks=num_total_blocks,
    )

    prefill_parallel_config = ParallelConfig()
    decode_parallel_config = ParallelConfig()

    prefill_instance = ray.remote(num_gpus=1)(Instance).remote(
        model_config=model_config,
        cache_config=prefill_cache_config,
        parallel_config=prefill_parallel_config,
    )
    decode_instance = ray.remote(num_gpus=1)(Instance).remote(
        model_config=model_config,
        cache_config=decode_cache_config,
        parallel_config=decode_parallel_config,
    )
    handler, offset = ray.get(prefill_instance.get_handlers.remote())

    ray.get(
        decode_instance.init_migration_manager.remote(
            prefill_cache_config, prefill_parallel_config
        )
    )

    ray.get(decode_instance.register_handler.remote([[handler]], [[offset]]))

    idx = []
    for i in range(num_idx):
        idx.append([0, i, i])

    sum = 0
    iter = 10
    for i in range(iter):
        sum += ray.get(decode_instance.migrate.remote(torch.tensor(idx)))
    bw = sum / iter
    ray.shutdown()
    return bw


pitches = []
bws = []
print("pitch 16, slice pitch 1024")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,       1024,         1,         16,  256
pitches.append(16)
bws.append(
    profile(
        32,
        1,
        16,
        1024,
        20000,
        256,
    )
)

print("pitch 32, slice pitch 512")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,       512,          1,        32,  256
pitches.append(32)
bws.append(
    profile(
        32,
        1,
        32,
        512,
        20000,
        256,
    )
)

print("pitch 64, slice pitch 256")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,       256,         1,         64,  256
pitches.append(64)
bws.append(
    profile(
        32,
        1,
        64,
        256,
        20000,
        256,
    )
)

print("pitch 128, slice pitch 128")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,       128,         1,        128,  256
pitches.append(128)
bws.append(
    profile(
        32,
        1,
        128,
        128,
        20000,
        256,
    )
)

print("pitch 256, slice pitch 64")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         64,         2,       128,  256
pitches.append(256)
bws.append(
    profile(
        32,
        2,
        128,
        64,
        20000,
        256,
    )
)


print("pitch 512, slice pitch 32")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         32,         4,       128,  256
pitches.append(512)
bws.append(
    profile(
        32,
        4,
        128,
        32,
        20000,
        256,
    )
)


print("pitch 1024, slice pitch 16")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         16,         8,       128,  256
pitches.append(1024)
bws.append(
    profile(
        32,
        8,
        128,
        16,
        20000,
        256,
    )
)

print("pitch 2048, slice pitch 8")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         8,         16,       128,  256
pitches.append(2048)
bws.append(
    profile(
        32,
        16,
        128,
        8,
        20000,
        256,
    )
)

print("pitch 4096, slice pitch 4")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         4,         32,       128,  256
pitches.append(4096)
bws.append(
    profile(
        32,
        32,
        128,
        4,
        20000,
        256,
    )
)

print("pitch 8192, slice pitch 2")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         2,         64,       128,  256
pitches.append(8192)
bws.append(
    profile(
        32,
        64,
        128,
        2,
        20000,
        256,
    )
)


print("pitch 16384, slice pitch 1")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         1,        128,       128,  256
pitches.append(16384)
bws.append(
    profile(
        32,
        128,
        128,
        1,
        20000,
        256,
    )
)


print("pitch 32768, slice pitch 1")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         1,        256,       128,  128
pitches.append(32768)
bws.append(
    profile(
        32,
        256,
        128,
        1,
        20000,
        128,
    )
)

print("pitch 65536, slice pitch 1")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         1,        512,       128,   64
pitches.append(65536)
bws.append(
    profile(
        32,
        512,
        128,
        1,
        10000,
        64,
    )
)

print("pitch 65536 * 2, slice pitch 1")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         1,        1024,       128,   32
pitches.append(65536 * 2)
bws.append(
    profile(
        32,
        1024,
        128,
        1,
        5000,
        32,
    )
)

print("pitch 65536 * 4, slice pitch 1")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         1,       2048,       128,   16
pitches.append(65536 * 4)
bws.append(
    profile(
        32,
        2048,
        128,
        1,
        2500,
        16,
    )
)

print("pitch 65536 * 8, slice pitch 1")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         1,       4096,       128,   8
pitches.append(65536 * 8)
bws.append(
    profile(
        32,
        4096,
        128,
        1,
        1250,
        8,
    )
)

print("pitch 65536 * 16, slice pitch 1")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         1,       8192,       128,   4
pitches.append(65536 * 16)
bws.append(
    profile(
        32,
        8192,
        128,
        1,
        625,
        4,
    )
)


print("pitch 65536 * 32, slice pitch 1")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         1,       16384,       128,   2
pitches.append(65536 * 32)
bws.append(
    profile(
        32,
        16384,
        128,
        1,
        312,
        2,
    )
)


print("pitch 65536 * 64, slice pitch 1")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         1,       32768,       128,   1

pitches.append(65536 * 64)
bws.append(
    profile(
        32,
        32768,
        128,
        1,
        150,
        1,
    )
)

print("pitch 65536 * 128, slice pitch 1")
# layers, kv, total_blocks, block_size, num_heads, head_size, #idx
#     32,  2,        20000,         1,       32768,       128,   1
pitches.append(65536 * 128)
bws.append(
    profile(
        16,
        65536,
        128,
        1,
        150,
        1,
    )
)

pitches.append(65536 * 128)
bws.append(
    profile(
        1,
        65536 * 16 * 150,
        128,
        1,
        1,
        1,
    )
)


threshold_bw = 600  # 阈值 (单位 GBps)

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 绘制曲线
ax.plot(pitches, bws, marker="o", linestyle="-", color="b", label="Bandwidth (GBps)")

# 绘制阈值线
ax.axhline(
    y=threshold_bw, color="r", linestyle="--", label=f"Threshold ({threshold_bw} GBps)"
)

ax.set_xscale("log")

# 添加标签和标题
ax.set_xlabel("Pitches (bytes)")
ax.set_ylabel("Bandwidth (GBps)")
ax.set_title("Pitches vs Bandwidth with Threshold Line")

# 添加图例
ax.legend()

# 显示图形
plt.grid(True)
plt.savefig("x.png")
