import os
import ray
import logging

import torch

from sgir_distserve import _C


class Source:
    def __init__(self,
                 value,
                 num_instance_prefill,
                 num_instance_decode,
                 rank,
                 num_layers,
                 num_gpu_blocks_prefill,
                 num_gpu_blocks_decode,
                 block_size,
                 total_num_heads,
                 head_size):
        self.rank = rank
        self.num_instance_prefill = num_instance_prefill
        self.num_instance_decode = num_instance_decode
        self.num_layers = num_layers
        self.num_gpu_blocks_prefill = num_gpu_blocks_prefill
        self.num_gpu_blocks_decode = num_gpu_blocks_decode
        self.block_size = block_size
        self.total_num_heads = total_num_heads
        self.head_size = head_size
        self.kv = torch.ones(num_layers,
                             2,
                             self.num_gpu_blocks_prefill,
                             self.block_size * self.total_num_heads // num_instance_prefill * self.head_size).half().cuda() * value
        self.num_bytes_per_elem = self.kv.itemsize
        self.kv = [self.kv[i] for i in range(num_layers)]

    def get_handles(self):
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


class Destination:
    def __init__(self,
                 value,
                 num_instance_prefill,
                 num_instance_decode,
                 rank,
                 num_layers,
                 num_gpu_blocks_prefill,
                 num_gpu_blocks_decode,
                 block_size,
                 total_num_heads,
                 head_size,
                 **kwargs):
        self.rank = rank
        self.num_instance_prefill = num_instance_prefill
        self.num_instance_decode = num_instance_decode
        self.num_layers = num_layers
        self.num_gpu_blocks_prefill = num_gpu_blocks_prefill
        self.num_gpu_blocks_decode = num_gpu_blocks_decode
        self.block_size = block_size
        self.total_num_heads = total_num_heads
        self.head_size = head_size
        self.kv = torch.ones(
            num_layers,
            2,
            self.num_gpu_blocks_decode,
            self.block_size * self.total_num_heads *
            self.head_size // self.num_instance_decode
        ).half().cuda() * value

        self.num_bytes_per_elem = self.kv.itemsize
        self.kv = [self.kv[i] for i in range(num_layers)]
        self.manager_ptr = _C.ops.init_migration_manager(
            self.rank,
            self.num_bytes_per_elem,
            self.num_layers,
            self.block_size,
            self.total_num_heads,
            self.head_size,
            self.num_instance_prefill,
            self.num_gpu_blocks_prefill,
            self.num_instance_decode,
            self.num_gpu_blocks_decode
        )

    def register_handles(self, handles, offsets):
        _C.ops.register_handler(self.manager_ptr,
                                handles,
                                offsets)

    def migrate(self, src_index, des_index):
        assert len(src_index) == len(des_index)
        _C.ops.migrate(self.manager_ptr,
                       self.kv,
                       torch.tensor(list(zip(des_index, src_index)))
                       )
        print("dump data")
        print(self.kv)
        print("dump data done")


ray.init()
src_remote_0 = ray.remote(num_gpus=1)(Source).remote(
    value=10,
    num_instance_prefill=4,
    num_instance_decode=2,
    rank=0,
    num_layers=2,
    num_gpu_blocks_prefill=40,
    num_gpu_blocks_decode=10,
    block_size=2,
    total_num_heads=4,
    head_size=4
)

src_remote_1 = ray.remote(num_gpus=1)(Source).remote(
    value=20,
    num_instance_prefill=4,
    num_instance_decode=2,
    rank=1,
    num_layers=2,
    num_gpu_blocks_prefill=40,
    num_gpu_blocks_decode=10,
    block_size=2,
    total_num_heads=4,
    head_size=4
)

src_remote_2 = ray.remote(num_gpus=1)(Source).remote(
    value=30,
    num_instance_prefill=4,
    num_instance_decode=2,
    rank=2,
    num_layers=2,
    num_gpu_blocks_prefill=40,
    num_gpu_blocks_decode=10,
    block_size=2,
    total_num_heads=4,
    head_size=4
)

src_remote_3 = ray.remote(num_gpus=1)(Source).remote(
    value=40,
    num_instance_prefill=4,
    num_instance_decode=2,
    rank=3,
    num_layers=2,
    num_gpu_blocks_prefill=40,
    num_gpu_blocks_decode=10,
    block_size=2,
    total_num_heads=4,
    head_size=4
)

des_remote_0 = ray.remote(num_gpus=1)(Destination).remote(
    value=1,
    num_instance_prefill=4,
    num_instance_decode=2,
    rank=0,
    num_layers=2,
    num_gpu_blocks_prefill=40,
    num_gpu_blocks_decode=10,
    block_size=2,
    total_num_heads=4,
    head_size=4
)

des_remote_1 = ray.remote(num_gpus=1)(Destination).remote(
    value=2,
    num_instance_prefill=4,
    num_instance_decode=2,
    rank=1,
    num_layers=2,
    num_gpu_blocks_prefill=40,
    num_gpu_blocks_decode=10,
    block_size=2,
    total_num_heads=4,
    head_size=4
)

handles_0, offsets_0 = ray.get(src_remote_0.get_handles.remote())
handles_1, offsets_1 = ray.get(src_remote_1.get_handles.remote())
handles_2, offsets_2 = ray.get(src_remote_2.get_handles.remote())
handles_3, offsets_3 = ray.get(src_remote_3.get_handles.remote())

des_remote_0.register_handles.remote(
    [handles_0, handles_1, handles_2, handles_3],
    [offsets_0, offsets_1, offsets_2, offsets_3])

des_remote_1.register_handles.remote(
    [handles_0, handles_1, handles_2, handles_3],
    [offsets_0, offsets_1, offsets_2, offsets_3])

des_remote_0.migrate.remote([0, 5, 3], [0, 8, 5])
des_remote_1.migrate.remote([0, 5, 3], [0, 8, 5])
