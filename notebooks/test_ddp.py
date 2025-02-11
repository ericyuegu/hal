import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# this solves the error
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"


def worker(rank, world_size) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # this causes the error
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:12355", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] has begun")

    tensor = torch.tensor([1.0]).cuda(rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] after all_reduce, tensor={tensor.item()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, nprocs=2, args=(2,), join=True)
