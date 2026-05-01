# %%
import multiprocessing as mproc
import os
import pickle

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def create_dataset(num_tensors=40, tensor_size=(1000000,)):
    # Create a list of large tensors
    dataset = [torch.randn(*tensor_size) for _ in range(num_tensors)]
    return dataset


class NumpySerializedList:
    def __init__(self, lst: list) -> None:
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        print(f"Serializing {len(lst)} elements to byte tensors and concatenating them all ...")
        self._lst = [_serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = np.cumsum(self._addr)
        self._lst = np.concatenate(self._lst)
        print(f"Serialized dataset takes {len(self._lst) / 1024**2:.2f} MiB")

    def __len__(self) -> int:
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr])
        return pickle.loads(bytes)


class TorchSerializedList(NumpySerializedList):
    def __init__(self, lst: list) -> None:
        super().__init__(lst)
        # Move data to shared memory
        self._addr = torch.from_numpy(self._addr).share_memory_()
        self._lst = torch.from_numpy(self._lst).share_memory_()

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return pickle.loads(bytes)


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def run(rank, world_size) -> None:
    # Initialize the process group
    dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29500", world_size=world_size, rank=rank)
    print(f"Rank {rank} initialized.")

    if rank == 0:
        print("Creating dataset")
        dataset_list = create_dataset()
        shared_dataset = TorchSerializedList(dataset_list)
        # Get the handles
        print("Dumping handles")
        handle = mproc.reduction.ForkingPickler.dumps((shared_dataset._addr, shared_dataset._lst))
    else:
        handle = None

    # Broadcast the handle
    handle_list = [handle]
    dist.broadcast_object_list(handle_list, src=0)
    handle = handle_list[0]

    if rank != 0:
        print("Deserializing dataset")
        shared_dataset = TorchSerializedList([])
        shared_dataset._addr, shared_dataset._lst = mproc.reduction.ForkingPickler.loads(handle)
        print(f"Rank {rank} obtained shared dataset.")

    # Create Dataset
    dataset = SharedDataset(shared_dataset)

    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=2, num_workers=2)

    # Measure memory usage
    import psutil

    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024**2
    print(f"Rank {rank} process {os.getpid()} memory usage: {mem_usage:.2f} MB")

    # Iterate over DataLoader
    for batch in data_loader:
        # Do something with the batch
        pass


def main() -> None:
    world_size = 2  # Number of DDP processes
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)


class SharedDataset(Dataset):
    def __init__(self, shared_data) -> None:
        self.data = shared_data
        print(f"Process {os.getpid()} initializing dataset with data id {id(self.data)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        # Print to verify that DataLoader workers share the data
        print(f"Process {os.getpid()} accessing data idx {idx} with data id {id(self.data)}")
        return self.data[idx]


if __name__ == "__main__":
    main()
