# %%
"""
Using DDP with TensorDict
=============================
"""

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from tensordict import MemoryMappedTensor
from tensordict import TensorDict
from torch.distributed import destroy_process_group
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision.transforms import ToTensor


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def setup(rank, world_size) -> None:
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    destroy_process_group()


def load_data() -> tuple[TensorDict, TensorDict]:
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    training_data_td = TensorDict(
        {
            "images": MemoryMappedTensor.empty(
                (len(training_data), *training_data[0][0].squeeze().shape),
                dtype=torch.float32,
            ),
            "targets": MemoryMappedTensor.empty((len(training_data),), dtype=torch.int64),
        },
        batch_size=[len(training_data)],
        device=get_device(),
    )
    test_data_td = TensorDict(
        {
            "images": MemoryMappedTensor.empty(
                (len(test_data), *test_data[0][0].squeeze().shape), dtype=torch.float32
            ),
            "targets": MemoryMappedTensor.empty((len(test_data),), dtype=torch.int64),
        },
        batch_size=[len(test_data)],
        device=get_device(),
    )

    for i, (img, label) in enumerate(training_data):
        training_data_td[i] = TensorDict({"images": img, "targets": label}, [])

    for i, (img, label) in enumerate(test_data):
        test_data_td[i] = TensorDict({"images": img, "targets": label}, [])

    return training_data_td, test_data_td


def train_ddp(rank, world_size, training_data_td: TensorDict, test_data_td: TensorDict) -> None:
    setup(rank, world_size)
    device = get_device()
    print(f"Using device: {device}")

    model = Net().to(device)
    model = DDP(model, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_sampler = DistributedSampler(training_data_td, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(training_data_td, batch_size=64, sampler=train_sampler, collate_fn=lambda x: x)
    size = len(train_dataloader)

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        for batch, data in enumerate(train_dataloader):
            X, y = data["images"].contiguous(), data["targets"].contiguous()
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current = batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    # Load data into shared memory
    training_data_td, test_data_td = load_data()

    # Use mp.spawn to launch the training process on each GPU
    mp.spawn(train_ddp, args=(world_size, training_data_td, test_data_td), nprocs=world_size, join=True)
