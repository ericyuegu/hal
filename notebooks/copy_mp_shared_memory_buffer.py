import time
from typing import List

import melee
import torch
import torch.multiprocessing as mp
from tensordict import TensorDict


class DummyModel(torch.nn.Module):
    def __init__(self, hidden_dim: int, time_steps: int, output_dim: int) -> None:
        super(DummyModel, self).__init__()
        self.fc = torch.nn.Linear(hidden_dim * time_steps, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, time_steps, hidden_dim]
        return x


def get_frame_data_from_gamestate(gamestate: melee.GameState | None = None, rank: int = 0) -> TensorDict:
    hidden_dim: int = 64  # Size of the hidden dimension
    device = "cpu"
    return TensorDict(
        {
            input_field: torch.full((hidden_dim,), rank, device=device)
            for input_field in ("gamestate", "ego_char", "ego_action", "opponent_char", "opponent_action")
        },
        batch_size=[],
        device=device,
    )


def get_model_output() -> TensorDict:
    output_dim: int = 10  # Size of the output dimension
    device = "cpu"
    time.sleep(0.007)
    return TensorDict(
        {output_field: torch.zeros(output_dim, device=device) for output_field in ("button", "main_stick", "c_stick")},
        batch_size=[],
        device=device,
    )


def cpu_worker(
    shared_input: TensorDict,
    shared_output: TensorDict,
    rank: int,
    data_ready_flags: List[mp.Event],
    output_ready_flags: List[mp.Event],
    stop_event: mp.Event,
    hidden_dim: int,
) -> None:
    """
    CPU worker that preprocesses data, writes it into shared memory,
    and reads the result after GPU processing.
    """
    while not stop_event.is_set():
        # Simulate data preprocessing (dummy data)
        frame_data = get_frame_data_from_gamestate(None, rank=rank)

        # Write data into shared_tensor at the specified index
        shared_input[rank].copy_(frame_data)

        # Signal that data is ready
        data_ready_flags[rank].set()

        # Wait for the output to be ready
        while not output_ready_flags[rank].is_set() and not stop_event.is_set():
            time.sleep(0.001)  # Sleep briefly to avoid busy waiting

        if stop_event.is_set():
            break

        # Read the output from shared_output
        output: torch.Tensor = shared_output[rank].clone()
        print(f"CPU Worker {rank} received output: {output}")

        # Clear the output ready flag for the next iteration
        output_ready_flags[rank].clear()


def gpu_worker(
    shared_input: torch.Tensor,
    shared_output: torch.Tensor,
    data_ready_flags: List[mp.Event],
    output_ready_flags: List[mp.Event],
    context_window_size: int,
    num_workers: int,
    stop_event: mp.Event,
    hidden_dim: int,
    output_dim: int,
    device: torch.device,
) -> None:
    """
    GPU worker that batches data from shared memory, updates the context window,
    performs forward operation with a dummy model, and writes output back to shared memory.
    """
    # Initialize the context window
    context_window = torch.stack(
        [
            torch.stack([get_frame_data_from_gamestate() for _ in range(context_window_size)], dim=0)
            for _ in range(num_workers)
        ],
        dim=0,
    )
    print(f"gamestate: {context_window['gamestate']}")

    # Initialize the dummy model
    model: DummyModel = DummyModel(hidden_dim, context_window_size, output_dim)
    model.to(device)

    iteration: int = 0
    while not stop_event.is_set():
        # Wait for all CPU workers to signal that data is ready
        for flag in data_ready_flags:
            while not flag.is_set() and not stop_event.is_set():
                time.sleep(0.001)  # Sleep briefly to avoid busy waiting

        if stop_event.is_set():
            break

        t0 = time.perf_counter()
        # Read data from shared_tensor
        batch_data: torch.Tensor = shared_input.clone().to(device)  # Shape: [num_workers, hidden_dim]

        # Update the context window by rolling and adding new data
        context_window[:, :-1] = context_window[:, 1:].clone()
        # Add new data at the last time step
        context_window[:, -1] = batch_data

        # Perform forward operation with the dummy model
        # output: torch.Tensor = model(context_window)
        output = get_model_output()
        # Write the output to shared_output
        shared_output.copy_(output.cpu())

        t1 = time.perf_counter()
        print(f"GPU Worker time: {t1 - t0}")

        # Signal to CPU workers that output is ready
        for flag in output_ready_flags:
            flag.set()

        # Clear data_ready_flags for the next iteration
        for flag in data_ready_flags:
            flag.clear()

        iteration += 1
        # For demonstration purposes, stop after a certain number of iterations
        if iteration >= 5:
            stop_event.set()
            break


def main() -> None:
    # Set the multiprocessing start method
    mp.set_start_method("spawn")

    num_workers: int = 4  # Number of CPU workers
    context_window_size: int = 16  # Size of the context window (time steps)
    hidden_dim: int = 64  # Size of the hidden dimension
    output_dim: int = 10  # Size of the output dimension

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize shared tensors in shared memory
    shared_input: TensorDict = torch.stack([get_frame_data_from_gamestate() for _ in range(num_workers)], dim=0)
    shared_input.share_memory_()
    print(f"shared_input: {shared_input}")

    shared_output: TensorDict = TensorDict(
        {
            output_field: torch.zeros(num_workers, output_dim, device=device)
            for output_field in ("button", "main_stick", "c_stick")
        },
        batch_size=[num_workers, output_dim],
        device=device,
    )
    shared_output.share_memory_()

    # Create events to signal when data is ready and when output is ready
    data_ready_flags: List[mp.Event] = [mp.Event() for _ in range(num_workers)]
    output_ready_flags: List[mp.Event] = [mp.Event() for _ in range(num_workers)]

    # Create an event to signal when to stop processing
    stop_event: mp.Event = mp.Event()

    # Start the GPU worker process
    gpu_process: mp.Process = mp.Process(
        target=gpu_worker,
        args=(
            shared_input,
            shared_output,
            data_ready_flags,
            output_ready_flags,
            context_window_size,
            num_workers,
            stop_event,
            hidden_dim,
            output_dim,
            device,
        ),
    )
    gpu_process.start()

    # Start CPU worker processes
    cpu_processes: List[mp.Process] = []
    for i in range(num_workers):
        p: mp.Process = mp.Process(
            target=cpu_worker,
            args=(
                shared_input,
                shared_output,
                i,
                data_ready_flags,
                output_ready_flags,
                stop_event,
                hidden_dim,
            ),
        )
        cpu_processes.append(p)
        p.start()

    # Wait for the GPU worker to finish processing
    gpu_process.join()

    # Signal CPU workers to stop
    stop_event.set()

    # Wait for all CPU workers to finish
    for p in cpu_processes:
        p.join()

    print("Processing complete.")


if __name__ == "__main__":
    main()
