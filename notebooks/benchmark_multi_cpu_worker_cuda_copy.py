import time
from typing import List

import torch
import torch.multiprocessing as mp
from tensordict import TensorDict
from torch.multiprocessing import Process
from torch.multiprocessing import Queue


class GPUWorker:
    def __init__(self, model, batch_size: int = 1024, feature_dim: int = 256) -> None:
        self.model = model.cuda()
        self.batch_size = batch_size
        self.feature_dim = feature_dim

    def setup_shared_output(self, in_cuda: bool = False) -> TensorDict:
        """Setup shared output TensorDict either in CUDA or CPU memory"""
        device = "cuda" if in_cuda else "cpu"
        td = TensorDict(
            {
                "embeddings": torch.empty(self.batch_size, self.feature_dim, device=device),
                "scores": torch.empty(self.batch_size, device=device),
                "processed": torch.zeros(self.batch_size, dtype=torch.bool, device=device),
            },
            batch_size=[self.batch_size],
        )

        # Share memory for CPU tensor or use CUDA shared memory
        if not in_cuda:
            td.share_memory_()
        return td

    def process_batch(self, input_data: torch.Tensor, output_td: TensorDict) -> None:
        """Run inference and store results"""
        with torch.no_grad():
            # Simulate model inference
            embeddings = self.model(input_data.cuda())
            scores = embeddings.norm(dim=1)

            if output_td["embeddings"].device.type == "cuda":
                # Pattern 1: Write directly to shared CUDA memory
                output_td["embeddings"].copy_(embeddings)
                output_td["scores"].copy_(scores)
                output_td["processed"].fill_(True)
            else:
                # Pattern 2: Copy to shared CPU memory
                output_td["embeddings"].copy_(embeddings.cpu())
                output_td["scores"].copy_(scores.cpu())
                output_td["processed"].fill_(True)


class CPUWorker:
    def __init__(self, worker_id: int, slice_size: int) -> None:
        self.worker_id = worker_id
        self.start_idx = worker_id * slice_size
        self.end_idx = (worker_id + 1) * slice_size

    def process_slice(self, output_td: TensorDict, result_queue: Queue) -> None:
        slice_dict = {"worker_id": self.worker_id, "start_idx": self.start_idx, "end_idx": self.end_idx}

        try:
            if output_td["embeddings"].device.type == "cuda":
                # Pattern 1: Each CPU worker needs to copy from CUDA
                embeddings = output_td["embeddings"][self.start_idx : self.end_idx].cpu()
                scores = output_td["scores"][self.start_idx : self.end_idx].cpu()
            else:
                # Pattern 2: Direct access to CPU memory
                embeddings = output_td["embeddings"][self.start_idx : self.end_idx]
                scores = output_td["scores"][self.start_idx : self.end_idx]

            # Simulate some CPU processing
            result = (embeddings * scores.unsqueeze(1)).mean().item()
            slice_dict["result"] = result

        except Exception as e:
            slice_dict["error"] = str(e)

        result_queue.put(slice_dict)


def benchmark_patterns(num_workers: int = 4, batch_size: int = 1024, feature_dim: int = 256) -> None:
    """Compare CUDA shared memory vs CPU shared memory patterns"""

    # Simple dummy model
    model = torch.nn.Linear(feature_dim, feature_dim).cuda()
    slice_size = batch_size // num_workers
    result_queue = Queue()

    def run_pattern(in_cuda: bool):
        # Setup workers
        gpu_worker = GPUWorker(model, batch_size, feature_dim)
        cpu_workers: List[Process] = []

        # Setup shared output
        output_td = gpu_worker.setup_shared_output(in_cuda=in_cuda)

        # Generate dummy input
        input_data = torch.randn(batch_size, feature_dim, device="cuda")

        # Time GPU inference and data transfer
        start_time = time.perf_counter()
        gpu_worker.process_batch(input_data, output_td)
        gpu_time = time.perf_counter() - start_time

        # Start CPU workers
        start_time = time.perf_counter()
        for i in range(num_workers):
            w = Process(target=CPUWorker(i, slice_size).process_slice, args=(output_td, result_queue))
            w.start()
            cpu_workers.append(w)

        # Collect results
        results = []
        for _ in range(num_workers):
            results.append(result_queue.get(timeout=5))

        # Wait for all workers
        for w in cpu_workers:
            w.join()

        cpu_time = time.perf_counter() - start_time

        return gpu_time, cpu_time, results

    # Compare patterns
    print("\nPattern 1: CUDA shared memory")
    gpu_time1, cpu_time1, _ = run_pattern(in_cuda=True)
    print(f"GPU time: {gpu_time1*1000:.2f}ms")
    print(f"CPU workers time: {cpu_time1*1000:.2f}ms")

    print("\nPattern 2: CPU shared memory")
    gpu_time2, cpu_time2, _ = run_pattern(in_cuda=False)
    print(f"GPU time: {gpu_time2*1000:.2f}ms")
    print(f"CPU workers time: {cpu_time2*1000:.2f}ms")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    benchmark_patterns()
