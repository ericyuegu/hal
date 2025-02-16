"""
Run closed loop evaluation of a model in the emulator.
"""
import argparse
import sys
import time
import traceback
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from typing import List
from typing import Optional

import torch
import torch.multiprocessing as mp
from loguru import logger
from tensordict import TensorDict

from hal.constants import Player
from hal.emulator_helper import EmulatorManager
from hal.emulator_helper import find_open_udp_ports
from hal.emulator_helper import get_replay_dir
from hal.eval.eval_helper import EpisodeStats
from hal.eval.eval_helper import Matchup
from hal.eval.eval_helper import deterministically_generate_random_matchups
from hal.eval.eval_helper import mock_framedata_as_tensordict
from hal.eval.eval_helper import share_and_pin_memory
from hal.gamestate_utils import extract_eval_gamestate_as_tensordict
from hal.preprocess.preprocessor import Preprocessor
from hal.training.config import TrainConfig
from hal.training.io import load_config_from_artifact_dir
from hal.training.io import load_model_from_artifact_dir


def setup_cpu_logger(debug: bool = False) -> None:
    logger.remove()
    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "rank={extra[rank]} - <level>{message}</level>"
    )
    logger.configure(extra={"rank": ""})  # Default values
    logger.add(sys.stderr, format=logger_format, level="DEBUG" if debug else "INFO")


def cpu_worker(
    shared_batched_model_input: TensorDict,
    shared_batched_model_output: TensorDict,
    rank: int,
    port: int,
    player: Player,
    matchup: Matchup,
    replay_dir: Path,
    preprocessor: Preprocessor,
    model_input_ready_flag: EventType,
    model_output_ready_flag: EventType,
    stop_event: EventType,
    episode_stats_queue: mp.Queue,
    enable_ffw: bool = True,
    debug: bool = False,
) -> None:
    """
    CPU worker that preprocesses data, writes it into shared memory,
    and sends controller inputs to the emulator from model predictions.
    """
    setup_cpu_logger(debug=debug)

    with logger.contextualize(rank=rank):
        try:
            emulator_manager = EmulatorManager(
                udp_port=port,
                player=player,
                replay_dir=replay_dir,
                opponent_cpu_level=9,
                matchup=matchup,
                enable_ffw=enable_ffw,
                debug=debug,
            )
            gamestate_generator = emulator_manager.run_game()
            gamestate = next(gamestate_generator)
            # Skip first N frames to match starting frame offset from training sequence sampling
            logger.debug(f"Skipping {preprocessor.eval_warmup_frames} starting frames to match training distribution")
            for _ in range(preprocessor.eval_warmup_frames):
                gamestate = next(gamestate_generator)

            i = 0
            while gamestate is not None:
                preprocess_start = time.perf_counter()
                gamestate_td = extract_eval_gamestate_as_tensordict(gamestate)
                model_inputs = preprocessor.preprocess_inputs(gamestate_td, player)
                preprocess_time = time.perf_counter() - preprocess_start

                transfer_start = time.perf_counter()
                sharded_model_input: TensorDict = shared_batched_model_input[rank]
                # Update our rank of the shared buffer with the last frame
                sharded_model_input.update_(model_inputs[-1], non_blocking=True)
                transfer_time = time.perf_counter() - transfer_start

                model_input_ready_flag.set()

                # Wait for the output to be ready
                while not model_output_ready_flag.is_set() and not stop_event.is_set():
                    time.sleep(0.0001)  # Sleep briefly to avoid busy waiting

                if stop_event.is_set():
                    break

                # Read model output and postprocess
                model_output = shared_batched_model_output[rank].clone()
                postprocess_start = time.perf_counter()
                controller_inputs = preprocessor.postprocess_preds(model_output)
                postprocess_time = time.perf_counter() - postprocess_start

                if debug and i % 60 == 0:
                    logger.debug(
                        f"Preprocess: {preprocess_time*1000:.2f}ms, Transfer: {transfer_time*1000:.2f}ms, Postprocess: {postprocess_time*1000:.2f}ms"
                    )

                # Send controller inputs to emulator, update gamestate
                gamestate = gamestate_generator.send(controller_inputs)

                # Clear the output ready flag for the next iteration
                model_output_ready_flag.clear()
                i += 1
        except StopIteration:
            logger.success(f"CPU worker {rank} episode complete.")
            logger.info(f"CPU worker {rank} episode stats: {emulator_manager.episode_stats}")
            episode_stats_queue.put(emulator_manager.episode_stats)
        except Exception as e:
            logger.error(
                f"CPU worker {rank} encountered an error: {e}\nTraceback:\n{''.join(traceback.format_tb(e.__traceback__))}"
            )
        finally:
            model_input_ready_flag.set()
            stop_event.set()
            logger.info(f"CPU worker {rank} stopped")


def gpu_worker(
    shared_batched_model_input_B: TensorDict,  # (n_workers,)
    shared_batched_model_output_B: TensorDict,  # (n_workers,)
    model_input_ready_flags: List[EventType],
    model_output_ready_flags: List[EventType],
    seq_len: int,
    stop_events: List[EventType],
    artifact_dir: Path,
    device: torch.device | str,
    checkpoint_idx: Optional[int] = None,
    cpu_flag_timeout: float = 5.0,
    debug: bool = False,
) -> None:
    """
    GPU worker that batches data from shared memory, updates the context window,
    performs inference with model, and writes output back to shared memory.
    """
    torch.set_float32_matmul_precision("high")
    model, _ = load_model_from_artifact_dir(Path(artifact_dir), idx=checkpoint_idx)
    model.eval()
    model.to(device)

    # Stack along time dimension
    # shape: (n_workers, seq_len)
    context_window_BL: TensorDict = torch.stack([shared_batched_model_input_B for _ in range(seq_len)], dim=-1).to(device)  # type: ignore
    logger.info(f"Context window shape: {context_window_BL.shape}, device: {context_window_BL.device}")

    # Warmup CUDA graphs with dummy inputs
    logger.info("Compiling model...")
    model = torch.compile(model, mode="default")
    with torch.no_grad():
        model(context_window_BL)
    logger.info("Warmup step finished")

    iteration = 0
    while not all(event.is_set() for event in stop_events):
        iteration_start = time.perf_counter()

        # Wait for all CPU workers to signal that data is ready
        flag_wait_start = time.perf_counter()
        for i, (input_flag, stop_event) in enumerate(zip(model_input_ready_flags, stop_events)):
            while not input_flag.is_set() and not stop_event.is_set():
                if not input_flag.is_set() and time.perf_counter() - flag_wait_start > cpu_flag_timeout:
                    logger.warning(f"CPU worker {i} input flag wait took too long, stopping episode")
                    input_flag.set()
                    stop_event.set()
                time.sleep(0.0001)  # Sleep briefly to avoid busy waiting

        if all(event.is_set() for event in stop_events):
            break

        transfer_start = time.perf_counter()
        if iteration < seq_len:
            # While context window is not full, fill in from the left
            context_window_BL[:, iteration].copy_(shared_batched_model_input_B, non_blocking=True)
        else:
            # Update the context window by rolling frame data left and adding new data on the right
            context_window_BL[:, :-1].copy_(context_window_BL[:, 1:].clone())
            context_window_BL[:, -1].copy_(shared_batched_model_input_B, non_blocking=True)
        # context_window_BL.save(f"/tmp/multishine_debugging/model_inputs_{iteration:06d}")
        transfer_time = time.perf_counter() - transfer_start

        inference_start = time.perf_counter()
        with torch.no_grad():
            outputs_BL: TensorDict = model(context_window_BL)
        # outputs_BL.save(f"/tmp/multishine_debugging/model_outputs_{iteration:06d}")
        seq_idx = min(seq_len - 1, iteration)
        outputs_B: TensorDict = outputs_BL[:, seq_idx]
        inference_time = time.perf_counter() - inference_start

        writeback_start = time.perf_counter()
        # Write last frame of model preds to shared buffer
        shared_batched_model_output_B.copy_(outputs_B)
        writeback_time = time.perf_counter() - writeback_start

        total_time = time.perf_counter() - iteration_start

        if iteration % 60 == 0:
            msg = f"Iteration {iteration}: Total: {total_time*1000:.2f}ms "
            if debug:
                msg += f"(Update context: {transfer_time*1000:.2f}ms, Inference: {inference_time*1000:.2f}ms, Writeback: {writeback_time*1000:.2f}ms)"
            logger.debug(msg)

        iteration += 1

        # Signal to CPU workers that output is ready
        for output_flag in model_output_ready_flags:
            output_flag.set()

        # Clear model_input_ready_flags for the next iteration
        for input_flag in model_input_ready_flags:
            input_flag.clear()


def flatten_replay_dir(replay_dir: Path) -> None:
    # Copy all files to base replay dir and clean up subdirs
    for file in replay_dir.glob("**/*.slp"):
        target_path = replay_dir / file.name
        counter = 1
        while target_path.exists():
            stem = file.stem
            target_path = replay_dir / f"{stem}_{counter}{file.suffix}"
            counter += 1

        try:
            file.replace(target_path)
        except OSError as e:
            logger.warning(f"Failed to move replay file {file}: {e}")

    for directory in sorted(replay_dir.glob("**/"), key=lambda x: len(str(x)), reverse=True):
        if directory != replay_dir:
            try:
                directory.rmdir()
            except OSError as e:
                logger.warning(f"Failed to remove directory {directory}: {e}")


def run_closed_loop_evaluation(
    artifact_dir: Path,
    n_workers: int,
    checkpoint_idx: Optional[int] = None,
    eval_stats_queue: Optional[mp.Queue] = None,
    player: Player = "p1",
    enable_ffw: bool = False,  # disable by default for emulator stability, TODO debug EXI inputs
    debug: bool = False,
) -> None:
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config: TrainConfig = load_config_from_artifact_dir(artifact_dir)
    preprocessor = Preprocessor(data_config=train_config.data)

    # Create events to signal when cpu and gpu workers are ready
    model_input_ready_flags: List[EventType] = [mp.Event() for _ in range(n_workers)]
    model_output_ready_flags: List[EventType] = [mp.Event() for _ in range(n_workers)]
    # Create events to signal when emulator episodes end
    stop_events: List[EventType] = [mp.Event() for _ in range(n_workers)]

    # Share and pin buffers in CPU memory for transferring model inputs and outputs
    mock_framedata_L = mock_framedata_as_tensordict(preprocessor.trajectory_sampling_len)
    # Store only a single time step to minimize copying
    mock_model_inputs_ = preprocessor.preprocess_inputs(mock_framedata_L, player)[-1]
    # batch_size == n_workers
    shared_batched_model_input_B: TensorDict = torch.stack(
        [mock_model_inputs_ for _ in range(n_workers)], dim=0  # type: ignore
    )
    shared_batched_model_input_B = share_and_pin_memory(shared_batched_model_input_B)
    shared_batched_model_output_B: TensorDict = torch.stack(
        [preprocessor.mock_preds_as_tensordict() for _ in range(n_workers)], dim=0  # type: ignore
    )
    shared_batched_model_output_B = share_and_pin_memory(shared_batched_model_output_B)

    gpu_process: mp.Process = mp.Process(
        target=gpu_worker,
        kwargs={
            "shared_batched_model_input_B": shared_batched_model_input_B,
            "shared_batched_model_output_B": shared_batched_model_output_B,
            "model_input_ready_flags": model_input_ready_flags,
            "model_output_ready_flags": model_output_ready_flags,
            "seq_len": preprocessor.seq_len,
            "stop_events": stop_events,
            "artifact_dir": artifact_dir,
            "device": device,
            "checkpoint_idx": checkpoint_idx,
            "debug": debug,
        },
    )
    gpu_process.start()

    cpu_processes: List[mp.Process] = []
    ports = find_open_udp_ports(n_workers)
    matchups = deterministically_generate_random_matchups(n=n_workers)
    episode_stats_queue: mp.Queue = mp.Queue()
    base_replay_dir = get_replay_dir(artifact_dir, step=checkpoint_idx)
    logger.info(f"Replays will be saved to {base_replay_dir}")
    for i, matchup in enumerate(matchups):
        replay_dir = base_replay_dir / f"{i:03d}"
        replay_dir.mkdir(exist_ok=True, parents=True)
        p: mp.Process = mp.Process(
            target=cpu_worker,
            kwargs={
                "shared_batched_model_input": shared_batched_model_input_B,
                "shared_batched_model_output": shared_batched_model_output_B,
                "rank": i,
                "port": ports[i],
                "player": player,
                "matchup": matchup,
                "replay_dir": replay_dir,
                "preprocessor": preprocessor,
                "model_input_ready_flag": model_input_ready_flags[i],
                "model_output_ready_flag": model_output_ready_flags[i],
                "stop_event": stop_events[i],
                "episode_stats_queue": episode_stats_queue,
                "enable_ffw": enable_ffw,
                "debug": debug,
            },
        )
        cpu_processes.append(p)
        p.start()

    gpu_process.join()

    for p in cpu_processes:
        p.join()

    # Clean up replay dir
    flatten_replay_dir(base_replay_dir)

    # Aggregate episode stats and return if requested
    episode_stats: List[EpisodeStats] = []
    while not episode_stats_queue.empty():
        episode_stats.append(episode_stats_queue.get())
    total_stats = sum(episode_stats, EpisodeStats(episodes=0))
    logger.info(f"Closed loop evaluation stats: {total_stats}")
    logger.info("Closed loop evaluation complete")

    if eval_stats_queue is not None:
        eval_stats_queue.put(total_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Melee in emulator")
    parser.add_argument("--model_dir", type=str, help="Path to model directory")
    parser.add_argument("--n_workers", type=int, help="Number of CPU workers")
    parser.add_argument("--enable_ffw", action="store_true", help="Enable fast forward mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    run_closed_loop_evaluation(
        artifact_dir=Path(args.model_dir),
        n_workers=args.n_workers,
        enable_ffw=args.enable_ffw,
        debug=args.debug,
    )
