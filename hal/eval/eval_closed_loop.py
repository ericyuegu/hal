import argparse
import sys
import time
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional

import attr
import melee
import torch
import torch.multiprocessing as mp
from loguru import logger
from tensordict import TensorDict

from hal.constants import PLAYER_1_PORT
from hal.constants import PLAYER_2_PORT
from hal.constants import Player
from hal.constants import get_opponent
from hal.data.stats import FeatureStats
from hal.data.stats import load_dataset_stats
from hal.eval.emulator_helper import console_manager
from hal.eval.emulator_helper import find_open_udp_ports
from hal.eval.emulator_helper import get_console_kwargs
from hal.eval.emulator_helper import self_play_menu_helper
from hal.eval.emulator_paths import REMOTE_CISO_PATH
from hal.eval.eval_helper import EpisodeStats
from hal.eval.eval_helper import mock_framedata_as_tensordict
from hal.eval.eval_helper import mock_preds_as_tensordict
from hal.eval.eval_helper import send_controller_inputs
from hal.eval.eval_helper import share_and_pin_memory
from hal.gamestate_utils import extract_gamestate_as_tensordict
from hal.training.config import TrainConfig
from hal.training.io import load_config_from_artifact_dir
from hal.training.io import load_model_from_artifact_dir
from hal.training.preprocess.registry import InputPreprocessFn
from hal.training.preprocess.registry import InputPreprocessRegistry
from hal.training.preprocess.registry import PredPostprocessFn
from hal.training.preprocess.registry import PredPostprocessingRegistry


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


@attr.s(auto_attribs=True)
class EmulatorManager:
    rank: int
    port: int
    player: Player
    episode_stats: EpisodeStats = EpisodeStats()
    max_steps: int = 8 * 60 * 60
    latency_warning_threshold: float = 14.0

    def gamestate_generator(self) -> Generator[Optional[melee.GameState], TensorDict, None]:
        """Generator that yields gamestates and receives controller inputs.

        Yields:
            Optional[melee.GameState]: The current game state, or None if the episode is over

        Sends:
            TensorDict: Controller inputs to be applied to the game
        """
        console_kwargs = get_console_kwargs(rank=self.rank, port=self.port)
        console = melee.Console(**console_kwargs)

        def _get_port(player: Player) -> int:
            return PLAYER_1_PORT if player == "p1" else PLAYER_2_PORT

        ego_controller = melee.Controller(
            console=console, port=_get_port(self.player), type=melee.ControllerType.STANDARD
        )
        opponent_controller = melee.Controller(
            console=console, port=_get_port(get_opponent(self.player)), type=melee.ControllerType.STANDARD
        )

        # Run the console
        console.run(iso_path=REMOTE_CISO_PATH)  # Do not pass dolphin_user_path to avoid overwriting init kwargs
        # Connect to the console
        logger.debug("Connecting to console...")
        if not console.connect():
            logger.debug("ERROR: Failed to connect to the console.")
            sys.exit(-1)
        logger.debug("Console connected")

        # Plug our controller in
        #   Due to how named pipes work, this has to come AFTER running dolphin
        #   NOTE: If you're loading a movie file, don't connect the controller,
        #   dolphin will hang waiting for input and never receive it
        logger.debug("Connecting controller 1 to console...")
        if not ego_controller.connect():
            logger.debug("ERROR: Failed to connect the controller.")
            sys.exit(-1)
        logger.debug("Controller 1 connected")
        logger.debug("Connecting controller 2 to console...")
        if not opponent_controller.connect():
            logger.debug("ERROR: Failed to connect the controller.")
            sys.exit(-1)
        logger.debug("Controller 2 connected")

        i = 0
        match_started = False
        try:
            with console_manager(console):
                logger.debug("Starting episode")
                while i < self.max_steps:
                    gamestate = console.step()
                    if gamestate is None:
                        logger.debug("Gamestate is None")
                        break

                    if console.processingtime * 1000 > self.latency_warning_threshold:
                        logger.debug("Last frame took " + str(console.processingtime * 1000) + "ms to process.")

                    if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                        if match_started:
                            break

                        self_play_menu_helper(
                            gamestate=gamestate,
                            controller_1=ego_controller,
                            controller_2=opponent_controller,
                            # TODO: select characters programmatically
                            character_1=melee.Character.FOX,
                            character_2=melee.Character.FOX,
                            stage_selected=melee.Stage.BATTLEFIELD,
                        )
                    else:
                        if not match_started:
                            match_started = True
                            logger.debug("Match started")

                        # Yield gamestate and receive controller inputs
                        controller_inputs = yield gamestate
                        if controller_inputs is not None:
                            send_controller_inputs(ego_controller, controller_inputs)

                        self.episode_stats.update(gamestate)
                        i += 1
        except Exception as e:
            logger.error(f"Episode {self.rank} encountered an error: {e}")
        finally:
            # Signal end of episode
            yield None


def cpu_worker(
    shared_batched_model_input: TensorDict,
    shared_batched_model_output: TensorDict,
    rank: int,
    port: int,
    player: Player,
    preprocess_inputs: InputPreprocessFn,
    postprocess_outputs: PredPostprocessFn,
    model_input_ready_flag: EventType,
    model_output_ready_flag: EventType,
    stop_event: EventType,
    train_config: TrainConfig,
    stats_by_feature_name: Dict[str, FeatureStats],
    episode_stats_queue: mp.Queue,
    debug: bool = False,
) -> None:
    """
    CPU worker that preprocesses data, writes it into shared memory,
    and reads the result after GPU processing.
    """
    setup_cpu_logger(debug=debug)

    with logger.contextualize(rank=rank):
        try:
            emulator_manager = EmulatorManager(rank=rank, port=port, player=player)
            gamestate_generator = emulator_manager.gamestate_generator()
            for i, gamestate in enumerate(gamestate_generator):
                if gamestate is None:
                    # First gamestate after match start is None for some reason
                    if i > 1:
                        break
                    continue

                preprocess_start = time.perf_counter()
                # Returns a TensorDict with shape (1,) for single frame
                gamestate_td = extract_gamestate_as_tensordict(gamestate)
                # Preprocess single frame
                data_config = attr.evolve(train_config.data, input_len=1, target_len=0)
                model_inputs = preprocess_inputs(gamestate_td, data_config, player, stats_by_feature_name)
                preprocess_time = time.perf_counter() - preprocess_start

                transfer_start = time.perf_counter()
                sharded_model_input: TensorDict = shared_batched_model_input[rank]
                # Update our rank of the shared buffer with the last frame
                sharded_model_input.update_(model_inputs[-1], non_blocking=True)
                transfer_time = time.perf_counter() - transfer_start

                if debug and i % 60 == 0:
                    logger.debug(f"Preprocess: {preprocess_time*1000:.2f}ms, Transfer: {transfer_time*1000:.2f}ms")

                model_input_ready_flag.set()

                # Wait for the output to be ready
                while not model_output_ready_flag.is_set() and not stop_event.is_set():
                    time.sleep(0.0001)  # Sleep briefly to avoid busy waiting

                if stop_event.is_set():
                    break

                # Read the output from shared_batched_model_output
                model_output = shared_batched_model_output[rank].clone()
                controller_inputs = postprocess_outputs(model_output)
                gamestate_generator.send(controller_inputs)

                # Clear the output ready flag for the next iteration
                model_output_ready_flag.clear()

            logger.success(f"CPU worker {rank} complete.")
        except Exception as e:
            logger.error(f"CPU worker {rank} encountered an error: {e}")
        finally:
            model_input_ready_flag.set()
            stop_event.set()
            logger.debug(f"CPU worker {rank} episode stats: {emulator_manager.episode_stats}")
            episode_stats_queue.put(emulator_manager.episode_stats)
            logger.info(f"CPU worker {rank} stopped")


def gpu_worker(
    shared_batched_model_input: TensorDict,  # (n_workers,)
    shared_batched_model_output: TensorDict,  # (n_workers,)
    model_input_ready_flags: List[EventType],
    model_output_ready_flags: List[EventType],
    seq_len: int,
    stop_events: List[EventType],
    model_dir: str,
    device: torch.device | str,
    checkpoint_idx: Optional[int] = None,
    cpu_flag_timeout: float = 10.0,
) -> None:
    """
    GPU worker that batches data from shared memory, updates the context window,
    performs inference with model, and writes output back to shared memory.
    """
    torch.set_float32_matmul_precision("high")
    model, _ = load_model_from_artifact_dir(Path(model_dir), idx=checkpoint_idx)
    model.eval()
    model.to(device)

    # Stack along time dimension
    # shape: (n_workers, seq_len)
    context_window: TensorDict = torch.stack([shared_batched_model_input for _ in range(seq_len)], dim=-1).to(device)  # type: ignore
    logger.info(f"Context window shape: {context_window.shape}, device: {context_window.device}")

    # Warmup CUDA graphs with dummy inputs
    logger.info("Compiling model...")
    model = torch.compile(model, mode="default")
    with torch.no_grad():
        model(context_window)
    logger.info("Warmup step finished")

    iteration = 0
    while not all(event.is_set() for event in stop_events):
        iteration_start = time.perf_counter()

        # Wait for all CPU workers to signal that data is ready
        flag_wait_start = time.perf_counter()
        for i, (flag, stop_event) in enumerate(zip(model_input_ready_flags, stop_events)):
            while not flag.is_set() and not stop_event.is_set():
                if not flag.is_set() and time.perf_counter() - flag_wait_start > cpu_flag_timeout:
                    logger.warning(f"CPU worker {i} input flag wait took too long, stopping episode")
                    flag.set()
                    stop_event.set()
                time.sleep(0.0001)  # Sleep briefly to avoid busy waiting

        if all(event.is_set() for event in stop_events):
            break

        # Update the context window by rolling and adding new data
        transfer_start = time.perf_counter()
        context_window[:, :-1].copy_(context_window[:, 1:].clone())
        context_window[:, -1].copy_(shared_batched_model_input, non_blocking=True)
        transfer_time = time.perf_counter() - transfer_start

        inference_start = time.perf_counter()
        with torch.no_grad():
            outputs: TensorDict = model(context_window)[:, -1]  # (n_workers,)
        inference_time = time.perf_counter() - inference_start

        writeback_start = time.perf_counter()
        # Write the output to shared_batched_model_output
        shared_batched_model_output.copy_(outputs)
        writeback_time = time.perf_counter() - writeback_start

        total_time = time.perf_counter() - iteration_start

        if iteration % 60 == 0:
            logger.debug(
                f"Iteration {iteration}: Total: {total_time*1000:.2f}ms "
                f"(Transfer: {transfer_time*1000:.2f}ms, "
                f"Inference: {inference_time*1000:.2f}ms, "
                f"Writeback: {writeback_time*1000:.2f}ms)"
            )

        iteration += 1

        # Signal to CPU workers that output is ready
        for flag in model_output_ready_flags:
            flag.set()

        # Clear model_input_ready_flags for the next iteration
        for flag in model_input_ready_flags:
            flag.clear()


def run_closed_loop_evaluation(
    model_dir: str,
    n_workers: int,
    checkpoint_idx: Optional[int] = None,
    eval_stats_queue: Optional[mp.Queue] = None,
    player: Player = "p1",
) -> None:
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config: TrainConfig = load_config_from_artifact_dir(Path(model_dir))
    seq_len = train_config.data.input_len
    preprocess_inputs = InputPreprocessRegistry.get(train_config.embedding.input_preprocessing_fn)
    stats_by_feature_name = load_dataset_stats(train_config.data.stats_path)
    postprocess_fn_name = getattr(train_config.embedding, "pred_postprocessing_fn", None) or train_config.embedding.target_preprocessing_fn  # backwards compatibility
    postprocess_outputs = PredPostprocessingRegistry.get(postprocess_fn_name)

    # Create events to signal when cpu and gpu workers are ready
    model_input_ready_flags: List[EventType] = [mp.Event() for _ in range(n_workers)]
    model_output_ready_flags: List[EventType] = [mp.Event() for _ in range(n_workers)]
    # Create events to signal when emulator episodes end
    stop_events: List[EventType] = [mp.Event() for _ in range(n_workers)]

    # Share and pin buffers in CPU memory for transferring model inputs and outputs
    mock_framedata = mock_framedata_as_tensordict(seq_len)
    # Store only a single time step to minimize memory transfer
    mock_model_inputs = preprocess_inputs(mock_framedata, train_config.data, player, stats_by_feature_name)[-1]
    shared_batched_model_input: TensorDict = torch.stack(
        [mock_model_inputs for _ in range(n_workers)], dim=0  # type: ignore
    )
    shared_batched_model_input = share_and_pin_memory(shared_batched_model_input)
    shared_batched_model_output: TensorDict = torch.stack(
        [mock_preds_as_tensordict(train_config.embedding) for _ in range(n_workers)], dim=0  # type: ignore
    )
    shared_batched_model_output = share_and_pin_memory(shared_batched_model_output)

    gpu_process: mp.Process = mp.Process(
        target=gpu_worker,
        kwargs={
            "shared_batched_model_input": shared_batched_model_input,
            "shared_batched_model_output": shared_batched_model_output,
            "model_input_ready_flags": model_input_ready_flags,
            "model_output_ready_flags": model_output_ready_flags,
            "seq_len": seq_len,
            "stop_events": stop_events,
            "model_dir": model_dir,
            "device": device,
            "checkpoint_idx": checkpoint_idx,
        },
    )
    gpu_process.start()

    cpu_processes: List[mp.Process] = []
    ports = find_open_udp_ports(n_workers)
    episode_stats_queue: mp.Queue = mp.Queue()
    for i in range(n_workers):
        p: mp.Process = mp.Process(
            target=cpu_worker,
            kwargs={
                "shared_batched_model_input": shared_batched_model_input,
                "shared_batched_model_output": shared_batched_model_output,
                "rank": i,
                "port": ports[i],
                "player": player,
                "preprocess_inputs": preprocess_inputs,
                "postprocess_outputs": postprocess_outputs,
                "model_input_ready_flag": model_input_ready_flags[i],
                "model_output_ready_flag": model_output_ready_flags[i],
                "stop_event": stop_events[i],
                "train_config": train_config,
                "stats_by_feature_name": stats_by_feature_name,
                "episode_stats_queue": episode_stats_queue,
            },
        )
        cpu_processes.append(p)
        p.start()

    gpu_process.join()

    for p in cpu_processes:
        p.join()

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
    args = parser.parse_args()
    run_closed_loop_evaluation(model_dir=args.model_dir, n_workers=args.n_workers)
