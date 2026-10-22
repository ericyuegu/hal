"""Evaluate an immutable O65 snapshot or the pinned O63 proxy against anchors.

Example: uv run python scripts/eval_o65.py --checkpoint runs/o65-eval/proxy.pt
--anchors runs/o65-eval/proxy.pt --output runs/o65-eval/check-1
--simulators dolphin --max-frames 1200 --opponent-player-id 3
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from hal.eval.conditioned_matchups import evaluate
from hal.eval.conditioned_matchups import pilot_protocol
from hal.eval.conditioned_matchups import read_protocol
from hal.eval.self_play_policy import PlayerCondition
from hal.models.self_play import DecoderPrecision


@dataclass(frozen=True, slots=True)
class Args:
    checkpoint: Path
    anchors: tuple[Path, ...]
    output: Path
    data_dir: Path | None = None
    protocol: Path | None = None
    lambda_grid: tuple[float, ...] = ()
    cadence_grid: tuple[str, ...] = ()
    split: Literal["selection", "held_out"] = "selection"
    simulators: tuple[Literal["msl", "dolphin"], ...] = ("msl", "dolphin")
    lambda_value: float = 1.0
    opponent_lambda: float = 1.0
    delay: int = 8
    replan: int = 4
    opponent_delay: int = 8
    opponent_replan: int = 4
    player_id: int = 0
    opponent_player_id: int = 0
    history_mode: Literal["block", "sliding"] = "block"
    compare_events: bool = False
    max_frames: int = 28924
    device: str = "cuda"
    execution: Literal["serial", "batched"] = "batched"
    dolphin_workers: int = 8
    decoder_precision: DecoderPrecision | None = None


def main(args: Args) -> None:
    torch.set_num_threads(2)
    grid = None
    if args.lambda_grid or args.cadence_grid:
        pairs = []
        for cadence in args.cadence_grid or (f"{args.delay}/{args.replan}",):
            delay, replan = (int(value) for value in cadence.split("/"))
            for value in args.lambda_grid or (args.lambda_value,):
                pairs.append(
                    (
                        PlayerCondition(value, delay, replan, args.player_id),
                        PlayerCondition(args.opponent_lambda, delay, replan, args.opponent_player_id),
                    )
                )
        grid = tuple(pairs)
    evaluate(
        args.checkpoint,
        args.anchors,
        args.output,
        protocol=pilot_protocol(max_frames=args.max_frames) if args.protocol is None else read_protocol(args.protocol),
        split=args.split,
        conditions=(
            PlayerCondition(args.lambda_value, args.delay, args.replan, args.player_id),
            PlayerCondition(args.opponent_lambda, args.opponent_delay, args.opponent_replan, args.opponent_player_id),
        ),
        simulators=args.simulators,
        data_dir=args.data_dir,
        device=args.device,
        history_mode=args.history_mode,
        compare_events=args.compare_events,
        condition_grid=grid,
        execution=args.execution,
        dolphin_workers=args.dolphin_workers,
        decoder_precision=args.decoder_precision,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
