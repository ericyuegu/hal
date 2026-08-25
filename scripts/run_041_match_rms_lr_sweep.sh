#!/usr/bin/env bash
set -euo pipefail

COMMON_ARGS=(
  --smoke
  --smoke-eval-matchups 0
  --cfg.target-loss-positions 262144000
  --cfg.muon-adjust-lr-fn match_rms_adamw
  --cfg.muon-weight-decay 0.0625
  --cfg.val-every 0
  --cfg.eval-every 0
  --cfg.ckpt-every 0
  --cfg.wandb-hist-every 0
  --cfg.layer-rms-every 0
)

uv run experiments/041_projectile_conditioning.py \
  "${COMMON_ARGS[@]}" \
  --cfg.muon-lr 6e-4 \
  --comment match-rms-lr-6e-4-2k

uv run experiments/041_projectile_conditioning.py \
  "${COMMON_ARGS[@]}" \
  --cfg.muon-lr 8.5e-4 \
  --comment match-rms-lr-8.5e-4-2k

uv run experiments/041_projectile_conditioning.py \
  "${COMMON_ARGS[@]}" \
  --cfg.muon-lr 1.2e-3 \
  --comment match-rms-lr-1.2e-3-2k
