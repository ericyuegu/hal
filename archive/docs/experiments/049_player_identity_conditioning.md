# Experiment 049: ego-player identity conditioning

O49 is a paired ablation of O43 arm A. It tests whether a small, learned
ego-player identity embedding improves the legacy-codec policy.

The two arms differ only in `treatment`:

- `control`: every example uses player ID 0.
- `conditioned`: ranked examples use the Platinum, Diamond, or Master aggregate;
  professional examples use the controlling player's exact connect code.

The model never receives the opponent's identity. This keeps inference valid
against players who are outside the training vocabulary.

## Identity contract

ID 0 is the fixed mask. IDs 1, 2, and 3 are Platinum, Diamond, and Master.
Professional IDs start at 4 and come from the sorted training-only connect-code
vocabulary.

Connect codes are identity keys. The builder removes outer whitespace but does
not change case. It does not use nicknames as fallback keys. Missing codes and
validation or test codes that are absent from training map to ID 0.

The professional manifests contain 487,080 replays and 974,160 player sides:

| Split | Replays | Sides | Code present | In train vocabulary | Nickname present |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 477,346 | 954,692 | 901,741 | 901,741 | 907,531 |
| Validation | 4,914 | 9,828 | 9,300 | 9,257 | 9,328 |
| Test | 4,820 | 9,640 | 9,132 | 9,084 | 9,188 |

There are 21,267 exact connect codes across all splits and 21,177 in training.
The training codes have zero casefold collision groups. For comparison only,
the manifests have 33,774 exact nicknames and 32,409 casefolded nicknames. No
casefolded nickname is used by the model.

The immutable sidecar is
`s3://hal/processed/player-identity-v1/professional-code-v1.jsonl.gz`.
Its SHA-256 is
`54ccf8a2497fe240313117297ca2ea31158e08db2cc53c67e7aa46853a8dac1c`.
The ordered vocabulary SHA-256 is
`c67c97c995ad033ea7f5b2223efce5b061394566439f091ff6e7aaa6a9d1cfd6`.

## Model and data

The learned embedding has width 32. A bias-free linear layer projects it to
O43's model width and adds it to every context token. Its weights start at zero,
so both O49 arms initially match O43 exactly. The checkpoint contains the
ordered connect-code vocabulary as a persistent buffer.

Training uses the natural replay-weighted mixture of all six ranked-anonymous
policy-world-v7 sources and all 38 professional policy-world-v7 sources. All
other model, optimizer, schedule, batch, context, loss, and evaluation settings
remain at the O43 arm-A defaults. Closed-loop evaluation uses Master unless an
explicit rank or connect code is supplied.

## Run and infer

```bash
uv run experiments/049_player_identity_conditioning.py --cfg.treatment control
uv run experiments/049_player_identity_conditioning.py --cfg.treatment conditioned
```

An explicit professional style needs only the checkpoint and a string connect
code:

```python
model, cfg, stats, _ = load_checkpoint("runs/<run>/final.pt")
engine = BF16Inference(model, cfg, player_code="MANG#0")
actions = engine.decode(context, horizon=1)
```

The lookup is exact after outer whitespace removal. An unknown explicit code
raises `KeyError`; it does not silently select the mask or another player.

The separate identity audit scores one deterministic window from every
validation replay twice, once from each ego side, across all 44 sources:

```bash
uv run experiments/049_player_identity_conditioning.py \
  --identity-audit runs/<run>/final.pt
```

It writes `identity_validation.json` beside the checkpoint with global and
per-identity teacher-forced NLL. The artifact records its checkpoint and
identity-sidecar hashes.
