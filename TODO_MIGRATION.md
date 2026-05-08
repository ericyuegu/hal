# Schema migration TODO

The new peppi-py-based pipeline (`hal/data/{extract,build_index,filter_replays,process_replays}.py`)
emits MDS shards under a new schema (`hal/data/schema.py`). The following subtrees still reference
the old libmelee-era schema and **will not run end-to-end** until they are rewritten:

- `hal/preprocess/` (`preprocessor.py`, `transformations.py`, `input_configs.py`, etc.)
- `hal/training/config.py` (imports the deleted `StreamRegistry`)
- `hal/eval/play.py` (imports `Preprocessor`; hardcodes deleted `hal/data/stats.json`)

## Column renames / removals (libmelee → peppi)

| Old | New | Notes |
|-----|-----|-------|
| `p{i}_facing` (0/1) | `p{i}_direction` (-1.0 / 1.0 float) | |
| `p{i}_main_stick_x/y` ∈ [0, 1], neutral=0.5 | `p{i}_main_stick_x/y` ∈ [-1, 1], neutral=0 | name is the same; **range changed**. |
| `p{i}_c_stick_x/y` ∈ [0, 1] | `p{i}_c_stick_x/y` ∈ [-1, 1] | same. |
| `p{i}_on_ground` | `p{i}_airborne` | inverted. |
| `p{i}_jumps_left` | `p{i}_jumps_used` | sense inverted. |
| `p{i}_invulnerable` | `p{i}_hurtbox_state` | 0=vulnerable, 1=invulnerable, 2=intangible. |
| `p{i}_l_shoulder` / `p{i}_r_shoulder` | `p{i}_trigger_l_physical` / `p{i}_trigger_r_physical` (+ `p{i}_trigger_logical`) | |
| `p{i}_shield_strength` | `p{i}_shield` | |
| `p{i}_speed_*` | (gone) | not provided by peppi; derive client-side if needed. |
| `p{i}_ecb_*` | (gone) | not provided by peppi. |
| `p{i}_hitstun_left` / `p{i}_invulnerability_left` | (gone) | not provided by peppi. |
| `p{i}_character`, `p{i}_port`, `stage`, `replay_uuid` | (gone from per-frame) | now per-replay only, in `manifest.jsonl` (`ReplayIndexEntry`). |
| — | `p{i}_main_stick_raw_x/y` (int8) | new; raw analog bytes for bit-exact controller reproduction. Mask-sentinel for slp < 1.2.0 / 3.15.0. |
| — | `p{i}_nana_*` | new; Ice Climbers follower. Mask-sentinel for non-IC players. |

## Mask conventions

The new extractor uses **dtype-aware masks**, not a single `NP_MASK_VALUE`:

- floats → `NaN`
- signed int < 4 bytes (e.g. int8) → `dtype.min`
- signed int ≥ 4 bytes → `INT32_MAX` (= `NP_MASK_VALUE`)
- unsigned int → `dtype.max`

Anything that previously did `arr >= NP_MASK_VALUE` to mask invalids needs to switch
to per-dtype handling (or normalize to NaN at load time).

## Stats / streams

- `hal/data/stats.json` is gone. The new pipeline does not auto-compute stats — the
  rewritten preprocessor should regenerate them against the new shards.
- `hal/data/streams.py` (`StreamRegistry`) is gone. The new pipeline writes local shards
  to `<output>/{train,val,test}/`. Remote streaming will be re-introduced when the
  training loop is rewritten; design it against `manifest.jsonl` rather than the
  hardcoded ranked/top-player paths.

## Per-replay metadata

Per-frame columns no longer carry replay-level scalars. Look these up via
`hal.data.manifest.ReplayIndexEntry` keyed by replay path / `replay_uuid`:
`stage`, `slp_version`, per-player `character` / `port` / `code` / `name`, `outcome`,
`rank_filename`, `frame_count`, etc.
