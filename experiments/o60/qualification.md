# Experiment 060 G4 qualification

Run the qualification only from a clean, pushed commit. The launcher checks both conditions.

```sh
uv run scripts/launch_gce.py \
  --project centering-star-502613-k3 \
  --zone us-central1-b \
  --machine-type g4-standard-96 \
  --gpu-count 2 \
  --disk 3000 \
  --disk-type hyperdisk-balanced \
  --no-spot \
  -- \
  uv run scripts/run_060_g4_qualification.py qualification/060-g4
```

The runner performs the hardware and P2P preflight, matched throughput measurements,
the 512-update production-shape smoke run, and the interrupted-resume comparison. It
uploads the final immutable record under `qualifications/060-g4-<git-sha>/record.json`.
The startup script shuts down the VM after the command exits and retains the boot disk.

Do not use this command to start the 497,664-update training run.
