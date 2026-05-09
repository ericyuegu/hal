# HAL Project Guidelines

# About the project

The goal of this project is to train Transformer models on Super Smash Bros. Melee using imitation learning & offline RL

We're rebooting the project after a long hiatus, and we're in the midst of rewriting everything from scratch.

For context, here is how we did it previously:
- We preprocess human .slp replays using libmelee and stored them as MDS shards following the schema in schema.py
- We sample trajectories from the dataset by choosing a random episode, random starting frame, and preprocessing seq_len subsequent frames to predict controller inputs as next-token prediction
- Preprocessing and target feature discretization are defined as functions in configs: input_configs.py, target_configs.py, postprocess_configs.py
    - Currently, the best working configs are `fine_main_analog_shoulder`, which discretizes the analog main stick into 37 joint x, y positions, predicts analog shoulder presses (no digital button L/R), all as single-label classification problems
- Model definitions are under models/gpt.py. Ignore lstm.py and mlp.py, they are deprecated
- We have a closed loop eval harness that runs Dolphin emulator and batches inputs on GPU in eval/eval.py. This is a very precise script that writes directly to shared memory buffers, be careful when touching it. 

Going forward, I would like to: 
- simplify & rewrite the data preprocessing pipeline from .slp to .mds for reliability, scalability & speed
- establish sanity checks for closed loop gamestate reproducibility from .slp to nparrays/tensors back through the melee.Controller interface into Dolphin
- modeling
    - use receding horizon control with flow matching action heads instead of classification
    - action chunk predictions should directly regress on continuous (float) values, either in time or frequency domain (i.e. DCT)
- revisit training loop
- revisit use of tensordicts (need to profile speed)
- revisit eval harness interface
- simplify and delete lots of old code, including old notebooks, feature preprocessing logic, model architectures, evals, and training loop
- investigate resuming from arbitrary frames in replay and forking/performing controller takeover in Dolphin to perform efficient rollouts for RL

# Principles

- Existing code is not precious. Code is tech debt. Delete liberally. The marginal cost of rewriting code rounds to zero, but the benefit of cleaner, better abstractions is high.
- Don't make references to "existing convention" from other parts of the repo in your comments unless asked.
- Invalid states should be impossible to represent.

## Code Style
- **Formatting**: Black with line_length=119, isort with black profile
- **Types**: Use type annotations everywhere. Return types required.
- **Imports**: Group order: stdlib, third-party, first-party (hal). Single line imports.
- **Naming**: snake_case for functions/variables, CamelCase for classes, UPPERCASE for constants
- **Error Handling**: Use descriptive exception messages, contextmanager for resources
    - Never swallow exceptions (i.e. just `pass`), never use bare `except`
    - Don't catch exceptions just to log and rethrow—only wrap an exception if that part of the stack can add helpful context for debugging
    - Always name the exceptions being caught, ideally with extremely specific clauses; do not write `except Exception` unless it is a crucial runtime code path that must never crash—these cases are uncommon but readily apparent
    - Avoid fallback logic or fallback values that silently change behavior or configuration
- **Type Annotations**: All functions, classes, and variables must specify explicit type annotations. Always include return types for functions. This ensures complete static type safety and clarity throughout the codebase.
    - We are on py314, don't use `from __future__ import annotations`

### Suggested Libraries
- Use `loguru` for logging
- Use MosaicML Streaming `streaming` and MDS format for datasets: https://docs.mosaicml.com/projects/streaming/en/stable/index.html
- Use `libmelee` to handle the Dolphin (emulator) lifecycle, Enet/spectator protocol, and blocking controller injection
- Use `peppi-py` to batch read .slp files offline
- Use `tyro` for CLIs

## Project Structure
This codebase is a machine learning project for Super Smash Bros Melee AI, with model training, 
data processing, and emulator integration components. Use loguru for logging and attrs for config.