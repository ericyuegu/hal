# HAL Project Guidelines

# About the project

- We are training a decoder-only Transformer on Super Smash Bros. Melee using imitation learning & offline RL
- We have preprocessed human replays using libmelee and stored them as MDS shards following the schema in schema.py
- We sample trajectories from the dataset by choosing a random episode, random starting frame, and preprocessing seq_len subsequent frames to predict controller inputs as next-token prediction
- Preprocessing and target feature discretization are defined as functions in configs: input_configs.py, target_configs.py, postprocess_configs.py
    - Currently, the best working configs are `fine_main_analog_shoulder`, which discretizes the analog main stick into 37 joint x, y positions, predicts analog shoulder presses (no digital button L/R), all as single-label classification problems
- Model definitions are under models/gpt.py. Ignore lstm.py and mlp.py, they are deprecated
- We have a closed loop eval harness that runs dolphin emulator and batches inputs on GPU in eval/eval.py. This is a very precise script that writes directly to shared memory buffers, do not touch it. 

# Making changes

- Never edit existing configs or model definitions to maintain backwards compatability & experiment reproducibility. To add a new class definition, append towards the bottom of the files (e.g. gpt.py or input_configs.py) and register a configuration with a sensible name at the bottom
    - I generally don't prioritize DRY across input/target configs or model versions
- Please feel free to test code by creating new scratch files in notebooks/ at the root of the repo
- Please help me keep the code readable, composable, and modular without sacrificing reproducibility and experimental isolation
- When generating edits to code, do not add comments or references to previous code or current instruction, unless you are explicitly told to do so

# Running experiments

- A training command looks like so (never run this without permission): `CUDA_VISIBLE_DEVICES=0 python hal/training/simple_trainer.py --n_gpus 1 --data.streams cody --data.stream_stats data/top_players/Cody/stats.json --arch GPTv5Controller-512-6-8-dropout --data.input_preprocessing_fn baseline_controller_fine_main_analog_shoulder_early_release --data.target_preprocessing_fn fine_main_analog_shoulder_early_release --data.pred_postprocessing_fn fine_main_analog_shoulder`

# Future directions

I want to train multi-token prediction, N frame latency, and an off-policy value model using offline RL. Eventually self-play PPO. 


## Code Style
- **Formatting**: Black with line_length=119, isort with black profile
- **Types**: Use type annotations everywhere. Return types required.
- **Imports**: Group order: stdlib, third-party, first-party (hal). Single line imports.
- **Naming**: snake_case for functions/variables, CamelCase for classes, UPPERCASE for constants
- **Error Handling**: Use descriptive exception messages, contextmanager for resources

## Project Structure
This codebase is a machine learning project for Super Smash Bros Melee AI, with model training, 
data processing, and emulator integration components. Use loguru for logging and attrs for config.