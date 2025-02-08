# %%

# Run hyperparam sweep

import os

device = 0

preprocessing_fns = (
    "inputs_v0",
    "inputs_v0_controller",  # controller
    "inputs_v1",  # action frame
    "inputs_v1_controller",  # action frame + controller
)

archs = (
    # abs position emb
    "GPTv1-256-4-4",
    "GPTv1Controller-256-4-4",
    # relative position emb
    "GPTv3-256-4-4",
    "GPTv3Controller-256-4-4",
)

# Change BASE_CMD to echo instead of running python
BASE_CMD = "echo 'Would run: CUDA_VISIBLE_DEVICES={device} python hal/training/simple_trainer.py --n_gpus 1 --data.data_dir data/multishine --arch {arch} --eval.n_workers 1 --embedding.input_preprocessing_fn {preprocess_fn}'"

if __name__ == "__main__":
    # Create a new tmux session if not already in one
    session_name = "hyperparam_sweep_test"  # Changed name to distinguish from real runs
    os.system(f"tmux new-session -d -s {session_name}")

    for i, arch in enumerate(archs):
        # Only use controller preprocessing fns with controller architectures
        valid_preprocess_fns = [fn for fn in preprocessing_fns if ("Controller" in arch) == ("controller" in fn)]

        for j, preprocess_fn in enumerate(valid_preprocess_fns):
            pane_name = f"{arch}-{preprocess_fn}"
            cmd = BASE_CMD.format(device=device, arch=arch, preprocess_fn=preprocess_fn)

            # Print the command that would be run
            print(f"Creating pane for: {pane_name}")
            print(f"Command: {cmd}")

            if i == 0 and j == 0:
                tmux_cmd = f"tmux send-keys -t {session_name} '{cmd}' C-m"
            else:
                tmux_cmd = f"tmux split-window -t {session_name} '{cmd}' && tmux select-layout -t {session_name} tiled"

            os.system(tmux_cmd)
            device += 1

    # Uncomment this line to automatically attach to the session
    os.system(f"tmux attach-session -t {session_name}")
