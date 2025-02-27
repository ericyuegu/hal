# %%
import numpy as np
from constants import ACTION_BY_IDX
from tensordict import TensorDict

from hal.preprocess.transformations import convert_multi_hot_to_one_hot_early_release
from hal.preprocess.transformations import encode_original_buttons_multi_hot

# np.set_printoptions(threshold=np.inf)

# %%
buttons_LD = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ]
)

convert_multi_hot_to_one_hot_early_release(buttons_LD)

# %%
buttons_LD = np.array(
    [
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ],
)

convert_multi_hot_to_one_hot_early_release(buttons_LD)

# %%
from streaming import StreamingDataset

# mds_path = "/opt/projects/hal2/data/ranked/diamond/train"
mds_path = "/opt/projects/hal2/data/top_players/Cody/test"
ds = StreamingDataset(local=mds_path, batch_size=1, shuffle=True)

x = ds[0]
# %%
orig = encode_original_buttons_multi_hot(TensorDict(x), "p1")
one_hot = convert_multi_hot_to_one_hot_early_release(orig.numpy())
print(f"{orig=}")
print(f"{one_hot=}")

# %%
x = ds[1]
# Find indices where left button is pressed
l_indices = np.where(x["p2_button_l"] == 1)[0]
l_indices
# %%
# Find indices where right button is pressed
r_indices = np.where(x["p2_button_r"] == 1)[0]
r_indices
# %%
common_indices = np.intersect1d(l_indices, r_indices)
common_indices
# %%
diff_indices = np.where((x["p2_r_shoulder"] != 0) & (x["p2_r_shoulder"] != 1))[0]
diff_indices.tolist()
x["p2_r_shoulder"][diff_indices]
# %%
diff_indices = np.where((x["p1_l_shoulder"] != 0) & (x["p1_l_shoulder"] != 1))[0]
diff_indices.tolist()
import numpy as np

# %%
# Find 3 roughly evenly spaced clusters in the non-binary shoulder values
from sklearn.cluster import KMeans

# Extract the non-binary shoulder values
values = x["p1_l_shoulder"][diff_indices]

# Apply KMeans clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(values.reshape(-1, 1))
cluster_centers = kmeans.cluster_centers_.flatten()
cluster_centers.sort()

print(f"3 evenly spaced cluster centers: {cluster_centers}")

# Visualize the distribution with the cluster centers
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.hist(values, bins=20)
for center in cluster_centers:
    plt.axvline(x=center, color="r", linestyle="--")
plt.title("Distribution of non-binary shoulder values with cluster centers")
plt.xlabel("Shoulder value")
plt.ylabel("Frequency")
plt.show()
# %%

# %%
# Plot the non-binary values of p1_l_shoulder
import matplotlib.pyplot as plt

values = x["p1_l_shoulder"][diff_indices]
plt.figure(figsize=(10, 5))
plt.bar(range(len(values)), values)
plt.xlabel("Index")
plt.ylabel("p1_l_shoulder value")
plt.title("Non-binary values of p1_l_shoulder")
plt.grid(True, alpha=0.3)
plt.show()

# %%
for i in diff_indices:
    print(f"{x['frame'][i]:<6}", end="")
    print(f"{x['p2_button_l'][i]:<6}", end="")
    print(f"{x['p2_button_r'][i]:<6}")

# %%
x["p2_character"]
# %%
x.keys()
# %%
# Print button and shoulder presses by frame for a range of frames
frame_range = range(0, 1000)

# Define button names for clearer output
button_names_by_key = {
    "button_a": "A",
    "button_b": "B",
    "button_x": "X",
    "button_y": "Y",
    "button_z": "Z",
    "button_l": "L_digital",
    "button_r": "R_digital",
    "l_shoulder": "L_analog",
    "r_shoulder": "R_analog",
    "main_stick_x": "main_x",
    "main_stick_y": "main_y",
}
# Create a table header
print(f"{'Frame':<6}", end="")
for name in button_names_by_key.values():
    print(f"{name:<8}", end="")
print()

# Print separator line
print("-" * 6, end="")
for _ in button_names_by_key.values():
    print("-" * 8, end="")
print()

# Print data rows
for frame in frame_range:
    print(f"{x['frame'][frame]:<6}", end="")
    for key, _ in button_names_by_key.items():
        field_name = f"p2_{key}"
        value = x[field_name][frame]
        # Format value based on type (float vs int)
        if isinstance(value, float):
            print(f"{value:8.3f}", end="")
        else:
            print(f"{value:<8}", end="")
    print()

# %%
# graph main_x, main_y on a unit circle for a range of frames, printing non-zero button press values
frame_range = range(400, 500)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Create a figure with two subplots - one for the stick positions and one for button presses
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Main Stick Positions and Button Presses for Frames 400-500")

# Get main stick positions for the frame range
main_x = [x[f"p2_main_stick_x"][frame] for frame in frame_range]
main_y = [x[f"p2_main_stick_y"][frame] for frame in frame_range]

# Create a colormap to show frame progression
colors = np.linspace(0, 1, len(frame_range))
cmap = LinearSegmentedColormap.from_list("frame_progression", ["blue", "red"])

# Plot the main stick positions with color indicating frame progression
scatter = ax1.scatter(main_x, main_y, c=colors, cmap=cmap, alpha=0.7)

# Add frame numbers as annotations to each point
for i, (x_pos, y_pos) in enumerate(zip(main_x, main_y)):
    # Only label every 5th point to avoid overcrowding
    if i % 5 == 0:
        ax1.annotate(
            str(frame_range[i]), (x_pos, y_pos), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8
        )

# Draw the unit circle with radius 0.5 centered at (0.5, 0.5)
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = 0.5 + 0.5 * np.cos(theta)
circle_y = 0.5 + 0.5 * np.sin(theta)
ax1.plot(circle_x, circle_y, "k--", label="Unit Circle (r=0.5)")

# Add a point at the center (0.5, 0.5)
ax1.scatter(0.5, 0.5, color="black", marker="x", s=100, label="Center (0.5, 0.5)")

# Set limits, labels, and grid for the first subplot
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_title("Main Stick Positions")
ax1.grid(True)
ax1.legend()
ax1.set_aspect("equal")

# Add a colorbar to show frame progression
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label("Frame Progression")

# Create a heatmap of button presses in the second subplot
button_keys = [
    "button_a",
    "button_b",
    "button_x",
    "button_y",
    "button_z",
    "button_l",
    "button_r",
]
button_data = np.zeros((len(frame_range), len(button_keys)))

# Fill the button data array
for i, frame in enumerate(frame_range):
    for j, key in enumerate(button_keys):
        field_name = f"p2_{key}"
        button_data[i, j] = x[field_name][frame]

# Plot the heatmap
im = ax2.imshow(button_data, aspect="auto", cmap="YlOrRd")
ax2.set_title("Button Presses")
ax2.set_xlabel("Button")
ax2.set_ylabel("Frame")

# Set y-ticks to show actual frame numbers
y_ticks = np.linspace(0, len(frame_range) - 1, 10, dtype=int)
ax2.set_yticks(y_ticks)
ax2.set_yticklabels([str(frame_range[i]) for i in y_ticks])

# Set x-ticks to show button names
ax2.set_xticks(np.arange(len(button_keys)))
ax2.set_xticklabels([key.replace("button_", "").replace("_shoulder", "") for key in button_keys], rotation=45)

# Add a colorbar for the heatmap
cbar2 = plt.colorbar(im, ax=ax2)
cbar2.set_label("Button Value")

plt.tight_layout()
plt.show()

# Print frames with non-zero button presses
print("\nFrames with non-zero button presses:")
print(f"{'Frame':<6}{'Buttons Pressed'}")
print("-" * 50)

for frame in frame_range:
    # Check for any non-zero button values
    button_presses = []
    for key in button_names_by_key.keys():
        field_name = f"p2_{key}"
        value = x[field_name][frame]
        if (isinstance(value, float) and value > 0.1) or (isinstance(value, (int, np.integer)) and value > 0):
            button_presses.append(
                f"{button_names_by_key[key]}:{value:.1f}"
                if isinstance(value, float)
                else f"{button_names_by_key[key]}"
            )

    if button_presses:
        print(f"{x['frame'][frame]:<6}{', '.join(button_presses)}")

# %%
multi_hot = encode_original_buttons_multi_hot(TensorDict(x), "p1")
# %%
multi_hot[862:875]
# %%
action_idx = x["p1_action"][862:875]
actions = [ACTION_BY_IDX[i] for i in action_idx]
actions
# %%
main_x, main_y = x["p1_main_stick_x"][862:875], x["p1_main_stick_y"][862:875]

# %%
x["p1_l_shoulder"][862:875].tolist()
# %%
x["p1_r_shoulder"][862:875].tolist()
# %%
import matplotlib.pyplot as plt
import numpy as np

# Create a figure
plt.figure(figsize=(8, 8))

# Plot the main stick positions
plt.scatter(main_x, main_y, color="blue", label="Main Stick Positions")

# Draw the unit circle with radius 0.5 centered at (0.5, 0.5)
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = 0.5 + 0.5 * np.cos(theta)
circle_y = 0.5 + 0.5 * np.sin(theta)
plt.plot(circle_x, circle_y, "r--", label="Unit Circle (r=0.5)")

# Add a point at the center (0.5, 0.5)
plt.scatter(0.5, 0.5, color="red", marker="x", s=100, label="Center (0.5, 0.5)")

# Set limits, labels, and grid
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Main Stick Positions with Unit Circle (r=0.5)")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.show()

import numpy as np

# %%
import torch
from tensordict import TensorDict

from hal.constants import SHOULDER_CLUSTER_CENTERS_V1
from hal.preprocess.transformations import get_closest_1D_clusters

x = np.array([[0.0, 0.2], [0.34, 0.7], [0.55, 0.9]])
get_closest_1D_clusters(x, SHOULDER_CLUSTER_CENTERS_V1)
# %%

from hal.preprocess.transformations import encode_original_shoulder_one_hot_finer
from hal.preprocess.transformations import encode_shoulder_one_hot_coarse

encode_original_shoulder_one_hot_finer(
    TensorDict({"p1_l_shoulder": torch.tensor([0.0, 0.34, 0.55]), "p1_r_shoulder": torch.tensor([0.4, 0.7, 0.98])}),
    "p1",
)
encode_shoulder_one_hot_coarse(
    TensorDict({"p1_l_shoulder": torch.tensor([0.0, 0.34, 0.55]), "p1_r_shoulder": torch.tensor([0.4, 0.7, 0.98])}),
    "p1",
).shape

# %%
actions = x["p2_action"][400:500]
actions = [ACTION_BY_IDX[i] for i in actions]
actions

# %%

import matplotlib.animation as animation

# %%
# Create a GIF animation showing main stick positions and button presses for each frame
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.patches import Rectangle

frame_range = range(400, 500)
button_keys = [
    "button_a",
    "button_b",
    "button_x",
    "button_y",
    "button_z",
    "button_l",
    "button_r",
]  # Removed l_shoulder and r_shoulder

# Create figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Main Stick Position and Button Presses")

# Setup for the stick position plot (left subplot)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_title("Main Stick Position")
ax1.grid(True)
ax1.set_aspect("equal")

# Draw the unit circle with radius 0.5 centered at (0.5, 0.5)
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = 0.5 + 0.5 * np.cos(theta)
circle_y = 0.5 + 0.5 * np.sin(theta)
ax1.plot(circle_x, circle_y, "k--", label="Unit Circle (r=0.5)")
ax1.scatter(0.5, 0.5, color="black", marker="x", s=100, label="Center (0.5, 0.5)")
ax1.legend(loc="upper right", fontsize="small")

# Setup for the button presses plot (right subplot)
ax2.set_xlim(-0.5, len(button_keys) - 0.5)
ax2.set_ylim(-0.5, 1.5)
ax2.set_title("Button Presses")
ax2.set_xticks(range(len(button_keys)))
ax2.set_xticklabels([key.replace("button_", "") for key in button_keys], rotation=45)
ax2.set_yticks([])
ax2.set_aspect(0.5)

# Initialize plots
stick_point = ax1.scatter([], [], color="red", s=100, zorder=5)
frame_text = ax1.text(
    0.05,
    0.95,
    "",
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
)

# Create rectangles for button states
button_rects = []
for i in range(len(button_keys)):
    rect = Rectangle((i - 0.4, 0), 0.8, 0, facecolor="red", alpha=0.7)
    ax2.add_patch(rect)
    button_rects.append(rect)


def init():
    stick_point.set_offsets(np.empty((0, 2)))
    frame_text.set_text("")
    for rect in button_rects:
        rect.set_height(0)
    return [stick_point, frame_text] + button_rects


def animate(i):
    frame = frame_range[i]

    # Update stick position
    main_x = x[f"p2_main_stick_x"][frame]
    main_y = x[f"p2_main_stick_y"][frame]
    stick_point.set_offsets([[main_x, main_y]])

    # Update frame text
    frame_text.set_text(f'Frame: {x["frame"][frame]}')

    # Update button states - only digital buttons
    for j, key in enumerate(button_keys):
        field_name = f"p2_{key}"
        value = x[field_name][frame]
        height = value  # Digital values (0 or 1)
        button_rects[j].set_height(height)
        button_rects[j].set_y(0)

    return [stick_point, frame_text] + button_rects


# Create animation
ani = animation.FuncAnimation(fig, animate, frames=len(frame_range), init_func=init, blit=True, interval=100)

plt.tight_layout()

# Save as GIF (requires ImageMagick)
ani.save("stick_and_buttons.gif", writer="pillow", fps=10)

# Display the animation in the notebook
HTML(ani.to_jshtml())
