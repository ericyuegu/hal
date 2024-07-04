# %%

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import seaborn as sns
from pyarrow import parquet as pq

np.set_printoptions(threshold=np.inf)

# %%
table: pa.Table = pq.read_table("/opt/projects/hal2/data/train.parquet")

# %%
table.column_names

# %%
uuid_filter = pc.field("replay_uuid") == 5393121284994579877
replay = table.filter(uuid_filter)

# p1_l_shoulder = replay["p1_l_shoulder"].to_pylist()
# p1_button_l = replay["p1_button_l"].to_pylist()
# for i, (analog, button) in enumerate(zip(p1_l_shoulder, p1_button_l)):
#     if math.ceil(analog) != button or math.floor(analog) != button:
#         print(f"{i=}, {analog=}, {button=}")

# %%
# p1_l_shoulder = replay["p1_l_shoulder"].to_numpy()
# p1_button_l = replay["p1_button_l"].to_numpy()
# print(f"{p1_l_shoulder.mean()=}")
# print(f"{p1_button_l.mean()=}")

# print(p1_l_shoulder)
# print(p1_button_l)

# %%
len(table)


# %%
def visualize_stick_heatmap(pyarrow_table: pa.Table) -> None:
    # Extract x and y values
    x = pyarrow_table["p1_main_stick_x"].to_numpy()
    y = pyarrow_table["p1_main_stick_y"].to_numpy()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a smooth heatmap using KDE
    sns.kdeplot(x=x, y=y, cmap="YlOrRd", fill=True, cbar=True, ax=ax)

    # Set labels and title
    ax.set_xlabel("Main Stick X")
    ax.set_ylabel("Main Stick Y")
    ax.set_title("Player 1 Main Stick Heatmap")

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    # Invert y-axis to match stick orientation
    ax.invert_yaxis()

    # Show the plot
    plt.show()


# Assuming 'replay' is your data structure containing the stick values
visualize_stick_heatmap(table)
