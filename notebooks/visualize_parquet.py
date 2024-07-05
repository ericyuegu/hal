# %%
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import seaborn as sns
from pyarrow import parquet as pq
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from hal.data.constants import ACTION_BY_IDX

np.set_printoptions(threshold=np.inf)

# %%
table: pa.Table = pq.read_table("/opt/projects/hal2/data/dev/train.parquet")
# randomly sample rows
# table = table.take(np.random.choice(len(table), 10000000, replace=False))

# %%
table.column_names

# %%

# %%
table["p1_position_y"].to_numpy().max()

# %%
# uuid_filter = pc.field("replay_uuid") == 5393121284994579877
# replay = table.filter(uuid_filter)

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
def visualize_position_heatmap(pyarrow_table: pa.Table, x_field: str, y_field: str, title: str) -> None:
    # Extract x and y values
    x = pyarrow_table[x_field].to_numpy()
    y = pyarrow_table[y_field].to_numpy()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a smooth heatmap using KDE
    sns.kdeplot(x=x, y=y, cmap="YlOrRd", fill=True, cbar=True, ax=ax)

    # Set labels and title
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    ax.set_title(title)

    # Set axis limits
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    # Invert y-axis to match stick orientation
    ax.invert_yaxis()

    # Show the plot
    plt.show()


# %%
# plot histogram of actions
actions = table["p1_action"].to_numpy()
actions = [ACTION_BY_IDX[action] for action in actions]

# %%
# rotate text and sort
plt.hist(actions, bins=185, rotation=90, sort=True)
plt.title("Player 1 Action Histogram")
plt.show()


# %%
visualize_position_heatmap(table, "p1_position_x", "p1_position_y", "Player 1 Position Heatmap")

# # %%
visualize_position_heatmap(table, "p1_main_stick_x", "p1_main_stick_y", "Player 1 Main Stick Heatmap")

# %%
# search k-means for p1_main_stick_x


# %%
def kmeans_hyperparameter_search(data, max_clusters=150, n_init=3):
    X = np.column_stack((data["p1_main_stick_x"].to_numpy(), data["p1_main_stick_y"].to_numpy()))

    results = []
    for n_clusters in tqdm(range(2, max_clusters + 1)):
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        cluster_labels = kmeans.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        inertia = kmeans.inertia_

        results.append(
            {"n_clusters": n_clusters, "silhouette_score": silhouette_avg, "inertia": inertia, "model": kmeans}
        )

    return results


# Perform hyperparameter search
search_results = kmeans_hyperparameter_search(table)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot([r["n_clusters"] for r in search_results], [r["silhouette_score"] for r in search_results], "bo-")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. Number of Clusters")

plt.subplot(1, 2, 2)
plt.plot([r["n_clusters"] for r in search_results], [r["inertia"] for r in search_results], "ro-")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Curve")

plt.tight_layout()
plt.show()

# Find the best model based on silhouette score
best_model = max(search_results, key=lambda x: x["silhouette_score"])
print(f"Best number of clusters: {best_model['n_clusters']}")
print(f"Best silhouette score: {best_model['silhouette_score']:.4f}")

# Visualize the best clustering result
X = np.column_stack((table["p1_main_stick_x"].to_numpy(), table["p1_main_stick_y"].to_numpy()))
cluster_labels = best_model["model"].predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels)
plt.xlabel("p1_main_stick_x")

plt.title("Best Clustering Result")
plt.show()


# %%
X = np.column_stack((table["p1_main_stick_x"].to_numpy(), table["p1_main_stick_y"].to_numpy()))
kmeans = KMeans(n_clusters=23, n_init=5, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# %%
cluster_centers = kmeans.cluster_centers_
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, s=50, cmap="viridis")
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c="red", s=200, alpha=0.75, marker="X")
plt.xlabel("p1_main_stick_x")
plt.ylabel("p1_main_stick_y")
plt.title("Cluster Locations")
plt.show()

# %%
print(cluster_centers)

# %%
