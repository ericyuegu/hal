# %%
import numpy as np
from matplotlib import pyplot as plt
from streaming import StreamingDataset


# %%
def assign_clusters(data: np.ndarray, centroids: np.ndarray, chunk_size: int = 100_000) -> np.ndarray:
    """
    Assign each data point to the nearest centroid using squared distances.
    Processes data in chunks to avoid large memory usage.

    Parameters:
      data:     (n_points, n_dim) array.
      centroids:(k, n_dim) array.
      chunk_size: number of points to process at once.

    Returns:
      labels: (n_points,) array of cluster indices.
    """
    n = data.shape[0]
    labels = np.empty(n, dtype=np.int32)

    # Precompute ||centroid||^2 for all centroids
    centroids_sq = np.sum(centroids**2, axis=1)  # Shape: (k,)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = data[start:end]

        # Compute squared distances:
        #   d(x, c)^2 = ||x||^2 + ||c||^2 - 2 * (x dot c)
        # (chunk**2).sum(axis=1, keepdims=True) has shape (chunk_size, 1)
        distances = np.sum(chunk**2, axis=1, keepdims=True) + centroids_sq - 2 * chunk.dot(centroids.T)

        # Assign the closest centroid (no need to take sqrt)
        labels[start:end] = np.argmin(distances, axis=1)

    return labels


def update_centroids(data: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """
    Compute new centroids as the mean of the points assigned to each cluster.
    Uses np.bincount to aggregate values over labels.

    Parameters:
      data:   (n_points, n_dim) array.
      labels: (n_points,) array of cluster indices.
      k:      number of clusters.

    Returns:
      new_centroids: (k, n_dim) array.
    """
    n_dim = data.shape[1]
    new_centroids = np.empty((k, n_dim), dtype=data.dtype)

    # Count how many points fall into each cluster
    counts = np.bincount(labels, minlength=k)

    # Compute the sum of coordinates for each cluster and then divide by the count
    for dim in range(n_dim):
        # For each dimension, sum the data values per cluster
        sums = np.bincount(labels, weights=data[:, dim], minlength=k)
        new_centroids[:, dim] = sums

    # Avoid division by zero: if a cluster is empty, reinitialize its centroid randomly.
    for j in range(k):
        if counts[j] > 0:
            new_centroids[j] /= counts[j]
        else:
            new_centroids[j] = data[np.random.choice(len(data))]

    return new_centroids


def k_means(data: np.ndarray, k: int, max_iterations: int = 100, chunk_size: int = 100_000) -> np.ndarray:
    """
    An optimized k-means implementation.

    Parameters:
      data:           (n_points, n_dim) array.
      k:              number of clusters.
      max_iterations: maximum iterations.
      chunk_size:     size of chunks for distance computations.

    Returns:
      centroids: (k, n_dim) array of centroids.
    """
    # Randomly initialize centroids from the data points
    indices = np.random.choice(len(data), size=k, replace=False)
    centroids = data[indices]

    for iteration in range(max_iterations):
        print(f"k={k}, iteration {iteration}")

        # Step 1: Assign clusters (using chunking to control memory use)
        labels = assign_clusters(data, centroids, chunk_size)

        # Step 2: Update centroids in a vectorized manner
        new_centroids = update_centroids(data, labels, k)

        # Check for convergence (you may adjust the tolerance)
        if np.allclose(centroids, new_centroids, rtol=1e-5, atol=1e-8):
            break

        centroids = new_centroids

    return centroids


# %%
mds_path = "/opt/projects/hal2/data/ranked/train"
ds = StreamingDataset(local=mds_path, batch_size=1, shuffle=True)

main_stick_x_tensors = []
main_stick_y_tensors = []
c_stick_x_tensors = []
c_stick_y_tensors = []

# %%
len(ds)

# %%
for i, sample in enumerate(ds):
    if i > 500:
        break
    if i % 100 == 0:
        print(f"Processing sample {i}")
    for player in ["p1", "p2"]:
        main_stick_x_tensors.append(sample[f"{player}_main_stick_x"])
        main_stick_y_tensors.append(sample[f"{player}_main_stick_y"])
        c_stick_x_tensors.append(sample[f"{player}_c_stick_x"])
        c_stick_y_tensors.append(sample[f"{player}_c_stick_y"])

# %%
len(main_stick_x_tensors)
# %%
main_stick_x = np.concatenate(main_stick_x_tensors)
main_stick_y = np.concatenate(main_stick_y_tensors)
c_stick_x = np.concatenate(c_stick_x_tensors)
c_stick_y = np.concatenate(c_stick_y_tensors)

# %%
main_stick = np.stack((main_stick_x, main_stick_y), axis=-1)
c_stick = np.stack((c_stick_x, c_stick_y), axis=-1)

# %%
main_stick.shape
# %%
main_stick_centroids = k_means(main_stick, k=21, max_iterations=10)
# %%
plt.scatter(main_stick_centroids[:, 0], main_stick_centroids[:, 1], color="red")

# %%
main_stick_centroids = k_means(main_stick, k=64, max_iterations=10)

# %%
plt.scatter(main_stick_centroids[:, 0], main_stick_centroids[:, 1], color="red")

# %%
main_stick_centroids = k_means(main_stick, k=128, max_iterations=10)

# %%
plt.scatter(main_stick_centroids[:, 0], main_stick_centroids[:, 1], color="red")

# %%
main_stick_centroids = k_means(main_stick, k=256, max_iterations=10)

# %%
plt.scatter(main_stick_centroids[:, 0], main_stick_centroids[:, 1], color="red")

# %%
