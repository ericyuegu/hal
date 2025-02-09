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
# %%
mds_path = "/opt/projects/hal2/data/mang0/train"
mang0_ds = StreamingDataset(local=mds_path, batch_size=1, shuffle=True)
len(mang0_ds)

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
    if i > 10000:
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
# randomly sample 1000000 points
main_stick = main_stick[np.random.choice(len(main_stick), size=1000000, replace=False)]
# %%
c_stick = c_stick[np.random.choice(len(c_stick), size=1000000, replace=False)]
# %%
c_stick.shape
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
main_stick_centroids
# %%
plt.scatter(main_stick_centroids[:, 0], main_stick_centroids[:, 1], color="red")

# %%
main_stick_centroids = k_means(main_stick, k=256, max_iterations=10)

# %%
plt.scatter(main_stick_centroids[:, 0], main_stick_centroids[:, 1], color="red")

# %%
pts = main_stick_centroids
# Define the center (about which symmetry is desired)
center = np.array([0.5, 0.5])

# A tolerance for “matching” points (you can adjust this)
tol = 0.05

N = len(pts)

# =====================================================
# 2. Force symmetry across the y–axis (vertical mirror)
# =====================================================
# Here we want that for any point (x,y), its partner should be (1-x, y).
# We loop over points and try to pair those that nearly satisfy this relation.
used = np.zeros(N, dtype=bool)
new_pts_y = pts.copy()  # will hold the adjusted points

for i in range(N):
    if used[i]:
        continue
    # If the point is already on the vertical axis, leave it.
    if np.abs(pts[i, 0] - 0.5) < 1e-6:
        used[i] = True
        continue
    # Define the “ideal mirror” of pts[i]
    target = np.array([1 - pts[i, 0], pts[i, 1]])
    # Look for an unpaired candidate that is close to the target.
    candidates = [j for j in range(N) if (not used[j]) and (j != i)]
    if len(candidates) == 0:
        continue
    dists = np.array([np.linalg.norm(pts[j] - target) for j in candidates])
    j_min = candidates[np.argmin(dists)]
    if dists[np.argmin(dists)] < tol:
        # We found a matching pair.
        # Average the y–coordinates and compute an average x–offset.
        avg_y = (pts[i, 1] + pts[j_min, 1]) / 2.0
        # For perfect symmetry across x=0.5, we require:
        #   x1 = 0.5 + d    and    x2 = 0.5 - d.
        d1 = pts[i, 0] - 0.5
        d2 = 0.5 - pts[j_min, 0]
        avg_d = (d1 + d2) / 2.0
        new_pts_y[i] = np.array([0.5 + avg_d, avg_y])
        new_pts_y[j_min] = np.array([0.5 - avg_d, avg_y])
        used[i] = True
        used[j_min] = True
    else:
        # If no partner is found within tolerance, you might decide to
        # simply “reflect” the point. Here we average it with its own mirror.
        new_pts_y[i] = (pts[i] + np.array([1 - pts[i, 0], pts[i, 1]])) / 2.0
        used[i] = True

# ======================================================
# 3. Force 4–way symmetry (add horizontal mirror symmetry)
# ======================================================
# Now we take the y–axis symmetric points and force symmetry with
# respect to the horizontal line y=0.5 (i.e. (x,y) and (x, 1-y) should match).
pts_temp = new_pts_y.copy()
used = np.zeros(N, dtype=bool)
new_pts_4 = pts_temp.copy()  # will hold the final 4–way symmetric points

for i in range(N):
    if used[i]:
        continue
    # If the point is already on the horizontal axis, leave it.
    if np.abs(pts_temp[i, 1] - 0.5) < 1e-6:
        used[i] = True
        continue
    # The mirror of pts_temp[i] across y=0.5 is:
    target = np.array([pts_temp[i, 0], 1 - pts_temp[i, 1]])
    # Look for an unpaired candidate near the target.
    candidates = [j for j in range(N) if (not used[j]) and (j != i)]
    if len(candidates) == 0:
        continue
    dists = np.array([np.linalg.norm(pts_temp[j] - target) for j in candidates])
    j_min = candidates[np.argmin(dists)]
    if dists[np.argmin(dists)] < tol:
        # We have found a matching pair.
        avg_x = (pts_temp[i, 0] + pts_temp[j_min, 0]) / 2.0
        # For perfect symmetry across y=0.5, we require:
        #   y1 = 0.5 + d    and    y2 = 0.5 - d.
        d1 = pts_temp[i, 1] - 0.5
        d2 = 0.5 - pts_temp[j_min, 1]
        avg_dy = (d1 + d2) / 2.0
        new_pts_4[i] = np.array([avg_x, 0.5 + avg_dy])
        new_pts_4[j_min] = np.array([avg_x, 0.5 - avg_dy])
        used[i] = True
        used[j_min] = True
    else:
        # If no partner is found, reflect the point across y=0.5.
        new_pts_4[i] = (pts_temp[i] + np.array([pts_temp[i, 0], 1 - pts_temp[i, 1]])) / 2.0
        used[i] = True

# ======================================================
# 4. Plot the Original, y–axis Symmetric, and 4–way Symmetric points
# ======================================================
plt.figure(figsize=(16, 5))

# Original points
plt.subplot(1, 3, 1)
plt.scatter(pts[:, 0], pts[:, 1], c="blue", s=40, label="Original")
plt.scatter(center[0], center[1], c="red", marker="x", s=100, label="Center")
plt.title("Original Points")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()

# y-axis symmetric points
plt.subplot(1, 3, 2)
plt.scatter(new_pts_y[:, 0], new_pts_y[:, 1], c="green", s=40, label="y–axis Symmetric")
plt.scatter(center[0], center[1], c="red", marker="x", s=100, label="Center")
plt.title("Forced Symmetry across y–axis")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()

# 4–way symmetric points
plt.subplot(1, 3, 3)
plt.scatter(new_pts_4[:, 0], new_pts_4[:, 1], c="magenta", s=40, label="4–way Symmetric")
plt.scatter(center[0], center[1], c="red", marker="x", s=100, label="Center")
plt.title("Forced 4–way Symmetry")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()

plt.tight_layout()
plt.show()


# %%
def remove_near_duplicates(points, rtol=1e-3):
    """
    Remove points that are nearly identical within a relative tolerance.

    Parameters:
        points: np.ndarray of shape (n_points, n_dimensions)
        rtol: relative tolerance for considering points as duplicates

    Returns:
        np.ndarray with near-duplicates removed
    """
    # Sort points lexicographically to make it easier to find duplicates
    sorted_idx = np.lexsort(points.T)
    sorted_points = points[sorted_idx]

    # Calculate distances between consecutive points
    diff = np.abs(sorted_points[1:] - sorted_points[:-1])
    tol = rtol * np.abs(sorted_points[:-1])

    # Points are duplicates if all their coordinates are within tolerance
    duplicates = np.all(diff <= tol, axis=1)
    keep = np.ones(len(points), dtype=bool)
    keep[sorted_idx[1:][duplicates]] = False

    return points[keep]


# Replace the np.unique line with:
deduped_new_pts_y = remove_near_duplicates(new_pts_y, rtol=1e-2)
len(deduped_new_pts_y)
# %%
plt.scatter(deduped_new_pts_y[:, 0], deduped_new_pts_y[:, 1], c="green", s=40, label="y–axis Symmetric")
plt.scatter(center[0], center[1], c="red", marker="x", s=100, label="Center")
plt.title("Forced Symmetry across y–axis")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()
plt.show()
# %%
deduped_new_pts_y
# %%
# Sort points by x and y coordinates while keeping rows together
sorted_points = deduped_new_pts_y[np.lexsort((deduped_new_pts_y[:, 1], deduped_new_pts_y[:, 0]))]
sorted_points
