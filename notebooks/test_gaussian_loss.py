# %%
import matplotlib.pyplot as plt
import torch
from constants import STICK_XY_CLUSTER_CENTERS_V0
from constants import STICK_XY_CLUSTER_CENTERS_V1

from hal.training.losses import Gaussian2DPointsLoss


# %%
# Plot probability mass at each cluster center
def plot_probs(actual_points: torch.Tensor, probs_BD: torch.Tensor, reference_points: torch.Tensor) -> None:
    for actual, probs in zip(actual_points.tolist(), probs_BD.tolist()):
        plt.figure(figsize=(8, 8))
        for (x, y), p in zip(reference_points, probs):
            plt.scatter(x, y, s=p * 1000, alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.title(f"Probability Mass at Cluster Centers, x,y={round(actual[0], 2)},{round(actual[1], 2)}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.show()


# %%
loss_fn = Gaussian2DPointsLoss(torch.tensor(STICK_XY_CLUSTER_CENTERS_V0), sigma=0.08)

actual_points = torch.tensor([[0.5, 0.5], [1.0, 0.5], [0.15, 0.3], [0.8338728, 0.13657573], [0.03, 0.5]])
probs = loss_fn.transform_to_probs(actual_points)

plot_probs(actual_points, probs, STICK_XY_CLUSTER_CENTERS_V0)

probs
# %%
probs.max(dim=1)

# %%
loss_fn = Gaussian2DPointsLoss(torch.tensor(STICK_XY_CLUSTER_CENTERS_V1), sigma=0.05)

probs = loss_fn.transform_to_probs(actual_points)

plot_probs(actual_points, probs, STICK_XY_CLUSTER_CENTERS_V1)

probs

# %%
loss_fn = Gaussian2DPointsLoss(torch.tensor(STICK_XY_CLUSTER_CENTERS_V0), sigma=0.08)
loss_fn.forward(
    logits=torch.tensor(
        [
            [
                4.1289e-09,
                2.8945e-06,
                6.3895e-29,
                2.9123e-05,
                6.3505e-30,
                4.5774e-05,
                2.1375e-17,
                1.4431e-05,
                6.7800e-17,
                1.0117e-16,
                7.3166e-00,
                3.9931e-18,
                5.5211e-34,
                1.2924e-32,
                2.0903e-11,
                1.9449e-00,
                1.2025e-22,
                2.4184e-21,
                3.4080e-32,
                1.0393e-12,
                7.3754e-01,
            ]
        ]
    ),
    target=torch.tensor([[0.8338728, 0.13657573]]),
)
