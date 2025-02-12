import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.special


class Gaussian2DHistogramLoss(nn.Module):
    """
    Constructs a Gaussian distribution over a 2D grid, then use cross-entropy against model logits.

    Stop Regressing: Training Value Functions via Classification for Scalable Deep RL, (Farebrother et al. 2024)
    https://arxiv.org/abs/2403.03950v1
    """

    # TODO refactor to use 2d grid
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        self.support = torch.linspace(min_value, max_value, num_bins + 1, dtype=torch.float32)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, self.transform_to_probs(target))

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        cdf_evals = torch.special.erf(
            (self.support - target.unsqueeze(-1)) / (torch.sqrt(torch.tensor(2.0)) * self.sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)


class Gaussian2DPointsLoss(nn.Module):
    """
    Constructs a distribution over a fixed set of reference 2D points based on
    distance to a target 2D point, then uses cross-entropy against model logits.
    """

    def __init__(self, cluster_centers: torch.Tensor, sigma: float) -> None:
        """
        Args:
            cluster_centers: (N, 2) tensor of the 2D reference points.
            sigma: Standard deviation for the Gaussian used to distribute mass.
        """
        super().__init__()
        # Ensure cluster_centers is a (N, 2) float tensor on the same device later
        self.register_buffer("cluster_centers", cluster_centers.float())
        self.sigma = sigma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, L, N) raw output from the model (one logit per reference point).
            target: (B, L, 2) ground-truth 2D points in the same space as self.cluster_centers.
        Returns:
            A scalar cross-entropy loss between the model's logits and the distance-based distribution.
        """
        # Transform targets to distributions over cluster centers
        target_distributions = self.transform_to_probs(target)  # (B, L, N)
        # Use cross_entropy in a 'soft' way (i.e. the target is already a distribution)
        # F.cross_entropy expects integer class labels, so we do a manual cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target_distributions * log_probs).sum(dim=-1).mean()
        return loss

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        """
        Converts each 2D target point into a probability distribution over the
        cluster centers, using a Gaussian kernel.

        Args:
            target: (B, 2) ground-truth 2D points
        Returns:
            (B, N) tensor of probabilities over the cluster centers.
        """
        # (N, 2) -> (1, N, 2) for broadcasting
        cluster_centers_expanded = self.cluster_centers.unsqueeze(0)  # (1, N, 2)
        # (B, 2) -> (B, 1, 2) for broadcasting
        target_expanded = target.unsqueeze(1)  # (B, 1, 2)
        dists_sq_BN = torch.sum((cluster_centers_expanded - target_expanded) ** 2, dim=-1)

        weights_BN = torch.exp(-dists_sq_BN / (2.0 * self.sigma**2))

        probs = weights_BN / (torch.sum(weights_BN, dim=-1, keepdim=True) + 1e-10)
        return probs

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Optional helper: recovers an expected 2D point from the distribution over cluster centers.

        Args:
            probs: (B, N) probability distribution over cluster centers
        Returns:
            (B, 2) the "mean" in 2D, i.e. sum_i probs_i * center_i
        """
        return torch.sum(probs.unsqueeze(-1) * self.cluster_centers, dim=1)
