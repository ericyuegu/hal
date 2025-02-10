import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.special


class GaussianHistogramLoss(nn.Module):
    """
    Stop Regressing: Training Value Functions via Classification for Scalable Deep RL, (Farebrother et al. 2024)
    https://arxiv.org/abs/2403.03950v1
    """

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
