import torch
from torch import nn


class ElboLoss:
    """Elbo loss in the Gaussian case

    The KL divergence is done with a N(0, I).
    """
    def __init__(self):
        # The networks don't have sigmoid at the end. Let's use BCE with Logits.
        self.reconstruction_loss = nn.BCEWithLogitsLoss(reduction="sum")

    def __call__(self, prediction, target):
        batch_size = target.shape[0]
        mu, log_sigma, reconstructed = prediction
        recon_loss = self.reconstruction_loss(reconstructed, target)

        # Compute the reg loss from our formula: 0.5 \sum_n \sum_i \sigma_i(x_n) + \mu_i(x_n)^2 - \log \sigma_i(x_n) - 1
        reg_loss = 0.5 * (torch.exp(log_sigma) + mu**2 - log_sigma - 1).sum()  # Could prune the constant term 1.
        return (recon_loss + reg_loss) / batch_size
