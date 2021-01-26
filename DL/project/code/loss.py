import torch
from torch import nn


class Loss:
    """Base class for handcrafted losses

    Attrs:
        base: Base criterion that the loss uses
    """
    base = nn.MSELoss(reduction="none")

    def __call__(self, prediction: torch.TensorType, target: torch.TensorType) -> torch.TensorType:
        """Compute the loss between the target and the prediction.

        Args:
            prediction (tensor[..., out]): The prediction to compare
            target (tensor[..., out]): Target of same size than prediction (typically: real glove position)

        Returns:
            tensor: The loss (the shape is often 1, if reduced.)
        """
        return self.base(prediction, target)


class LastLoss(Loss):
    """Sequence loss. Keep only the last moments of a sequence


    Make the assumption that the beginning is quite hard to predict
    and track only final errors.
    """
    def __init__(self, last: int = 1, reduction: str = "sum"):
        super().__init__()
        self.last = last
        self.reduction = reduction

    def __call__(self, prediction: torch.TensorType, target: torch.TensorType) -> torch.TensorType:
        unreduced = self.base(prediction[..., -self.last,:], target[..., -self.last,:])

        if self.reduction == "mean":
            return unreduced.mean()
        if self.reduction == "sum":
            return unreduced.sum() * target.shape[-2] / self.last  # Normalize this loss
        if self.reduction == "none":
            return unreduced  # Warning: Shape: (..., last, out_size) != (..., seq_len, out_size)
        raise ValueError(f"Unknown reduction method: {self.reduction}")


class DecayLoss(Loss):
    """Sequence Loss with decay.

    Gives much more importance to the end of the sequence:
    Make the assumption that the beginning is quite hard to predict
    and allow errors at the beginning.
    """
    def __init__(self, decay_power:int = 1, reduction: str = "sum"):
        super().__init__()
        self.decay_power = decay_power
        self.reduction = reduction

    def __call__(self, prediction: torch.TensorType, target: torch.TensorType) -> torch.TensorType:
        seq_len = target.shape[-2]   # target.shape = (..., seq_len, out_size)
        decay = torch.arange(seq_len, device=target.device)**self.decay_power
        decay = seq_len * decay / decay.sum()  # Renormalize this loss

        unreduced = self.base(prediction, target) * decay.view(seq_len, 1)

        if self.reduction == "mean":
            return unreduced.mean()
        if self.reduction == "sum":
            return unreduced.sum()
        if self.reduction == "none":
            return unreduced
        raise ValueError(f"Unknown reduction method: {self.reduction}")

class MovementLoss(Loss):
    """Sequence Loss that focus on movement.

    Gives much more importance the moment where there is movement

    Goal: Prevent to learn to predict the mean of the signal.
    """
    def __init__(self, movement_power: int = 1, epsilon: float = 0.005, decay_power:int = 0, reduction: str = "sum"):
        super().__init__()
        self.base = DecayLoss(decay_power, reduction="none")
        self.epsilon = epsilon
        self.movement_power = movement_power
        self.reduction = reduction

    def __call__(self, prediction: torch.TensorType, target: torch.TensorType) -> torch.TensorType:
        movements = torch.abs(target[..., :-1, :] - target[..., 1:, :])  # Shape: (..., seq_len - 1, out_size)
        movements = movements ** self.movement_power + self.epsilon
        movements *= 30  # Empirical normalization: mean(mvt(glove)) ~ 0.03

        unreduced = self.base(prediction[..., 1:, :], target[..., 1:, :]) * movements

        if self.reduction == "mean":
            return unreduced.mean()
        if self.reduction == "sum":
            return unreduced.sum()
        if self.reduction == "none":
            return unreduced
        raise ValueError(f"Unknown reduction method: {self.reduction}")
