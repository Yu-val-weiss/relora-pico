"""File containing ReLoRA training utils."""

import torch


@torch.no_grad()
def random_prune_(tensor: torch.Tensor, pruning_ratio: float) -> None:
    """Randomly prune the tensor to the desired ratio, in place.

    Args:
        tensor (torch.Tensor): Tensor to prune.
        pruning_ratio (float): Pruning ratio.
    """
    pruning_mask = torch.rand_like(tensor) > pruning_ratio
    tensor.mul_(pruning_mask)


def reset_optimizer_for_relora(
    optimizer: torch.optim.Optimizer,
    *,
    reset_params: list[torch.nn.Parameter],
    optimizer_state_keys: list[str],
) -> None:
    """Resets optimizer for relora.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to reset.
        reset_params: list[torch.nn.Parameter],
        optimizer_state_keys: list[str],
    """
    for p in reset_params:
        param_state = optimizer.state[p]
        for key in optimizer_state_keys:
            random_prune_(param_state[key], 0.999)
