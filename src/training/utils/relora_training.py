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

    # For regular optimizers:
    #   - optimizer.state is a dict[torch.nn.Parameter, dict[str, torch.Tensor]]
    #   - optimizer.state[p] is a dict[str, torch.Tensor] where str is
    #   - an optimizer state key e.g., "exp_avg", "exp_avg_sq"

    # For ZeroRedundancyOptimizer, it works differently:
    #   - ZeroRedundancyOptimizer.state always maps to empty dicts.
    #   - instead, it uses optimizer.optim.state for rank-local updates.
    # also, fabric wraps the optimizer in lightning.fabric.wrappers.FabricDeepSpeedZeroOptimizer
    # to get to where we need, we need to call optimizer.optimizer.optim.state
    # is_zero_opt = "DeepSpeedZero" in optimizer.__class__.__name__
    # if is_zero_opt:
    #     print(dir(optimizer))
    # optimizer_state = optimizer.state if not is_zero_opt else optimizer.optimizer.optimizer.state_dict()
    reset_count = 0
    print("opt state", len(optimizer.state))
    print("opt opt state", len(optimizer.optimizer.optimizer.state))
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Group {i}: {len(param_group['params'])} parameters")
    for i, param_group in enumerate(optimizer.optimizer.optimizer.param_groups):
        print(f"Group inside {i}: {len(param_group['params'])} parameters")
    for n, p in reset_params:
        if id(p) in optimizer.state:
            param_state = optimizer.state[id(p)]
            for key in optimizer_state_keys:
                random_prune_(param_state[key], 0.999)
            reset_count += 1
    print(f"reset count: {reset_count} from {torch.cuda.current_device()}")
