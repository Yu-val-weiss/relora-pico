"""File containing ReLoRA training utils."""

import torch
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

PRUNING_RATIO = 0.999


@torch.no_grad()
def random_prune_(tensor: torch.Tensor, pruning_ratio: float) -> None:
    """Randomly prune the tensor to the desired ratio, in place.

    Args:
        tensor (torch.Tensor): Tensor to prune.
        pruning_ratio (float): Pruning ratio.
    """
    pruning_mask = torch.rand_like(tensor) > pruning_ratio
    tensor.mul_(pruning_mask)


def _simple_reset(
    optimizer: torch.optim.Optimizer,
    named_reset_params: list[tuple[str, torch.nn.Parameter]],
    optimizer_state_keys: list[str],
) -> tuple[int, int]:
    non_zero_sum = 0
    zeroed = 0

    for _, p in named_reset_params:
        param_state = optimizer.state[p]
        for key in optimizer_state_keys:
            non_zero = torch.count_nonzero(param_state[key]).item()
            non_zero_sum += non_zero
            random_prune_(param_state[key], PRUNING_RATIO)
            zeroed += non_zero - torch.count_nonzero(param_state[key]).item()

    return zeroed, non_zero_sum


def _zero_opt_reset(
    optimizer: torch.optim.Optimizer,
    named_reset_params: list[tuple[str, torch.nn.Parameter]],
    optimizer_state_keys: list[str],
) -> tuple[int, int]:
    # deep speed zero optimizer stores state for all parameters in a single flat tensor

    optimizer: DeepSpeedZeroOptimizer = optimizer.optimizer
    # parameters in this partition of the optimizer
    partition_params = set(optimizer.params_in_partition[0])
    # maps a slice of the flat tensor relating to a specific paramter
    slice_mappings = optimizer._param_slice_mappings[0]

    assert len(optimizer.state) == 1, "expected single tensor in DeepSpeedZeroOptimizer state"

    state_dict = next(iter(optimizer.state.values()))

    mask_tensors: dict[str, torch.Tensor] = {
        key: torch.ones_like(state_dict[key]) for key in optimizer_state_keys
    }

    # curr_dev = torch.cuda.current_device()

    # print(f"{curr_dev} -- {mask_tensors['exp_avg']} -- {torch.sum(mask_tensors['exp_avg'] == 1)}")
    non_zero_sum = 0

    for n, p in named_reset_params:
        if p in partition_params:
            fixed_name = n.split(".module.")[-1]
            param_slice_map = slice_mappings[fixed_name]
            param_size = param_slice_map.numel
            param_slice = slice(param_slice_map.start, param_slice_map.start + param_size)
            for key in optimizer_state_keys:
                mask_tensors[key][param_slice] = torch.rand(param_size, device=state_dict[key].device)
                non_zero_sum += torch.count_nonzero(state_dict[key][param_size]).item()

    # print(f"{curr_dev} -- {mask_tensors['exp_avg']} -- {torch.sum(mask_tensors['exp_avg'] == 1)}")

    zeroed = 0

    with torch.no_grad():
        for key in optimizer_state_keys:
            mask = mask_tensors[key] > PRUNING_RATIO
            non_zero = torch.count_nonzero(state_dict[key]).item()
            state_dict[key].mul_(mask)
            zeroed += non_zero - torch.count_nonzero(state_dict[key]).item()

    return zeroed, non_zero_sum


def reset_optimizer_for_relora(
    optimizer: torch.optim.Optimizer,
    *,
    named_reset_params: list[tuple[str, torch.nn.Parameter]],
    optimizer_state_keys: list[str],
) -> tuple[int, int]:
    """Resets optimizer for relora.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to reset.
        named_reset_params: list[tuple[str, torch.nn.Parameter]], includes names of params.
        optimizer_state_keys: list[str].
    """

    is_zero_opt = "DeepSpeedZero" in optimizer.__class__.__name__
    if is_zero_opt:
        return _zero_opt_reset(optimizer, named_reset_params, optimizer_state_keys)
    return _simple_reset(optimizer, named_reset_params, optimizer_state_keys)
