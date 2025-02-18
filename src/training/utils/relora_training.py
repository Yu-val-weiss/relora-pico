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
) -> None:
    for _, p in named_reset_params:
        param_state = optimizer.state[p]
        for key in optimizer_state_keys:
            random_prune_(param_state[key], PRUNING_RATIO)


def _zero_opt_reset(
    optimizer: torch.optim.Optimizer,
    named_reset_params: list[tuple[str, torch.nn.Parameter]],
    optimizer_state_keys: list[str],
) -> None:
    # deep speed zero optimizer stores state for all parameters in a single flat tensor

    optimizer: DeepSpeedZeroOptimizer = optimizer.optimizer
    partition_params = set(optimizer.params_in_partition[0])  # parameters in this partition of the optimizer
    slice_mappings = optimizer._param_slice_mappings[0]

    st_key = next(iter(optimizer.state.keys()))

    st_dict = optimizer.state[st_key]
    mask_tensors: dict[str, torch.Tensor] = {
        key: torch.ones_like(st_dict[key]) for key in optimizer_state_keys
    }

    for n, p in named_reset_params:
        if p in partition_params:
            fixed_name = n.split(".module.")[-1]
            param_slice = slice_mappings[fixed_name]
            for key in optimizer_state_keys:
                mask_tensors[key][param_slice.start : param_slice.start + param_slice.numel] = torch.rand(
                    param_slice.numel, device=st_dict[key].device
                )

    with torch.no_grad():
        for key in optimizer_state_keys:
            mask = mask_tensors[key] > PRUNING_RATIO
            mask = mask.type_as(st_dict[key])
            print(mask)
            print(
                f"from {torch.cuda.current_device()} -- {key} before = {torch.count_nonzero(st_dict[key])}"
            )
            st_dict[key].mul_(mask)
            print(f"from {torch.cuda.current_device()} -- {key} after = {torch.count_nonzero(st_dict[key])}")


def reset_optimizer_for_relora(
    optimizer: torch.optim.Optimizer,
    *,
    named_reset_params: list[tuple[str, torch.nn.Parameter]],
    optimizer_state_keys: list[str],
) -> None:
    """Resets optimizer for relora.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to reset.
        named_reset_params: list[tuple[str, torch.nn.Parameter]], includes names of params.
        optimizer_state_keys: list[str].
    """

    is_zero_opt = "DeepSpeedZero" in optimizer.__class__.__name__
    if is_zero_opt:
        _zero_opt_reset(optimizer, named_reset_params, optimizer_state_keys)
    else:
        _simple_reset(optimizer, named_reset_params, optimizer_state_keys)
