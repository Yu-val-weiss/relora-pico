"""File containing ReLoRA training utils."""

import torch
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
import lightning as L

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
    fabric: L.Fabric,
    named_reset_params: list[tuple[str, torch.nn.Parameter]],
    optimizer_state_keys: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    non_zero_sum = torch.Tensor(0, device=fabric.device)
    zeroed = torch.Tensor(0, device=fabric.device)

    for _, p in named_reset_params:
        param_state = optimizer.state[p]
        for key in optimizer_state_keys:
            non_zero = torch.count_nonzero(param_state[key])
            non_zero_sum += non_zero
            random_prune_(param_state[key], PRUNING_RATIO)
            zeroed += non_zero - torch.count_nonzero(param_state[key])

    return zeroed, non_zero_sum


def _zero_opt_reset(
    optimizer: torch.optim.Optimizer,
    fabric: L.Fabric,
    named_reset_params: list[tuple[str, torch.nn.Parameter]],
    optimizer_state_keys: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resets optimizer states for DeepSpeed Zero optimizer.
    DeepSpeed zero optimizer stores state for all parameters in a single flat tensor, so requires special logic.
    """

    #

    optimizer: DeepSpeedZeroOptimizer = optimizer.optimizer  # get inner optimizer
    # parameters in this partition of the optimizer
    partition_params = set(optimizer.params_in_partition[0])
    # maps a slice of the flat tensor relating to a specific paramter
    slice_mappings = optimizer._param_slice_mappings[0]

    assert len(optimizer.state) == 1, "expected single tensor in DeepSpeedZeroOptimizer state"

    state_dict = next(iter(optimizer.state.values()))

    mask_tensors: dict[str, torch.Tensor] = {
        key: torch.ones_like(state_dict[key]) for key in optimizer_state_keys
    }

    non_zero_sum = torch.tensor(0, device=fabric.device)

    for n, p in named_reset_params:
        if p in partition_params:
            fixed_name = n.split(".module.")[-1]
            param_slice_map = slice_mappings[fixed_name]
            param_size = param_slice_map.numel
            param_slice = slice(param_slice_map.start, param_slice_map.start + param_size)
            for key in optimizer_state_keys:
                mask_tensors[key][param_slice] = torch.rand(param_size, device=state_dict[key].device)
                non_zero_sum += torch.count_nonzero(state_dict[key][param_slice])

    zeroed = torch.tensor(0, device=fabric.device)

    with torch.no_grad():
        for key in optimizer_state_keys:
            mask = mask_tensors[key] > PRUNING_RATIO
            non_zero = torch.count_nonzero(state_dict[key])
            state_dict[key].mul_(mask)
            zeroed += non_zero - torch.count_nonzero(state_dict[key])

    return zeroed, non_zero_sum


def reset_optimizer_for_relora(
    optimizer: torch.optim.Optimizer,
    fabric: L.Fabric,
    *,
    named_reset_params: list[tuple[str, torch.nn.Parameter]],
    optimizer_state_keys: list[str],
) -> torch.Tensor:
    """Resets optimizer for relora.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to reset.
        fabric (Fabric): fabric instance.
        named_reset_params: list[tuple[str, torch.nn.Parameter]], includes names of params.
        optimizer_state_keys: list[str].
    """

    is_zero_opt = "DeepSpeedZero" in optimizer.__class__.__name__
    if is_zero_opt:
        tup = _zero_opt_reset(optimizer, fabric, named_reset_params, optimizer_state_keys)
        fabric.all_reduce(tup, reduce_op="sum")
        zeroed, non_zero_sum = tup
    else:
        zeroed, non_zero_sum = _simple_reset(optimizer, fabric, named_reset_params, optimizer_state_keys)

    return zeroed / non_zero_sum
