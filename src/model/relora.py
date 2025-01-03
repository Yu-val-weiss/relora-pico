"""Pico ReLoRA - adding ReLoRA to Pico! (see src/model/pico.py for pico's implementation).

References:
    - ReLoRA: https://arxiv.org/pdf/2307.05695

Adapted from:
    - ReLoRA: https://github.com/Guitaricet/relora
"""

import math
from typing import Optional, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import ModelConfig
from model.pico import PicoHFConfig


def functional_merge_and_reinit(module: nn.Module) -> None:
    """Functionally merge and reinit a ReLoRALinear module. If module is a different type to ReLoRALinear,
    this function is a no-op.

    Args:
        module (nn.Module): module to merge and reinit.
    """
    if not isinstance(module, ReLoRALinear):
        return

    # merge
    delta = (module.B.weight @ module.A.weight) * module._scale()
    module.weight.data += delta

    # reinit
    nn.init.kaiming_uniform_(module.A.weight, a=math.sqrt(5))
    nn.init.zeros_(module.B.weight)
    if module.trainable_scaling:
        nn.init.zeros_(module.s)


class ReLoRALinear(nn.Module):
    """Linear ReLoRA layer. δW = s * W_A * W_B."""

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        model_config: Union["ModelConfig", "PicoHFConfig"],
        fabric: Optional["L.Fabric"] = None,
        *,
        bias: bool = True,
        bias_data: Optional[torch.Tensor] = None,
        weight_data: Optional[torch.Tensor] = None,
    ):
        """Wraps a linear layer into a ReLoRA layer."""
        super().__init__()

        self.fabric = fabric

        relora_config = model_config.relora

        assert relora_config is not None

        device = self.fabric.device if self.fabric is not None else None

        # set up bias and weight
        if relora_config.lora_only is True:
            self.bias = None
            self.weight = None
        else:
            if bias_data is None:
                if bias:
                    bias_data = torch.zeros(
                        out_feats,
                        device=device,
                        requires_grad=True,
                    )
            self.bias = nn.Parameter(bias_data) if bias and bias_data is not None else None

        if weight_data is None:
            # NOTE trainable weights are W_a and W_b
            weight_data = torch.zeros(out_feats, in_feats, device=device)

        self.weight = nn.Parameter(weight_data, requires_grad=False)

        # initialise other parameters
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.r = relora_config.r
        self.lora_alpha = relora_config.lora_alpha
        self.dropout = nn.Dropout(p=relora_config.lora_dropout)
        self.lora_only = relora_config.lora_only
        self.trainable_scaling = relora_config.trainable_scaling

        # δW = s * W_A * W_B
        self.A = nn.Linear(self.in_feats, self.r, bias=False)
        self.B = nn.Linear(self.r, self.out_feats, bias=False)

        # init A and B
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

        # init s, the scaling factor
        if self.trainable_scaling:
            self.s = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        else:
            self.s = self.lora_alpha / self.r

        # freeze the weight
        if not self.lora_only and self.weight is not None:
            self.weight.requires_grad = False

    def _scale(self) -> torch.Tensor | float:
        if self.trainable_scaling:
            return self.s.tanh()
        return self.s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass."""
        lora_result = self.B(self.A(self.dropout(x))) * self._scale()
        if self.lora_only:
            return lora_result
        return F.linear(x, self.weight, self.bias) + lora_result

    @torch.no_grad()
    def merge_and_reinit(self):
        """Performs ReLoRA merge step, and reinitialises. No-op if self.lora_only == True."""
        if self.lora_only:
            return

        # merge into weight (NOTE B @ A to preserve correct dimensions)
        self.weight.data += self.B.weight @ self.A.weight

        # reinitialise A and B
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        if self.trainable_scaling:
            nn.init.zeros_(self.s)
