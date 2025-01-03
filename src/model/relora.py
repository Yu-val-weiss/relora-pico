"""Pico ReLoRA - adding ReLoRA to Pico! (see src/model/pico.py for pico's implementation).

References:
    - ReLoRA: https://arxiv.org/pdf/2307.05695

Adapted from:
    - ReLoRA: https://github.com/Guitaricet/relora
"""

import math
from dataclasses import asdict
from typing import Any, Optional, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.model_config import ModelConfig, ReLoRAConfig
from src.model.pico import Pico, PicoHF, PicoHFConfig


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
        in_features: int,
        out_features: int,
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
                        out_features,
                        device=device,
                        requires_grad=True,
                    )
            self.bias = nn.Parameter(bias_data) if bias and bias_data is not None else None

        if weight_data is None:
            # NOTE trainable weights are W_a and W_b
            weight_data = torch.zeros(out_features, in_features, device=device)

        self.weight = nn.Parameter(weight_data, requires_grad=False)

        # initialise other parameters
        self.in_feats = in_features
        self.out_feats = out_features
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


class ReLoRAPico(Pico):
    """ReLoRA wrapper for the Pico model."""

    def __init__(
        self,
        model_config: Union["ModelConfig", "ReLoRAPicoHFConfig"],
        fabric: Optional["L.Fabric"] = None,
    ):
        """Initialise the Pico model.

        Args:
            model_config (Union[ModelConfig, PicoHFConfig]): Model config.
            fabric (L.Fabric, optional): Fabric instance to use. Defaults to None.
        """
        super().__init__(model_config, fabric)

        relora_conf = self.config.relora

        if relora_conf is None:
            raise ValueError("Cannot init ReLoRAPico model with None ReLoRA config.")

        for module_name, module in self.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in relora_conf.target_modules):
                continue

            weight_data = module.weight.data if relora_conf.keep_original_weights else None
            bias_data = None
            if module.bias is not None:
                bias_data = module.bias.data if relora_conf.keep_original_weights else None

            relora_module = ReLoRALinear(
                module.in_features,
                module.out_features,
                self.config,
                self.fabric,
                bias=module.bias is not None,
                bias_data=bias_data,
                weight_data=weight_data,
            )

            if relora_conf.keep_original_weights:
                nn.init.zeros_(relora_module.A.weight)  # no need to do B, since already init at 0

            if relora_conf.lora_only:
                assert (
                    not relora_conf.keep_original_weights
                ), "Only one of lora_only and keep_original_weights can be true"
                module.weight = None

            del module

            parent = self.get_parent(module_name)
            child_suffix = module_name.split[-1]
            setattr(parent, child_suffix, relora_module)

    def get_parent(self, module_name: str) -> nn.Module:
        """Gets the module's parent.

        Args:
            module_name (str): Child's fully qualified name.

        Returns:
            nn.Module: Parent module.
        """
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        return self.get_submodule(parent_name)

    def merge_and_reinit(self) -> None:
        """Merge and reinit all ReLoRALinear layers."""

        for module in self.modules():
            if isinstance(module, ReLoRALinear):
                module.merge_and_reinit()

    def convert_to_hf_model(self) -> "ReLoRAPicoHF":
        """Convert the Lightning model to a HuggingFace model."""
        # Create HF config without fabric-specific settings
        hf_config = ReLoRAPicoHFConfig.from_dataclass(self.config)

        # Create new HF model
        hf_model = ReLoRAPicoHF(hf_config)

        # Copy state dict, excluding fabric-specific keys
        hf_model.load_state_dict(self.state_dict(prefix="pico."))

        return hf_model


class ReLoRAPicoHFConfig(PicoHFConfig):
    """ReLoRA wrapper for HFConfig."""

    model_type = "relora-pico"

    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        config_dict = super().to_dict()
        if config_dict.get("relora") is not None:
            config_dict["relora"] = asdict(self.relora)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs) -> "ReLoRAPicoHFConfig":
        """Create a PicoHFConfig from a dict.

        Args:
            config_dict (Dict[str, Any]): Config dict to use.
            **kwargs: Keyword arguments, see below.

        Keyword arguments:
            return_unused_kwargs (bool, optional): If set to True, returns all unused kwargs as a dictionary
            in second element of tuple. Defaults to False.

        Returns:
            ReLoRAPicoHFConfig: Config for HuggingFace-compatible version of ReLoRAPico.
        """
        # NOTE The typical from_dict method doesn't actually set the attributes unless they are
        # defined in the constructor.

        return_unused_kwargs = kwargs.get("return_unused_kwargs", False)

        pico_config = super().from_dict(config_dict)
        if return_unused_kwargs:
            pico_config, unused_kwargs = pico_config

        if "relora" in config_dict:
            pico_config.relora = ReLoRAConfig(**config_dict["relora"])

        if return_unused_kwargs:
            return pico_config, unused_kwargs
        return pico_config


class ReLoRAPicoHF(PicoHF):
    """ReLoRA wrapper for HuggingFace wrapper for Pico model."""

    config_class = ReLoRAPicoHFConfig
    _no_split_modules = ["PicoBlock", "Attention", "SwiGLU", "RMSNorm", "ReLoRALinear"]

    def __init__(self, config: ReLoRAPicoHFConfig):
        """Initialise HuggingFace wrapper for Pico model.

        Args:
            config (ReLoRaPicoHFConfig): Config to initialise from.
        """
        super().__init__(config)
        self.pico = ReLoRAPico(config)


ReLoRAPicoHFConfig.register_for_auto_class()
ReLoRAPicoHF.register_for_auto_class("AutoModel")
ReLoRAPicoHF.register_for_auto_class("AutoModelForCausalLM")
