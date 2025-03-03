"""PicoDecoder ReLoRA - adding ReLoRA to PicoDecoder!
(see src/model/pico_decoder.py for PicoDecoder's implementation).

References:
    - ReLoRA: https://arxiv.org/pdf/2307.05695

Adapted from:
    - ReLoRA: https://github.com/Guitaricet/relora
"""

import math
import os
from dataclasses import asdict, make_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import upload_file

from .pico_decoder import PicoDecoder, PicoDecoderHF, PicoDecoderHFConfig

try:
    if TYPE_CHECKING:
        from src.config.model_config import ModelConfig
except ImportError:
    pass


def functional_merge_and_reinit(module: nn.Module) -> None:
    """Functionally merge and reinit a ReLoRALinear module. If module is a different type to ReLoRALinear,
    this function is a no-op.

    Args:
        module (nn.Module): module to merge and reinit.
    """
    if not isinstance(module, ReLoRALinear):
        return

    # merge
    delta = (module.B_lora.weight @ module.A_lora.weight) * module._scale()
    module.weight.data += delta

    # reinit
    nn.init.kaiming_uniform_(module.A_lora.weight, a=math.sqrt(5))
    nn.init.zeros_(module.B_lora.weight)
    if module.trainable_scaling:
        nn.init.ones_(module.s)


class ReLoRALinear(nn.Module):
    """Linear ReLoRA layer. δW = s * W_A * W_B."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        model_config: Union["ModelConfig", "PicoDecoderHFConfig"],
        module_device: torch.device,
        *,
        bias: bool = True,
        bias_data: Optional[torch.Tensor] = None,
        weight_data: Optional[torch.Tensor] = None,
    ):
        """Wraps a linear layer into a ReLoRA layer."""
        super().__init__()

        relora_config = model_config.relora

        assert relora_config is not None

        # set up bias and weight
        if relora_config.lora_only is True:
            self.bias = None
            self.weight = None
        else:
            if bias_data is None:
                if bias:
                    bias_data = torch.zeros(
                        out_features,
                        device=module_device,
                        requires_grad=True,
                    )
            self.bias = nn.Parameter(bias_data) if bias and bias_data is not None else None

        if weight_data is None:
            # NOTE trainable weights are W_a and W_b
            weight_data = torch.zeros(out_features, in_features, device=module_device)

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
        self.A_lora = nn.Linear(self.in_feats, self.r, bias=False)
        self.B_lora = nn.Linear(self.r, self.out_feats, bias=False)

        # init A and B
        nn.init.kaiming_uniform_(self.A_lora.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_lora.weight)

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
        lora_result = self.B_lora(self.A_lora(self.dropout(x))) * self._scale()
        if self.lora_only:
            return lora_result
        return F.linear(x, self.weight, self.bias) + lora_result

    @torch.no_grad()
    def merge_and_reinit(self):
        """Performs ReLoRA merge step, and reinitialises. No-op if self.lora_only == True."""
        if self.lora_only:
            return

        # merge into weight (NOTE B @ A to preserve correct dimensions)
        self.weight.data += self.B_lora.weight @ self.A_lora.weight

        # reinitialise A and B
        nn.init.kaiming_uniform_(self.A_lora.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_lora.weight)
        if self.trainable_scaling:
            nn.init.ones_(self.s)


class PicoReLoRADecoder(PicoDecoder):
    """ReLoRA wrapper for the PicoDecoder model."""

    MODEL_TYPE = "pico_relora_decoder"

    def __init__(self, model_config: Union["ModelConfig", "PicoReLoRADecoderHFConfig"]):
        """Initialise the PicoDecoder model.

        Args:
            model_config (Union[ModelConfig, PicoDecoderHFConfig]): Model config.
        """
        super().__init__(model_config)

        relora_conf = self.config.relora

        if relora_conf is None:
            raise ValueError("Cannot init ReLoRAPico model with None ReLoRA config.")

        for module_name, module in self.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(
                target_key.lower() in module_name.lower() for target_key in relora_conf.target_modules
            ):
                continue

            weight_data = module.weight.data if relora_conf.keep_original_weights else None
            bias_data = None
            if module.bias is not None:
                bias_data = module.bias.data if relora_conf.keep_original_weights else None

            relora_module = ReLoRALinear(
                module.in_features,
                module.out_features,
                self.config,
                module.weight.device,
                bias=module.bias is not None,
                bias_data=bias_data,
                weight_data=weight_data,
            )

            if relora_conf.keep_original_weights:
                nn.init.zeros_(relora_module.A_lora.weight)  # no need to do B, since already init at 0

            if relora_conf.lora_only:
                if relora_conf.keep_original_weights:
                    msg = "Only one of lora_only and keep_original_weights can be true"
                    raise ValueError(msg)
                module.weight = None

            del module

            parent = self.get_parent(module_name)
            child_suffix = module_name.split(".")[-1]
            setattr(parent, child_suffix, relora_module)

            if os.environ.get("VERBOSE_RELORA", "false").lower() == "true":
                print(f"RELORAD {module_name}")
                for n, p in relora_module.named_parameters():
                    print(f"{n=}, {p.size()=}, {p.requires_grad=}")

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

    def convert_to_hf_model(self) -> "ReLoRAPicoDecoderHF":
        """Convert the Lightning model to a HuggingFace model."""
        # Create HF config without fabric-specific settings
        hf_config = PicoReLoRADecoderHFConfig.from_dataclass(self.config)

        # Create new HF model
        hf_model = ReLoRAPicoDecoderHF(hf_config)

        # Copy state dict, excluding fabric-specific keys
        hf_model.load_state_dict(self.state_dict(prefix="pico_decoder."))

        return hf_model


class PicoReLoRADecoderHFConfig(PicoDecoderHFConfig):
    """ReLoRA wrapper for HFConfig."""

    model_type = PicoReLoRADecoder.MODEL_TYPE

    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        config_dict = super().to_dict()
        relora_conf = config_dict.get("relora")
        if relora_conf is not None and not isinstance(relora_conf, dict):
            config_dict["relora"] = asdict(self.relora)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs) -> "PicoReLoRADecoderHFConfig":
        """Create a PicoDecoderHFConfig from a dict.

        Args:
            config_dict (Dict[str, Any]): Config dict to use.
            **kwargs: Keyword arguments, see below.

        Keyword arguments:
            return_unused_kwargs (bool, optional): If set to True, returns all unused kwargs as a dictionary
            in second element of tuple. Defaults to False.

        Returns:
            ReLoRAPicoDecoderHFConfig: Config for HuggingFace-compatible version of ReLoRAPico.
        """
        # NOTE The typical from_dict method doesn't actually set the attributes unless they are
        # defined in the constructor.

        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        relora_dict: dict | None = config_dict.pop("relora", None)

        # handle parent class from_dict
        result = super().from_dict(config_dict, return_unused_kwargs=return_unused_kwargs)

        if return_unused_kwargs:
            pico_config, unused_kwargs = result
        else:
            pico_config = result
            unused_kwargs = {}

        # Add ReLoRA config if present
        if relora_dict:
            # Create dataclass dynamically from relora config, to avoid import from src.config.model_config
            ReLoRAConfig = make_dataclass(
                "ReLoRAConfig", [(name, type(v)) for name, v in relora_dict.items()]
            )

            pico_config.relora = ReLoRAConfig(**relora_dict)

        if return_unused_kwargs:
            return pico_config, unused_kwargs
        return pico_config


class ReLoRAPicoDecoderHF(PicoDecoderHF):
    """ReLoRA wrapper for HuggingFace wrapper for PicoDecoder model."""

    config_class = PicoReLoRADecoderHFConfig
    _no_split_modules = ["PicoBlock", "Attention", "SwiGLU", "RMSNorm", "ReLoRALinear"]

    def __init__(self, config: PicoReLoRADecoderHFConfig):
        """Initialise HuggingFace wrapper for PicoDecoder model.

        Args:
            config (ReLoRaPicoDecoderHFConfig): Config to initialise from.
        """
        super().__init__(config)
        self.pico_decoder = PicoReLoRADecoder(config)

    def push_to_hub(self, repo_id, commit_message, revision=None, token=None, **kwargs):
        """Override to push PicoDecoder as well"""
        super().push_to_hub(
            repo_id=repo_id, commit_message=commit_message, revision=revision, token=token, **kwargs
        )

        pico_path = Path(__file__).resolve().parent / "pico_decoder.py"

        upload_file(
            path_or_fileobj=pico_path,
            path_in_repo="pico_decoder.py",
            repo_id=repo_id,
            revision=revision,
            commit_message=commit_message,
            **kwargs,
        )


PicoReLoRADecoderHFConfig.register_for_auto_class()
ReLoRAPicoDecoderHF.register_for_auto_class("AutoModel")
ReLoRAPicoDecoderHF.register_for_auto_class("AutoModelForCausalLM")
