"""
Model Config

Specifies the hyperparameters for the Pico model/model architecture.
Includes hyperparameters for ReLoRA.
"""

from dataclasses import dataclass
from typing import Optional

from ._constants import BATCH_SIZE, MAX_SEQ_LEN, VOCAB_SIZE


@dataclass
class ReLoRAConfig:
    """Config dataclass for ReLoRA.

    Hyperparameters taken from ReLoRA's [source implementation](https://github.com/Guitaricet/relora/blob/main/peft_pretraining/relora.py).
    """

    target_modules: list[str]
    reset_frequency: int  # reset frequency is measured in optimizer steps NOT global steps
    r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    keep_original_weights: bool = True
    lora_only: bool = False
    trainable_scaling: bool = False

    def __post_init__(self):
        """Validate post initialisation."""
        if self.r <= 0:
            raise ValueError("ReLoRA r must be positive!")


@dataclass
class ModelConfig:
    """Config dataclass for the Model."""

    d_model: int = 192
    n_layers: int = 12

    vocab_size: int = VOCAB_SIZE
    batch_size: int = BATCH_SIZE
    max_seq_len: int = MAX_SEQ_LEN

    attention_n_heads: int = 12
    attention_n_kv_heads: Optional[int] = 4

    activation_hidden_dim: int = 768

    norm_eps: float = 1e-6

    position_emb_theta: float = 10000.0

    relora: Optional[ReLoRAConfig] = None
