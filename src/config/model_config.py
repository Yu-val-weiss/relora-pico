"""
Model Config

Specifies the hyperparameters for the Pico model/model architecture.
"""

from dataclasses import dataclass
from typing import Optional

from ._constants import BATCH_SIZE, MAX_SEQ_LEN, VOCAB_SIZE


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
