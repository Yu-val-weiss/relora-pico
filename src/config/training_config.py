"""
Training Config

Specifies the hyperparameters for the training process, i.e. the optimizer, learning rate, etc.
"""

from dataclasses import dataclass, field
from typing import Literal

from ._constants import GRADIENT_ACCUMULATION_STEPS


@dataclass
class FabricConfig:
    """Config dataclass for Fabric."""

    # Configure nodes/devices for parallelised training
    num_nodes: int = 1
    num_devices: int = 1
    precision: str = "bf16-mixed"
    # Hardware accelerator to use, can be cpu/cuda/mps etc.
    accelerator: str = "cuda"


@dataclass
class OptimizationConfig:
    """Config dataclass for Optimization."""

    # Optimizer
    optimizer: str = "adamw"
    lr: float = 3e-4

    # Learning Rate Scheduler
    lr_scheduler: Literal["linear_with_warmup", "relora_jagged_cosine"] = "linear_with_warmup"
    lr_warmup_steps: int = 2500

    # Extra LR Config for relora_jagged_cosine
    restart_warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    # Extra LR Config for relora_jagged_cosine
    restart_warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    # Define number of gradient accumulation steps
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS


@dataclass
class TrainingConfig:
    """Config dataclass for Training."""

    fabric: FabricConfig = field(default_factory=FabricConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    max_steps: int = 200_000
