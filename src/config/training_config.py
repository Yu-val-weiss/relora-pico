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

    num_nodes: int = 1
    num_devices: int = 1
    precision: str = "16-mixed"
    accelerator: str = "cuda"
    strategy: str = "deepspeed_stage_2"


@dataclass
class OptimizationConfig:
    """Config dataclass for Optimization."""

    # Optimizer
    optimizer: str = "adamw"
    lr: float = 1e-5

    # Learning Rate Scheduler
    lr_scheduler: Literal["linear_with_warmup", "relora_jagged_cosine"] = "linear_with_warmup"
    lr_warmup_steps: int = 50_000

    # Extra LR Config for relora_jagged_cosine
    restart_warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    # Gradient Accumulation
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS


@dataclass
class TrainingConfig:
    """Config dataclass for Training."""

    fabric: FabricConfig = field(default_factory=FabricConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    strategy: str = "deepspeed"
    max_steps: int = 200_000
