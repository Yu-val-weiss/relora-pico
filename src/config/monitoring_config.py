"""
Monitoring Config

Specifies the monitoring process, e.g. how to log metrics and keep track of training progress.
"""

from dataclasses import dataclass, field


@dataclass
class LoggingConfig:
    """Logging config Dataclass."""

    log_level: str = "INFO"
    log_every_n_steps: int = 100


@dataclass
class WandbConfig:
    """Weights and Biases configuation Dataclass"""

    # configure logging to Weights and Biases
    project: str = ""
    entity: str = ""


@dataclass
class MonitoringConfig:
    """Monitoring config dataclass."""

    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Weights and Biases
    save_to_wandb: bool = False
    wandb: WandbConfig = field(default_factory=WandbConfig)
