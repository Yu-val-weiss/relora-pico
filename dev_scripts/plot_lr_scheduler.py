"""Script to plot a learning rate scheduler"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import SGD

from src.config.training_config import OptimizationConfig, TrainingConfig
from src.training.utils.initialization import initialize_lr_scheduler

STEPS = 300_000
LR_WARMUP_STEPS = 10_000
RESTART_WARMUP_STEPS = 100
RELORA_RESET_FREQ = 20_000


def main():
    """Plot learning rate"""
    optimizer = SGD([torch.tensor(1)], lr=1)

    training_config = TrainingConfig(
        None,
        OptimizationConfig(
            lr=1,
            lr_scheduler="relora_jagged_cosine",
            lr_warmup_steps=LR_WARMUP_STEPS,
            restart_warmup_steps=RESTART_WARMUP_STEPS,
            min_lr_ratio=0.1,
        ),
        strategy="auto",
        max_steps=STEPS,
    )

    training_config.relora_reset_freq = RELORA_RESET_FREQ

    scheduler = initialize_lr_scheduler(training_config, optimizer)

    fig, ax = plt.subplots()

    lrs = []
    for _ in range(STEPS):
        optimizer.step()
        lrs.append(scheduler.get_last_lr())
        scheduler.step()

    ax.plot(lrs)
    ax.set_xlabel("Step (1000s)")
    ax.set_ylabel("Learning rate multiplier")
    ax.grid(True)
    ticks = np.arange(0, STEPS + 1, step=RELORA_RESET_FREQ)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks // 1000)

    plt.show()


if __name__ == "__main__":
    main()
