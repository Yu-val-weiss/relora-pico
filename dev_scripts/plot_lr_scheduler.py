"""Script to plot a learning rate scheduler"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import SGD

from src.config.training_config import OptimizationConfig, TrainingConfig
from src.training.utils.initialization import initialize_lr_scheduler

MAX_STEPS = 30_000
LR_WARMUP_STEPS = 2_000
RESTART_WARMUP_STEPS = 100
RELORA_RESET_FREQ = 2_000

rc_fonts = {
    "text.usetex": True,
    "text.latex.preamble": "\n".join([r"\usepackage{libertine}"]),
}
matplotlib.rcParams.update(rc_fonts)


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
        # strategy="auto",
        max_steps=MAX_STEPS,
    )

    training_config.relora_reset_freq = RELORA_RESET_FREQ

    scheduler = initialize_lr_scheduler(training_config, optimizer)

    fig, ax = plt.subplots(figsize=(14, 8))

    lrs = []
    for _ in range(MAX_STEPS):
        optimizer.step()
        lrs.append(scheduler.get_last_lr())
        scheduler.step()

    ax.plot(lrs)
    ax.set_xlabel("Step (1000s)", size=14)
    ax.set_ylabel("Learning rate multiplier", size=14)
    ax.grid(True)
    ticks = np.arange(0, MAX_STEPS + 1, step=RELORA_RESET_FREQ)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks // 1000)

    fig.tight_layout()

    fig.savefig("graphs/lr_scheduler.pdf")
    plt.show()


if __name__ == "__main__":
    main()
