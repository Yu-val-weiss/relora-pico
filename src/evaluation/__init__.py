"""
Pico Evaluation Package

This package implements the evaluation pipeline for the Pico language model. It provides
functionality to evaluate model performance using various metrics and handles the complete
evaluation workflow.

We recommend that each evaluation metric should have its own config, and should be
implemented as a module in the `evaluation/tasks` directory that exposes a `run_<metric_name>` function.

NOTE: Out of the box we only support Paloma, but the structure is designed to be flexible and
you are meant to add whatever metrics you want. One of the main reasons we store out
the model in the HuggingFace format is so that its easy to use third-party evaluation
libraries/frameworks.
"""

from __future__ import annotations

import os

import torch
from lightning.fabric import Fabric
from torch import nn

from src.config import CheckpointingConfig, EvaluationConfig

from .tasks.blimp import run_blimp_evaluation
from .tasks.paloma import run_paloma_evaluation


def run_evaluation(
    evaluation_config: EvaluationConfig,
    checkpointing_config: CheckpointingConfig,
    fabric: Fabric,
    model: nn.Module,
) -> None:
    """Run model evaluation using specified metrics in `evaluation_config`.

    This function orchestrates the complete evaluation pipeline by:
    1. Resolving the model checkpoint path (either specified or latest) to load the model from;
        during training, this is the path to the latest checkpoint in the run directory.
    2. Iterating over each evaluation metric, and running the corresponding evaluation function.
        NOTE: we suggest you follow the pattern of the Paloma evaluation function, and implement
        your own evaluation function for each metric in the `evaluation/tasks` directory.
    3. Aggregating results across all metrics in a dictionary, and returning it.

    Args:
        evaluation_config (EvaluationConfig): Configuration object containing:
            - metrics (List[str]): Metrics to evaluate; each metric should have its
                own config. Currently supported: ["paloma"];
            - paloma (PalomaConfig): Configuration for Paloma evaluation
                - max_length (int): Maximum sequence length
                - limit_eval_examples (Optional[int]): Number of examples to evaluate
        checkpointing_config (CheckpointingConfig): Configuration object containing:
        fabric (Fabric): Lightning Fabric instance
        model (nn.Module): Original model instance

    Returns:
        Dict[str, float]: Dictionary mapping metric names to their values
            Example: {"paloma": 3.45}

    Raises:
        ValueError: If an unsupported evaluation metric is requested

    Example:
        results = run_evaluation(
            EvaluationConfig(
                run_name="experiment_1",
                metrics=["paloma"],
                paloma=PalomaConfig(max_length=2048, batch_size=16)
            )
        )

    """

    fabric.barrier()

    model.to("cpu")  # Offloading model to CPU

    evaluation_results = {}

    # NOTE: Evaluation is only run on first processes to enable third-party evaluation libraries
    # to determine how to handle distributed evaluation.
    if fabric.global_rank == 0:
        run_name = checkpointing_config.run_name
        model_path = (
            f"{os.getcwd()}/{checkpointing_config.runs_dir}/"
            f"{run_name}/{checkpointing_config.checkpoints_dir}/latest"
        )
        os.makedirs(model_path, exist_ok=True)

        for metric in evaluation_config.metrics:
            # NOTE: add your own metrics here
            if metric == "paloma":
                eval_result = run_paloma_evaluation(model_path, evaluation_config.paloma)
            elif metric == "blimp":
                eval_result = run_blimp_evaluation(model_path, evaluation_config.blimp)
            else:
                raise ValueError(f"Metric {metric} not supported")

            evaluation_results[metric] = eval_result

    torch.cuda.empty_cache()

    fabric.barrier()

    model.to(fabric.device)

    return evaluation_results
