"""
Evaluation Config

Specifies the hyperparameters for the evaluation process, i.e. what metrics to compute, etc.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from src.config._constants import MAX_SEQ_LEN


@dataclass
class PalomaEvaluationConfig:
    """Config dataclass for Paloma Evaluation."""

    max_length: int = MAX_SEQ_LEN
    batch_size: int = 16


@dataclass
class EvaluationConfig:
    """Config dataclass for Evaluation."""

    # Evaluation metrics to compute: by default, we compute the perplexity of the model
    metrics: Optional[List[str]] = field(default_factory=lambda: ["paloma"])

    # NOTE: Add other evaluation configs here
    # Each evaluation metric should have its own config
    paloma: PalomaEvaluationConfig = field(default_factory=PalomaEvaluationConfig)
