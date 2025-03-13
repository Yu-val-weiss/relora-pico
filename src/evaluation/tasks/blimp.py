"""
BLiMP (Benchmark of Linguistic Minimal Pairs) is a challenge set for evaluating
what language models know about major grammatical phenomena in English.

To evaluate on BLiMP, we use the Hugging Face evaluation framework.

For more details, see: https://github.com/alexwarstadt/blimp
"""

import evaluate

from src.config.evaluation_config import BlimpEvaluationConfig


def run_blimp_evaluation(model_path: str, blimp_config: BlimpEvaluationConfig):
    """Run Blimp evaluation on the Blimp evaluation dataset.

    We use the HuggingFace evaluate library to load in and compute the Blimp metric.

    Args:
        model_path (str): Path to the model checkpoint to be evaluated
        blimp_config (BlimpEvaluationConfig): Configuration for BLiMP evaluation
    """

    blimp = evaluate.load("pico-lm/blimp")

    blimp_result = blimp.compute(
        model_id=model_path,
        predictions=blimp_config.metric_uids,
        batch_size=blimp_config.batch_size,
        samples_per_set=blimp_config.samples_per_set,
        trust_remote_code=True,
    )

    return blimp_result
