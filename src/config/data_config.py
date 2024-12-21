"""
Data Config

Specifies the hyperparameters for the dataset, dataloader, and tokenizer.
"""

from dataclasses import dataclass, field

from ._constants import BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, MAX_SEQ_LEN, VOCAB_SIZE


@dataclass
class DatasetConfig:
    """Config dataclass for the Dataset."""

    name: str = "pico-lm/pretokenized-dolma"


@dataclass
class DataLoaderConfig:
    """Config dataclass for the Data Loader."""

    # NOTE: You should only change these values jointly with the training config; so that the
    # sub-batch size is consistent with the gradient accumulation steps
    full_batch_size: int = BATCH_SIZE
    sub_batch_size: int = BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS
    max_seq_len: int = MAX_SEQ_LEN


@dataclass
class TokenizerConfig:
    """Config dataclass for the Tokenizer."""

    name: str = "allenai/OLMo-7B-0724-hf"
    vocab_size: int = VOCAB_SIZE


@dataclass
class DataConfig:
    """Config dataclass for the Data."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
