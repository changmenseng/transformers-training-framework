from dataclasses import dataclass, field
from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        metadata={"help": ""}
    )
    n_classes: int = field(
        default=256, metadata={"help": ""}
    )

@dataclass
class DataArguments:

    data_file: str = field(
        default=None, metadata={"help": ""}
    )
    data_dir: str = field(
        default=None, metadata={"help": ""}
    )

    reference_file: str = field(
        default=None, metadata={"help": ""}
    )

    max_length: int = field(
        default=512,
        metadata={
            "help": ""
        },
    )

@dataclass
class TrainingArguments(HFTrainingArguments):
    lora_tune: bool = field(
        default=True, metadata={"help": "Whether to use lora"}
    )
    lora_rank: int = field(
        default=32, metadata={"help": "Lora rank, only valid when `lora_tune=True`"}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "Lora dropout, only valid when `lora_tune=True`"}
    )
