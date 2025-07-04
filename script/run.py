import sys
import logging
import os
from pathlib import Path
import torch

from transformers import (
    AutoConfig, 
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from peft import LoraConfig, get_peft_model

from src.arguments import (
    ModelArguments, 
    DataArguments, 
    TrainingArguments
)
from src.data import (
    Dataset, 
    Collator
)
from src.model import (
    GPT2ClassifierConfig
    GPT2Classifier
)
from transformers import Trainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    ################################
    # 初始化分词器
    ################################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################################
    # 初始化模型
    ################################
    model_config = GPT2ClassifierConfig.from_pretrained(
        model_args.config_name_or_path,
        n_classes=model_args.n_classes # 新参数需要手动输入，下一步相关新权重会自动初始化
    )
    logger.info('Model config: %s', model_config)

    if torch.cuda.get_device_name(0) == 'Tesla V100-SXM2-32GB':
        torch_dtype = torch.float16
        attn_implementation = None
    else:
        torch_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
        
    model = GPT2Classifier.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        config=model_config
    )
    if training_args.lora_tune:
        raise NotImplementedError("需要仔细看看target_modules")
        lora_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=2 * training_args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=training_args.lora_dropout,
            bias="none",
            modules_to_save=["class_head"]
        )
        logger.info('Lora config: %s', lora_config)
        model = get_peft_model(model, lora_config)
        # model.enable_input_require_grads()
        model.print_trainable_parameters()

    ################################
    # 初始化数据
    ################################
    train_dataset = Dataset(
        data_file=data_args.data_file,
        data_dir=data_args.data_dir,
        reference_file=data_args.reference_file
    )

    ################################
    # 初始化训练
    ################################
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=Collator(
            tokenizer=tokenizer,
            max_length=data_args.max_length
        ),
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
