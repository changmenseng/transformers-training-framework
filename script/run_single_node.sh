#!/bin/bash
source /etc/ssd1/jiangzhongtao/anaconda3/bin/activate
echo $(which torchrun)

set -e

save_root="" # 目标根目录
output_name="" # 保存文件夹名
model_name_or_path="" # 模型hf
reference_file=""
description=""


distributed_args="
    --nproc_per_node $gpus_per_node
"

model_args="
    --model_name_or_path $model_name_or_path \
    --n_classes 3 \
"

data_args="
    --reference_file $reference_file \
    --max_length 512 \
"

train_args="
    --deepspeed /etc/ssd1/jiangzhongtao/code_framework/train/transformers/script/config_zero1.json \
    --lora_tune True \
    --lora_rank 64 \
    --lora_dropout 0.1 \
    --no_remove_unused_columns \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --bf16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --save_steps 3673 \
    --gradient_checkpointing True \
    --dataloader_drop_last False \
    --report_to tensorboard \
    --output_dir $save_root/${output_name} \
"

torchrun \
    $distributed_args \
    /etc/ssd1/jiangzhongtao/code_framework/train/transformers/script/run.py \
    $model_args \
    $data_args \
    $train_args

cp $0 $save_root/$output_name/.
cp $reference_file $save_root/$output_name/reference_files.txt
