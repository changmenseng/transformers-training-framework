#!/bin/bash
source /etc/ssd1/jiangzhongtao/anaconda3/bin/activate
echo $(which torchrun)

set -e

gpus_per_node=8
master_addr=10.82.137.14 # /etc/mpi/hostfile 第一个ip地址
master_port=12889
nnodes=4
node_rank=0
world_size=$(($gpus_per_node*$nnodes))
save_root="" # 目标根目录
output_name="" # 保存文件夹名
model_name_or_path="" # 模型hf
train_data_file=""
description=""

distributed_args="
    --nproc_per_node $gpus_per_node \
    --nnodes $nnodes \
    --node_rank $node_rank \
    --master_addr $master_addr \
    --master_port $master_port
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

if test $node_rank = 0; then
echo "This is rank 0, copying $0"
cp $0 $save_root/$output_name/.
cp $reference_file $save_root/$output_name/reference_file.txt
fi

# export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

# python /etc/ssd1/jiangzhongtao/baai_embedding_tune/script/eval.py \
#     --model_path $save_root/$output_name/sentence_transformer \
#     --save_path $save_root/$output_name/mteb_results \
#     --mp_size 8 \
#     --split_id ${node_rank} \
#     --n_splits ${nnodes} \
#     --dtype bfloat16

# set +x
