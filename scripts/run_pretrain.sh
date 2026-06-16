#!/usr/bin/env bash
set -euo pipefail

cd /ducor

# This stage reads already-downloaded LLaVA-Med alignment JSON/images.
# Set IMAGE_ROOT to your local downloaded image dir.
PRETRAIN_JSON=${PRETRAIN_JSON:-data/pmc/llava_med_alignment_500k_filter.json}
IMAGE_ROOT=${IMAGE_ROOT:-data/pmc/images}
OUT_DIR=${OUT_DIR:-checkpoint/pretrain}
GPUS=${GPUS:-0,1}
NUM_PROCESSES=${NUM_PROCESSES:-2}
MODEL_TYPE=${MODEL_TYPE:-stablelm}
PRETRAIN_PATH=${PRETRAIN_PATH:pre_checkpoints}
PRETRAIN_BATCH_SIZE=${PRETRAIN_BATCH_SIZE:-16}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-3}
PRETRAIN_GRAD_ACC=${PRETRAIN_GRAD_ACC:-1}
PRETRAIN_LR=${PRETRAIN_LR:-2e-3}
PRETRAIN_WD=${PRETRAIN_WD:-0.0}
PRETRAIN_WARMUP_RATIO=${PRETRAIN_WARMUP_RATIO:-0.03}
PRETRAIN_SCHEDULER=${PRETRAIN_SCHEDULER:-cosine}
PRETRAIN_WORKERS=${PRETRAIN_WORKERS:-4}

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPUS} \
accelerate launch \
  --config_file default_config.yaml \
  --main_process_port 29531 \
  --num_processes=${NUM_PROCESSES} \
  --num_machines=1 \
  --mixed_precision=fp16 \
  --deepspeed_config_file=zero2.json \
  main.py \
  --method=pretrain \
  --phase=train \
  --dataset=PMC15M \
  --pretrain_path=${PRETRAIN_PATH} \
  --pretrain_batch_size=${PRETRAIN_BATCH_SIZE} \
  --pretrain_epochs=${PRETRAIN_EPOCHS} \
  --pretrain_gradient_accumulation_steps=${PRETRAIN_GRAD_ACC} \
  --pretrain_lr=${PRETRAIN_LR} \
  --pretrain_weight_decay=${PRETRAIN_WD} \
  --pretrain_warmup_ratio=${PRETRAIN_WARMUP_RATIO} \
  --pretrain_lr_scheduler_type=${PRETRAIN_SCHEDULER} \
  --pretrain_dataloader_num_workers=${PRETRAIN_WORKERS} \
  --pad=right \
  --model_type=${MODEL_TYPE} \
  --llmlora=lora \
  --vislora=lora \
  --pretrain_data_path=${PRETRAIN_JSON} \
  --pretrain_image_root=${IMAGE_ROOT} \
  --out_dir=${OUT_DIR}
