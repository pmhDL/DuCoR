#!/usr/bin/env bash
set -euo pipefail

cd /ducor

# This stage reads already-downloaded LLaVA-Med instruction JSON/images.
# Set IMAGE_ROOT to your local downloaded image dir.
SFT_JSON=${SFT_JSON:-data/pmc/llava_med_instruct_60k_inline_mention_filter.json}
IMAGE_ROOT=${IMAGE_ROOT:-data/pmc/images}
OUT_DIR=${OUT_DIR:-checkpoint/sft}
GPUS=${GPUS:-0,1}
NUM_PROCESSES=${NUM_PROCESSES:-2}
MODEL_TYPE=${MODEL_TYPE:-stablelm}
SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-16}
SFT_EPOCHS=${SFT_EPOCHS:-5}
SFT_GRAD_ACC=${SFT_GRAD_ACC:-2}
SFT_LR=${SFT_LR:-2e-5}
SFT_WD=${SFT_WD:-0.0}
SFT_WARMUP_RATIO=${SFT_WARMUP_RATIO:-0.03}
SFT_SCHEDULER=${SFT_SCHEDULER:-cosine}
SFT_WORKERS=${SFT_WORKERS:-4}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-}

if [[ -z "${INIT_CHECKPOINT}" ]]; then
  echo "Set INIT_CHECKPOINT to the pretrain checkpoint, e.g. checkpoint/pretrain/.../checkpoint_pretrain.pt" >&2
  exit 1
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPUS} \
accelerate launch \
  --config_file default_config.yaml \
  --main_process_port 29532 \
  --num_processes=${NUM_PROCESSES} \
  --num_machines=1 \
  --mixed_precision=fp16 \
  --deepspeed_config_file=zero2.json \
  main.py \
  --method=sft \
  --phase=train \
  --dataset=LLaVA_Med_Instruct \
  --sft_batch_size=${SFT_BATCH_SIZE} \
  --sft_epochs=${SFT_EPOCHS} \
  --sft_gradient_accumulation_steps=${SFT_GRAD_ACC} \
  --sft_lr=${SFT_LR} \
  --sft_weight_decay=${SFT_WD} \
  --sft_warmup_ratio=${SFT_WARMUP_RATIO} \
  --sft_lr_scheduler_type=${SFT_SCHEDULER} \
  --sft_dataloader_num_workers=${SFT_WORKERS} \
  --pad=right \
  --model_type=${MODEL_TYPE} \
  --llmlora=lora \
  --vislora=lora \
  --init_checkpoint=${INIT_CHECKPOINT} \
  --sft_data_path=${SFT_JSON} \
  --sft_image_root=${IMAGE_ROOT} \
  --out_dir=${OUT_DIR}
