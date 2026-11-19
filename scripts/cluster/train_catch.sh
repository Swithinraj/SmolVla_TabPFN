#!/bin/bash
# Fine-tune SmolVLA on a catch dataset (one GPU). Usage:
#   bash scripts/cluster/train_catch.sh INIT_PRETRAINED_DIR DATASET_DIR OUTPUT_DIR STEPS [BATCH] [SAVE_FREQ]
# Example (the big run):  train_catch.sh checkpoints/4_catch_pilot_003000/pretrained_model data/so101_catch_v2 outputs/catch_big 12000 4 4000
set -e
source "$(dirname "$0")/env.sh"
INIT=$1; DATA=$2; OUT=$3; STEPS=${4:-12000}; BATCH=${5:-4}; SAVE=${6:-4000}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TFS='{brightness: {weight: 1, type: ColorJitter, kwargs: {brightness: [0.8, 1.2]}}, contrast: {weight: 1, type: ColorJitter, kwargs: {contrast: [0.8, 1.2]}}}'
RENAME='{observation.images.top: observation.images.camera1, observation.images.front: observation.images.camera2}'
lerobot-train \
  --policy.path="$INIT" --policy.device=cuda \
  --policy.input_features=null --policy.output_features=null \
  --policy.train_expert_only=true --policy.freeze_vision_encoder=true --policy.push_to_hub=false \
  --policy.optimizer_lr=0.0001 --policy.scheduler_warmup_steps=200 \
  --policy.scheduler_decay_steps=$STEPS --policy.scheduler_decay_lr=0.0000025 \
  --dataset.repo_id=local/catch --dataset.root="$DATA" --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true --dataset.image_transforms.backend=dataloader \
  --dataset.image_transforms.max_num_transforms=1 "--dataset.image_transforms.tfs=$TFS" \
  "--rename_map=$RENAME" --output_dir="$OUT" --steps=$STEPS --batch_size=$BATCH \
  --num_workers=4 --env_eval_freq=0 --log_freq=50 --save_freq=$SAVE
