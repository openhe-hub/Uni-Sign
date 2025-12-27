#!/bin/bash

# CSLR Training Script - Baseline (without Hungarian loss)
# Dataset: CSL_Daily (only dataset with gloss annotations)
# Task: Continuous Sign Language Recognition (CSLR)

output_dir=out/cslr_baseline
ckpt_path=out/stage1_pretraining/best_checkpoint.pth

# Create output directory if not exists
mkdir -p $output_dir

deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset CSL_Daily \
  --task CSLR \
  --label_smoothing 0.2

echo "Training completed. Results saved to: $output_dir"
