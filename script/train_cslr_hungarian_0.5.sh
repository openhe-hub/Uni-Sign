#!/bin/bash

# CSLR Training Script with Hungarian Loss
# Dataset: CSL_Daily (only dataset with gloss annotations)
# Task: Continuous Sign Language Recognition (CSLR)
# Hungarian loss helps handle order-invariant gloss matching

output_dir=out/cslr_hungarian_0.5
ckpt_path=out/stage1_pretraining/best_checkpoint.pth

# Hungarian weight: 0.5 means equal weight for CE and Hungarian loss
# Total loss = 0.5 * CrossEntropy + 0.5 * Hungarian
hungarian_weight=0.5

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
  --label_smoothing 0.2 \
  --use_hungarian \
  --hungarian_weight $hungarian_weight

echo "Training completed with Hungarian loss (weight=$hungarian_weight)."
echo "Results saved to: $output_dir"
