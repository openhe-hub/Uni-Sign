#!/bin/bash

# CSLR Training Script - Pure Hungarian Loss (100%) with SHUFFLED gloss order
# Dataset: CSL_Daily (only dataset with gloss annotations)
# Task: Continuous Sign Language Recognition (CSLR)
# Purpose: Test Hungarian loss order-invariance with 100% Hungarian weight

output_dir=out/cslr_hungarian_1.0_shuffled
ckpt_path=out/stage1_pretraining_csl_daily/best_checkpoint.pth

# Hungarian weight: 1.0 means PURE Hungarian loss (no CE component)
# Total loss = 0.0 * CrossEntropy + 1.0 * Hungarian
# This is the correct setup for shuffled data!
hungarian_weight=1.0

# Create output directory if not exists
mkdir -p $output_dir

echo "Training CSLR with PURE Hungarian Loss (100%) on SHUFFLED labels..."
echo "Hungarian weight: $hungarian_weight (PURE Hungarian, no CE)"
echo "Output: $output_dir"
echo ""
echo "This is the theoretically correct setup for shuffled data:"
echo "  - CE loss would give wrong training signal (position mismatch)"
echo "  - Hungarian loss finds optimal matching regardless of order"
echo ""

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
  --hungarian_weight $hungarian_weight \
  --shuffle_labels

echo ""
echo "Training completed with PURE Hungarian loss (weight=$hungarian_weight) and shuffled labels."
echo "Results saved to: $output_dir"
echo "Check log: cat $output_dir/log.txt"
