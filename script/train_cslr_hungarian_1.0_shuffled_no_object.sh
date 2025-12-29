#!/bin/bash

# CSLR Training Script - Pure Hungarian Loss with NO OBJECT matching
# Dataset: CSL_Daily
# Task: Continuous Sign Language Recognition (CSLR)
# Features:
#   - 100% Hungarian loss (no CE)
#   - Shuffled labels (test order-invariance)
#   - No object matching enabled (penalize excess predictions)

output_dir=out/cslr_hungarian_1.0_shuffled_no_object
ckpt_path=out/stage1_pretraining_csl_daily/best_checkpoint.pth

hungarian_weight=1.0
no_object_weight=0.1  # Weight for penalizing unmatched predictions

mkdir -p $output_dir

echo "Training CSLR with PURE Hungarian + NO OBJECT matching..."
echo "Hungarian weight: $hungarian_weight"
echo "No object weight: $no_object_weight"
echo "Shuffled labels: YES"
echo "Output: $output_dir"
echo ""
echo "No object matching will:"
echo "  - Find unmatched predictions (when L > T)"
echo "  - Penalize them to predict PAD token"
echo "  - Reduce insertion rate"
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
  --shuffle_labels \
  --use_no_object \
  --no_object_weight $no_object_weight

echo ""
echo "Training completed!"
echo "Results saved to: $output_dir"
echo "Check log: cat $output_dir/log.txt"
echo ""
echo "Compare with baseline (no no_object):"
echo "  cat out/cslr_hungarian_1.0_shuffled/log.txt"
