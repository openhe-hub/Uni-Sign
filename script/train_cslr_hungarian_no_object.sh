#!/bin/bash

# CSLR Training Script - PURE Hungarian with NO OBJECT matching
# Dataset: CSL_Daily
# Task: Continuous Sign Language Recognition (CSLR)
# Features:
#   - 100% Hungarian (no CE)
#   - No label shuffle (preserve semantic order)
#   - No object matching enabled (penalize excess predictions)
#   - Optional dummy column for explicit PAD matches

output_dir=out/cslr_hungarian_1.0_no_object_v2
ckpt_path=out/stage1_pretraining_csl_daily/best_checkpoint.pth

hungarian_weight=1.0          # 0.0=only CE, 1.0=only Hungarian
no_object_weight=0.5          # Weight for penalizing unmatched predictions
allow_null_match_flag="--allow_null_match"
no_object_cost=2.0            # Cost for dummy column (lower = easier to match PAD)

mkdir -p $output_dir

echo "Training CSLR with PURE Hungarian + NO OBJECT matching..."
echo "Hungarian weight: $hungarian_weight"
echo "No object weight: $no_object_weight"
echo "Allow null match: $allow_null_match_flag (cost=${no_object_cost})"
echo "Shuffled labels: NO"
echo "Output: $output_dir"
echo ""
echo "No object matching will:"
echo "  - Find unmatched predictions (when L > T)"
echo "  - Penalize them to predict PAD token"
echo "  - Reduce insertion rate"
echo ""
echo "Labels will keep original order (no shuffle) to avoid breaking semantic alignment."

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
  --use_no_object \
  --no_object_weight $no_object_weight \
  $allow_null_match_flag \
  --no_object_cost $no_object_cost

echo ""
echo "Training completed!"
echo "Results saved to: $output_dir"
echo "Check log: cat $output_dir/log.txt"
