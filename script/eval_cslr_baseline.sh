#!/bin/bash

# CSLR Evaluation Script - Baseline
# Evaluates the baseline model (CE loss only) on CSL_Daily test set

ckpt_path=out/cslr_baseline/best_checkpoint.pth
output_dir=out/cslr_baseline_eval

echo "Evaluating CSLR Baseline on test set..."
echo "Checkpoint: $ckpt_path"
echo "Output: $output_dir"
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
   --eval

echo ""
echo "Evaluation completed!"
echo "Results saved to: $output_dir"
echo "Check test results:"
echo "  cat $output_dir/log.txt"
