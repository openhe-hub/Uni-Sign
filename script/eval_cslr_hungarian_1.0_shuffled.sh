#!/bin/bash

# CSLR Evaluation Script - Pure Hungarian Loss (100%) with SHUFFLED gloss order
# Evaluates the pure Hungarian shuffled model on test set

ckpt_path=out/cslr_hungarian_1.0_shuffled/best_checkpoint.pth
output_dir=out/cslr_hungarian_1.0_shuffled_eval
hungarian_weight=1.0

echo "Evaluating CSLR Pure Hungarian (100%, shuffled) on test set..."
echo "Checkpoint: $ckpt_path"
echo "Hungarian weight: $hungarian_weight (PURE Hungarian)"
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
   --use_hungarian \
   --hungarian_weight $hungarian_weight \
   --shuffle_labels \
   --eval

echo ""
echo "Evaluation completed!"
echo "Results saved to: $output_dir"
echo "Check test results:"
echo "  cat $output_dir/log.txt"
