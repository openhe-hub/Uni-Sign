#!/bin/bash

# CSLR Evaluation Script - Hungarian Loss (weight=0.5)
# Evaluates the Hungarian loss model on CSL_Daily test set

ckpt_path=out/cslr_hungarian_0.5/best_checkpoint.pth
output_dir=out/cslr_hungarian_0.5_eval
hungarian_weight=0.5

echo "Evaluating CSLR Hungarian 0.5 on test set..."
echo "Checkpoint: $ckpt_path"
echo "Hungarian weight: $hungarian_weight"
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
   --eval

echo ""
echo "Evaluation completed!"
echo "Results saved to: $output_dir"
echo "Check test results:"
echo "  cat $output_dir/log.txt"
