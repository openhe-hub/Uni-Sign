#!/bin/bash

# CSLR Training Script with Hungarian Loss (Heavy weight)
# Hungarian weight: 0.7 means more emphasis on order-invariant matching
# Total loss = 0.3 * CrossEntropy + 0.7 * Hungarian

output_dir=out/cslr_hungarian_0.7
ckpt_path=out/stage1_pretraining/best_checkpoint.pth
hungarian_weight=0.7

mkdir -p $output_dir

deepspeed --include localhost:0--master_port 29511 fine_tuning.py \
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
