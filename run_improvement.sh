#!/usr/bin/env bash
set -e

python improvement.py \
  --model_name_or_path=facebook/dinov2-large \
  --model_name=davit_pneumonia_detection.bin \
  --output_dir=./saved_models \
  --do_train \
  --do_test \
  --train_data_file=../../data/improvement_data/train \
  --eval_data_file=../../data/improvement_data/val \
  --test_data_file=../../data/improvement_data/test \
  --epochs 30 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --learning_rate 1e-4 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 123456 \
  --focal_gamma 2.0 \
  --focal_alpha -1.0 2>&1 | tee davit_improvement_train.log
