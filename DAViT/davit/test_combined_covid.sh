#!/usr/bin/env bash
set -e

python main.py \
  --model_name_or_path=facebook/dinov2-large \
  --model_name=davit_pneumonia_detection.bin \
  --output_dir=./saved_models \
  --do_test \
  --test_data_file=../../data/data2/combined_data \
  --eval_batch_size 8 \
  --seed 123456 2>&1 | tee davit_combined_covid_test.log
