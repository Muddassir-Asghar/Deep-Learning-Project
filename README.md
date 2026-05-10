# Cross-Domain DAViT Pneumonia Detection

This project evaluates and improves a DAViT-based chest X-ray pneumonia detection pipeline. The baseline follows the paper "DAViT: A Domain-Adapted Vision Transformer for Automated Pneumonia Detection and Explanation Using Chest X-Ray Images" and the improved version adds stronger augmentation, focal loss, dropout regularization, and selective backbone freezing.

## Required Files

- `train.py`: training entry point.
- `inference.py`: single-image inference entry point.
- `config.yaml`: default model, path, and training settings.
- `notebooks/01_inference_demo.ipynb`: minimal inference notebook.
- `results/baseline_metrics.json`: baseline DAViT metrics.
- `results/improved_metrics.json`: improved model metrics.
- `results/training_log.csv`: CSV summary of logged runs.

## Baseline Paper

DAViT: A Domain-Adapted Vision Transformer for Automated Pneumonia Detection and Explanation Using Chest X-Ray Images  
https://doi.org/10.1109/ACCESS.2025.3579314
