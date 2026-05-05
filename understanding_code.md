# Code Understanding Guide

This guide explains the baseline and improved DAViT code so someone new can follow how data flows, how the model is built, and what each module does.

## Baseline

### File: DAViT/davit/model_davit.py

Purpose: Define the baseline model architecture and its forward pass.

Modules and components:
- CNN
  - conv1, bn1: 1x1 projection to expand channels from 1024 to 2048.
  - conv2, bn2: 3x3 depthwise grouped conv (groups=32) to mix local spatial features.
  - conv3, bn3: 1x1 projection to refine features.
  - relu: non-linearity after the block.
  - forward: applies the three conv blocks and ReLU in sequence.

- Model
  - vit: pretrained DINOv2 backbone that outputs token embeddings.
  - cnn: the shallow CNN block that processes the spatial feature map from ViT tokens.
  - avgpool: global average pooling to collapse spatial dimensions.
  - fc: linear layer that maps 2048 channels to 1000 features.
  - classifier: final linear layer to 2 classes (NORMAL vs PNEUMONIA).
  - forward:
    - runs ViT to get token embeddings.
    - removes CLS token and reshapes tokens into a 2D feature map.
    - passes through CNN, pools, flattens, and runs through fc and classifier.
    - uses CrossEntropyLoss when labels are provided, otherwise returns softmax probabilities.

### File: DAViT/davit/main.py

Purpose: Training, evaluation, and testing pipeline for the baseline model.

Modules and components:
- TextDataset
  - Loads images from NORMAL and PNEUMONIA folders.
  - Applies train/val/test transforms.
  - Builds labels using filename checks (bacteria/virus) for detection.

- convert_examples_to_features
  - Loads image and applies the transform pipeline.

- train
  - Builds a dataloader and optimizer (AdamW) with weight decay groups.
  - Runs epochs with gradient accumulation and warmup schedule.
  - Saves the best checkpoint by evaluation loss.

- evaluate
  - Computes average loss on the evaluation split.

- test
  - Runs inference and prints metrics (accuracy, F1, precision, recall, specificity, AUC).

- main
  - Configures device and seeds.
  - Instantiates processor, ViT backbone, and Model.
  - Loads a pretrained state dict from saved_models/checkpoint-best-f1/domain_adapted_davit.bin.
  - Runs training and testing based on flags.

## Improvements

### File: DAViT/davit/improved_model.py

Purpose: Define the improved model with Focal Loss and dropout.

Modules and components:
- FocalLoss
  - A replacement for standard CrossEntropyLoss that down-weights easy examples.
  - Uses gamma to focus learning on harder cases.
  - Optional alpha for class weighting.

- CNN
  - Same structure as baseline CNN.

- Model
  - Same ViT backbone and CNN pipeline.
  - Adds dropout with p=0.5 before the final classifier to reduce overfitting.
  - Uses FocalLoss instead of standard CrossEntropyLoss.

### File: DAViT/davit/improvement.py

Purpose: Training, evaluation, and testing pipeline for the improved model with domain fine-tuning.

Modules and components:
- TextDataset
  - Adds ColorJitter (brightness and contrast) to training transforms for domain robustness.
  - Uses folder name (NORMAL vs PNEUMONIA) to set labels.

- freeze_backbone
  - Freezes ViT parameters to avoid updating the large backbone.
  - Trains only the shallow CNN, fc, and classifier layers.

- train
  - Same logic as baseline, but optimizer only includes trainable parameters.

- evaluate and test
  - Same logic as baseline for loss and metrics.

- main
  - Loads the improved model and pretrained weights.
  - Applies the backbone freezing.
  - Runs training/testing based on flags.

## Improvements Summary

- Frozen ViT backbone
  - The ViT has ~300M parameters and is expensive to train on limited hardware.
  - Freezing it focuses training on the shallow CNN and classifier only.

- Focal Loss instead of standard CrossEntropyLoss
  - The improved model uses FocalLoss to emphasize hard-to-classify images.
  - This helps with subtle domain shift cases and improves robustness.

- Dropout before the classifier
  - Dropout with p=0.5 reduces overfitting on the small COVID subset.

- Hospital-agnostic augmentation
  - ColorJitter in training reduces sensitivity to contrast and exposure differences.
