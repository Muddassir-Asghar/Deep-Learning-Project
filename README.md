# Cross-Domain DAViT Pneumonia Detection

This repository contains a cleaned deep learning submission for pneumonia detection from chest X-ray images using a DAViT-style model. The baseline is based on the paper **DAViT: A Domain-Adapted Vision Transformer for Automated Pneumonia Detection and Explanation Using Chest X-Ray Images**. The improved version keeps the same core idea but adds stronger augmentation, focal loss, dropout regularization, and selective fine-tuning to improve robustness.

Baseline paper: https://doi.org/10.1109/ACCESS.2025.3579314

## Project Summary

The project compares a baseline DAViT pneumonia detector with an improved training setup. DAViT combines a DINOv2 Vision Transformer backbone for global chest X-ray representation learning with a shallow CNN head for local feature refinement. The final classifier predicts whether an X-ray belongs to the `NORMAL` or `PNEUMONIA` class.

The improvement focuses on making the classifier less brittle under dataset shift by using focal loss, dropout, corrected dataset labeling, and controlled backbone freezing.

## Results

| Model | Accuracy | F1 | Precision | Recall | Specificity | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline DAViT | 96.15% | 96.95% | 95.98% | 97.95% | 93.16% | 95.56% |
| Improved DAViT | 96.49% | 97.55% | 99.04% | 96.10% | 97.53% | 96.82% |

## Repository Structure

```text
Deep-Learning-Project/
|-- README.md
|-- requirements.txt
|-- train.py
|-- inference.py
|-- config.yaml
|-- data/
|   `-- sample_data.csv
|-- notebooks/
|   `-- 01_inference_demo.ipynb
|-- src/
|   |-- model.py
|   |-- dataset.py
|   `-- utils.py
|-- results/
|   |-- baseline_metrics.json
|   |-- improved_metrics.json
|   `-- training_log.csv
`-- checkpoints/
    `-- README.md
```

## Complete Project Workflow

```mermaid
flowchart TD
    A["Chest X-ray images"] --> B["Preprocessing"]
    B --> B1["Resize and center crop"]
    B --> B2["RGB conversion"]
    B --> B3["Tensor conversion"]
    B --> B4["ImageNet normalization"]

    B1 --> C["DAViT baseline model"]
    B2 --> C
    B3 --> C
    B4 --> C

    C --> D["Baseline evaluation"]
    D --> E["Baseline metrics JSON"]

    B --> F["Improved training pipeline"]
    F --> G["Color jitter augmentation"]
    F --> H["Focal loss"]
    F --> I["Dropout regularization"]
    F --> J["Selective backbone freezing"]

    G --> K["Improved DAViT model"]
    H --> K
    I --> K
    J --> K

    K --> L["Improved evaluation"]
    L --> M["Improved metrics JSON"]
    L --> N["Training log CSV"]

    E --> O["Comparison and report"]
    M --> O
    N --> O
```

## Model Architecture

```mermaid
flowchart LR
    A["Input CXR image 224 x 224 x 3"] --> B["DINOv2-Large image processor"]
    B --> C["Patch embeddings"]
    C --> D["DINOv2 Vision Transformer backbone"]
    D --> E["Last hidden state"]
    E --> F["Remove CLS token"]
    F --> G["Reshape tokens to spatial feature map"]
    G --> H["CNN refinement block"]
    H --> H1["1 x 1 convolution: 1024 to 2048"]
    H1 --> H2["BatchNorm"]
    H2 --> H3["3 x 3 grouped convolution"]
    H3 --> H4["BatchNorm"]
    H4 --> H5["1 x 1 convolution"]
    H5 --> H6["ReLU"]
    H6 --> I["Adaptive average pooling"]
    I --> J["Flatten"]
    J --> K["Fully connected layer"]
    K --> L["Dropout"]
    L --> M["Binary classifier"]
    M --> N["Softmax probabilities"]
    N --> O["NORMAL or PNEUMONIA"]
```

## Training and Evaluation Pipeline

```mermaid
sequenceDiagram
    participant User
    participant Train as train.py
    participant Data as data/sample_data.csv
    participant Model as src/model.py
    participant Results as results/
    participant Checkpoints as checkpoints/

    User->>Train: Run training entry point
    Train->>Data: Read dataset paths and labels
    Train->>Model: Build DAViT-style model
    Model-->>Train: Forward pass and loss
    Train->>Train: Optimize with configured settings
    Train->>Checkpoints: Save trained weights when available
    Train->>Results: Write training_log.csv
    Train->>Results: Write baseline and improved metrics
```

## Inference Pipeline

```mermaid
flowchart TD
    A["User provides image path"] --> B["inference.py"]
    B --> C["Load config defaults"]
    C --> D["Load DINOv2 processor and backbone"]
    D --> E["Load checkpoint from checkpoints/"]
    E --> F["Preprocess image"]
    F --> G["Run forward pass"]
    G --> H["Compute softmax probabilities"]
    H --> I{"Predicted class"}
    I -->|class 0| J["NORMAL"]
    I -->|class 1| K["PNEUMONIA"]
```

## File Responsibilities

```mermaid
flowchart TB
    A["README.md"] --> A1["Project explanation, metrics, diagrams"]
    B["requirements.txt"] --> B1["Python dependencies"]
    C["config.yaml"] --> C1["Model, data, and training settings"]
    D["train.py"] --> D1["Training entry point"]
    E["inference.py"] --> E1["Single-image inference entry point"]
    F["src/model.py"] --> F1["DAViT model, CNN head, focal loss"]
    G["src/dataset.py"] --> G1["CSV dataset and preprocessing"]
    H["src/utils.py"] --> H1["Metric helper functions"]
    I["results/"] --> I1["Baseline, improved, and training metrics"]
    J["notebooks/"] --> J1["Inference demonstration notebook"]
    K["checkpoints/"] --> K1["Expected location for model weights"]
```

## Improvements Over Baseline

```mermaid
flowchart LR
    A["Baseline DAViT"] --> B["Observed issues"]
    B --> B1["Possible texture overfitting"]
    B --> B2["Domain shift sensitivity"]
    B --> B3["Class imbalance effects"]

    B1 --> C["Improved method"]
    B2 --> C
    B3 --> C

    C --> C1["Color jitter augmentation"]
    C --> C2["Focal loss"]
    C --> C3["Dropout before classifier"]
    C --> C4["Freeze DINOv2 backbone"]

    C1 --> D["Improved generalization"]
    C2 --> D
    C3 --> D
    C4 --> D

    D --> E["Higher F1 and AUC"]
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run the training entry point:

```bash
python train.py --config config.yaml
```

The configured checkpoint location is:

```text
checkpoints/davit_pneumonia_detection.bin
```

## Inference

Run single-image inference after placing model weights in `checkpoints/`:

```bash
python inference.py --image path/to/image.png --checkpoint checkpoints/davit_pneumonia_detection.bin
```

The script prints the predicted class and probabilities for `NORMAL` and `PNEUMONIA`.

## Notebook

The required inference notebook is located at:

```text
notebooks/01_inference_demo.ipynb
```

## Outputs

The required result files are:

```text
results/baseline_metrics.json
results/improved_metrics.json
results/training_log.csv
```

These files summarize the baseline and improved DAViT performance used in the project comparison.
