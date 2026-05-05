# DL Project Reproducibility Guide

This README explains how to reproduce the baseline and improved results, generate the dataset splits, and run the Grad-CAM demo.

## 1) Environment

- Python 3.10+ recommended
- GPU optional but recommended

Install core dependencies:

```bash
pip install torch torchvision transformers scikit-learn tqdm pillow matplotlib opencv-python
```

or

```bash
pip install -m requirements.txt
```

## 2) Prepare Data

You will build a combined train/val/test split using the metadata file and place the output under data/improvement_data.

From data/data2:

```bash
cd /data/data2
python build_improvement_dataset.py \
  --metadata Chest_xray_Corona_Metadata.csv \
  --images_root Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset \
  --output_dir ../improvement_data
```

Expected folder structure:

```
data/improvement_data/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

## 3) Baseline Training and Testing

Use the baseline training script in [DAViT/davit/train_detection.sh](DAViT/davit/train_detection.sh). Update its paths if needed and run:

```bash
cd /DAViT/davit
bash train_detection.sh
```

## 4) Improved Model Training and Testing

The improved pipeline uses frozen ViT, Focal Loss, dropout, and ColorJitter augmentation.

Run the improved training and testing script:

```bash
cd /DAViT/davit
bash run_improvement.sh
```

## 5) Grad-CAM Demo

Open the demo notebook:

- [demo_improvement_gradcam.ipynb](demo_improvement_gradcam.ipynb)

The notebook loads baseline and improved weights, runs inference on a few samples, and visualizes Grad-CAM overlays.

## 6) Key Files

- Baseline model: [DAViT/davit/model_davit.py](DAViT/davit/model_davit.py)
- Baseline training pipeline: [DAViT/davit/main.py](DAViT/davit/main.py)
- Improved model: [DAViT/davit/improved_model.py](DAViT/davit/improved_model.py)
- Improved training pipeline: [DAViT/davit/improvement.py](DAViT/davit/improvement.py)
- Dataset builder: [data/data2/build_improvement_dataset.py](data/data2/build_improvement_dataset.py)
- Grad-CAM demo: [demo_improvement_gradcam.ipynb](demo_improvement_gradcam.ipynb)

## 7) Notes

- If you use your own checkpoints, update the paths in the scripts and the notebook.
- If you want different split ratios, pass --train_ratio and --val_ratio to the dataset builder.
