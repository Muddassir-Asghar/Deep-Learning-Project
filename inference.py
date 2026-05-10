import argparse
import os

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

from improved_model import Model


def load_image(path, processor):
    crop_size = (processor.size["shortest_edge"], processor.size["shortest_edge"])
    transform = transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="./checkpoints/davit_pneumonia_detection.bin")
    parser.add_argument("--model_name_or_path", default="facebook/dinov2-large")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)
    vit = AutoModel.from_pretrained(args.model_name_or_path)
    model = Model(vit, processor, argparse.Namespace(focal_alpha=-1.0, focal_gamma=2.0))

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    pixel_values = load_image(args.image, processor).to(device)
    with torch.no_grad():
        probs = model(pixel_values=pixel_values)[0]
    pred = int(torch.argmax(probs).item())
    label = "PNEUMONIA" if pred == 1 else "NORMAL"
    print({"label": label, "probabilities": {"NORMAL": float(probs[0]), "PNEUMONIA": float(probs[1])}})


if __name__ == "__main__":
    main()
