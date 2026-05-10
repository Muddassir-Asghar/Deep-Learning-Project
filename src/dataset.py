import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ChestXrayCsvDataset(Dataset):
    def __init__(self, csv_path, processor):
        self.csv_path = Path(csv_path)
        self.root = self.csv_path.parent.parent
        with self.csv_path.open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))

        crop_size = (processor.size["shortest_edge"], processor.size["shortest_edge"])
        self.transform = transforms.Compose(
            [
                transforms.Resize(crop_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self.root / row["image_path"]
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image), torch.tensor(int(row["label"])).long()
