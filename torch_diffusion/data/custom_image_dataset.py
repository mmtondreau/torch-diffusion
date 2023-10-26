import torch

import torch
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
from PIL import Image
import glob
import os


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.files = glob.glob(os.path.join(img_dir, "*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(0).to(torch.int64)
        return image, label
