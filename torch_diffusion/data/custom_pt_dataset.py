import torch

import os
from torch.utils.data import Dataset
import glob


class CustomPTDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.files = glob.glob(os.path.join(dir, "*.pt"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item_pt = self.files[idx]
        item, noise, t = torch.load(item_pt)
        if self.transform:
            item = self.transform(item)
        label = torch.tensor(0).to(torch.int64)

        return item, noise, t, label
