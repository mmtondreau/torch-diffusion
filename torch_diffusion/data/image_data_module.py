import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch_diffusion.data.custom_pt_dataset import CustomPTDataset
from torch.utils.data import random_split
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        width=128,
        height=192,
        data_dir: str = "../preprocessed_data",
        batch_size: int = 16,
        num_workers=2,
        validation_split=0.2,
        test_split=0.1,
    ):
        super().__init__()
        self.data_dir = os.path.join(data_dir, f"{width}x{height}")
        logger.info(f"Loadig from {self.data_dir }")
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.num_workers = num_workers
        # load on main thread so data gets shared across processes.
        dataset = CustomPTDataset(self.data_dir, transform=None)
        # Calculate the size of splits
        self.total_len = len(dataset)
        self.val_len = int(self.validation_split * self.total_len)
        self.test_len = int(self.test_split * self.total_len)
        self.train_len = self.total_len - self.val_len - self.test_len

        # Split the dataset into training, validation, and test sets
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [self.train_len, self.val_len, self.test_len]
        )

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):  # Define test dataloader
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
