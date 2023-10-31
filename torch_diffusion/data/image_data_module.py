import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch_diffusion.data.custom_pt_dataset import CustomPTDataset
from torch.utils.data import random_split


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "../preprocessed_data",
        batch_size: int = 16,
        num_workers=2,
        validation_split=0.2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.num_workers = num_workers
        # load on main thread so data gets shared across processes.
        dataset = CustomPTDataset(self.data_dir, transform=None)

        # Calculate the size of the validation set
        num_val_samples = int(self.validation_split * len(dataset))
        num_train_samples = len(dataset) - num_val_samples

        # Split the dataset into training and validation sets
        self.train_dataset, self.val_dataset = random_split(
            dataset, [num_train_samples, num_val_samples]
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
