import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch_diffusion.data.custom_image_dataset import CustomImageDataset
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./preprocessed_data",
        batch_size: int = 16,
        num_workers=2,
        val_split=0.2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        # load on main thread so data gets shared across processes.
        dataset = CustomImageDataset(self.data_dir, transform=self.transform)

        # Calculate the size of the validation set
        num_val_samples = int(self.val_split * len(dataset))
        num_train_samples = len(dataset) - num_val_samples

        # Split the dataset into training and validation sets
        self.train_dataset, self.val_dataset = random_split(
            dataset, [num_train_samples, num_val_samples]
        )

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        pass

    def transform(self, img, target_width=128, target_height=192):
        # Apply the other transformations
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert the image to a PyTorch tensor
                transforms.Normalize((0.5,), (0.5,)),  # range [-1,1]
            ]
        )

        img = transform(img)

        return img

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
