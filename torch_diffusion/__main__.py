import pytorch_lightning as pl

from torch_diffusion.data.image_data_module import ImageDataModule
from torch_diffusion.model.difussion_model import DiffusionModule
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import neptune
from lightning.pytorch.loggers import NeptuneLogger
import torch

torch.set_float32_matmul_precision("medium")


def setup_logger():
    return NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_KEY"],
        project="mmtondreau/diffusion",
        name="diffusion",
    )


MODEL_CKPT_FILE = "model"
MODEL_CKPT_DIRPATH = "model_checkpoints/"
MDDEL_CKPT_FILE_PATH = os.path.join(MODEL_CKPT_DIRPATH, MODEL_CKPT_FILE)

if __name__ == "__main__":
    dm = ImageDataModule(batch_size=32, num_workers=2, val_split=0.2)
    model = DiffusionModule()
    if os.path.exists(MDDEL_CKPT_FILE_PATH):
        print(f"Found checkpoint file {MDDEL_CKPT_FILE_PATH}, loading...")
        model.load_from_checkpoint(MDDEL_CKPT_FILE_PATH)

    trainer = pl.Trainer(
        logger=setup_logger(),
        devices=1,
        accelerator="gpu",
        max_epochs=100,
        callbacks=[
            EarlyStopping(
                monitor="ptl/val_loss", mode="min", patience=5, min_delta=0.0001
            ),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                filename=MODEL_CKPT_FILE,
                dirpath=MODEL_CKPT_DIRPATH,
            ),
        ],
    )
    trainer.fit(model, datamodule=dm)
