import pytorch_lightning as pl
from torch_diffusion.callbacks.slack_alert import SlackAlert
from torch_diffusion.config import Config

from torch_diffusion.data.image_data_module import ImageDataModule
from torch_diffusion.data.preprocessor import PreProcessor
from torch_diffusion.model.difussion_model import DiffusionModule, DiffusionModuleConfig
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, BatchSizeFinder
import neptune
from lightning.pytorch.loggers import NeptuneLogger
import torch
import sys
import argparse
from dataclasses import dataclass

torch.set_float32_matmul_precision("medium")


def setup_logger():
    return NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_KEY"],
        project="mmtondreau/diffusion",
        name="diffusion",
    )


MODEL_CKPT_FILE = "{epoch}-{val_loss:.2f}"
MODEL_CKPT_DIRPATH = "model_checkpoints/"


@dataclass
class TorchDiffusionArgs:
    checkpoint: str = None
    learning_rate: float = None
    batch_size: int = None
    num_workers: int = None
    preprocess: bool = None
    preprocess_output: str = None
    preprocess_input: str = None


def parse_args() -> TorchDiffusionArgs:
    parser = argparse.ArgumentParser(prog="torch_diffusion")
    parser.add_argument(
        "-c", "--checkpoint", help="The model checkpoint to restore from"
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        default=0.001,
        type=float,
        help="The learning rate to use for training",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=32,
        type=int,
        help="The mini-batch size to utilize",
    )
    parser.add_argument(
        "-n",
        "--num_workers",
        default=1,
        type=int,
        help="The number of workers for processing data",
    )
    parser.add_argument(
        "-p",
        "--preprocess",
        help="Only run the processing of the immages",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-o",
        "--preprocess_output",
        help="The directroy to output the preprocessed images",
        default="./preprocessed_data",
    )
    parser.add_argument(
        "-i",
        "--preprocess_input",
        help="The directroy to take unpreprocessed images from",
        default="../data",
    )
    return parser.parse_args(namespace=TorchDiffusionArgs())
    # args = parser.parse_args()
    # return TorchDiffusionArgs(
    #     checkpoint=args.checkpoint,
    #     learning_rate=float(args.learning_rate),
    #     batch_size=int(args.batch_size),
    #     preprocess=args.preprocess,
    #     preprocess_output=args.preprocess_output,
    #     preprocess_input=args.preprocess_input,
    # )


def training(args: TorchDiffusionArgs):
    dm = ImageDataModule(
        batch_size=args.batch_size, num_workers=args.num_workers, val_split=0.2
    )

    if args.checkpoint is not None:
        print(f"Found checkpoint file {args.checkpoint}, loading...")
        model = DiffusionModule.load_from_checkpoint(
            args.checkpoint,
            config=DiffusionModuleConfig(learning_rate=args.learning_rate),
        )
    else:
        model = DiffusionModule(
            config=DiffusionModuleConfig(learning_rate=args.learning_rate)
        )
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
                enable_version_counter=True,
            ),
            # BatchSizeFinder(mode="binsearch", init_val=args.batch_size),
            SlackAlert(config=Config(), model_name="diffusion"),
        ],
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    args = parse_args()
    if args.preprocess:
        PreProcessor(
            data_dir=args.preprocess_input, output_dir=args.preprocess_output
        ).process()
    else:
        training(args)
