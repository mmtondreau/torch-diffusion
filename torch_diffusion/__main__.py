import pytorch_lightning as pl
from torch_diffusion.callbacks.slack_alert import SlackAlert


from torch_diffusion.data.image_data_module import ImageDataModule
from torch_diffusion.data.preprocessor import PreProcessor
from torch_diffusion.model.difussion_model import DiffusionModule, DiffusionModuleConfig
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
import torch
from omegaconf import DictConfig
import hydra
from datetime import datetime

torch.set_float32_matmul_precision("medium")


def setup_logger(cfg: DictConfig):
    return NeptuneLogger(
        api_key=cfg.neptune.api_key,
        project="mmtondreau/diffusion",
        name="diffusion",
    )


MODEL_CKPT_FILE = datetime.now().isoformat()
MODEL_CKPT_DIRPATH = "model_checkpoints/"


def find_latest_checkpoint(cfg: DictConfig):
    if cfg.training.checkpoint is not None:
        checkpoint_file = cfg.training.checkpoint
        print(f"Using checkpoint {checkpoint_file}")
        return checkpoint_file
    file_list = os.listdir(cfg.training.checkpoint_dir)
    if len(file_list) > 0:
        file_list.sort(reverse=True)
        checkpoint_file = os.path.join(cfg.training.checkpoint_dir, file_list[0])
        print(f"Found checkpoint file {checkpoint_file}")
        return checkpoint_file
    else:
        return None


def training(cfg: DictConfig):
    dm = ImageDataModule(
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        val_split=0.2,
    )

    if cfg.training.checkpoint_dir is not None:
        checkpoint_file = find_latest_checkpoint(cfg)
        model = DiffusionModule.load_from_checkpoint(
            checkpoint_file,
            config=DiffusionModuleConfig(
                learning_rate=float(cfg.training.learning_rate)
            ),
        )
    else:
        model = DiffusionModule(
            config=DiffusionModuleConfig(
                learning_rate=float(cfg.training.learning_rate)
            )
        )
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename=MODEL_CKPT_FILE,
            dirpath=MODEL_CKPT_DIRPATH,
            enable_version_counter=True,
        ),
        # BatchSizeFinder(mode="binsearch", init_val=args.batch_size),
        SlackAlert(cfg=cfg, model_name="diffusion"),
    ]
    if cfg.training.early_stopping.enabled == "True":
        callbacks.append(
            EarlyStopping(
                monitor="ptl/val_loss",
                mode="min",
                patience=int(cfg.training.early_stopping.patience),
                min_delta=float(cfg.training.early_stopping.min_delta),
            ),
        )
    trainer = pl.Trainer(
        logger=setup_logger(cfg),
        devices=1,
        accelerator="gpu",
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=dm)


@hydra.main(config_name="config.yaml", config_path="config", version_base=None)
def main(cfg):
    if cfg.preprocess == "True":
        PreProcessor(
            data_dir=cfg.preprocess.input_dir, output_dir=cfg.preprocess.output_dir
        ).process()
    else:
        training(cfg)


if __name__ == "__main__":
    main()
