import pytorch_lightning as pl
from torch_diffusion.callbacks.slack_alert import SlackAlert
import glob

from torch_diffusion.data.image_data_module import ImageDataModule
from torch_diffusion.data.preprocessor import PreProcessor
from torch_diffusion.model.context_unit import ContextUnitConfig, ContextUnitLayerCOnfig
from torch_diffusion.model.difussion_model import (
    DiffusionModule,
    DiffusionModuleConfig,
    DiffusionModuleMetrics,
)
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)

from lightning.pytorch.loggers import NeptuneLogger
import torch
from omegaconf import DictConfig
import hydra
from datetime import datetime
from dataclasses import asdict

from torch_diffusion.model.inference import Inference, InferenceConfig

torch.set_float32_matmul_precision("medium")

MODEL_CONFIG = ContextUnitConfig(
    features=64,
    layers=[
        ContextUnitLayerCOnfig(features=256, kernel_size=9),
        ContextUnitLayerCOnfig(features=512, kernel_size=5),
        ContextUnitLayerCOnfig(features=1024, kernel_size=3),
    ],
)


def setup_logger(cfg: DictConfig):
    return NeptuneLogger(
        api_key=cfg.neptune.api_key,
        project="mmtondreau/diffusion",
        name="diffusion",
        log_model_checkpoints=True,
        prefix="",
    )


MODEL_CKPT_DIRPATH = "model_checkpoints"


def find_latest_checkpoint(cfg: DictConfig, model_hash: str):
    if cfg.training.checkpoint is not None:
        checkpoint_file = cfg.training.checkpoint
        print(f"Using checkpoint {checkpoint_file}")
        return checkpoint_file
    file_list = glob.glob(
        os.path.join(cfg.training.checkpoint_dir, f"{model_hash}*.ckpt")
    )
    if len(file_list) > 0:
        file_list.sort(reverse=True)
        checkpoint_file = file_list[0]
        print(f"Found checkpoint file {checkpoint_file}")
        return checkpoint_file
    else:
        return None


def get_model_hash(config):
    return DiffusionModule(config=config, model_config=MODEL_CONFIG).get_model_hash()


def training(cfg: DictConfig):
    batch_size = int(cfg.training.batch_size)
    dm = ImageDataModule(
        batch_size=batch_size,
        num_workers=int(cfg.training.num_workers),
        validation_split=float(cfg.training.validation_split),
        test_split=float(cfg.training.test_split),
        data_dir=cfg.preprocess.output_dir,
        width=cfg.model.width,
        height=cfg.model.height,
        truncate=2000,
    )

    model_config = get_model_config(cfg)

    model_hash = get_model_hash(model_config)

    model_ckpt_file = f"{model_hash}:{datetime.now().isoformat()}"
    checkpoint_file = None
    if cfg.training.checkpoint_dir is not None:
        checkpoint_file = find_latest_checkpoint(cfg, model_hash)
        if checkpoint_file is not None:
            model = DiffusionModule.load_from_checkpoint(
                checkpoint_file, config=model_config
            )
    if checkpoint_file is None:
        model = DiffusionModule(
            config=model_config,
            model_config=MODEL_CONFIG,
        )

    callbacks = [
        ModelCheckpoint(
            monitor=DiffusionModuleMetrics.VALIDATION_EPOCH_LOSS,
            mode="min",
            filename=model_ckpt_file,
            dirpath=MODEL_CKPT_DIRPATH,
            enable_version_counter=True,
        ),
        # BatchSizeFinder(mode="binsearch", init_val=args.batch_size),
        SlackAlert(
            cfg=cfg,
            model_name="diffusion",
            monitor=DiffusionModuleMetrics.VALIDATION_EPOCH_LOSS,
        ),
        # SWA causes massive memory dump at end of training
        # StochasticWeightAveraging(
        #     swa_lrs=float(cfg.training.stochastic_weight_averaging)
        # ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if cfg.training.early_stopping.enabled == "True":
        callbacks.append(
            EarlyStopping(
                monitor=DiffusionModuleMetrics.VALIDATION_EPOCH_LOSS,
                mode="min",
                patience=int(cfg.training.early_stopping.patience),
                min_delta=float(cfg.training.early_stopping.min_delta),
            ),
        )
    neptune_logger = setup_logger(cfg)
    neptune_logger.log_hyperparams(params=asdict(model_config))
    neptune_logger.experiment["signature"] = model_hash
    trainer = pl.Trainer(
        logger=neptune_logger,
        devices="auto",
        accelerator="auto",
        max_epochs=int(cfg.training.max_epochs),
        callbacks=callbacks,
        accumulate_grad_batches=config_accumulate_grad_batches(cfg),
        gradient_clip_val=config_gradient_clip_val(cfg),
        enable_progress_bar=True,
        log_every_n_steps=5,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    neptune_logger.log_model_summary(model=model, max_depth=-1)


def get_model_config(cfg):
    return DiffusionModuleConfig(
        learning_rate=float(cfg.training.learning_rate),
        height=int(cfg.model.height),
        width=int(cfg.model.width),
    )


def config_gradient_clip_val(cfg):
    return float(
        cfg.training.gradient_clip_val
        if cfg.training.gradient_clip_val is not None
        else None
    )


def config_accumulate_grad_batches(cfg):
    return (
        int(cfg.training.accumulate_grad_batches)
        if cfg.training.accumulate_grad_batches is not None
        else None
    )


def inference(cfg: DictConfig):
    model_config = get_model_config(cfg)
    model_hash = get_model_hash(model_config)
    checkpoint_file = find_latest_checkpoint(cfg, model_hash)
    Inference(
        InferenceConfig(
            checkpoint_file=checkpoint_file,
            height=int(cfg.model.height),
            width=int(cfg.model.width),
            features=int(cfg.model.features),
            timesteps=int(cfg.diffusion.timesteps),
            beta1=float(cfg.diffusion.beta1),
            beta2=float(cfg.diffusion.beta2),
        )
    ).infer()


@hydra.main(config_name="config.yaml", config_path="config", version_base=None)
def main(cfg):
    if cfg.preprocess.enabled == "True":
        PreProcessor(
            data_dir=cfg.preprocess.input_dir,
            output_dir=cfg.preprocess.output_dir,
            target_height=int(cfg.model.height),
            target_width=int(cfg.model.width),
            samples_per_image=int(cfg.preprocess.samples_per_image),
        ).process()
    elif cfg.inference.enabled == "True":
        inference(cfg)
    else:
        training(cfg)


if __name__ == "__main__":
    main()
