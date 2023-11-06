from typing import Dict, List
import pytorch_lightning as pl
import torch
from torch_diffusion.model.context_unit import ContextUnet
import torch.nn.functional as F
import torchvision.transforms as transforms
from dataclasses import dataclass
from PIL.Image import Image
import time
from lightning.pytorch.utilities.model_summary import ModelSummary
import hashlib


@dataclass(frozen=True)
class DiffusionModuleMetrics:
    VALIDATION_EPOCH_LOSS = "val/epoch/loss"
    VALIDATION_BATCH_LOSS = "val/batch/loss"
    TRAINING_EPOCH_LOSS = "train/epoch/loss"
    TRAINING_BATCH_LOSS = "train/batch/loss"
    TEST_BATCH_LOSS = "test/batch/loss"
    TEST_EPOCH_LOSS = "test/epoch/loss"


@dataclass
class DiffusionModuleConfig:
    learning_rate: float = 0.001
    features: int = 512
    height: int = 128
    width: int = 192


@dataclass
class DiffusionEvaluation:
    perturb: torch.Tensor
    predicted: torch.Tensor
    truth: torch.Tensor
    loss: torch.Tensor
    time_step: torch.Tensor


class DiffusionModule(pl.LightningModule):
    _val_loss: List[float]
    _train_loss: List[float]
    _pil: Dict[str, Image]
    _learning_rate: float

    def __init__(self, config: DiffusionModuleConfig):
        super().__init__()
        self.model = ContextUnet(
            in_channels=3,
            n_feat=config.features,
            width=config.width,
            height=config.height,
        )
        self._learning_rate = (
            0.001 if config.learning_rate is None else config.learning_rate
        )
        self._val_loss = []
        # diffusion hyperparameters

        self.example_input_array = (
            torch.Tensor(16, 3, config.height, config.width),
            torch.Tensor(16, 1),
        )
        self._val_loss = []
        self._train_loss = []
        self._pil = {}
        self._timesteps = 500
        self.save_hyperparameters()

    def setup(self, stage=None):
        pass

    def forward(self, x, t):
        x = self.model(x, t)
        return x

    def training_step(self, batch, batch_idx):
        eval = self._shared_eval(batch, "train")
        self._train_loss.append(eval.loss)
        return eval.loss

    def on_train_epoch_start(self):
        self._train_loss.clear()
        torch.manual_seed(time.time())

    def on_train_epoch_end(self):
        loss = torch.stack(self._train_loss).mean()
        self.log(DiffusionModuleMetrics.TRAINING_EPOCH_LOSS, loss)

    def on_validation_epoch_start(self):
        self._val_loss.clear()
        torch.manual_seed(0)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        eval = self._shared_eval(batch, "val")
        self._val_loss.append(eval.loss)
        self._log_images(eval, batch_idx)
        # self.log("val/loss", loss, sync_dist=True)

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self._val_loss).mean()
        self.log(DiffusionModuleMetrics.VALIDATION_EPOCH_LOSS, avg_loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        eval = self._shared_eval(batch, "test")
        self.log(DiffusionModuleMetrics.TEST_BATCH_LOSS, eval.loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self._learning_rate, weight_decay=0.001
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            max_lr=0.01,
            optimizer=optimizer,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            epochs=self.trainer.max_epochs,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def get_model_hash(self):
        summary = str(ModelSummary(self, max_depth=-1))
        return hashlib.sha256(summary.encode("utf-8")).hexdigest()

    def _shared_eval(self, batch, stage) -> DiffusionEvaluation:
        x, noise, t, _ = batch

        predictions = self(x, t / self._timesteps)

        loss = F.mse_loss(predictions, noise)
        self.logger.experiment[f"{stage}/batch/loss"].append(loss)
        return DiffusionEvaluation(
            loss=loss, perturb=x, truth=noise, predicted=predictions, time_step=t
        )

    def _log_images(self, evaluation: DiffusionEvaluation, batch_idx: int):
        image_names = ["pred", "truth", "perturb"]

        if batch_idx % 10 == 0:
            images = {
                "pred": evaluation.predicted[0],
                "truth": evaluation.perturb[0],
                "perturb": evaluation.truth[0],
            }
            for image_name in image_names:
                self._publish_image(
                    image_name,
                    evaluation.time_step[0],
                    images[image_name],
                    batch_idx,
                )

    def _publish_image(self, type, t, image, batch_idx):
        pil = self._to_pil(image)
        self.logger.experiment[f"val_image_{type}_{batch_idx}"].append(
            pil, description=f"t: {t}"
        )
        self._pil[type] = pil

    def _denoise_ddim(self, x, t, t_prev, pred_noise):
        ab = self.ab_t[t]
        ab_prev = self.ab_t[t_prev]

        x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
        dir_xt = (1 - ab_prev).sqrt() * pred_noise

        return x0_pred + dir_xt

    def _undo_normalize(self, image):
        image = (image * 0.5) + 0.5
        return image

    def _to_pil(self, image):
        first_image = self._undo_normalize(image)
        first_image_pil = transforms.ToPILImage()(first_image)
        return first_image_pil
