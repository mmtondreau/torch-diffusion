import pytorch_lightning as pl
import torch
from torch_diffusion.model.context_unit import ContextUnet
import torch.nn.functional as F
import torchvision.transforms as transforms
from dataclasses import dataclass


@dataclass
class DiffusionModuleConfig:
    learning_rate: float = 0.001


class DiffusionModule(pl.LightningModule):
    def __init__(self, config: DiffusionModuleConfig):
        super().__init__()
        self.model = ContextUnet(in_channels=3)
        self.learning_rate = (
            0.001 if config.learning_rate is None else config.learning_rate
        )
        self.val_loss = []
        # diffusion hyperparameters
        self.timesteps = 500
        self.beta1 = 1e-4
        self.beta2 = 0.02
        self.example_input_array = [torch.Tensor(16, 3, 192, 128), torch.Tensor(16, 1)]
        self.val_loss = []
        self.pil = {}
        self.save_hyperparameters()

    def setup(self, stage=None):
        # construct DDPM noise schedule
        self.b_t = (self.beta2 - self.beta1) * torch.linspace(
            0, 1, self.timesteps + 1, device=self.device
        ) + self.beta1
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()
        self.ab_t[0] = 1

    def forward(self, x, t):
        x = self.model(x, t)
        return x

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx, "train")
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_loss.clear()

    def on_validation_epoch_start(self):
        self.val_loss.clear()

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        loss = self._shared_eval(batch, batch_idx, "val")
        self.log("val_loss", loss, prog_bar=True)
        self.val_loss.append(loss)

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_loss).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.logger.experiment["ptl/val_loss"].append(avg_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        loss = self._shared_eval(batch, batch_idx, "test")
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=0.001
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=5, factor=0.1, mode="min"
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }

    def _shared_eval(self, batch, batch_idx, stage):
        x, _ = batch

        tensorboard = self.logger.experiment

        # perturb data
        noise = torch.randn_like(x)
        t = torch.randint(1, self.timesteps + 1, (x.shape[0],), device=self.device)
        x_pert = self._perturb_input(x, t, noise)

        predictions = self(x_pert, t / self.timesteps)

        self.log_images(batch_idx, stage, x, tensorboard, t, x_pert, predictions)

        loss = F.mse_loss(predictions, noise)
        tensorboard[f"{stage}_loss"].append(loss)
        return loss

    def log_images(self, batch_idx, stage, x, tensorboard, t, x_pert, predictions):
        image_names = [f"pred", f"truth", f"perturb"]
        if tensorboard.exists(f"{stage}_image_pred") and batch_idx == 0:
            self.clear_image_logs(stage, tensorboard)
            self.pil = {name: [] for name in image_names}

        if batch_idx % 10 == 0:
            images = {
                "pred": predictions[0],
                "truth": x[0],
                "perturb": x_pert[0],
            }
            for type, image in images.items():
                self.log_image_tpye(stage, type, t,image)

    def log_image_tpye(self, stage, type, t, image):
        pil = self.to_pil(image)
        self.logger.experiment[f"{stage}_image_{type}"].append(
            pil,
            name=f"t: {t}",
        )
        if stage == "val":
            self.pil[type] = pil

    def clear_image_logs(self, stage, tensorboard):
        del tensorboard[f"{stage}_image_pred"]
        del tensorboard[f"{stage}_image_truth"]
        del tensorboard[f"{stage}_image_perturb"]

    def _denoise_ddim(self, x, t, t_prev, pred_noise):
        ab = self.ab_t[t]
        ab_prev = self.ab_t[t_prev]

        x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
        dir_xt = (1 - ab_prev).sqrt() * pred_noise

        return x0_pred + dir_xt

    # helper function: perturbs an image to a specified noise level
    def _perturb_input(self, x, t, noise):
        # print(f"t device: {t.device}, x device: {x.device}, ab_t device: {self.ab_t.device}")
        return (
            self.ab_t.sqrt()[t, None, None, None] * x
            + (1 - self.ab_t[t, None, None, None]) * noise
        )

    def undo_normalize(self, image):
        image = (image * 0.5) + 0.5
        return image

    def to_pil(self, image):
        first_image = self.undo_normalize(image)
        first_image_pil = transforms.ToPILImage()(first_image)
        return first_image_pil
