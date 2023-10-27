from typing import Any, Dict
from pytorch_lightning.callbacks import Callback
from torch_diffusion.config import Config
from torch_diffusion.slack.client import SlackChannel, SlackClient


class SlackAlert(Callback):
    _slack_client: SlackClient
    _state: Dict[str, Any]
    _model_name: str

    def __init__(self, config: Config, model_name: str):
        self._slack_client = SlackClient(config=config)
        self._model_name = model_name

    def on_train_start(self, trainer, pl_module):
        self._slack_client.send(
            SlackChannel.MONITORING,
            f"{self._model_name} Training Started",
        )

    def on_train_end(self, trainer, pl_module):
        self._slack_client.send(
            SlackChannel.MONITORING,
            f"{self._model_name} Training Ended",
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        val_loss = trainer.callback_metrics.get("ptl/val_loss")
        ts = self._slack_client.send(
            SlackChannel.MONITORING,
            f"{self._model_name} Completed Epoch: {epoch}, Val Loss: {val_loss}",
        )
        self._slack_client.send_image(
            SlackChannel.MONITORING,
            pl_module.pil["pred"],
            message_text="val_predicted",
            parent_ts=ts,
        )
        self._slack_client.send_image(
            SlackChannel.MONITORING,
            pl_module.pil["perturb"],
            message_text="val_perturb",
            parent_ts=ts,
        )
        self._slack_client.send_image(
            SlackChannel.MONITORING,
            pl_module.pil["truth"],
            message_text="val_turth",
            parent_ts=ts,
        )
