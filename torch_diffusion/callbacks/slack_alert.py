from typing import Any, Dict
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from torch_diffusion.slack.client import SlackChannel, SlackClient


class SlackAlert(Callback):
    _slack_client: SlackClient
    _state: Dict[str, Any]
    _model_name: str
    _monitor: str

    def __init__(self, cfg: DictConfig, model_name: str, monitor: str):
        self._slack_client = SlackClient(config=cfg)
        self._model_name = model_name
        self._monitor = monitor

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
        metric = trainer.callback_metrics.get(self._monitor)
        ts = self._slack_client.send(
            SlackChannel.MONITORING,
            f"{self._model_name} Completed Epoch: {epoch}, {self._monitor}: {metric}",
        )

        # self._slack_client.send_images(
        #     SlackChannel.MONITORING,
        #     images={
        #         "predicted": pl_module._pil["pred"],
        #         "perturb": pl_module._pil["perturb"],
        #         "truth": pl_module._pil["truth"],
        #     },
        #     parent_ts=ts,
        # )
