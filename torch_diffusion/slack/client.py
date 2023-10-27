from io import BytesIO
from slack_sdk import WebClient
from typing import Any, Optional
import os
from enum import Enum

from torch_diffusion.config import Config
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class SlackChannel(str, Enum):
    ALERTS = "alerts"
    GENERAL = "general"
    MONITORING = "monitoring"
    DEPLOYMENTS = "deployments"


class SlackClient:
    _config: Config
    _slack_web_client: WebClient

    def __init__(self, config: Config):
        self._config = config
        self._slack_web_client = WebClient(token=config.get_slack_token())

    def send(self, channel: SlackChannel, message: str) -> str:
        try:
            channel_id = self.__get_channel_id(channel)
            response = self._slack_web_client.chat_postMessage(channel=channel_id, text=message)
            return response['ts']  # Return the timestamp (thread_ts) of the posted message
        except Exception as e:
            logger.error(e)
            return ""

    def send_image(self, channel: SlackChannel, pil_image: Image.Image, message_text: str, parent_ts: str) -> None:
        try:
            channel_id = self.__get_channel_id(channel)

            # Convert PIL image to bytes
            image_bytes = BytesIO()
            pil_image.save(image_bytes, format="PNG")
            image_bytes.seek(0)

            # Upload the image to Slack and add it as an attachment in a reply
            response = self._slack_web_client.files_upload(
                channels=channel_id,
                file=image_bytes,
                filename="image.png",
                initial_comment=message_text,
                thread_ts=parent_ts  # Specify the parent message's timestamp to create a thread
            )

            logger.info(f"Image posted as a reply to parent message in channel {channel}")
        except Exception as e:
            logger.error(f"Error posting image: {str(e)}")

    def __get_channel_id(self, channel: SlackChannel) -> str:
        return self._config.get_slack_channelid(channel)
