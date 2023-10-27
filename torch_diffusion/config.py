import configparser


class Config:
    _config: configparser.ConfigParser

    def __init__(self) -> None:
        self._config = configparser.ConfigParser()
        self._config.read("config.ini")

    def get_slack_token(self) -> str:
        return self._config.get("Slack", "slack_token")

    def get_slack_channelid(self, channel_name: str) -> str:
        return self._config.get("Slack.Channels", channel_name)
