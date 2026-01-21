from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from typing_extensions import Self


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parents[1] / ".env",
        env_file_encoding="UTF-8",
        extra='ignore'
    )