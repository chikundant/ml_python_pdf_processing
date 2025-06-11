import base64
from functools import cached_property
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings():
    model_config = SettingsConfigDict()

    SERVICE_NAME: str = "pdf-processor"
    DEBUG: str = True
    SERVICE_VERSION: str = "0.1"
    PATH = "/pdf-processor"
    NUMBER_OF_WORKERS = 5
    SERVICE_HOST: str = "localhost"
    RELOAD: bool = True
    SERVICE_PORT: int = 8000
    SERVICE_URL: str = f"http://{SERVICE_HOST}:{SERVICE_PORT}{PATH}"

    @property
    def docs_url(self) -> str:
        return f"{self.PATH}/docs"
