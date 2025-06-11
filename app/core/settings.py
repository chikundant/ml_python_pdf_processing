from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="app/system_configs/.env", env_file_encoding="utf-8"
    )
    SERVICE_NAME: str = "Otel Sentry Integration"
    SERVICE_HOST: str = "0.0.0.0"
    SERVICE_PORT: int = 8001
    NUMBER_OF_WORKERS: int = 1
    DEBUG: bool = False
    TESTING: bool = False
    RELOAD: bool = False
    SENTRY_DSN: str = "https://893b2b14ce582fc9d879ce3c038d5162@o4506421920464896.ingest.sentry.io/4506421939732480"
