from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

class DBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")

    ASYNC_DRIVER: str = "postgresql+asyncpg"
    SYNC_DRIVER: str = "postgresql"
    HOST: str = "db"
    PORT: int = 5432
    USER: str = "postgres"
    PASSWORD: str = "password"
    NAME: str = "postgres"
    POOL_SIZE: int = 5
    POOL_OVERFLOW: int = 10
    POOL_RECYCLE: int = 3600

    ECHO: bool = True

    @property
    def ASYNC_DSN(self) -> URL:
        return URL.create(
            self.ASYNC_DRIVER,
            self.USER,
            self.PASSWORD,
            self.HOST,
            self.PORT,
            self.NAME,
        )

    @property
    def SYNC_DSN(self) -> str:
        return URL.create(
            self.SYNC_DRIVER,
            self.USER,
            self.PASSWORD,
            self.HOST,
            self.PORT,
            self.NAME,
        ).render_as_string(hide_password=False)

    @property
    def PATH(self) -> str:
        return f"{self.HOST}:{self.PORT}/{self.NAME}"

class Settings(BaseSettings):
    model_config = SettingsConfigDict()

    SERVICE_NAME: str = "pdf-processor"
    DEBUG: bool = True
    SERVICE_VERSION: str = "0.1"
    PATH: str = "/pdf-processor"
    NUMBER_OF_WORKERS: int = 5
    SERVICE_HOST: str = "0.0.0.0"  # Bind to all interfaces
    RELOAD: bool = True
    SERVICE_PORT: int = 8000
    SERVICE_URL: str = f"http://{SERVICE_HOST}:{SERVICE_PORT}{PATH}"

    db: DBSettings = DBSettings()

    @property
    def docs_url(self) -> str:
        return f"{self.PATH}/docs"

