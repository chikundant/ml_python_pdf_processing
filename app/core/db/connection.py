import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Coroutine

from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.settings import DBSettings
from app.models.knowledge_base import Base


class AsyncDBPoolProvisionError(Exception):
    """Exception raised when the async DB pool cannot be provisioned."""


CoroutineType = Callable[[], Coroutine]


def _db_json(obj: Any) -> str:
    if isinstance(obj, datetime):
        return f"{obj.isoformat()}Z"
    return obj


async def provide_async_db_pool(  # type: ignore
    settings: DBSettings, max_retries: int = 3, retry_interval: int = 2
) -> AsyncEngine:
    for attempt in range(max_retries + 1):
        try:
            pool = create_async_engine(
                settings.ASYNC_DSN,
                echo=settings.ECHO,
                pool_size=settings.POOL_SIZE,
                max_overflow=settings.POOL_OVERFLOW,
                pool_recycle=settings.POOL_RECYCLE,
                json_serializer=lambda obj: json.dumps(obj, default=_db_json),
            )
            async with pool.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                await conn.execute(text("SELECT 1;"))

            return pool
        except Exception as exc:
            if attempt < max_retries:
                await asyncio.sleep(retry_interval)
            else:
                raise AsyncDBPoolProvisionError(
                    f"failed to connect to db at {settings.PATH}"
                ) from exc


def init_db_pool(app: FastAPI, settings: DBSettings) -> CoroutineType:
    async def _init() -> None:
        pool = await provide_async_db_pool(settings)
        app.state.db_pool = pool

    return _init


def close_db_pool(app: FastAPI) -> CoroutineType:
    async def _close() -> None:
        if hasattr(app.state, "db_pool"):
            await app.state.db_pool.dispose()

    return _close


@asynccontextmanager
async def provide_db_pool(
    settings: DBSettings,
) -> AsyncGenerator[AsyncEngine, None]:
    pool = await provide_async_db_pool(settings)

    try:
        yield pool
    finally:
        await pool.dispose()
