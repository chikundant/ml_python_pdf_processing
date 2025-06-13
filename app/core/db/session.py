from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from starlette.requests import HTTPConnection


def _get_db_pool(request: HTTPConnection) -> AsyncEngine:
    return request.app.state.db_pool


async def get_session(
    pool: AsyncEngine = Depends(_get_db_pool),
) -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSession(pool, expire_on_commit=False) as session:
        yield session
