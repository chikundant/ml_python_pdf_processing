from fastapi import FastAPI, APIRouter

from app.core.settings import Settings
from contextlib import asynccontextmanager

from app.routers.rest import provide_api_v1_router
from app.utils.sentry import init_sentry

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_sentry(settings)
    setup_routers(app)
    yield


def setup_routers(app: FastAPI):
    api_v1_router = provide_api_v1_router()
    api_router = APIRouter()
    api_router.include_router(api_v1_router, prefix="/api/v1")
    app.include_router(api_router, prefix='/otel-sentry')
