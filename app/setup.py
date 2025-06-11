from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.settings import Settings
from app.routers.rest import provide_api_v1_router

settings = Settings()

origins = ["*"]


def provide_app(settings: Settings) -> FastAPI:
    app = FastAPI(
        title=settings.SERVICE_NAME,
        debug=settings.DEBUG,
        version=settings.SERVICE_VERSION,
        docs_url=settings.docs_url,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings

    api_v1_router = provide_api_v1_router()
    app.include_router(api_v1_router)
    return app


app = provide_app(settings)