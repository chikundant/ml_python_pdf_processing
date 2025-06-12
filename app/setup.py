from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.db.connection import close_db_pool, init_db_pool
from app.core.settings import Settings
from app.routers.rest import provide_api_v1_router

settings = Settings()

origins = ["*"]


def provide_app(settings: Settings) -> FastAPI:
    app = FastAPI(
        title=settings.SERVICE_NAME,
        debug=settings.DEBUG,
        version=settings.SERVICE_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings

    app.add_event_handler("startup", init_db_pool(app, settings.db))
    
    api_v1_router = provide_api_v1_router()
    app.include_router(api_v1_router)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    app.add_event_handler("shutdown", close_db_pool(app))
    return app


app = provide_app(settings)
