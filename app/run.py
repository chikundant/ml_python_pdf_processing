import uvicorn
from fastapi import FastAPI
from app import lifespan, settings, setup_routers

app = FastAPI(lifespan=lifespan)

if __name__ == "__main__":
    uvicorn.run(
        app="app.run:app",
        workers=settings.NUMBER_OF_WORKERS,
        host=settings.SERVICE_HOST,
        port=settings.SERVICE_PORT,
        reload=settings.RELOAD,
    )
