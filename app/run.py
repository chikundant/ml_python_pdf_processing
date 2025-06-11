import os

import uvicorn

from app.setup import settings


if __name__ == "__main__":
    uvicorn.run(
        app="app.setup:app",
        workers=settings.NUMBER_OF_WORKERS,
        host=settings.SERVICE_HOST,
        port=os.getenv("PORT", default=settings.SERVICE_PORT),
        reload=settings.RELOAD,
    )