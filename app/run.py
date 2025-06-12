import os

import uvicorn

from app.setup import settings


if __name__ == "__main__":
    uvicorn.run(
        app="app.setup:app",
        workers=settings.NUMBER_OF_WORKERS,
        host=settings.SERVICE_HOST,  # Ensure it binds to 0.0.0.0
        port=os.getenv("PORT", default=settings.SERVICE_PORT),  # Ensure the port matches the Docker configuration
        reload=settings.RELOAD,
    )
