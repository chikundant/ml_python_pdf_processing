from fastapi import APIRouter

from app.routers.rest.v1.document import router as user_router


def provide_api_v1_router() -> APIRouter:
    router = APIRouter()

    router.include_router(user_router, prefix="/user", tags=["user"])

    return router
