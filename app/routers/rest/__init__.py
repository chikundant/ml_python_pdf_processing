from fastapi import APIRouter

from app.routers.rest.v1.document import router as document_rounter
from app.routers.rest.v1.chatbot import router as chatbot_router
from app.routers.rest.v1.index import router as index_router


def provide_api_v1_router() -> APIRouter:
    router = APIRouter()

    router.include_router(document_rounter, prefix="/documents", tags=["document"])
    router.include_router(chatbot_router, prefix="/chatbot", tags=["chatbot"])
    router.include_router(index_router, prefix="", tags=["index"])

    return router
