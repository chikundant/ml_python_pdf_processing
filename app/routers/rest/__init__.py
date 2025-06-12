from fastapi import APIRouter

from app.routers.rest.v1.document import router as document_rounter
from app.routers.rest.v1.chatbot import router as chatbot_router


def provide_api_v1_router() -> APIRouter:
    router = APIRouter()

    router.include_router(document_rounter, prefix="/document", tags=["document"])
    router.include_router(chatbot_router, prefix="/chatbot", tags=["chatbot"])

    return router
