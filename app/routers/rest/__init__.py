from fastapi import APIRouter

from app.routers.rest.v1.document import router as document_rounter


def provide_api_v1_router() -> APIRouter:
    router = APIRouter()

    router.include_router(document_rounter, prefix="/document", tags=["document"])

    return router
