from fastapi import APIRouter, Depends
from app.core.db.session import get_session
from app.services.ml_service import MLService
from app.dependancies.service import get_s3_bucket_service
router = APIRouter()

@router.post("/init/")
async def init_knowledge_base(s3_bucket_service=Depends(get_s3_bucket_service), db_session=Depends(get_session)):
    """"Initialize the knowledge base by processing documents from S3 bucket."""
    ml_service = MLService(s3_bucket_service, db_session)
    await ml_service.initialize()
    return {"message": "Knowledge base initialized successfully"}

@router.get("/stats/")
async def get_stats(s3_bucket_service=Depends(get_s3_bucket_service), db_session=Depends(get_session)):
    """Get statistics about the knowledge base."""
    ml_service = MLService(s3_bucket_service, db_session)
    return await ml_service.get_knowledge_base_stats()

@router.get("/ask/")
async def ask(question:str, s3_bucket_service=Depends(get_s3_bucket_service), db_session=Depends(get_session)):
    """Ask a question to the knowledge base."""
    ml_service = MLService(s3_bucket_service, db_session)
    return await ml_service.answer_question(question=question)
