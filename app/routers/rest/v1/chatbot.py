from fastapi import APIRouter, Depends
from app.core.db.session import get_session
from app.services.ml_service import MLService
from app.dependancies.service import get_s3_bucket_service
router = APIRouter()


# @router.post("/load-documents/")
# async def load_documents(s3_bucket_service=Depends(get_s3_bucket_service)):
#     ml_service = MLService(s3_bucket_service)
#     await ml_service.load_documents_from_s3()
#     return {"message": "Documents loaded successfully"}


# @router.post("/train-model/")
# async def train_model(s3_bucket_service=Depends(get_s3_bucket_service)):
#     ml_service = MLService(s3_bucket_service)
#     await ml_service.load_documents_from_s3()
#     ml_service.build_knowledge_base()
#     return {"message": "Model trained successfully"}


# @router.post("/ask/")
# async def ask_question(question: str, s3_bucket_service=Depends(get_s3_bucket_service)):
#     ml_service = MLService(s3_bucket_service)
#     return {"answer": await ml_service.answer_question(question)}

@router.post("/init/")
async def init_knowledge_base(s3_bucket_service=Depends(get_s3_bucket_service), db_session=Depends(get_session)):
    ml_service = MLService(s3_bucket_service, db_session)
    await ml_service.initialize()
    return {"message": "Knowledge base initialized successfully"}

@router.get("/stats/")
async def get_stats(s3_bucket_service=Depends(get_s3_bucket_service), db_session=Depends(get_session)):
    ml_service = MLService(s3_bucket_service, db_session)
    return await ml_service.get_knowledge_base_stats()

@router.get("/ask/")
async def get_stats(question:str, s3_bucket_service=Depends(get_s3_bucket_service), db_session=Depends(get_session)):
    ml_service = MLService(s3_bucket_service, db_session)
    return await ml_service.answer_question(question=question)
