from fastapi import APIRouter, Depends
from fastapi import UploadFile, File
from app.dependancies.service import get_ml_service, get_s3_bucket_service
from app.services.ml_service import MLService
from app.services.s3_bucket_service import S3BucketService

router = APIRouter()


@router.post("/files/")
async def upload_file(
    file: UploadFile = File(...),
    s3_bucket: S3BucketService = Depends(get_s3_bucket_service),
    ml_service: MLService = Depends(get_ml_service),
):
    response = await s3_bucket.upload_file(file.filename, file.file)
    ml_service.load_documents_from_s3()
    return response


@router.get("/list-files")
async def list_files(s3_bucket: S3BucketService = Depends(get_s3_bucket_service)):
    return await s3_bucket.list_files()


@router.get("/files")
async def get_file(
    filename: str, s3_bucket: S3BucketService = Depends(get_s3_bucket_service)
):
    return await s3_bucket.extract_text_from_s3_pdf(filename)
