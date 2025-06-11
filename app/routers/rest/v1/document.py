from fastapi import APIRouter, Depends
import os
from fastapi import FastAPI, UploadFile, File, Form
from app.dependancies.service import get_s3_bucket_service
from app.services.s3_bucket_service import S3BucketService
from utils.extract_test import extract_text_from_pdf
import boto3

router = APIRouter()


@router.post("/files/")
async def upload_file(file: UploadFile = File(...), s3_bucket: S3BucketService = Depends(get_s3_bucket_service)):
    response = await s3_bucket.upload_file(file.filename, file.file)

    return response


@router.get("/list-files")
async def list_files(s3_bucket: S3BucketService = Depends(get_s3_bucket_service)):
    return await s3_bucket.list_files()


@router.get("/files")
async def get_file(filename:str, s3_bucket: S3BucketService = Depends(get_s3_bucket_service)):
    return await s3_bucket.extract_text_from_s3_pdf(filename)