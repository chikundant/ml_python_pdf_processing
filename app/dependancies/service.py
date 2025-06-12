import boto3
from fastapi import Depends

from app.services.ml_service import MLService
from app.services.s3_bucket_service import S3BucketService


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url="http://localstack:4566",  # Use the Docker service name for LocalStack
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


def get_s3_bucket_service(
    boto3_client=Depends(get_s3_client),
) -> S3BucketService:
    return S3BucketService(boto3_client, "pdf-bucket")


def get_ml_service(
    s3_bucket_service: S3BucketService = Depends(get_s3_bucket_service),
) -> MLService:
    return MLService(s3_bucket_service)
