import aioboto3
from fastapi import Depends
from app.services.s3_bucket_service import S3BucketService

async def get_s3_client():
    session = aioboto3.Session()
    async with session.resource(
        "s3",
        endpoint_url="http://localhost:4566",
        aws_access_key_id="test",
        aws_secret_access_key="test"
    ) as s3_resource:
        return s3_resource

def get_s3_bucket_service(
    s3_resource=Depends(get_s3_client),
) -> S3BucketService:
    return S3BucketService(s3_resource, "pdf-bucket")
