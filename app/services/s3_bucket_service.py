import base64
import pdfplumber
import io

def extract_text_from_pdf(file_path: str) -> str:
    with pdfplumber.open(file_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

class S3BucketService:
    def __init__(self, s3_client, bucket_name):
        self._s3_client = s3_client
        self._bucket_name = bucket_name

    async def upload_file(self, file_name: str, file_content):
        return self._s3_client.put_object(
            Bucket=self._bucket_name,
            Key=file_name,
            Body=file_content
        )
    async def list_files(self):
        response = self._s3_client.list_objects_v2(Bucket=self._bucket_name)
        files = response.get("Contents", [])
        
        return [{"filename": file["Key"], "size": file["Size"]} for file in files]
    
    async def delete_file(self, file_name: str):
        return self._s3_client.delete_object(
            Bucket=self._bucket_name,
            Key=file_name
        )   
    
    async def get_file(self, file_name: str):
        response = self._s3_client.get_object(
            Bucket=self._bucket_name,
            Key=file_name
        )
        
        file_content = response["Body"].read()
        encoded_content = base64.b64encode(file_content).decode("utf-8")  # Encode content in Base64
        metadata = {
            "file_name": file_name,
            "content_type": response.get("ContentType"),
            "content_length": response.get("ContentLength"),
        }
        return {"content": encoded_content, "metadata": metadata}
    
    async def extract_text_from_s3_pdf(self, file_name: str) -> str:
        response = self._s3_client.get_object(
            Bucket=self._bucket_name,
            Key=file_name
        )
        
        file_content = response["Body"].read()
        
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        return text