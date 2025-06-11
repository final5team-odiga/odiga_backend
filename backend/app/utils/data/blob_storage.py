import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContentSettings
from pathlib import Path


load_dotenv()  # Add override=True

class BlobStorageManager:
    def __init__(self, user_id: str, magazine_id: str):
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in .env")

        self.container_name = "user"  # Changed from environment variable
        self.user_id = user_id        # Added
        self.magazine_id = magazine_id # Added
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)
    
    def get_images(self):
        """images 폴더에서 모든 이미지 가져오기"""
        # Changed prefix to include user_id and magazine_id
        prefix = f"{self.user_id}/magazine/{self.magazine_id}/images/"
        print(f"DEBUG: Listing blobs with prefix: {prefix}")
        blob_list = self.container_client.list_blobs(name_starts_with=prefix)
        return sorted([blob for blob in blob_list], key=lambda x: x.name)
      
    def get_texts(self):
        """texts 폴더에서 모든 텍스트 파일 가져오기"""
        # Changed prefix to include user_id and magazine_id
        prefix = f"{self.user_id}/magazine/{self.magazine_id}/texts/"
        blob_list = self.container_client.list_blobs(name_starts_with=prefix)
        return sorted([blob for blob in blob_list], key=lambda x: x.name)
    
    def get_image_url(self, blob):
        """이미지 URL 생성"""
        if isinstance(blob, str):
            blob_name = blob
        else:
            blob_name = blob.name
        return f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"
    
    def read_text_file(self, blob):
        """텍스트 파일 내용 읽기"""
        if isinstance(blob, str):
            blob_name = blob
        else:
            blob_name = blob.name
        
        blob_client = self.container_client.get_blob_client(blob_name)
        download_stream = blob_client.download_blob()
        return download_stream.readall().decode('utf-8')
    
    # Helper methods (optional but recommended)
    def build_image_path(self, filename: str) -> str:
        """이미지 파일의 전체 경로 생성"""
        return f"{self.user_id}/magazine/{self.magazine_id}/images/{filename}"
    
    def build_text_path(self, filename: str) -> str:
        """텍스트 파일의 전체 경로 생성"""
        return f"{self.user_id}/magazine/{self.magazine_id}/texts/{filename}"
    
    def save_to_blob(self, content, filename: str, category: str = "texts", content_type="text/plain"):
        """콘텐츠를 Blob Storage에 저장 - category 파라미터 추가"""
        if category == "images":
            blob_name = self.build_image_path(filename)
        elif category == "texts":
            blob_name = self.build_text_path(filename)
        else:
            blob_name = f"{self.user_id}/magazine/{self.magazine_id}/{category}/{filename}"
        
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_client.upload_blob(
            content, 
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type)
        )
        return blob_client.url


