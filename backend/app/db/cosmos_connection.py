import os
import json
from datetime import datetime
from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경변수에서 값 불러오기
COSMOS_ENDPOINT = os.getenv("AZURE_COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("AZURE_COSMOS_KEY")
DATABASE_NAME = os.getenv("DATABASE_NAME")
MAGAZINE_CONTAINER = os.getenv("MAGAZINE_CONTAINER")
IMAGE_CONTAINER = os.getenv("IMAGE_CONTAINER")
LOGGING_CONTAINER = os.getenv("LOGGING_CONTAINER")

# Cosmos 클라이언트 초기화
try:
    client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    database = client.get_database_client(DATABASE_NAME)
    print(f"✅ Database '{DATABASE_NAME}' 연결 성공")
except Exception as e:
    print("❌ 연결 실패:", str(e))

magazine_container = database.get_container_client(MAGAZINE_CONTAINER)
image_container = database.get_container_client(IMAGE_CONTAINER)
logging_container = database.get_container_client(LOGGING_CONTAINER)
