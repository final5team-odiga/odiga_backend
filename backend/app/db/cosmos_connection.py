import os
from azure.cosmos import CosmosClient
from dotenv import load_dotenv


load_dotenv()

# 환경변수에서 값 불러오기
COSMOS_ENDPOINT = os.getenv("AZURE_COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("AZURE_COSMOS_KEY")
DATABASE_NAME = os.getenv("DATABASE_NAME")
MAGAZINE_CONTAINER = os.getenv("MAGAZINE_CONTAINER")
IMAGE_CONTAINER = os.getenv("IMAGE_CONTAINER")
LOGGING_CONTAINER = os.getenv("LOGGING_CONTAINER")
TEMPLATE_CONTAINER = os.getenv("TEMPLATE_CONTAINER")
JSX_CONTAINER = os.getenv("JSX_CONTAINER", "jsx_components")  # 새로운 JSX 컴포넌트 컨테이너

# Cosmos 클라이언트 초기화
try:
    client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    database = client.get_database_client(DATABASE_NAME)
    print(f"✅ Database '{DATABASE_NAME}' 연결 성공")
except Exception as e:
    print("❌ 연결 실패:", str(e))

try:
    magazine_container = database.get_container_client(MAGAZINE_CONTAINER)
    image_container = database.get_container_client(IMAGE_CONTAINER)
    logging_container = database.get_container_client(LOGGING_CONTAINER)
    template_container = database.get_container_client(TEMPLATE_CONTAINER)
    jsx_container = database.get_container_client(JSX_CONTAINER)
    print("✅ 모든 컨테이너 연결 성공")
except Exception as e:
    print("❌ 컨테이너 연결 실패:", str(e))




