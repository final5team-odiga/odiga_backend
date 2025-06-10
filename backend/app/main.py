import os
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from app.crud.data.database import create_tables
from api import create_api_router
from api.middleware import setup_middleware

# 환경 변수 로드 및 기본 설정
load_dotenv()

# ✅ Windows 환경 설정 추가
if os.name == 'nt':  # Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CRUD & Magazine Generation API")

# 미들웨어 설정
setup_middleware(app)

# Static 파일 경로
if not os.path.isdir("static"):
    os.makedirs("static/profiles", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# API 라우터 등록
api_router = create_api_router()
app.include_router(api_router)

# ✅ 시스템 초기화
@app.on_event("startup")
async def on_startup():
    """애플리케이션 시작 시 초기화"""
    await create_tables()
    
    # ✅ 매거진 생성 시스템 초기화
    try:
        from utils.log.hybridlogging import get_hybrid_logger
        startup_logger = get_hybrid_logger("Startup")
        startup_logger.info("=== 매거진 생성 시스템 초기화 완료 ===")
    except Exception as e:
        logger.warning(f"매거진 시스템 초기화 중 경고: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
