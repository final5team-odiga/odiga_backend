import os
import logging
import sys
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from backend.app.crud.data.database import create_tables
from backend.app.api import create_api_router
from backend.app.api.middleware import setup_middleware

# 환경 변수 로드 및 기본 설정
load_dotenv()

# Windows 환경 설정 추가
if os.name == 'nt':  # Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # stdout/stderr 재설정
    try:
        import io
        # 기존 stdout을 안전하게 교체
        if hasattr(sys.stdout, 'buffer') and sys.stdout.buffer:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, 
                encoding='utf-8', 
                errors='replace',
                line_buffering=True
            )
        if hasattr(sys.stderr, 'buffer') and sys.stderr.buffer:
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, 
                encoding='utf-8', 
                errors='replace',
                line_buffering=True
            )
    except Exception:
        pass

# ✅ 로깅을 파일 우선으로 설정
try:
    # 로그 디렉토리 생성
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log', encoding='utf-8'),  # 파일 우선
            logging.StreamHandler()  # 콘솔은 보조
        ],
        force=True  # 기존 로깅 설정 강제 재설정
    )
except Exception:
    # 최소한의 로깅 설정
    logging.basicConfig(level=logging.WARNING, force=True)

logger = logging.getLogger(__name__)


app = FastAPI(title="CRUD & Magazine Generation API")

# 미들웨어 설정
setup_middleware(app)

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# API 라우터 등록
api_router = create_api_router()
app.include_router(api_router)

# 루트 경로에서 대시보드 UI 제공
@app.get("/")
async def read_root(request: Request):
    """통합 대시보드 UI 제공"""
    return templates.TemplateResponse("index.html", {"request": request})

# 시스템 초기화
@app.on_event("startup")
async def on_startup():
    """애플리케이션 시작 시 초기화"""
    await create_tables()
    
    # 매거진 생성 시스템 초기화
    try:
        from backend.app.utils.log.hybridlogging import get_hybrid_logger
        startup_logger = get_hybrid_logger("Startup")
        startup_logger.info("=== 매거진 생성 시스템 초기화 완료 ===")
    except Exception as e:
        logger.warning(f"매거진 시스템 초기화 중 경고: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 