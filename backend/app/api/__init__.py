from fastapi import APIRouter
from .routes import auth, articles, comments, profiles, speech, storage, analytics, daily

def create_api_router() -> APIRouter:
    """모든 라우터를 통합하는 API 라우터 생성"""
    api_router = APIRouter()
    
    # 각 라우터 등록
    api_router.include_router(auth.router)
    api_router.include_router(articles.router)
    api_router.include_router(comments.router)
    api_router.include_router(profiles.router)
    api_router.include_router(speech.router)
    api_router.include_router(storage.router)
    api_router.include_router(analytics.router)
    api_router.include_router(magazine.router)
    api_router.include_router(daily.router)
    return api_router