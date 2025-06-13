# middleware.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
import os

def setup_middleware(app: FastAPI):
    """미들웨어 설정 (최종 수정안)"""

    # 1. CORS 설정
    # 다른 도메인에서의 API 요청을 허용합니다.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://brave-grass-0233c7200.6.azurestaticapps.net", # Static Web App 도메인
            "http://localhost:3000"  # 로컬 개발 환경용
        ],
        allow_credentials=True,
        allow_methods=["*"], # 모든 HTTP 메서드 허용
        allow_headers=["*"], # 모든 헤더 허용
    )

    # 2. 세션 미들웨어 설정
    # 세션 쿠키의 동작 방식을 제어합니다.
    SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY")
    if not SESSION_SECRET_KEY:
        raise ValueError("SESSION_SECRET_KEY 환경 변수가 설정되지 않았습니다.")

    app.add_middleware(
        SessionMiddleware,
        secret_key=SESSION_SECRET_KEY,
        # ⭐️ 이 세 줄이 문제를 해결하는 핵심입니다.
        same_site="none",                            # 교차 사이트 요청에서도 쿠키를 보낼 수 있도록 허용
        https_only=True,                             # same_site="none"을 위한 필수 보안 설정
        # ⭐️ 백엔드 도메인을 명시하여 쿠키의 유효 범위를 알려줍니다.
        # 주의: 앞에 점(.)을 붙이지 마세요.
        domain="odigawepapp.azurewebsites.net"
    )
