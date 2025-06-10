from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
import os

def setup_middleware(app: FastAPI):
    """미들웨어 설정"""
    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 세션 미들웨어
    SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY")
    app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY)
