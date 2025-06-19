FROM python:3.12.9

# 환경 변수 설정 - 포트 관련 추가
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/app \
    PORT=8000 \
    WEBSITES_PORT=8000 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ make libffi-dev libpq-dev libssl-dev \
    ffmpeg libcairo2-dev libpango1.0-dev libgdk-pixbuf2.0-dev \
    libfontconfig1-dev curl gnupg wget chromium \
    && rm -rf /var/lib/apt/lists/*

# Node.js 설치
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Node.js 의존성 설치
COPY package*.json ./
RUN npm install --production || true

# 프로젝트 전체 복사
COPY . .

# 디렉토리 생성 및 권한 설정 (한 번만)
RUN mkdir -p logs static templates service && \
    useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

USER app

EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
