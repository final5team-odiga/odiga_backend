from fastapi import APIRouter, Depends, Request, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ...crud.data.database import get_db
from ...crud.models.models import Daily
from ...crud.utils.schemas import DailyCreate, DailyRead

from datetime import datetime

router = APIRouter()

# 유틸: 현재 로그인된 사용자 가져오기 (main.py와 동일)
async def get_current_user(request: Request) -> str | None:
    return request.session.get("user")

# Daily 레코드 생성 함수
async def create_daily(db: AsyncSession, user_id: str, daily_data: DailyCreate):
    db_daily = Daily(
        userID=user_id,
        date=daily_data.date,
        season=daily_data.season,
        weather=daily_data.weather,
        temperature=daily_data.temperature,
        mood=daily_data.mood,
        country=daily_data.country
    )
    db.add(db_daily)
    await db.commit()
    await db.refresh(db_daily)
    return db_daily

# 내 캘린더 기록 모두 조회 함수
async def get_dailies_for_user(db: AsyncSession, user_id: str):
    result = await db.execute(
        select(Daily).where(Daily.userID == user_id).order_by(Daily.date.asc())
    )
    return result.scalars().all()

@router.post("/mypage/daily/", response_model=DailyRead)
async def add_daily(
    request: Request,
    date: str = Form(...),
    season: str = Form(...),
    weather: str = Form(...),
    temperature: float = Form(...),
    mood: str = Form(None),
    country: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    user_id = await get_current_user(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    daily_create = DailyCreate(
        date=datetime.fromisoformat(date),
        season=season,
        weather=weather,
        temperature=temperature,
        mood=mood,
        country=country
    )
    daily = await create_daily(db, user_id, daily_create)
    return daily

@router.get("/mypage/daily/", response_model=list[DailyRead])
async def daily_calendar(request: Request, db: AsyncSession = Depends(get_db)):
    user_id = await get_current_user(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    dailies = await get_dailies_for_user(db, user_id)
    return dailies
