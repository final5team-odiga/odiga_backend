from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func

from app.crud.data.database import get_db
from app.models import Article

router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/country-counts/", summary="나라별 게시글 수 반환")
async def country_counts(db: AsyncSession = Depends(get_db)):
    """전체 게시글에서 여행 국가(travelCountry)별로 게시글 개수를 집계하여 반환합니다."""
    # travelCountry별 count 집계
    result = await db.execute(
        select(
            Article.travelCountry,
            func.count(Article.articleID).label("count")
        ).group_by(Article.travelCountry)
    )
    rows = result.all()

    # JSON 직렬화
    data = [
        {"country": country or "Unknown", "count": count}
        for country, count in rows
    ]

    return JSONResponse(status_code=200, content={"success": True, "data": data})
