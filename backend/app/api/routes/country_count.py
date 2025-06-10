### app/routes/articles.py
from fastapi import APIRouter, Request, Form, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.crud.database import get_db
from app.crud.schemas import ArticleCreate, ArticleUpdate
from app.crud.crud import create_article, update_article, delete_article
from app.crud.models import Article
from app.crud.main import get_current_user

######### 나라별 article 개수 ##########
@app.get("/articles/country-counts/", summary="나라별 게시글 수 반환")
async def country_counts(db: AsyncSession = Depends(get_db)):
    """
    전체 게시글에서 여행 국가(travelCountry)별로
    게시글 개수를 집계하여 반환합니다.
    """
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