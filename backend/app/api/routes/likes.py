### app/routes/likes.py
from fastapi import APIRouter, Request, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.database import get_db
from app.crud.crud import toggle_like
from app.crud.main import get_current_user

router = APIRouter(tags=["likes"])



@app.post("/articles/{article_id}/like/")
async def like_article(
    request: Request,
    article_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    좋아요 토글 (로그인 필요)
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"success": False, "message": "로그인이 필요합니다."}
        )

    result = await toggle_like(db, article_id, user_id)
    # result는 {"success": bool, "liked": bool, "totalLikes": int} 형태라고 가정
    return JSONResponse(status_code=200, content=result)