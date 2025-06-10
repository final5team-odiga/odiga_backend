### app/routes/comments.py
from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.crud.database import get_db
from app.crud.schemas import CommentCreate, CommentUpdate
from app.crud.crud import create_comment, update_comment, delete_comment
from app.crud.models import Article, Comment
from app.crud.main import get_current_user

router = APIRouter(tags=["comments"])


@app.post("/articles/{article_id}/comments/")
async def post_comment(
    request: Request,
    article_id: str,
    content: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """
    댓글 생성 (로그인 필요)
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"success": False, "message": "로그인이 필요합니다."}
        )

    # 게시글 존재 여부 확인
    result = await db.execute(select(Article).where(Article.articleID == article_id))
    if not result.scalars().first():
        raise HTTPException(status_code=404, detail="Article not found")

    await create_comment(db, CommentCreate(
        articleID=article_id,
        commentAuthor=user_id,
        content=content
    ))
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"success": True, "message": "댓글 작성 성공"}
    )


@app.put("/articles/{article_id}/comments/{comment_id}")
async def edit_comment(
    request: Request,
    article_id: str,
    comment_id: int,
    content: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """
    댓글 수정 (로그인 & 댓글 작성자 확인)
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"success": False, "message": "로그인이 필요합니다."}
        )

    # 게시글 & 댓글 존재 여부 확인
    art_res = await db.execute(select(Article).where(Article.articleID == article_id))
    if not art_res.scalars().first():
        raise HTTPException(status_code=404, detail="Article not found")

    com_res = await db.execute(select(Comment).where(Comment.commentID == comment_id))
    comment = com_res.scalars().first()
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    if comment.commentAuthor != user_id:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"success": False, "message": "해당 댓글을 수정할 권한이 없습니다."}
        )

    await update_comment(db, comment_id, CommentUpdate(content=content))
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"success": True, "message": "댓글 수정 성공"}
    )


@app.delete("/articles/{article_id}/comments/{comment_id}")
async def delete_comment_endpoint(
    request: Request,
    article_id: str,
    comment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    댓글 삭제 (로그인 필요)
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"success": False, "message": "로그인이 필요합니다."}
        )

    # 게시글 & 댓글 존재 여부 확인
    art_res = await db.execute(select(Article).where(Article.articleID == article_id))
    if not art_res.scalars().first():
        raise HTTPException(status_code=404, detail="Article not found")

    com_res = await db.execute(select(Comment).where(Comment.commentID == comment_id))
    comment = com_res.scalars().first()
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    if comment.commentAuthor != user_id:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"success": False, "message": "해당 댓글을 삭제할 권한이 없습니다."}
        )

    await delete_comment(db, comment_id)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"success": True, "message": "댓글 삭제 성공"}
    )