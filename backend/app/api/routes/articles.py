from fastapi import APIRouter, Request, Form, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ...crud.data.database import get_db
from ...crud.models.models import Article
from ...crud.utils.schemas import ArticleCreate, ArticleUpdate
from ...crud.crud import create_article, update_article, delete_article
from ..dependencies import get_current_user, require_auth

from ...crud.models.models import Comment

from ...crud.crud import check_user_liked

router = APIRouter(prefix="/articles", tags=["articles"])

@router.get("/")
async def list_articles(db: AsyncSession = Depends(get_db)):
    """모든 글 목록을 최신 순서로 반환"""
    result = await db.execute(select(Article).order_by(Article.createdAt.desc()))
    articles = result.scalars().all()

    article_list = []
    for art in articles:
        article_list.append({
            "articleID": art.articleID,
            "articleTitle": art.articleTitle,
            "articleAuthor": art.articleAuthor,
            "imageURL": art.imageURL,
            "travelCountry": art.travelCountry,
            "travelCity": art.travelCity,
            "shareLink": art.shareLink,
            "price": art.price,
            "view_count": art.view_count,
            "createdAt": art.createdAt.isoformat(),
            "updatedAt": art.modifiedAt.isoformat() if art.modifiedAt else None
        })

    return JSONResponse(status_code=200, content={"success": True, "articles": article_list})

@router.get("/{article_id}")
async def article_detail(
    article_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """특정 글의 상세 정보 + 댓글 목록 반환"""
    result = await db.execute(select(Article).where(Article.articleID == article_id))
    article = result.scalars().first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # 조회수 1 증가
    article.view_count += 1
    await db.commit()

    # 댓글 로드
    comments_result = await db.execute(
        select(Comment).where(Comment.articleID == article_id).order_by(Comment.createdAt.asc())
    )
    comments = comments_result.scalars().all()

    comment_list = []
    for com in comments:
        comment_list.append({
            "commentID": com.commentID,
            "articleID": com.articleID,
            "commentAuthor": com.commentAuthor,
            "content": com.content,
            "createdAt": com.createdAt.isoformat(),
            "updatedAt": com.modifiedAt.isoformat() if com.modifiedAt else None
        })

    # 로그인 유저가 좋아요를 눌렀는지 여부
    user_id = await get_current_user(request)
    user_liked = False
    if user_id:
        user_liked = await check_user_liked(db, article_id, user_id)

    article_data = {
        "articleID": article.articleID,
        "articleTitle": article.articleTitle,
        "articleAuthor": article.articleAuthor,
        "content": article.content,
        "imageURL": article.imageURL,
        "travelCountry": article.travelCountry,
        "travelCity": article.travelCity,
        "shareLink": article.shareLink,
        "price": article.price,
        "view_count": article.view_count,
        "createdAt": article.createdAt.isoformat(),
        "updatedAt": article.modifiedAt.isoformat() if article.modifiedAt else None,
        "comments": comment_list,
        "likes": article.likes,
        "userLiked": user_liked
    }

    return JSONResponse(status_code=200, content={"success": True, "article": article_data})

@router.post("/")
async def create_article_endpoint(
    articleTitle: str = Form(...),
    imageURL: str = Form(None),
    content: str = Form(...),
    travelCountry: str = Form(...),
    travelCity: str = Form(...),
    shareLink: str = Form(None),
    price: float = Form(None),
    user_id: str = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """새 글 생성 (로그인 필요)"""
    article_in = ArticleCreate(
        articleTitle=articleTitle,
        articleAuthor=user_id,
        content=content,
        imageURL=imageURL,
        travelCountry=travelCountry,
        travelCity=travelCity,
        shareLink=shareLink,
        price=price,
    )
    db_article = await create_article(db, article_in)

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "success": True,
            "message": "게시글 생성 성공",
            "articleID": db_article.articleID
        }
    )

@router.put("/{article_id}")
async def edit_article(
    article_id: str,
    articleTitle: str = Form(None),
    content: str = Form(None),
    imageURL: str = Form(None),
    travelCountry: str = Form(None),
    travelCity: str = Form(None),
    shareLink: str = Form(None),
    price: float = Form(None),
    user_id: str = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """글 수정 (로그인 & 작성자 확인)"""
    result = await db.execute(select(Article).where(Article.articleID == article_id))
    article = result.scalars().first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    if article.articleAuthor != user_id:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"success": False, "message": "해당 글을 수정할 권한이 없습니다."}
        )

    update_in = ArticleUpdate(
        articleTitle=articleTitle,
        content=content,
        imageURL=imageURL,
        travelCountry=travelCountry,
        travelCity=travelCity,
        shareLink=shareLink,
        price=price
    )
    await update_article(db, article_id, update_in)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"success": True, "message": "게시글 수정 성공"}
    )

@router.delete("/{article_id}")
async def delete_article_endpoint(
    article_id: str,
    user_id: str = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """글 삭제 (로그인 & 작성자 확인)"""
    result = await db.execute(select(Article).where(Article.articleID == article_id))
    article = result.scalars().first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    if article.articleAuthor != user_id:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"success": False, "message": "해당 글을 삭제할 권한이 없습니다."}
        )

    await delete_article(db, article_id)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"success": True, "message": "게시글 삭제 성공"}
    )
