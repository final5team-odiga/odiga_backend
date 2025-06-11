from fastapi import APIRouter, Request, Form, Depends, UploadFile, File, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete, update
from passlib.context import CryptContext

from ...crud.data.database import get_db
from ...crud.models.models import User, Article, Comment, Like
from ...crud.utils.azure_utils import upload_profile_image, is_image_safe_for_upload
from ..dependencies import require_auth

router = APIRouter(prefix="/profile", tags=["profiles"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.get("/mypage/")
async def mypage(
    user_id: str = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """마이페이지: 현재 로그인된 사용자의 정보 + 작성한 게시글 목록 반환"""
    # 사용자 정보 로드
    result_user = await db.execute(select(User).where(User.userID == user_id))
    user = result_user.scalars().first()
    if not user:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"success": False, "message": "유효하지 않은 세션입니다. 다시 로그인해주세요."}
        )

    # 본인이 쓴 글 로드
    result_articles = await db.execute(
        select(Article)
        .where(Article.articleAuthor == user_id)
        .order_by(Article.createdAt.desc())
    )
    my_articles = result_articles.scalars().all()

    article_list = []
    for art in my_articles:
        article_list.append({
            "articleID": art.articleID,
            "articleTitle": art.articleTitle,
            "createdAt": art.createdAt.isoformat(),
            "updatedAt": art.modifiedAt.isoformat() if art.modifiedAt else None
        })

    user_data = {
        "userID": user.userID,
        "userName": user.userName,
        "userEmail": user.userEmail,
        "userCountry": user.userCountry,
        "userLanguage": user.userLanguage,
        "profileImage": user.profileImage,
        "createdAt": user.createdAt.isoformat(),
        "updatedAt": user.modifiedAt.isoformat() if user.modifiedAt else None,
        "myArticles": article_list
    }

    return JSONResponse(status_code=200, content={"success": True, "user": user_data})

@router.put("/")
async def edit_profile(
    userName: str = Form(...),
    userEmail: str = Form(...),
    userCountry: str = Form(None),
    userLanguage: str = Form(None),
    password: str = Form(None),
    profile_image: UploadFile = File(None),
    user_id: str = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """프로필 수정 (로그인 필요)"""
    # DB에서 사용자 로드
    result = await db.execute(select(User).where(User.userID == user_id))
    user = result.scalars().first()
    if not user:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"success": False, "message": "유효하지 않은 세션입니다. 다시 로그인해주세요."}
        )

    update_data = {
        "userName": userName,
        "userEmail": userEmail,
        "userCountry": userCountry,
        "userLanguage": userLanguage,
    }
    if password:
        update_data["userPasswordHash"] = pwd_context.hash(password)

    # 프로필 이미지가 업로드되었다면 저장하고 URL 업데이트
    if profile_image:
        content = await profile_image.read()

        # Optional: Safety check
        is_safe, analysis = is_image_safe_for_upload(content, profile_image.filename)
        if not is_safe:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "message": "부적절한 이미지로 판단되어 업로드가 차단되었습니다.", "details": analysis}
            )
        
        # Upload and get SAS-protected URL
        image_url = upload_profile_image(user_id, content)
        update_data["profileImage"] = image_url

    await db.execute(
        update(User)
        .where(User.userID == user_id)
        .values(**update_data)
        .execution_options(synchronize_session="fetch")
    )
    await db.commit()

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"success": True, "message": "프로필 수정 성공"}
    )

@router.delete("/delete_account/")
async def delete_account(
    request: Request,
    user_id: str = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """회원 탈퇴 (로그인 필요)"""
    # 댓글, 좋아요, 게시글, 사용자 순서로 삭제
    await db.execute(delete(Comment).where(Comment.commentAuthor == user_id))
    await db.execute(delete(Like).where(Like.userID == user_id))
    await db.execute(delete(Article).where(Article.articleAuthor == user_id))
    await db.execute(delete(User).where(User.userID == user_id))
    await db.commit()

    request.session.clear()
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"success": True, "message": "회원 탈퇴 성공"}
    )
