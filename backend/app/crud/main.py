import os
import logging
import tempfile
import uuid

from fastapi import (
    FastAPI,
    Request,
    Form,
    HTTPException,
    Depends,
    UploadFile,
    File,
    status
)
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy import delete, func, update
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from pydantic import EmailStr
from typing import List

from passlib.context import CryptContext

from app.database import get_db, create_tables
from app.models import User, Article, Comment, Like
from app.schemas import (
    UserCreate,
    ArticleCreate,
    ArticleUpdate,
    CommentCreate,
    CommentUpdate,
    LikeCreate
)
from app.crud import (
    create_user,
    create_article,
    update_article,
    delete_article,
    create_comment,
    update_comment,
    delete_comment,
    toggle_like,
    check_user_liked
)
from app.stt import transcribe_audio
from app.tts import lan_det, request_tts
from app.azure_utils import (
    upload_image_if_not_exists,
    delete_image,
    list_images,
    generate_blob_sas_url,
    list_output_files,
    upload_output_file,
    is_image_safe_for_upload,
    upload_profile_image,
    delete_interview_result
)
from azure.core.exceptions import ResourceNotFoundError

# ---------------------------------------------------
# 환경 변수 로드 및 기본 설정
# ---------------------------------------------------
load_dotenv()
SPEECH_SERVICE_KEY = os.getenv("SPEECH_SERVICE_KEY")
SPEECH_REGION      = os.getenv("SPEECH_REGION")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="CRUD & STT Web App (JSON API)")

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 미들웨어
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY)

# Static 파일 경로 (프로필 이미지 업로드 등)
if not os.path.isdir("static"):
    os.makedirs("static/profiles", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------
# 앱 시작 시 DB 테이블 생성
# ---------------------------------------------------
@app.on_event("startup")
async def on_startup():
    await create_tables()


# ---------------------------------------------------
# 유틸: 현재 로그인된 사용자 가져오기
# ---------------------------------------------------
async def get_current_user(request: Request) -> str | None:
    return request.session.get("user")


# ---------------------------------------------------
# 인증: 회원가입 / 로그인 / 로그아웃
# ---------------------------------------------------


@app.get("/check_userid/", summary="userID 중복 확인")
async def check_userid(userID: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.userID == userID))
    exists = result.scalar_one_or_none() is not None
    return JSONResponse(status_code=200, content={"available": not exists})


@app.post("/signup/", status_code=status.HTTP_201_CREATED)
async def signup(
    userID: str = Form(...),
    userName: str = Form(...),
    password: str = Form(...),
    userEmail: EmailStr = Form(...),
    userCountry: str = Form(...),
    userLanguage: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    # 이메일 중복 확인만 수행
    result_email = await db.execute(select(User).where(User.userEmail == userEmail))
    if result_email.scalar_one_or_none():
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "이미 존재하는 이메일입니다."}
        )

    # 비밀번호 해싱
    hashed_pw = pwd_context.hash(password)
    user_in = UserCreate(
        userID=userID,
        userName=userName,
        userPasswordHash=hashed_pw,
        userEmail=userEmail,
        userCountry=userCountry,
        userLanguage=userLanguage,
    )

    try:
        await create_user(db, user_in)
    except IntegrityError:
        # userID 중복인 경우
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "이미 존재하는 사용자ID입니다."}
        )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"success": True, "message": "회원가입 성공"}
    )

@app.post("/login/")
async def login(
    request: Request,
    userID: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.userID == userID))
    user = result.scalars().first()
    if not user or not pwd_context.verify(password, user.userPasswordHash):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "아이디 또는 비밀번호가 잘못되었습니다."}
        )

    request.session["user"] = user.userID
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"success": True, "message": "로그인 성공", "userID": user.userID}
    )


@app.post("/logout/")
async def logout(request: Request):
    request.session.clear()
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"success": True, "message": "로그아웃 성공"}
    )


# ---------------------------------------------------
# 게시판 CRUD (Articles / Comments / Likes)
# ---------------------------------------------------

@app.get("/articles/")
async def list_articles(db: AsyncSession = Depends(get_db)):
    """
    모든 글 목록을 최신 순서로 반환
    """
    result = await db.execute(select(Article).order_by(Article.createdAt.desc()))
    articles = result.scalars().all()

    # JSON 직렬화용: Pydantic Schema를 따로 쓰지 않고, 간단히 dict 변환
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


@app.get("/articles/{article_id}")
async def article_detail(
    article_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    특정 글의 상세 정보 + 댓글 목록 반환
    """
    result = await db.execute(select(Article).where(Article.articleID == article_id))
    article = result.scalars().first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # 조회수 1 증가 (view_count를 1 올리고 DB에 반영)
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


@app.post("/articles/")
async def create_article_endpoint(
    request: Request,
    articleTitle: str = Form(...),
    imageURL: str = Form(None),
    content: str = Form(...),
    travelCountry: str = Form(...),
    travelCity: str = Form(...),
    shareLink: str = Form(None),
    price: float = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    새 글 생성 (로그인 필요)
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"success": False, "message": "로그인이 필요합니다."}
        )

    article_in = ArticleCreate(
        articleTitle=articleTitle,
        articleAuthor=user_id,
        content = content,
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


@app.put("/articles/{article_id}")
async def edit_article(
    request: Request,
    article_id: str,
    articleTitle: str = Form(None),
    content: str = Form(None),
    imageURL: str = Form(None),
    travelCountry: str = Form(None),
    travelCity: str = Form(None),
    shareLink: str = Form(None),
    price: float = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    글 수정 (로그인 & 작성자 확인)
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"success": False, "message": "로그인이 필요합니다."}
        )

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


@app.delete("/articles/{article_id}")
async def delete_article_endpoint(
    request: Request,
    article_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    글 삭제 (로그인 & 작성자 확인)
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"success": False, "message": "로그인이 필요합니다."}
        )

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


# ---------------------------------------------------
# 마이페이지 / 프로필 / 회원 탈퇴
# ---------------------------------------------------

@app.get("/mypage/")
async def mypage(request: Request, db: AsyncSession = Depends(get_db)):
    """
    마이페이지: 현재 로그인된 사용자의 정보 + 작성한 게시글 목록 반환
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"success": False, "message": "로그인이 필요합니다."}
        )

    # 사용자 정보 로드
    result_user = await db.execute(select(User).where(User.userID == user_id))
    user = result_user.scalars().first()
    if not user:
        request.session.clear()
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
        "content": art.content,  
        "profileImage": user.profileImage,
        "view_count": art.view_count,
        "createdAt": user.createdAt.isoformat(),
        "updatedAt": user.modifiedAt.isoformat() if user.modifiedAt else None,
        "myArticles": article_list
    }

    return JSONResponse(status_code=200, content={"success": True, "user": user_data})


@app.put("/profile/")
async def edit_profile(
    request: Request,
    userName: str = Form(...),
    userEmail: str = Form(...),
    userCountry: str = Form(None),
    userLanguage: str = Form(None),
    password: str = Form(None),
    profile_image: UploadFile = File(None),
    db: AsyncSession = Depends(get_db),
):
    """
    프로필 수정 (로그인 필요)
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"success": False, "message": "로그인이 필요합니다."}
        )

    # DB에서 사용자 로드
    result = await db.execute(select(User).where(User.userID == user_id))
    user = result.scalars().first()
    if not user:
        request.session.clear()
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
    .execution_options(synchronize_session="fetch"))
    # 프로필 이미지가 업로드되었다면 저장하고 URL 업데이트
    # if profile_image:
    #     ext = os.path.splitext(profile_image.filename)[1]
    #     save_dir = "static/profiles"
    #     os.makedirs(save_dir, exist_ok=True)
    #     save_path = f"{save_dir}/{user_id}{ext}"
    #     content = await profile_image.read()
    #     with open(save_path, "wb") as f:
    #         f.write(content)
    #     update_data["profileImage"] = f"/static/profiles/{user_id}{ext}"

    # await db.execute(
    #     (select(User))
    #     .where(User.userID == user_id)
    #     .execution_options(synchronize_session="fetch")
    #     .update(update_data)
    # )
    await db.commit()

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"success": True, "message": "프로필 수정 성공"}
    )


@app.delete("/delete_account/")
async def delete_account(request: Request, db: AsyncSession = Depends(get_db)):
    """
    회원 탈퇴 (로그인 필요)
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"success": False, "message": "로그인이 필요합니다."}
        )

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


# ---------------------------------------------------
# STT 엔드포인트 (JSON)
# ---------------------------------------------------

@app.post("/transcribe/")
async def transcribe(
    request: Request,
    audio_file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Multipart-form-data로 받은 음성 파일을 STT 처리하고,
    JSON(정상 응답: { success, detected_language, transcription }) 형태로 반환.
    """
    try:
        # 로그인 체크
        user_id = await get_current_user(request)
        if not user_id:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"success": False, "message": "Login required"}
            )

        # 업로드 파일 검사
        if not audio_file.filename:
            logger.error("No filename provided")
            raise HTTPException(status_code=400, detail="No file uploaded")

        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        allowed_formats = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.webm', '.mp4']
        if file_ext not in allowed_formats:
            logger.error(f"Unsupported file format: {file_ext}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(allowed_formats)}"
            )

        content = await audio_file.read()
        file_size = len(content)
        logger.info(f"Received audio file: {audio_file.filename} ({file_size} bytes)")

        if file_size > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 100MB)")
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        logger.info(f"Saved temporary file: {temp_path}")

        try:
            # STT 수행
            logger.info("Starting STT processing...")
            result = transcribe_audio(
                filepath=temp_path,
                key=SPEECH_SERVICE_KEY,
                region=SPEECH_REGION
            )
            logger.info("STT processing completed successfully")

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "success": True,
                    "detected_language": result.get("detected_language"),
                    "transcription": result.get("transcription", "")
                }
            )
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.info(f"Cleaned up temporary file: {temp_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file: {cleanup_error}")

    except HTTPException:
        # HTTPException은 그대로 상위로 전달되어 JSON으로 응답됨
        raise

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"success": False, "message": f"Audio file not found: {str(e)}"}
        )

    except ValueError as e:
        logger.error(f"Audio processing error: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "message": str(e), "error_details": "Audio format conversion failed"}
        )

    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        msg = str(e)
        if "FFmpeg" in msg:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "message": "Audio conversion service unavailable",
                    "error_details": "FFmpeg이 필요합니다. 관리자에게 문의하세요."
                }
            )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": f"Runtime error: {msg}"}
        )

    except Exception as e:
        logger.error(f"Unexpected transcription error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": "Transcription service temporarily unavailable",
                "error_details": str(e)
            }
        )


# ---------------------------------------------------
# TTS 엔드포인트 (JSON)
# ---------------------------------------------------

@app.post("/tts/")
async def tts_api(
    text_input: str = Form(...)
):
    """
    text_input(폼 필드) → 언어 감지 → TTS 음성(bytes) → Base64 인코딩된 Data URI 반환
    """
    # 언어 감지
    lang_code = lan_det(text_input)
    if not lang_code:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "message": "Language detection failed"}
        )

    # TTS 생성
    audio_bytes = request_tts(text_input, lang_code)
    if not audio_bytes:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": "TTS generation failed"}
        )

    # Base64 인코딩
    import base64
    b64_str = base64.b64encode(audio_bytes).decode("utf-8")
    data_uri = f"data:audio/mpeg;base64,{b64_str}"

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success": True,
            "language": lang_code,
            "audio_data_uri": data_uri
        }
    )


@app.get("/tts-info/")
async def tts_info():
    """
    클라이언트에게 TTS API 사용법을 JSON으로 안내
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "description": "POST /tts/ 에 폼 필드 'text_input'을 보내면, Base64 인코딩된 오디오 Data URI를 반환합니다.",
            "method": "POST",
            "endpoint": "/tts/",
            "form_fields": {
                "text_input": "TTS로 변환할 문자열"
            },
            "response_example": {
                "success": True,
                "language": "<감지된 언어 코드>",
                "audio_data_uri": "data:audio/mpeg;base64,...."
            }
        }
    )

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


# ---------------------------------------------------
# 블롭 스토리지 업로드 조회 삭제 엔드포인트 (JSON)
# ---------------------------------------------------


@app.get("/images/")
async def list_user_images(request: Request, magazine_id: str):
    """
    Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/images 폴더 속 이미지 조회
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"success": False, "message": "Login required"})

    image_names = list_images(user_id, magazine_id)
    image_urls = [generate_blob_sas_url(user_id, magazine_id, "images", name) for name in image_names]


    return JSONResponse(
        status_code=200,
        content={"success": True, "images": [{"name": n, "url": u} for n, u in zip(image_names, image_urls)]}
    )


@app.post("/images/upload/")
async def upload_user_images(request: Request, magazine_id: str = Form(...), files: List[UploadFile] = File(...)):
    """
    Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/images 폴더에 이미지 업로드
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"success": False, "message": "Login required"})

    uploaded = []
    skipped = []

    for file in files:
        try:
            content = await file.read()

            # The updated function now handles safety check, processing, and naming internally
            success, final_filename = upload_image_if_not_exists(user_id, magazine_id, file.filename, content)
            
            if success:
                uploaded.append({
                    "original_filename": file.filename,
                    "stored_filename": final_filename
                })
            else:
                skipped.append({"filename": file.filename, "reason": "Upload failed"})
                
        except ValueError as e:
            # Handle validation errors (unsupported format, content safety, etc.)
            skipped.append({"filename": file.filename, "reason": str(e)})
        except Exception as e:
            # Handle other errors
            skipped.append({"filename": file.filename, "reason": f"Upload error: {str(e)}"})

    return JSONResponse(
        status_code=207,
        content={
            "success": True,
            "uploaded": uploaded,
            "skipped": skipped,
            "message": f"{len(uploaded)} uploaded, {len(skipped)} skipped."
        }
    )


@app.delete("/images/delete/")
async def delete_user_image(request: Request, magazine_id: str = Form(...), filename: str = Form(...)):
    """
    Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/images 폴더 속 이미지 삭제
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"success": False, "message": "Login required"})

    delete_image(user_id, magazine_id, filename)
    return JSONResponse(status_code=200, content={"success": True, "message": "Image deleted successfully."})


@app.get("/outputs/list/")
async def list_outputs(request: Request, magazine_id: str):
    """
    Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/outputs 폴더 속 파일 목록 조회
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"success": False, "message": "Login required"})

    files = list_output_files(user_id, magazine_id)
    return JSONResponse(status_code=200, content={"success": True, "files": files})


@app.get("/download-output/")
async def download_output_file(request: Request, filename: str, magazine_id: str):
    """
    Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/outputs 폴더 속 파일 다운로드
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"success": False, "message": "Login required"})

    try:
        download_url = generate_blob_sas_url(user_id, magazine_id, "outputs", filename, expiry_minutes=60)
        return JSONResponse(
            status_code=200,
            content={"success": True, "download_url": download_url, "filename": filename}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": "Failed to generate download URL", "error": str(e)})


@app.post("/outputs/upload/")
async def upload_output_file_endpoint(request: Request, magazine_id: str = Form(...), file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """
    Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/outputs 폴더 속 파일 업로드
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"success": False, "message": "Login required"})

    content = await file.read()

    # Upload the PDF to Azure Blob Storage
    upload_output_file(user_id, magazine_id, file.filename, content)

    # Generate SAS URL
    pdf_url = generate_blob_sas_url(user_id, magazine_id, "outputs", file.filename, expiry_minutes=60)

    # Save SAS URL to User.outputPdf
    await db.execute(
        update(User)
        .where(User.userID == user_id)
        .values(outputPdf=pdf_url)
        .execution_options(synchronize_session="fetch")
    )
    await db.commit()

    return JSONResponse(
        status_code=201,
        content={"success": True, "message": f"Uploaded '{file.filename}' and saved URL to user profile", "pdf_url": pdf_url}
    )


@app.post("/texts/upload/")
async def upload_interview_text(request: Request, magazine_id: str = Form(...), text: str = Form(...)):
    """
    Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/texts 폴더 속 텍스트 파일 업로드
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(status_code=401, content={"success": False, "message": "Login required"})
    
    from app.azure_utils import upload_interview_result
    blob_path = upload_interview_result(user_id, magazine_id, text.encode("utf-8"))
    
    # Extract the actual filename from the blob path
    final_filename = blob_path.split("/")[-1]
    
    return JSONResponse(status_code=201, content={
        "success": True, 
        "message": f"Uploaded '{final_filename}'",
        "filename": final_filename
    })


@app.delete("/texts/delete/")
async def delete_interview_text(request: Request, magazine_id: str = Form(...), filename: str = Form(...)):
    """
    Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/texts 폴더 속 텍스트 파일 삭제
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(status_code=401, content={"success": False, "message": "Login required"})

    try:
        delete_interview_result(user_id, magazine_id, filename)
        return JSONResponse(status_code=200, content={"success": True})
    except ResourceNotFoundError:
        return JSONResponse(status_code=404, content={"success": False, "message": "File not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


@app.get("/texts/list/")
async def list_interview_texts(request: Request, magazine_id: str):
    """
    Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/texts 폴더 속 텍스트 파일 조회
    """
    user_id = await get_current_user(request)
    if not user_id:
        return JSONResponse(status_code=401, content={"success": False, "message": "Login required"})

    from app.azure_utils import list_text_files
    files = list_text_files(user_id, magazine_id)
    return JSONResponse(status_code=200, content={"success": True, "files": files})
