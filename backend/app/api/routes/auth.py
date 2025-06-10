from fastapi import APIRouter, Request, Form, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError
from pydantic import EmailStr
from passlib.context import CryptContext

from app.crud.data.database import get_db
from app.models import User
from app.crud.utils.schemas import UserCreate
from app.crud.crud import create_user

router = APIRouter(prefix="/auth", tags=["authentication"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.get("/check_userid/", summary="userID 중복 확인")
async def check_userid(userID: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.userID == userID))
    exists = result.scalar_one_or_none() is not None
    return JSONResponse(status_code=200, content={"available": not exists})

@router.post("/signup/", status_code=status.HTTP_201_CREATED)
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
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "이미 존재하는 사용자ID입니다."}
        )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"success": True, "message": "회원가입 성공"}
    )

@router.post("/login/")
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

@router.post("/logout/")
async def logout(request: Request):
    request.session.clear()
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"success": True, "message": "로그아웃 성공"}
    )
