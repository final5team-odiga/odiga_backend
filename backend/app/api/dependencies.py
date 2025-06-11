from fastapi import Request

async def get_current_user(request: Request) -> str | None:
    """현재 로그인된 사용자 가져오기"""
    return request.session.get("user")

async def require_auth(request: Request) -> str:
    """로그인 필수 의존성"""
    user_id = await get_current_user(request)
    if not user_id:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인이 필요합니다."
        )
    return user_id
