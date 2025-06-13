# dependencies.py

from fastapi import Request, HTTPException, status

async def get_current_user(request: Request) -> str | None:
    """
    서버의 세션 저장소에서 현재 로그인된 사용자의 ID를 가져옵니다.
    로그인하지 않았다면 None을 반환합니다.
    """
    user_id = request.session.get("user")
    
    # 디버깅을 위한 로그 추가
    if user_id:
        print(f"✅ [Auth] User found in session: {user_id}")
    else:
        print("⚠️  [Auth] No user found in session. Request is anonymous.")
        
    return user_id

async def require_auth(request: Request) -> str:
    """
    사용자가 로그인했는지 확인하는 핵심 의존성 함수입니다.
    이 함수는 인증이 필요한 모든 API 엔드포인트에서 사용됩니다.
    로그인하지 않았다면 401 Unauthorized 에러를 발생시킵니다.
    """
    user_id = await get_current_user(request)
    
    if not user_id:
        # 세션에 사용자 정보가 없으면, 인증 실패로 간주합니다.
        print("❌ [Auth] Authentication required, but request is anonymous. Raising 401.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인이 필요합니다."
        )
    
    # 인증 성공 시, 사용자 ID를 API 함수로 전달합니다.
    print(f"✅ [Auth] Authentication successful for user: {user_id}")
    return user_id
