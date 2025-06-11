from fastapi import APIRouter, Request, Form, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import os
import logging

from ...crud.data.database import get_db
from ...agents.system_coordinator import SystemCoordinator
from ...utils.log.hybridlogging import get_hybrid_logger
from ..dependencies import require_auth


router = APIRouter(prefix="/magazine", tags=["magazine"])

# ✅ 안전한 로거 초기화
try:
    logger = get_hybrid_logger("MagazineAPI")
except Exception:
    logger = logging.getLogger("MagazineAPI")

@router.get("/test")
async def test_endpoint():
    """테스트 엔드포인트"""
    return {"message": "Magazine API is working", "status": "ok"}

@router.post("/generate/")
async def generate_magazine(
    magazine_id: str = Form(...),
    user_input: str = Form(None),
    image_folder: str = Form(None),
    generate_pdf: bool = Form(True),
    user_id: str = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """완전 통합 멀티모달 매거진 생성 API"""
    
    try:
        logger.info(f"매거진 생성 요청 - 사용자: {user_id}, 매거진 ID: {magazine_id}")
        
        # ✅ 안전한 SystemCoordinator 초기화
        try:
            system_coordinator = SystemCoordinator(user_id=user_id, magazine_id=magazine_id)
        except Exception as e:
            logger.error(f"SystemCoordinator 초기화 실패: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "message": "시스템 초기화 실패",
                    "error": str(e)
                }
            )
        
        # 매거진 생성 실행
        final_result = await system_coordinator.coordinate_complete_magazine_generation(
            user_input=user_input,
            image_folder=image_folder,
            generate_pdf=generate_pdf
        )
        
        # 결과 처리
        if "error" in final_result:
            logger.error(f"매거진 생성 실패: {final_result['error']}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "message": "매거진 생성 실패",
                    "error": final_result['error'],
                    "magazine_id": final_result.get('magazine_id')
                }
            )
        
        result_data = final_result.get("result", {})
        pdf_result = result_data.get("pdf_generation", {})
        
        response_data = {
            "success": True,
            "message": "매거진 생성 완료",
            "magazine_id": final_result.get("magazine_id"),
            "magazine_title": result_data.get("magazine_title"),
            "magazine_subtitle": result_data.get("magazine_subtitle"),
            "total_images_used": result_data.get("total_images_used", 0),
            "image_placement_success": result_data.get("image_placement_success", False),
            "pdf_generation": {
                "enabled": generate_pdf,
                "success": pdf_result.get("success", False),
                "output_path": pdf_result.get("output_path"),
                "message": pdf_result.get("message")
            }
        }
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=response_data
        )
        
    except Exception as e:
        logger.error(f"매거진 생성 API 오류: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": "서버 내부 오류",
                "error": str(e)
            }
        )
