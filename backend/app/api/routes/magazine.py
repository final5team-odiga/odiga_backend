from fastapi import APIRouter, Request, Form, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import os

from app.crud.data.database import get_db
from agents.system_coordinator import SystemCoordinator
from utils.log.hybridlogging import get_hybrid_logger
from api.dependencies import require_auth

router = APIRouter(prefix="/magazine", tags=["magazine"])
logger = get_hybrid_logger("MagazineAPI")

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
        
        # ✅ user_id와 magazine_id를 사용하여 SystemCoordinator 초기화
        system_coordinator = SystemCoordinator(user_id=user_id, magazine_id=magazine_id)
        
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
        
        if pdf_result.get("success"):
            logger.info(f"✅ 전체 프로세스 완료 - PDF: {pdf_result.get('output_path')}")
        else:
            logger.warning(f"⚠️ 매거진 생성은 완료되었으나 PDF 생성 실패: {pdf_result.get('message')}")
        
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

@router.post("/generate-async/")
async def generate_magazine_async(
    background_tasks: BackgroundTasks,
    magazine_id: str = Form(...),
    user_input: str = Form(None),
    image_folder: str = Form(None),
    generate_pdf: bool = Form(True),
    user_id: str = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """비동기 매거진 생성 (백그라운드 작업)"""
    
    async def background_magazine_generation():
        """백그라운드에서 실행될 매거진 생성 함수"""
        try:
            logger.info(f"백그라운드 매거진 생성 시작 - 사용자: {user_id}, 매거진 ID: {magazine_id}")
            
            system_coordinator = SystemCoordinator(user_id=user_id, magazine_id=magazine_id)
            
            final_result = await system_coordinator.coordinate_complete_magazine_generation(
                user_input=user_input,
                image_folder=image_folder,
                generate_pdf=generate_pdf
            )
            
            if "error" not in final_result:
                logger.info(f"백그라운드 매거진 생성 완료 - 사용자: {user_id}, 매거진 ID: {magazine_id}")
            else:
                logger.error(f"백그라운드 매거진 생성 실패 - 사용자: {user_id}, 오류: {final_result['error']}")
                
        except Exception as e:
            logger.error(f"백그라운드 매거진 생성 중 오류: {e}")
    
    # 백그라운드 작업 추가
    background_tasks.add_task(background_magazine_generation)
    
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "success": True,
            "message": "매거진 생성이 백그라운드에서 시작되었습니다.",
            "user_id": user_id,
            "magazine_id": magazine_id,
            "status": "processing"
        }
    )

@router.get("/status/{magazine_id}")
async def get_magazine_status(
    magazine_id: str,
    user_id: str = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """매거진 생성 상태 확인"""
    try:
        # MagazineDBUtils를 사용하여 상태 확인
        from db.magazine_db_utils import MagazineDBUtils
        
        magazine_data = await MagazineDBUtils.get_magazine_content(magazine_id)
        
        if not magazine_data:
            raise HTTPException(
                status_code=404,
                detail="매거진을 찾을 수 없습니다."
            )
        
        # 권한 확인
        if magazine_data.get("user_id") != user_id:
            raise HTTPException(
                status_code=403,
                detail="해당 매거진에 접근할 권한이 없습니다."
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "magazine_id": magazine_id,
                "status": magazine_data.get("status", "unknown"),
                "created_at": magazine_data.get("created_at"),
                "error": magazine_data.get("error")
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"매거진 상태 확인 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "상태 확인 중 오류가 발생했습니다.",
                "error": str(e)
            }
        )
