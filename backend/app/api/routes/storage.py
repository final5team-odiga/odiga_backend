from typing import List
from fastapi import APIRouter, Request, Form, Depends, UploadFile, File, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update
from azure.core.exceptions import ResourceNotFoundError

from ...crud.data.database import get_db
from ...crud.models.models import User
from ...crud.utils.azure_utils import (
    upload_image_if_not_exists,
    delete_image,
    list_images,
    generate_blob_sas_url,
    list_output_files,
    upload_output_file,
    upload_interview_result,
    delete_interview_result,
    list_text_files,
    list_user_folders
)
from ..dependencies import require_auth

router = APIRouter(prefix="/storage", tags=["storage"])

@router.get("/images/")
async def list_user_images(
    magazine_id: str,
    user_id: str = Depends(require_auth)
):
    """Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/images 폴더 속 이미지 조회"""
    image_names = list_images(user_id, magazine_id)
    image_urls = [generate_blob_sas_url(user_id, magazine_id, "images", name) for name in image_names]

    return JSONResponse(
        status_code=200,
        content={"success": True, "images": [{"name": n, "url": u} for n, u in zip(image_names, image_urls)]}
    )

@router.post("/images/upload/")
async def upload_user_images(
    magazine_id: str = Form(...),
    files: List[UploadFile] = File(...),
    user_id: str = Depends(require_auth)
):
    """Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/images 폴더에 이미지 업로드"""
    uploaded = []
    skipped = []

    for file in files:
        try:
            content = await file.read()
            success, final_filename = upload_image_if_not_exists(user_id, magazine_id, file.filename, content)
            
            if success:
                uploaded.append({
                    "original_filename": file.filename,
                    "stored_filename": final_filename
                })
            else:
                skipped.append({"filename": file.filename, "reason": "Upload failed"})
                
        except ValueError as e:
            skipped.append({"filename": file.filename, "reason": str(e)})
        except Exception as e:
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

@router.delete("/images/delete/")
async def delete_user_image(
    magazine_id: str = Form(...),
    filename: str = Form(...),
    user_id: str = Depends(require_auth)
):
    """Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/images 폴더 속 이미지 삭제"""
    delete_image(user_id, magazine_id, filename)
    return JSONResponse(status_code=200, content={"success": True, "message": "Image deleted successfully."})

@router.get("/outputs/list/")
async def list_outputs(
    magazine_id: str,
    user_id: str = Depends(require_auth)
):
    """Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/outputs 폴더 속 파일 목록 조회"""
    files = list_output_files(user_id, magazine_id)
    return JSONResponse(status_code=200, content={"success": True, "files": files})

@router.get("/download-output/")
async def download_output_file(
    filename: str,
    magazine_id: str,
    user_id: str = Depends(require_auth)
):
    """Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/outputs 폴더 속 파일 다운로드"""
    try:
        download_url = generate_blob_sas_url(user_id, magazine_id, "outputs", filename, expiry_minutes=60)
        return JSONResponse(
            status_code=200,
            content={"success": True, "download_url": download_url, "filename": filename}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": "Failed to generate download URL", "error": str(e)})

@router.post("/outputs/upload/")
async def upload_output_file_endpoint(
    magazine_id: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/outputs 폴더 속 파일 업로드"""
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

@router.post("/texts/upload/")
async def upload_interview_text(
    magazine_id: str = Form(...),
    text: str = Form(...),
    user_id: str = Depends(require_auth)
):
    """Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/texts 폴더 속 텍스트 파일 업로드"""
    try:
        blob_path = upload_interview_result(user_id, magazine_id, text.encode("utf-8"))
        final_filename = blob_path.split("/")[-1]
        
        return JSONResponse(status_code=201, content={
            "success": True, 
            "message": f"Uploaded '{final_filename}'",
            "filename": final_filename
        })
    except ValueError as e:
        return JSONResponse(
            status_code=400, 
            content={
                "success": False, 
                "message": "Text content failed safety check",
                "error": str(e)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={
                "success": False, 
                "message": "Upload failed",
                "error": str(e)
            }
        )

@router.delete("/texts/delete/")
async def delete_interview_text(
    magazine_id: str = Form(...),
    filename: str = Form(...),
    user_id: str = Depends(require_auth)
):
    """Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/texts 폴더 속 텍스트 파일 삭제"""
    try:
        delete_interview_result(user_id, magazine_id, filename)
        return JSONResponse(status_code=200, content={"success": True})
    except ResourceNotFoundError:
        return JSONResponse(status_code=404, content={"success": False, "message": "File not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})

@router.get("/texts/list/")
async def list_interview_texts(
    magazine_id: str,
    user_id: str = Depends(require_auth)
):
    """Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/{magazine_id}/texts 폴더 속 텍스트 파일 조회"""
    files = list_text_files(user_id, magazine_id)
    return JSONResponse(status_code=200, content={"success": True, "files": files})


@router.get("/magazines/list/")
async def list_user_magazines(
    user_id: str = Depends(require_auth)
):
    """Azure Blob Storage의 "user" Container 아래 {user_id}/magazine/ 폴더 속 magazine_id 목록 조회"""
    try:
        magazine_folders = list_user_folders(user_id)
        return JSONResponse(
            status_code=200, 
            content={
                "success": True, 
                "magazines": magazine_folders
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={
                "success": False, 
                "message": "Failed to retrieve magazine folders", 
                "error": str(e)
            }
        )