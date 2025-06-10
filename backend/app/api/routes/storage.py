### app/routes/storage.py

from fastapi import APIRouter, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List

from app.crud.azure_utils import (
    upload_image_if_not_exists,
    delete_image,
    list_images,
    generate_blob_sas_url,
    list_output_files,
    upload_output_file,
    upload_interview_result,
    list_text_files
)
from app.main import get_current_user

router = APIRouter(tags=["storage"])

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