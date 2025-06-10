import os
import tempfile
import logging
from fastapi import APIRouter, Request, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.data.database import get_db
from app.service.stt import transcribe_audio
from app.service.tts import lan_det, request_tts
from api.dependencies import require_auth

router = APIRouter(prefix="/speech", tags=["speech"])
logger = logging.getLogger(__name__)

SPEECH_SERVICE_KEY = os.getenv("SPEECH_SERVICE_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")

@router.post("/transcribe/")
async def transcribe(
    audio_file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Multipart-form-data로 받은 음성 파일을 STT 처리하고, JSON 형태로 반환"""
    try:
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
        raise
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

@router.post("/tts/")
async def tts_api(text_input: str = Form(...)):
    """text_input(폼 필드) → 언어 감지 → TTS 음성(bytes) → Base64 인코딩된 Data URI 반환"""
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

@router.get("/tts-info/")
async def tts_info():
    """클라이언트에게 TTS API 사용법을 JSON으로 안내"""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "description": "POST /speech/tts/ 에 폼 필드 'text_input'을 보내면, Base64 인코딩된 오디오 Data URI를 반환합니다.",
            "method": "POST",
            "endpoint": "/speech/tts/",
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
