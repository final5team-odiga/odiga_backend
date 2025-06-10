import asyncio
import json
import traceback
from typing import Dict, List, Any
from app.utils.log.hybridlogging import get_hybrid_logger
from app.utils.data.blob_storage import BlobStorageManager
from app.utils.log.logging_manager import LoggingManager

from app.agents.image_analyzer import ImageAnalyzerAgent
from app.agents.contents.content_creator import ContentCreatorV2Crew
from app.utils.data.pdf_vector_manager import PDFVectorManager
from app.agents.Editor.unified_multimodal_agent import UnifiedMultimodalAgent
from app.db.cosmos_connection import logging_container, template_container, jsx_container
from app.db.db_utils import save_to_cosmos, save_jsx_components
from uuid import uuid4
from app.db.magazine_db_utils import MagazineDBUtils
from datetime import datetime
from app.service.pdf.pdf_generater import PDFGenerationService

def sanitize_coroutines(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: sanitize_coroutines(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_coroutines(item) for item in data]
    elif asyncio.iscoroutine(data):
        return f"COROUTINE_OBJECT_REMOVED: {str(data)}"
    return data

class SystemCoordinator:
    """통합 시스템 조율자 - 완전 통합 아키텍처 적용"""

    def __init__(self, user_id: str, magazine_id: str):
        self.user_id = user_id
        self.magazine_id = magazine_id
        self.logger = get_hybrid_logger(self.__class__.__name__)
        
        # ✅ user_id와 magazine_id를 사용하여 BlobStorageManager 초기화
        self.blob_manager = BlobStorageManager(
            user_id=self.user_id,
            magazine_id=self.magazine_id
        )
        
        self.logging_manager = LoggingManager(self.logger)
        self.vector_manager = PDFVectorManager(isolation_enabled=True)
        self.image_analyzer = ImageAnalyzerAgent()
        self.content_creator = ContentCreatorV2Crew()
        
        # ✅ 통합된 멀티모달 에이전트 (템플릿 선택 + JSX 생성 포함)
        self.multimodal_agent = UnifiedMultimodalAgent(self.vector_manager, self.logger)
        
        # ✅ PDF 생성 서비스 추가
        self.pdf_service = PDFGenerationService()

    async def coordinate_complete_magazine_generation(self, user_input: str = None,
                                                      image_folder: str = None,
                                                      generate_pdf: bool = True,
                                                      output_pdf_path: str = None) -> Dict:
        """✅ 완전 통합 매거진 생성 프로세스 (PDF 생성 포함)"""

        self.logger.info("=== 📝 완전 통합 아키텍처 기반 매거진 생성 시작 ===")
        
        try:
            # === Phase 1: 콘텐츠 초안 생성 ===
            self.logger.info("--- 🚀 Phase 1: 콘텐츠 초안 생성 ---")
            image_analysis_results = await self._execute_image_analysis_stage()
            
            # ✅ 이미지 분석 결과 저장 및 로깅
            if image_analysis_results:
                self.logger.info(f"✅ 이미지 분석 완료: {len(image_analysis_results)}개 이미지")
                await MagazineDBUtils.save_combined_image_analysis({
                    "id": str(uuid4()), 
                    "magazine_id": self.magazine_id,
                    "created_at": str(datetime.now()), 
                    "analysis_count": len(image_analysis_results),
                    "image_analyses": image_analysis_results
                })
            else:
                self.logger.warning("❌ 이미지 분석 결과가 없습니다. 기본 이미지를 사용합니다.")
            
            raw_content_json = await self._execute_content_generation_stage(user_input, image_analysis_results)
            raw_content = json.loads(raw_content_json)
            raw_content['magazine_id'] = self.magazine_id
            
            await MagazineDBUtils.save_magazine_content({
                "id": self.magazine_id, 
                "user_id": self.user_id, 
                "status": "phase1_completed",
                "content": raw_content
            })
            self.logger.info(f"✅ Phase 1 완료. Magazine ID: {self.magazine_id}")

            # === ✅ Phase 2: 이미지 배치가 포함된 멀티모달 처리 ===
            self.logger.info("--- 🎨 Phase 2: 이미지 배치 포함 멀티모달 처리 ---")
            
            # ✅ 이미지 분석 결과를 명시적으로 전달
            final_result = await self.multimodal_agent.process_magazine_unified(
                raw_content, 
                image_analysis_results,  # ✅ 실제 이미지 데이터 전달
                user_id=self.user_id
            )
            
            # ✅ 이미지 배치 결과 검증
            jsx_components = final_result.get("content_sections", [])
            image_placement_success = self._verify_image_placement(jsx_components)
            
            self.logger.info(f"✅ 이미지 배치 검증: {'성공' if image_placement_success else '실패'}")
            
            if not final_result or "content_sections" not in final_result:
                raise ValueError("통합 멀티모달 처리 실패 또는 결과가 비어있습니다.")
            
            # ✅ 최종 결과 구성
            complete_result = {
                "magazine_id": self.magazine_id,
                "magazine_title": raw_content.get("magazine_title", "제목 없음"),
                "magazine_subtitle": raw_content.get("magazine_subtitle", ""),
                "components": jsx_components,
                "user_id": self.user_id,
                "processing_summary": final_result.get("processing_metadata", {}),
                "content_sections": jsx_components,
                "image_placement_success": image_placement_success,
                "total_images_used": self._count_images_in_jsx(jsx_components)
            }
            
            # ✅ 결과 저장
            await self._save_results_with_file_manager({
                "magazine_id": self.magazine_id,
                "jsx_components": jsx_components,
                "template_data": {
                    "user_id": self.user_id,
                    "content_sections": jsx_components
                }
            })
            
            # ✅ Phase 3: PDF 생성 (Blob Storage에 저장)
            if generate_pdf:
                self.logger.info("--- 📄 Phase 3: PDF 생성 ---")
                pdf_result = await self._execute_pdf_generation_stage(output_pdf_path)
                complete_result["pdf_generation"] = pdf_result
            
            self.logger.info("🎉✅ 완전 통합 처리 완료!")
            return {"magazine_id": self.magazine_id, "result": complete_result}
        
        except Exception as e:
            self.logger.error(f"매거진 생성 실패: {e}\n{traceback.format_exc()}")
            await MagazineDBUtils.update_magazine_content(self.magazine_id, {
                "status": "failed", "error": str(e)
            })
            return {"error": str(e), "magazine_id": self.magazine_id}

    async def _execute_pdf_generation_stage(self, output_pdf_path: str = None) -> Dict:
        """✅ Phase 3: PDF 생성 실행 (Blob Storage에 저장)"""
        try:
            if not output_pdf_path:
                output_pdf_path = f"magazine_result_{self.user_id}_{self.magazine_id}.pdf"
            
            self.logger.info(f"PDF 생성 시작: {self.magazine_id} -> {output_pdf_path}")
            
            # ✅ PDF 생성 후 Blob Storage의 outputs 폴더에 저장
            success = await self.pdf_service.generate_pdf_from_cosmosdb(
                magazine_id=self.magazine_id,
                output_pdf_path=output_pdf_path
            )
            
            if success:
                # ✅ 생성된 PDF를 Blob Storage의 outputs 폴더에 저장
                import os
                if os.path.exists(output_pdf_path):
                    with open(output_pdf_path, 'rb') as pdf_file:
                        pdf_content = pdf_file.read()
                    
                    # Blob Storage의 outputs 폴더에 저장
                    blob_url = self.blob_manager.save_to_blob(
                        content=pdf_content,
                        filename=os.path.basename(output_pdf_path),
                        category="outputs",
                        content_type="application/pdf"
                    )
                    
                    # 로컬 파일 삭제
                    os.remove(output_pdf_path)
                    
                    self.logger.info(f"✅ PDF 생성 완료 및 Blob Storage 저장: {blob_url}")
                    return {
                        "success": True,
                        "output_path": blob_url,
                        "message": "PDF 생성 및 Blob Storage 저장 성공"
                    }
                else:
                    self.logger.error("❌ PDF 파일이 생성되지 않았습니다.")
                    return {
                        "success": False,
                        "output_path": None,
                        "message": "PDF 파일 생성 실패"
                    }
            else:
                self.logger.error("❌ PDF 생성 실패")
                return {
                    "success": False,
                    "output_path": None,
                    "message": "PDF 생성 실패"
                }
                
        except Exception as e:
            self.logger.error(f"PDF 생성 중 오류 발생: {e}")
            return {
                "success": False,
                "output_path": None,
                "message": f"PDF 생성 오류: {str(e)}"
            }

    def _verify_image_placement(self, jsx_components: List[Dict]) -> bool:
        """JSX 컴포넌트에 이미지가 포함되었는지 검증"""
        for component in jsx_components:
            jsx_code = component.get("jsx_code", "")
            if "<img" in jsx_code and "src=" in jsx_code:
                return True
        return False

    def _count_images_in_jsx(self, jsx_components: List[Dict]) -> int:
        """JSX 컴포넌트에 포함된 이미지 개수 계산"""
        total_images = 0
        for component in jsx_components:
            jsx_code = component.get("jsx_code", "")
            total_images += jsx_code.count("<img")
        return total_images

    async def _execute_image_analysis_stage(self) -> List[Dict]:
        """1단계: 이미지 분석 실행"""
        self.logger.info("1단계: 이미지 분석 시작")

        try:
            images = self.blob_manager.get_images()
            self.logger.info(f"이미지 {len(images)}개 발견")

            if not images:
                self.logger.warning("분석할 이미지가 없습니다.")
                return []
            
            if hasattr(self.image_analyzer, 'analyze_images_batch_async'):
                results = await self.image_analyzer.analyze_images_batch_async(
                    images, 
                    user_id=self.user_id, 
                    magazine_id=self.magazine_id, 
                    max_concurrent=5
                )
            else:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, self.image_analyzer.analyze_single_image_async, images)

            return results

        except Exception as e:
            self.logger.error(f"이미지 분석 실패: {e}")
            return []

    async def _execute_content_generation_stage(self, user_input: str, image_analysis_results: List[Dict]) -> str:
        """2단계: 콘텐츠 생성 실행"""
        self.logger.info("2단계: 콘텐츠 생성 시작")

        try:
            text_blobs = self.blob_manager.get_texts()
            texts = [self.blob_manager.read_text_file(text_blob) for text_blob in text_blobs]

            if not texts:
                self.logger.warning("처리할 텍스트가 없습니다.")
                return self._create_default_content()

            # 여러 텍스트 파일을 하나의 문자열로 결합
            combined_text = "\n\n".join(texts)
            
            if hasattr(self.content_creator, 'execute_content_creation') and asyncio.iscoroutinefunction(self.content_creator.execute_content_creation):
                magazine_content = await self.content_creator.execute_content_creation(combined_text, image_analysis_results)
            else:
                magazine_content = await asyncio.get_event_loop().run_in_executor(
                    None, self.content_creator.execute_content_creation_sync, combined_text, image_analysis_results
                )

            if not magazine_content:
                self.logger.warning("콘텐츠 생성 결과가 없습니다.")
                return self._create_default_content()

            if isinstance(magazine_content, dict):
                try:
                    magazine_content = json.dumps(magazine_content, ensure_ascii=False)
                except (TypeError, ValueError) as e:
                    self.logger.error(f"콘텐츠 직렬화 실패: {e}")
                    return self._create_default_content()
            
            if not isinstance(magazine_content, str):
                magazine_content = str(magazine_content)

            try:
                json.loads(magazine_content)
            except json.JSONDecodeError:
                self.logger.warning(f"생성된 콘텐츠가 유효한 JSON이 아닙니다. 기본 콘텐츠를 사용합니다. Content: {magazine_content[:200]}...")
                return self._create_default_content()

            self.logger.info(f"2단계: 콘텐츠 생성 완료 - {len(magazine_content)}자")
            return magazine_content

        except Exception as e:
            self.logger.error(f"콘텐츠 생성 실패: {e}\n{traceback.format_exc()}")
            return self._create_default_content()

    def _create_default_content(self) -> str:
        """기본 매거진 콘텐츠 생성"""
        default_content = {
            "mag_id": "default_magazine",
            "magazine_title": "fallback",
            "magazine_subtitle": "fallback",
            "sections": [
                {
                    "title": "fallback",
                    "subtitle": "fallback",
                    "content": "fallback."
                },
                {
                    "title": "fallback",
                    "subtitle": "fallback",
                    "content": "fallback"
                }
            ]
        }
        return json.dumps(default_content, ensure_ascii=False)

    async def _save_results_with_file_manager(self, final_result: Dict) -> None:
        """결과 저장 (JSX 저장 로직 개선)"""
        try:
            # 1. 기본 JSON 저장
            outputs_data = {
                "processing_summary": final_result.get("processing_summary", {}),
                "timestamp": asyncio.get_event_loop().time()
            }

            if 'session_id' not in outputs_data:
                outputs_data['session_id'] = final_result.get('session_id', 'unknown_session')

            save_to_cosmos(logging_container, outputs_data, partition_key_field='session_id')
            self.logger.info("✅ outputs 데이터 Cosmos DB 저장 완료")

            # 2. template_data Cosmos DB 저장
            template_data = final_result.get("template_data", {})
            if template_data and template_data.get("content_sections"):
                save_to_cosmos(template_container, template_data, partition_key_field='user_id')
                self.logger.info(f"✅ template_data Cosmos DB 저장 완료: {len(template_data.get('content_sections', []))}개 섹션")

            # ✅ 3. JSX 컴포넌트 저장 로직 개선
            jsx_components = final_result.get("jsx_components", [])
            
            if jsx_components:
                # JSX 메타데이터를 Template 컨테이너에 저장
                jsx_data = {
                    "id": str(uuid4()),
                    "user_id": template_data.get("user_id", "unknown_user"),
                    "total_components": len(jsx_components),
                    "components": jsx_components,
                    "creation_timestamp": asyncio.get_event_loop().time()
                }
                
                save_to_cosmos(template_container, jsx_data, partition_key_field='user_id')
                self.logger.info(f"✅ JSX 컴포넌트 메타데이터를 Template 컨테이너에 저장 완료")
                
                # ✅ JSX 전용 컨테이너에는 순수 JSX 코드만 저장
                magazine_id = final_result.get("magazine_id", str(uuid4()))
                
                # 순수 JSX 컴포넌트만 추출하여 저장
                pure_jsx_components = []
                for i, component in enumerate(jsx_components):
                    pure_jsx_data = {
                        "title": component.get("title", f"섹션 {i+1}"),
                        "jsx_code": component.get("jsx_code", ""),
                        "metadata": component.get("metadata", {})
                    }
                    pure_jsx_components.append(pure_jsx_data)
                
                saved_ids = save_jsx_components(jsx_container, magazine_id, pure_jsx_components, order_matters=True)
                self.logger.info(f"✅ JSX 컴포넌트 {len(saved_ids)}개를 JSX 전용 컨테이너에 저장 완료")
                
                # ✅ magazine_id를 final_result에 추가하여 PDF 생성에서 사용할 수 있도록 함
                final_result["magazine_id"] = magazine_id
                
            else:
                self.logger.warning("저장할 JSX 컴포넌트가 없습니다.")

        except Exception as e:
            self.logger.error(f"결과 저장 실패: {e}")
            raise
