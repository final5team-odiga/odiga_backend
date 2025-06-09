import asyncio
import json
import traceback
import time
from typing import Dict, List, Any
from utils.log.hybridlogging import get_hybrid_logger
from utils.data.blob_storage import BlobStorageManager
from utils.log.logging_manager import LoggingManager

from agents.image_analyzer import ImageAnalyzerAgent
from agents.contents.content_creator import ContentCreatorV2Crew
from utils.data.pdf_vector_manager import PDFVectorManager
from agents.Editor.unified_multimodal_agent import UnifiedMultimodalAgent
from db.cosmos_connection import logging_container, template_container, jsx_container
from db.db_utils import save_to_cosmos, save_jsx_components
from crewai import Crew
from uuid import uuid4
from db.magazine_db_utils import MagazineDBUtils
from datetime import datetime

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

    def __init__(self):
        self.logger = get_hybrid_logger(self.__class__.__name__)
        self.blob_manager = BlobStorageManager()
        self.logging_manager = LoggingManager(self.logger)

        self.vector_manager = PDFVectorManager(isolation_enabled=True)

        self.image_analyzer = ImageAnalyzerAgent()
        self.content_creator = ContentCreatorV2Crew()
        
        # ✅ 통합된 멀티모달 에이전트 (템플릿 선택 + JSX 생성 포함)
        self.multimodal_agent = UnifiedMultimodalAgent(self.vector_manager, self.logger)

    async def coordinate_complete_magazine_generation(self, user_input: str = None,
                                                      image_folder: str = None,
                                                      user_id: str = "unknown_user") -> Dict:
        """✅ 완전 통합 매거진 생성 프로세스 (2단계로 간소화)"""

        self.logger.info("=== 📝 완전 통합 아키텍처 기반 매거진 생성 시작 ===")
        magazine_id = str(uuid4())
        
        try:
            # === Phase 1: 콘텐츠 초안 생성 ===
            self.logger.info("--- 🚀 Phase 1: 콘텐츠 초안 생성 ---")
            image_analysis_results = await self._execute_image_analysis_stage()
            
            if image_analysis_results:
                await MagazineDBUtils.save_combined_image_analysis({
                    "id": str(uuid4()), "magazine_id": magazine_id,
                    "created_at": str(datetime.now()), "analysis_count": len(image_analysis_results),
                    "image_analyses": image_analysis_results
                })
            
            raw_content_json = await self._execute_content_generation_stage(user_input, image_analysis_results)
            raw_content = json.loads(raw_content_json)
            raw_content['magazine_id'] = magazine_id
            
            await MagazineDBUtils.save_magazine_content({
                "id": magazine_id, "user_id": user_id, "status": "phase1_completed",
                "content": raw_content
            })
            self.logger.info(f"✅ Phase 1 완료. Magazine ID: {magazine_id}")

            # === ✅ Phase 2: 완전 통합된 멀티모달 처리 (템플릿 선택 + JSX 생성 포함) ===
            self.logger.info("--- 🎨 Phase 2: 완전 통합 멀티모달 처리 ---")
            
            final_result = await self.multimodal_agent.process_magazine_unified(
                raw_content, image_analysis_results, user_id=user_id
            )
            
            if not final_result or "content_sections" not in final_result:
                raise ValueError("통합 멀티모달 처리 실패 또는 결과가 비어있습니다.")
            
            # ✅ JSX 컴포넌트 추출 (이미 통합 처리에서 생성됨)
            jsx_components = []
            for section in final_result.get("content_sections", []):
                jsx_component = section.get("jsx_component", {})
                if jsx_component:
                    jsx_components.append(jsx_component)
            
            # ✅ 최종 결과 구성
            complete_result = {
                "magazine_id": magazine_id,
                "magazine_title": raw_content.get("magazine_title", "제목 없음"),
                "magazine_subtitle": raw_content.get("magazine_subtitle", ""),
                "components": jsx_components,
                "user_id": user_id,
                "processing_summary": final_result.get("processing_metadata", {}),
                "content_sections": final_result.get("content_sections", [])
            }
            
            
            # ✅ 결과 저장
            await self._save_results_with_file_manager({
                "magazine_id": magazine_id,
                "jsx_components": jsx_components,
                "template_data": {
                    "user_id": user_id,
                    "content_sections": final_result.get("content_sections", [])
                }
            })
            
            self.logger.info("🎉✅ 완전 통합 처리 완료!")
            return {"magazine_id": magazine_id, "result": complete_result}
            
        except Exception as e:
            self.logger.error(f"매거진 생성 실패: {e}\n{traceback.format_exc()}")
            await MagazineDBUtils.update_magazine_content(magazine_id, {
                "status": "failed", "error": str(e)
            })
            return {"error": str(e), "magazine_id": magazine_id}

    async def _execute_image_analysis_stage(self) -> List[Dict]:
        """1단계: 이미지 분석 실행"""
        self.logger.info("1단계: 이미지 분석 시작")

        try:
            images = self.blob_manager.get_images()
            self.logger.info(f"이미지 {len(images)}개 발견")

            if not images:
                self.logger.warning("분석할 이미지가 없습니다.")
                return []

            crew = Crew(agents=[self.image_analyzer.create_agent()], verbose=False)
            
            if hasattr(self.image_analyzer, 'analyze_images_batch_async'):
                results = await self.image_analyzer.analyze_images_batch_async(images, max_concurrent=5)
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
            self.logger.error(f"콘텐츠 생성 실패: {e}\\n{traceback.format_exc()}")
            return self._create_default_content()

    def _create_default_content(self) -> str:
        """기본 매거진 콘텐츠 생성"""
        default_content = {
            "mag_id": "default_magazine",
            "magazine_title": "베니스 여행 이야기",
            "magazine_subtitle": "아름다운 수상 도시에서의 특별한 순간들",
            "sections": [
                {
                    "title": "베니스의 겨울",
                    "subtitle": "안개 속 신비로운 도시",
                    "content": "겨울의 베니스는 또 다른 매력을 선사합니다. 안개에 쌓인 운하와 고딕 건축물들은 마치 동화 속 한 장면 같습니다."
                },
                {
                    "title": "카니발의 열기",
                    "subtitle": "화려한 가면과 축제",
                    "content": "세계적으로 유명한 베니스 카니발은 도시 전체를 축제의 장으로 만듭니다. 전통 의상과 아름다운 가면은 시간 여행을 하는 듯한 느낌을 줍니다."
                }
            ]
        }
        return json.dumps(default_content, ensure_ascii=False)

    async def _save_results_with_file_manager(self, final_result: Dict) -> None:
        """결과 저장 (완전히 개선된 File Manager 활용)"""

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

            jsx_components = final_result.get("jsx_components", [])
            if not jsx_components:
                jsx_components = final_result.get("result", {}).get("content_sections", [])
            if jsx_components:
                jsx_data = {
                    "id": str(uuid4()),
                    "user_id": template_data.get("user_id", "unknown_user"),
                    "total_components": len(jsx_components),
                    "components": jsx_components,
                    "creation_timestamp": asyncio.get_event_loop().time()
                }
                
                save_to_cosmos(template_container, jsx_data, partition_key_field='user_id')
                self.logger.info(f"✅ JSX 컴포넌트 메타데이터를 Template 컨테이너에 저장 완료")
                
                magazine_id = final_result.get("magazine_id", str(uuid4()))
                saved_ids = save_jsx_components(jsx_container, magazine_id, jsx_components, order_matters=True)
                self.logger.info(f"✅ JSX 컴포넌트 {len(saved_ids)}개를 JSX 전용 컨테이너에 저장 완료")

        except Exception as e:
            self.logger.error(f"결과 저장 실패: {e}")
            raise
    
