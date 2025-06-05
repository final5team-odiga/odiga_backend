import asyncio
import os
import json
from typing import Dict, List

from crewai import Crew
from utils.hybridlogging import get_hybrid_logger
from utils.file_manager import FileManager
from utils.blob_storage import BlobStorageManager
from utils.logging_manager import LoggingManager

from agents.image_analyzer import ImageAnalyzerAgent
from agents.contents.content_creator import ContentCreatorV2Crew

from agents.Editor.unified_multimodal_agent import UnifiedMultimodalAgent
from agents.Editor.semantic_analysis_engine import SemanticAnalysisEngine
from agents.Editor.realtime_layout_generator import RealtimeLayoutGenerator
from agents.jsx.unified_jsx_generator import UnifiedJSXGenerator
from utils.template_scanner import TemplateScanner

# Cosmos DB 관련 import
from db.cosmos_connection import image_container, magazine_container, logging_container, temmplate_container
from db.db_utils import save_to_cosmos


class SystemCoordinator:
    """통합 시스템 조율자 - 기존 시스템과 새로운 시스템 완전 통합"""

    def __init__(self):
        self.logger = get_hybrid_logger(self.__class__.__name__)
        self.file_manager = FileManager(
            output_folder=os.getenv("OUTPUT_FOLDER", "./output"))
        self.blob_manager = BlobStorageManager()
        self.logging_manager = LoggingManager()

        # 기존 에이전트들 (1-2단계용)
        self.image_analyzer = ImageAnalyzerAgent()
        self.content_creator = ContentCreatorV2Crew()

        # 새로운 통합 에이전트들 (3단계용)
        self.multimodal_agent = UnifiedMultimodalAgent()
        self.semantic_engine = SemanticAnalysisEngine()
        self.layout_generator = RealtimeLayoutGenerator()
        self.jsx_generator = UnifiedJSXGenerator()
        self.template_scanner = TemplateScanner()

    async def coordinate_complete_magazine_generation(self, user_input: str = None,
                                                      image_folder: str = None,
                                                      available_templates: List[str] = None) -> Dict:
        """완전 통합 매거진 생성 프로세스"""

        self.logger.info("=== 완전 통합 매거진 생성 프로세스 시작 ===")

        try:
            # 템플릿이 제공되지 않은 경우 동적 스캔
            if not available_templates:
                self.logger.info("템플릿 목록이 제공되지 않음. 동적 스캔 실행")
                available_templates = await self.template_scanner.scan_jsx_templates()

            # 템플릿 메타데이터 로깅
            template_metadata = await self.template_scanner.get_template_metadata()
            self.logger.info(f"템플릿 메타데이터: {template_metadata}")

            # ✅ 실제 처리 단계들 실행
            # 1단계: 이미지 분석
            image_results = await self._execute_image_analysis_stage()

            # 2단계: 콘텐츠 생성
            magazine_content = await self._execute_content_creation_stage(image_results)

            # 3단계: 멀티모달 처리
            final_result = await self._execute_multimodal_processing_stage(
                magazine_content, image_results, available_templates
            )

            self.logger.info("=== 완전 통합 매거진 생성 프로세스 완료 ===")
            return final_result

        except Exception as e:
            self.logger.error(f"완전 통합 매거진 생성 실패: {e}")

    async def _execute_image_analysis_stage(self) -> List[Dict]:
        """1단계: 이미지 분석 실행 (기존 시스템 활용)"""
        self.logger.info("1단계: 이미지 분석 시작")

        images = self.blob_manager.get_images()
        self.logger.info(f"이미지 {len(images)}개 발견")

        crew = Crew(agents=[self.image_analyzer.create_agent()], verbose=False)
        try:
            if hasattr(self.image_analyzer, 'analyze_images_batch_async'):
                results = await self.image_analyzer.analyze_images_batch_async(images, max_concurrent=5)
            else:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, self.image_analyzer.analyze_images, images, crew)
            for result in results:
                # user_id가 없다면 기본값 부여 (예: 'unknown_user')

                if 'user_id' not in result:
                    result['user_id'] = 'unknown_user'
                save_to_cosmos(image_container, result,
                               partition_key_field='user_id')

            # # ✅ 결과 저장 (기존 file_manager 활용)
            #     analysis_path = os.path.join(self.file_manager.output_folder, "image_analysis_results.json")
            #     self.file_manager.save_json(results, analysis_path)

            # ✅ 새로운 로깅 방식 적용
            await self.logging_manager.log_image_analysis_completion(len(images), len(results))

            self.logger.info(f"1단계: 이미지 분석 완료 - {len(results)}개 결과")
            return results

        except Exception as e:
            self.logger.error(f"이미지 분석 실패: {e}")
            return []

    async def _execute_content_creation_stage(self, image_results: List[Dict]) -> str:
        """2단계: 콘텐츠 생성 실행 (기존 시스템 활용)"""
        self.logger.info("2단계: 콘텐츠 생성 시작")

        text_blobs = self.blob_manager.get_texts()
        texts = [self.blob_manager.read_text_file(
            text_blob) for text_blob in text_blobs]

        try:
            if hasattr(self.content_creator, 'execute_content_creation') and asyncio.iscoroutinefunction(self.content_creator.execute_content_creation):
                magazine_content = await self.content_creator.execute_content_creation(texts, image_results)
            else:
                magazine_content = await asyncio.get_event_loop().run_in_executor(
                    None, self.content_creator.execute_content_creation_sync, texts, image_results
                )

            if isinstance(magazine_content, str):
                try:
                    magazine_content_dict = json.loads(magazine_content)
                except Exception:
                    magazine_content_dict = {"content": magazine_content}
            else:
                magazine_content_dict = magazine_content

            if 'mag_id' not in magazine_content_dict:
                magazine_content_dict['mag_id'] = magazine_content_dict.get(
                    'mag_id', 'unknown_mag')

            save_to_cosmos(magazine_container,
                           magazine_content_dict, partition_key_field='mag_id')

            # # ✅ 결과 저장 (기존 file_manager 활용)
            # content_path = os.path.join(
            #     self.file_manager.output_folder, "magazine_content.json")
            # self.file_manager.save_magazine_content_json(
            #     magazine_content, content_path)

            # ✅ 새로운 로깅 방식 적용
            await self.logging_manager.log_content_creation_completion(len(texts), len(image_results), len(magazine_content))

            self.logger.info(f"2단계: 콘텐츠 생성 완료 - {len(magazine_content)}자")
            return magazine_content

        except Exception as e:
            self.logger.error(f"콘텐츠 생성 실패: {e}")
            return "기본 여행 매거진 콘텐츠"

    async def _execute_multimodal_processing_stage(self, magazine_content: str,
                                                   image_results: List[Dict],
                                                   available_templates: List[str]) -> Dict:
        """3단계: 통합 멀티모달 처리 실행 (이미지 다양성 최적화 포함)"""
        self.logger.info("3단계: 통합 멀티모달 처리 시작")

        try:
            parsed_content = self._parse_magazine_content_to_sections(
                magazine_content)
            self.logger.info(
                f"파싱된 섹션 수: {len(parsed_content.get('sections', []))}")

            input_data = {
                "magazine_content": parsed_content,
                "image_analysis": image_results
            }

            # ✅ 이미지 다양성 최적화 로깅
            self.logger.info(f"이미지 다양성 최적화 대상: {len(image_results)}개 이미지")

            # 의미적 분석 수행
            semantic_analysis = await self.semantic_engine.analyze_text_image_semantics(
                input_data["magazine_content"],
                input_data["image_analysis"]
            )

            # ✅ 두 가지 키 모두 지원
            text_semantics = semantic_analysis.get('text_semantics', [])
            if not text_semantics:
                # 이전 버전 호환성
                semantic_mappings = semantic_analysis.get(
                    'semantic_mappings', [])
                text_semantics = semantic_mappings  # 매핑을 텍스트 의미로 사용

            self.logger.info(f"의미 분석된 텍스트 섹션: {len(text_semantics)}")

            # ✅ 의미적 분석 로깅
            await self.logging_manager.log_semantic_analysis_completion(semantic_analysis)

            self.logger.info(
                f"의미 분석된 텍스트 섹션: {len(semantic_analysis.get('text_semantics', []))}")

            # 실시간 레이아웃 생성
            layout_results = await self.layout_generator.generate_optimized_layouts(
                semantic_analysis,
                available_templates
            )

            # ✅ 레이아웃 생성 로깅
            await self.logging_manager.log_layout_generation_completion(layout_results)

            # ✅ 멀티모달 에이전트로 통합 처리 (이미지 다양성 최적화 포함)
            unified_results = await self.multimodal_agent.process_magazine_unified(
                input_data["magazine_content"],
                input_data["image_analysis"],
                available_templates
            )

            # ✅ 멀티모달 처리 로깅
            await self.logging_manager.log_multimodal_processing_completion(unified_results)

            # ✅ 다양성 최적화 결과 로깅
            if unified_results.get("diversity_optimization_applied"):
                optimization_stats = unified_results.get(
                    "optimization_stats", {})
                total_used = unified_results.get("total_images_used", 0)
                utilization_rate = total_used / \
                    len(image_results) if image_results else 0

                await self.logging_manager.log_diversity_optimization_completion({
                    "utilization_rate": utilization_rate,
                    "total_images_processed": len(image_results),
                    "total_images_used": total_used,
                    "optimization_stats": optimization_stats
                })

                self.logger.info(f"✅ 이미지 다양성 최적화 적용 - 활용률: {utilization_rate:.2%}, "
                                 f"CLIP 사용: {optimization_stats.get('clip_available', False)}")

            # 결과 통합
            integrated_data = await self._integrate_processing_results(
                semantic_analysis, layout_results, unified_results
            )

            self.logger.info(
                f"통합된 콘텐츠 섹션: {len(integrated_data.get('content_sections', []))}")

            # JSX 컴포넌트 생성
            jsx_results = await self.jsx_generator.generate_jsx_with_multimodal_context(
                integrated_data
            )

            # ✅ JSX 생성 로깅
            await self.logging_manager.log_jsx_generation_completion(
                len(jsx_results.get('jsx_components', [])), jsx_results
            )

            self.logger.info(
                f"생성된 JSX 컴포넌트: {len(jsx_results.get('jsx_components', []))}")

            # 최종 결과 패키징
            final_result = await self._package_final_results(
                integrated_data, jsx_results, input_data
            )

            # ✅ 다양성 최적화 메타데이터 추가
            final_result["diversity_optimization"] = {
                "applied": unified_results.get("diversity_optimization_applied", False),
                "total_images_processed": len(image_results),
                "total_images_used": unified_results.get("total_images_used", 0),
                "optimization_stats": unified_results.get("optimization_stats", {})
            }

            # 결과 저장
            await self._save_results_with_file_manager(final_result)

            self.logger.info("3단계: 통합 멀티모달 처리 완료 (이미지 다양성 최적화 포함)")
            return final_result

        except Exception as e:
            self.logger.error(f"멀티모달 처리 실패: {e}")

    def _parse_magazine_content_to_sections(self, magazine_content: str) -> Dict:
        """매거진 콘텐츠를 섹션으로 파싱 (JSON 우선 처리)"""
        try:
            # JSON 형식 파싱 시도
            import json
            parsed_json = json.loads(magazine_content)

            if isinstance(parsed_json, dict) and "sections" in parsed_json:
                # JSON 구조가 올바른 경우
                sections = []
                for section in parsed_json["sections"]:
                    if isinstance(section, dict):
                        sections.append({
                            "title": section.get("title", "제목 없음"),
                            "subtitle": section.get("subtitle", ""),
                            "content": section.get("body", ""),
                            "image_keywords": section.get("image_keywords", [])
                        })

                self.logger.info(f"JSON 파싱 성공: {len(sections)}개 섹션")
                return {
                    "title": parsed_json.get("magazine_title", "여행 매거진"),
                    "subtitle": parsed_json.get("magazine_subtitle", ""),
                    "sections": sections
                }

        except json.JSONDecodeError:
            self.logger.info("JSON 파싱 실패, 텍스트 기반 파싱 시도")

        # 기존 텍스트 기반 파싱 (폴백)
        if "===" in magazine_content:
            return self._split_by_headers(magazine_content)
        else:
            # 길이 단축
            return self._split_by_length(magazine_content, max_length=500)

    def _split_by_headers(self, content: str) -> Dict:
        """헤더 기반 섹션 분할 (개선된 버전)"""
        sections = []
        lines = content.split('\n')
        current_section = {"title": "", "content": ""}

        for line in lines:
            line = line.strip()

            # 구조적 마커 제거
            if "magazine layout design structure" in line.lower():
                continue

            if line.startswith("===") and line.endswith("==="):
                # 이전 섹션 저장
                if current_section["content"]:
                    # 내용 길이 제한
                    if len(current_section["content"]) > 500:
                        current_section["content"] = current_section["content"][:497] + "..."
                    sections.append(current_section)

                # 새 섹션 시작
                title = line.replace("===", "").strip()
                current_section = {"title": title, "content": ""}
            else:
                if line:
                    current_section["content"] += line + " "

        # 마지막 섹션 저장
        if current_section["content"]:
            if len(current_section["content"]) > 500:
                current_section["content"] = current_section["content"][:497] + "..."
            sections.append(current_section)

        self.logger.info(f"헤더 기반 파싱: {len(sections)}개 섹션")
        return {
            "title": "여행 매거진",
            "subtitle": "",
            "sections": sections
        }

    def _split_by_length(self, content: str, max_length: int = 1000) -> List[Dict]:
        """길이 기반 섹션 분할"""

        sections = []
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        current_section = ""
        section_count = 1

        for paragraph in paragraphs:
            if len(current_section + paragraph) > max_length and current_section:
                # 현재 섹션 저장
                sections.append({
                    "title": f"여행 이야기 {section_count}",
                    "content": current_section.strip()
                })
                current_section = paragraph
                section_count += 1
            else:
                current_section += "\n\n" + paragraph if current_section else paragraph

        # 마지막 섹션 저장
        if current_section:
            sections.append({
                "title": f"여행 이야기 {section_count}",
                "content": current_section.strip()
            })

        return sections

    async def _integrate_processing_results(self, semantic_analysis: Dict,
                                            layout_results: Dict,
                                            unified_results: Dict) -> Dict:
        """처리 결과들 통합"""

        integrated_data = {
            "selected_templates": unified_results.get("selected_templates", []),
            "content_sections": unified_results.get("content_sections", []),
            "semantic_analysis": semantic_analysis,
            "optimized_layouts": layout_results.get("optimized_layouts", []),
            "integration_metadata": {
                "multimodal_processing": True,
                "semantic_optimization": True,
                "layout_optimization": True,
                "processing_timestamp": asyncio.get_event_loop().time(),
                "total_sections": len(unified_results.get("content_sections", [])),
                "semantic_confidence": semantic_analysis.get("analysis_metadata", {}).get("mapping_confidence", 0.0),
                "ai_search_enhanced": semantic_analysis.get("analysis_metadata", {}).get("ai_search_enhanced", False)
            }
        }

        return integrated_data

    async def _package_final_results(self, integrated_data: Dict,
                                     jsx_results: Dict,
                                     input_data: Dict) -> Dict:
        """최종 결과 패키징"""

        final_result = {
            "template_data": integrated_data,
            "jsx_components": jsx_results.get("jsx_components", []),
            "processing_summary": {
                "total_sections": len(integrated_data.get("content_sections", [])),
                "total_jsx_components": len(jsx_results.get("jsx_components", [])),
                "semantic_confidence": integrated_data.get("integration_metadata", {}).get("semantic_confidence", 0.0),
                "multimodal_optimization": True,
                "responsive_design": True,
                "style_optimization": True
            },
            "execution_logs": {
                "image_analysis_completed": True,
                "content_creation_completed": True,
                "semantic_analysis_completed": True,
                "layout_generation_completed": True,
                "multimodal_processing_completed": True,
                "jsx_generation_completed": True,
                "integration_completed": True
            },
            "source_data": {
                "magazine_content_sections": len(input_data.get("magazine_content", {}).get("sections", [])),
                "image_analysis_count": len(input_data.get("image_analysis", [])),
                "available_templates": integrated_data.get("selected_templates", [])
            }
        }

        return final_result

    async def _save_results_with_file_manager(self, final_result: Dict) -> None:
        """결과 저장 (완전히 개선된 File Manager 활용)"""

        try:
            # 1. 기본 JSON 저장 (SystemCoordinator 역할)
            outputs_data = {
                "processing_summary": final_result.get("processing_summary", {}),
                "execution_logs": final_result.get("execution_logs", {}),
                "timestamp": asyncio.get_event_loop().time()
            }

            if "source_data" in final_result:
                outputs_data["source_data"] = final_result["source_data"]
            else:
                outputs_data["source_data"] = {
                    "magazine_content_sections": 0,
                    "image_analysis_count": 0,
                    "available_templates": []
                }

            # session_id 파티션 키
            if 'session_id' not in outputs_data:
                outputs_data['session_id'] = final_result.get(
                    'session_id', 'unknown_session')

            # Cosmos DB에 저장 (파티션 키: session_id)
            save_to_cosmos(logging_container, outputs_data,
                           partition_key_field='session_id')
            self.logger.info("✅ outputs 데이터 Cosmos DB 저장 완료")

            # outputs_path = os.path.join(
            #     self.file_manager.output_folder, "latest_outputs.json")
            # self.file_manager.save_json(outputs_data, outputs_path)

            # ✅ 2. template_data.json 저장 (File Manager에 위임)
            # template_data = final_result.get("template_data", {})
            # if template_data and template_data.get("content_sections"):
            #     template_path = os.path.join(
            #         self.file_manager.output_folder, "template_data.json")
            #     await self.file_manager.save_template_data_async(template_data, template_path)
            #     self.logger.info(
            #         f"✅ template_data.json 저장: {len(template_data.get('content_sections', []))}개 섹션")

            # ✅ 2. template_data를 Cosmos DB에 저장 (로컬 파일 대신)
            template_data = final_result.get("template_data", {})
            if template_data and template_data.get("content_sections"):
                # template_data에 필요한 메타데이터 추가
                template_document = {
                    "id": f"template_{int(asyncio.get_event_loop().time())}",
                    "template_id": template_data.get("template_id", f"template_{int(asyncio.get_event_loop().time())}"),
                    "content_sections": template_data.get("content_sections", []),
                    "selected_templates": template_data.get("selected_templates", []),
                    "semantic_analysis": template_data.get("semantic_analysis", {}),
                    "optimized_layouts": template_data.get("optimized_layouts", []),
                    "integration_metadata": template_data.get("integration_metadata", {}),
                    "created_at": asyncio.get_event_loop().time(),
                    "document_type": "template_data"
                }

                # Cosmos DB에 저장 (파티션 키: template_id)
                save_to_cosmos(temmplate_container, template_document,
                               partition_key_field='user_id')

                self.logger.info(
                    f"✅ template_data Cosmos DB 저장 완료: {len(template_data.get('content_sections', []))}개 섹션")

            #  3. React 앱 생성 요청
            jsx_components = final_result.get("jsx_components", [])
            if template_data.get("content_sections") and jsx_components:
                project_name = f"magazine_app_{int(asyncio.get_event_loop().time())}"
                project_folder = self.file_manager.create_project_folder(
                    project_name)

                # JSX 저장은 여기서만 한 번만 수행
                self.file_manager.create_magazine_react_app(
                    project_folder=project_folder,
                    saved_components=jsx_components,
                    template_data=template_data
                )

                self.logger.info(f"✅ React 앱 생성 완료: {project_folder}")
                self.logger.info(
                    f"📱 실행 방법: cd {project_folder} && npm install && npm run dev")

        except Exception as e:
            self.logger.error(f"결과 저장 실패: {e}")
            raise
