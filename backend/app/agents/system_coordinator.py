import asyncio
import os
import json
import traceback
from typing import Dict, List, Any, Optional
from utils.log.hybridlogging import get_hybrid_logger
from service.file_manager import FileManager
from utils.data.blob_storage import BlobStorageManager
from utils.log.logging_manager import LoggingManager

from agents.image_analyzer import ImageAnalyzerAgent
from agents.contents.content_creator import ContentCreatorV2Crew

from agents.Editor.unified_multimodal_agent import UnifiedMultimodalAgent
from agents.Editor.semantic_analysis_engine import SemanticAnalysisEngine
from agents.Editor.realtime_layout_generator import RealtimeLayoutGenerator
from agents.jsx.unified_jsx_generator import UnifiedJSXGenerator
from agents.jsx.template_selector import SectionStyleAnalyzer
from db.cosmos_connection import logging_container, template_container, jsx_container
from db.db_utils import save_to_cosmos, save_jsx_components
from crewai import Crew
from uuid import uuid4
from db.magazine_db_utils import MagazineDBUtils
from datetime import datetime

# Helper function to sanitize coroutine objects from data structures
def sanitize_coroutines(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: sanitize_coroutines(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_coroutines(item) for item in data]
    elif asyncio.iscoroutine(data):
        # Log or identify that a coroutine was found and stringified
        # This helps in debugging which part of the async chain was not awaited
        return f"COROUTINE_OBJECT_REMOVED: {str(data)}"
    return data

class SystemCoordinator:
    """통합 시스템 조율자 - 기존 시스템과 새로운 시스템 완전 통합"""

    def __init__(self):
        self.logger = get_hybrid_logger(self.__class__.__name__)
        self.file_manager = FileManager(
            output_folder=os.getenv("OUTPUT_FOLDER", "./output"))
        self.blob_manager = BlobStorageManager()
        self.logging_manager = LoggingManager(self.logger)

        # 기존 에이전트들 (1-2단계용)
        self.image_analyzer = ImageAnalyzerAgent()
        self.content_creator = ContentCreatorV2Crew()

        # 새로운 통합 에이전트들 (3-4단계용)
        self.layout_generator = RealtimeLayoutGenerator() # 레이아웃 전략 생성기
        self.multimodal_agent = UnifiedMultimodalAgent()
        self.semantic_engine = SemanticAnalysisEngine()
        self.template_selector = SectionStyleAnalyzer()
        self.jsx_generator = UnifiedJSXGenerator()

    async def coordinate_complete_magazine_generation(self, user_input: str = None,
                                                      image_folder: str = None,
                                                      user_id: str = "unknown_user") -> Dict:
        """완전 통합 매거진 생성 프로세스 (새로운 아키텍처)"""

        self.logger.info("=== 📝 신규 아키텍처 기반 매거진 생성 시작 ===")
        magazine_id = str(uuid4())
        
        try:
            # === 1단계: 콘텐츠 초안 생성 ===
            self.logger.info("--- 🚀 Phase 1: 콘텐츠 초안 생성 ---")
            # 이미지 분석은 콘텐츠 생성과 병렬 또는 직전에 수행될 수 있음
            image_analysis_results = await self._execute_image_analysis_stage()
            
            # 이미지 분석 결과를 하나의 문서로 images 컨테이너에 저장
            if image_analysis_results:
                combined_analysis = {
                    "id": str(uuid4()),
                    "magazine_id": magazine_id,
                    "created_at": str(datetime.now()),
                    "analysis_count": len(image_analysis_results),
                    "image_analyses": image_analysis_results
                }
                await MagazineDBUtils.save_combined_image_analysis(combined_analysis)
            
            raw_content_json = await self._execute_content_generation_stage(user_input, image_analysis_results)
            raw_content = json.loads(raw_content_json)
            
            # 초기 상태 저장 - 매거진 콘텐츠만 magazine_container에 저장
            await MagazineDBUtils.save_magazine_content({
                "id": magazine_id,
                "user_id": user_id,
                "status": "phase1_completed",
                "raw_content": raw_content
            })
            
            self.logger.info(f"✅ Phase 1 완료. Magazine ID: {magazine_id}")

            # === 2단계: 편집 및 의미/스타일 확정 ===
            self.logger.info("--- 🎨 Phase 2: 편집 및 의미/스타일 확정 ---")
            
            # 매거진 콘텐츠를 섹션으로 파싱
            parsed_content = self._parse_magazine_content_to_sections(raw_content_json)
            self.logger.info(f"파싱된 섹션 수: {len(parsed_content.get('sections', []))}")
            
            # 의미적 분석 수행
            semantic_analysis = await self.semantic_engine.analyze_text_image_semantics(
                parsed_content,
                image_analysis_results
            )
            
            if not semantic_analysis:
                self.logger.warning("의미적 분석 결과가 없습니다.")
                semantic_analysis = {
                    "text_semantics": [],
                    "semantic_mappings": [],
                    "analysis_metadata": {
                        "sections_processed": 0,
                        "images_processed": 0,
                        "success": False
                    }
                }
                
            self.logger.info(f"의미 분석된 텍스트 섹션: {len(semantic_analysis.get('text_semantics', []))}")
            
            # 멀티모달 에이전트로 통합 처리 (편집 단계)
            unified_results = await self.multimodal_agent.process_magazine_unified(
                parsed_content,
                image_analysis_results,
                [],  # 템플릿은 더 이상 여기서 전달하지 않음
                user_id=user_id
            )
            
            if not unified_results:
                self.logger.warning("멀티모달 통합 처리 결과가 없습니다.")
                unified_results = {
                    "status": "error",
                    "message": "멀티모달 통합 처리 실패",
                    "user_id": user_id
                }
                
            # Phase 2 결과 저장
            final_sections = []
            for section_idx, section in enumerate(parsed_content.get('sections', [])):
                # 해당 섹션에 대한 의미 분석 데이터 찾기
                section_semantics = None
                for sem in semantic_analysis.get('text_semantics', []):
                    if sem.get('section_index') == section_idx:
                        section_semantics = sem
                        break
                
                # 최종 편집된 콘텐츠와 메타데이터
                final_section_data = {
                    "title": section.get('title'),
                    "subtitle": section.get('subtitle', ''),
                    "final_content": section.get('content'),  # 실제로는 Editor 에이전트들이 수정한 최종본
                    "metadata": {
                        "style": section_semantics.get('style', '') if section_semantics else '',
                        "emotion": section_semantics.get('emotional_tone', '') if section_semantics else '',
                        "keywords": section_semantics.get('keywords', []) if section_semantics else [],
                        "image_count": len(section_semantics.get('related_images', [])) if section_semantics else 0
                    }
                }
                final_sections.append(final_section_data)
                
            await MagazineDBUtils.update_magazine_content(magazine_id, {
                "status": "phase2_completed",
                "final_content": final_sections,
                "semantic_analysis": semantic_analysis,
                "unified_results": unified_results
            })
            self.logger.info("✅ Phase 2 완료.")

            # === 2.5단계: 섹션별 레이아웃 전략 생성 ===
            self.logger.info("--- 🧠 Phase 2.5: 섹션별 레이아웃 전략 생성 ---")
            layout_strategies = []
            for section_data in final_sections:
                # RealtimeLayoutGenerator를 사용해 각 섹션의 이상적인 레이아웃 전략을 생성
                strategy = await self.layout_generator.generate_layout_strategy_for_section(
                    section_data  # 전체 섹션 데이터 전달
                )
                layout_strategies.append(strategy)
            
            await MagazineDBUtils.update_magazine_content(magazine_id, {
                "status": "phase2.5_completed",
                "layout_strategies": layout_strategies
            })
            self.logger.info("✅ Phase 2.5 완료.")


            # === 3단계: 지능형 템플릿 매칭 ===
            self.logger.info("--- 🧩 Phase 3: 지능형 템플릿 매칭 ---")
            content_template_pairs = []
            
            for i, section_data in enumerate(final_sections):
                # 생성된 레이아웃 전략을 template_selector에 전달하여 최적의 템플릿 검색
                template_code = await self.template_selector.analyze_and_select_template(
                    section_data, 
                    layout_strategies[i]
                )
                content_template_pairs.append({
                    "content": section_data,
                    "template_code": template_code,
                    "layout_strategy": layout_strategies[i]
                })
            
            await MagazineDBUtils.update_magazine_content(magazine_id, {
                "status": "phase3_completed",
                "content_template_pairs": content_template_pairs
            })
            self.logger.info("✅ Phase 3 완료.")
            
            # === 4단계: 최종 JSX 어셈블리 ===
            self.logger.info("--- 🛠️ Phase 4: 최종 JSX 어셈블리 ---")
            final_jsx_components = []
            
            for pair in content_template_pairs:
                # UnifiedJSXGenerator는 콘텐츠와 템플릿 코드를 받아 결합
                jsx_component = await self.jsx_generator.generate_jsx_from_template(
                    pair['content'], 
                    pair['template_code']
                )
                final_jsx_components.append(jsx_component)
                
            final_result = {
                "magazine_id": magazine_id,
                "magazine_title": parsed_content.get("magazine_title", "제목 없음"),
                "magazine_subtitle": parsed_content.get("magazine_subtitle", ""),
                "components": final_jsx_components,
                "user_id": user_id,
                "processing_summary": {
                    "total_sections": len(parsed_content.get("sections", [])),
                    "semantic_confidence": semantic_analysis.get("analysis_metadata", {}).get("success", False),
                    "multimodal_optimization": True,
                    "responsive_design": True
                }
            }
            
            await MagazineDBUtils.update_magazine_content(magazine_id, {
                "status": "completed",
                "final_result": final_result
            })
            
            # ✅ NEW: JSX 컴포넌트 별도 저장 (파일 시스템 대신 Cosmos DB JSX 컨테이너에 저장)
            await self._save_results_with_file_manager({
                "magazine_id": magazine_id,
                "jsx_components": final_jsx_components,
                "template_data": {
                    "user_id": user_id,
                    "content_sections": final_sections,
                    "selected_templates": [pair.get("template_code", "") for pair in content_template_pairs]
                }
            })
            
            self.logger.info("🎉✅ Phase 4 완료. 전체 프로세스 성공!")
            
            return {"magazine_id": magazine_id, "result": final_result}
            
        except Exception as e:
            self.logger.error(f"매거진 생성 실패: {e}\n{traceback.format_exc()}")
            await MagazineDBUtils.update_magazine_content(magazine_id, {
                "status": "failed",
                "error": str(e)
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
            
            # 이미지 분석 실행
            if hasattr(self.image_analyzer, 'analyze_images_batch_async'):
                results = await self.image_analyzer.analyze_images_batch_async(images, max_concurrent=5)
            else:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, self.image_analyzer.analyze_images_batch, images)

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

            # 콘텐츠 생성 실행
            if hasattr(self.content_creator, 'execute_content_creation') and asyncio.iscoroutinefunction(self.content_creator.execute_content_creation):
                magazine_content = await self.content_creator.execute_content_creation(texts, image_analysis_results)
            else:
                magazine_content = await asyncio.get_event_loop().run_in_executor(
                    None, self.content_creator.execute_content_creation_sync, texts, image_analysis_results
                )

            if not magazine_content:
                self.logger.warning("콘텐츠 생성 결과가 없습니다.")
                return self._create_default_content()

            # 결과 정규화: DB 저장 로직을 제거하고, 반환값을 JSON 문자열로 일관성 있게 만듭니다.
            if isinstance(magazine_content, dict):
                try:
                    magazine_content = json.dumps(magazine_content, ensure_ascii=False)
                except (TypeError, ValueError) as e:
                    self.logger.error(f"콘텐츠 직렬화 실패: {e}")
                    return self._create_default_content()
            
            # 생성된 콘텐츠가 문자열이 아닌 경우 문자열로 변환
            if not isinstance(magazine_content, str):
                magazine_content = str(magazine_content)

            # 최종 반환 전에 유효한 JSON인지 확인
            try:
                # 여기서는 로드만 해보고, 실제 객체는 사용하지 않습니다.
                # 상위 메소드에서 다시 파싱해서 사용합니다.
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

    def _parse_magazine_content_to_sections(self, magazine_content: str) -> Dict:
        """매거진 콘텐츠를 섹션으로 파싱"""
        self.logger.info("매거진 콘텐츠 섹션 파싱 시작")

        try:
            # JSON 문자열인 경우
            content_dict = json.loads(magazine_content)
            
            # 이미 섹션 구조가 있는지 확인
            if "sections" in content_dict:
                self.logger.info(f"기존 섹션 구조 발견: {len(content_dict['sections'])}개 섹션")
                
                # 하위 섹션이 있는 경우 처리
                has_sub_sections = False
                section_count = len(content_dict['sections'])
                total_section_count = section_count
                
                for section in content_dict['sections']:
                    if 'sub_sections' in section:
                        has_sub_sections = True
                        total_section_count += len(section['sub_sections'])
                
                if has_sub_sections:
                    self.logger.info(f"하위 섹션 포함 총 {total_section_count}개 섹션 발견")
                    
                    # 하위 섹션을 별도의 섹션으로 변환하여 추가 (멀티모달 에이전트 호환성)
                    flattened_sections = []
                    
                    for section in content_dict['sections']:
                        if 'sub_sections' in section:
                            # 부모 섹션 정보 추가 (본문 없이 제목만)
                            parent_section = {
                                "title": section.get('title', ''),
                                "subtitle": section.get('subtitle', ''),
                                "content": f"이 섹션은 {len(section['sub_sections'])}개의 하위 섹션으로 구성되어 있습니다.",
                                "is_parent": True,
                                "parent_id": section.get('section_id', '')
                            }
                            flattened_sections.append(parent_section)
                            
                            # 하위 섹션 추가
                            for sub_section in section['sub_sections']:
                                flattened_sections.append({
                                    "title": sub_section.get('title', ''),
                                    "subtitle": sub_section.get('subtitle', ''),
                                    "content": sub_section.get('body', ''),
                                    "is_sub_section": True,
                                    "parent_id": section.get('section_id', ''),
                                    "sub_section_id": sub_section.get('sub_section_id', '')
                                })
                        else:
                            # 일반 섹션 추가
                            flattened_sections.append({
                                "title": section.get('title', ''),
                                "subtitle": section.get('subtitle', ''),
                                "content": section.get('body', '')
                            })
                    
                    return {
                        "magazine_title": content_dict.get("magazine_title", "여행 매거진"),
                        "magazine_subtitle": content_dict.get("magazine_subtitle", "특별한 순간들"),
                        "sections": flattened_sections,
                        "has_dynamic_sections": True,
                        "original_section_count": section_count,
                        "total_section_count": total_section_count
                    }
                
                # 기존 섹션 구조 변환
                sections = []
                for section in content_dict['sections']:
                    sections.append({
                        "title": section.get('title', ''),
                        "subtitle": section.get('subtitle', ''),
                        "content": section.get('body', '')
                    })
                
                return {
                    "magazine_title": content_dict.get("magazine_title", "여행 매거진"),
                    "magazine_subtitle": content_dict.get("magazine_subtitle", "특별한 순간들"),
                    "sections": sections
                }
            
            # 섹션 구조가 없는 경우
            self.logger.info("섹션 구조 없음, 콘텐츠 분석 시작")
            
            # 콘텐츠 형식에 따라 분할 방식 결정
            if isinstance(content_dict, dict) and "content" in content_dict:
                content = content_dict["content"]
                if isinstance(content, str):
                    # 헤더로 분할 시도
                    sections = self._split_by_headers(content)
                    if len(sections["sections"]) > 1:
                        return sections
                    
                    # 헤더가 없으면 길이로 분할
                    return self._split_by_length(content)
                else:
                    self.logger.warning("콘텐츠가 문자열이 아님")
                    return {"sections": [{"title": "여행 이야기", "content": str(content)}]}
            else:
                # 직접 콘텐츠로 사용
                return {"sections": [{"title": "여행 이야기", "content": json.dumps(content_dict, ensure_ascii=False)}]}
                
        except json.JSONDecodeError:
            # 일반 텍스트인 경우
            self.logger.info("JSON이 아닌 일반 텍스트로 처리")
            
            # 헤더로 분할 시도
            sections = self._split_by_headers(magazine_content)
            if len(sections["sections"]) > 1:
                return sections
            
            # 헤더가 없으면 길이로 분할
            return self._split_by_length(magazine_content)
        except Exception as e:
            self.logger.error(f"매거진 콘텐츠 파싱 실패: {e}")
            return {"sections": [{"title": "여행 이야기", "content": str(magazine_content)[:1000]}]}

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
            "user_id": unified_results.get("user_id", "unknown_user"),
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
                "total_jsx_components": len(jsx_results.get("jsx_components", []) if isinstance(jsx_results, dict) else 0),
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
                "templates_used": len(integrated_data.get("selected_templates", []))
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
                    "templates_used": 0
                }

            # session_id 파티션 키
            if 'session_id' not in outputs_data:
                outputs_data['session_id'] = final_result.get(
                    'session_id', 'unknown_session')

            # Cosmos DB에 저장 (파티션 키: session_id)
            save_to_cosmos(logging_container, outputs_data,
                           partition_key_field='session_id')
            self.logger.info("✅ outputs 데이터 Cosmos DB 저장 완료")

            # ✅ 2. template_data Cosmos DB 저장
            template_data = final_result.get("template_data", {})
            if template_data and template_data.get("content_sections"):
                # Cosmos DB에 저장 (파티션 키: user_id)
                save_to_cosmos(template_container, template_data,
                             partition_key_field='user_id')
                self.logger.info(
                    f"✅ template_data Cosmos DB 저장 완료: {len(template_data.get('content_sections', []))}개 섹션")

            # 3. JSX 컴포넌트 Cosmos DB 저장 
            jsx_components = final_result.get("jsx_components", [])
            if jsx_components:
                # JSX 컴포넌트 데이터 구조화
                jsx_data = {
                    "id": str(uuid4()),
                    "user_id": template_data.get("user_id", "unknown_user"),
                    "total_components": len(jsx_components),
                    "components": jsx_components,
                    "creation_timestamp": asyncio.get_event_loop().time()
                }
                
                # Template 컨테이너에 메타데이터 저장
                save_to_cosmos(template_container, jsx_data, partition_key_field='user_id')
                self.logger.info(f"✅ JSX 컴포넌트 메타데이터를 Template 컨테이너에 저장 완료")
                
                # ✅ NEW: JSX 컴포넌트를 전용 컨테이너에 개별적으로 저장
                magazine_id = final_result.get("magazine_id", str(uuid4()))
                saved_ids = save_jsx_components(
                    jsx_container, 
                    magazine_id, 
                    jsx_components, 
                    order_matters=True
                )
                self.logger.info(f"✅ JSX 컴포넌트 {len(saved_ids)}개를 JSX 전용 컨테이너에 저장 완료")

        except Exception as e:
            self.logger.error(f"결과 저장 실패: {e}")
            raise
