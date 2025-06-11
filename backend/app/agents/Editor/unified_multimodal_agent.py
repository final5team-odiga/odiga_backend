import asyncio
import json
import re
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from crewai import Agent, Task, Crew
from ...custom_llm import get_azure_llm

from .semantic_analysis_engine import SemanticAnalysisEngine
from .image_diversity_manager import ImageDiversityManager
from ..jsx.template_selector import SectionStyleAnalyzer
from ..jsx.unified_jsx_generator import UnifiedJSXGenerator

from ...utils.isolation.ai_search_isolation import AISearchIsolationManager
from ...utils.data.pdf_vector_manager import PDFVectorManager
from ...utils.isolation.session_isolation import SessionAwareMixin
from ...utils.isolation.agent_communication_isolation import InterAgentCommunicationMixin
from ...utils.log.logging_manager import LoggingManager

from ...db.magazine_db_utils import MagazineDBUtils

class UnifiedMultimodalAgent(SessionAwareMixin, InterAgentCommunicationMixin):
    """통합 멀티모달 에이전트 - 하이브리드 방식: 요약 없는 CrewAI 분석 + 이미지 배치 강화"""
    
    def __init__(self, vector_manager: PDFVectorManager, logger: Any):
        self.llm = get_azure_llm()
        self.logger = logger

        self.isolation_manager = AISearchIsolationManager()
        self.vector_manager = vector_manager
        
        # ✅ CLIP 세션 공유를 위한 초기화
        self._initialize_shared_clip_session()
        
        # ✅ 공유 CLIP 세션을 사용하는 컴포넌트들
        self.semantic_engine = SemanticAnalysisEngine(self.logger, self.shared_clip_session)
        self.image_diversity_manager = ImageDiversityManager(
            self.vector_manager, self.logger
        )
        
        # CLIP 세션 주입
        self.image_diversity_manager.set_external_clip_session(
            self.shared_clip_session.get("onnx_session"),
            self.shared_clip_session.get("clip_preprocess")
        )
        
        # ✅ 새로 통합되는 컴포넌트 (수정된 초기화)
        self.template_selector = SectionStyleAnalyzer()
        self.jsx_generator = UnifiedJSXGenerator(
            logger=logger, 
            vector_manager=vector_manager
        )
        
        self.logging_manager = LoggingManager(self.logger)
        self.__init_session_awareness__()
        self.__init_inter_agent_communication__()
        
        # CrewAI 에이전트들
        self.content_structure_agent = self._create_content_structure_agent_with_ai_search()
        self.image_layout_agent = self._create_image_layout_agent_with_ai_search()
        self.semantic_coordinator_agent = self._create_semantic_coordinator_agent_with_ai_search()

        # ✅ 원본 데이터 보존용 변수
        self.current_magazine_content = None
        self.current_image_analysis = None
        self.image_allocation_result = None

    def _initialize_shared_clip_session(self):
        """✅ 공유 CLIP 세션 초기화 (중복 방지)"""
        try:
            import open_clip
            import onnxruntime as ort
            import os
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent  # backend/app까지 올라가기
            onnx_model_path = project_root / "model" / "clip_onnx" / "clip_visual.quant.onnx"
            
            # PyTorch 텍스트 모델
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k', device="cpu"
            )
            clip_model.eval()
            
            # ONNX 이미지 모델
            onnx_session = None
            if os.path.exists(onnx_model_path):
                onnx_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
            
            self.shared_clip_session = {
                "clip_model": clip_model,
                "clip_preprocess": clip_preprocess,
                "onnx_session": onnx_session,
                "clip_available": onnx_session is not None
            }
            
            self.logger.info("✅ UnifiedMultimodalAgent: 공유 CLIP 세션 초기화 성공")
            
        except Exception as e:
            self.logger.error(f"공유 CLIP 세션 초기화 실패: {e}")
            self.shared_clip_session = {
                "clip_model": None,
                "clip_preprocess": None,
                "onnx_session": None,
                "clip_available": False
            }

    async def process_data(self, input_data):
        result = await self._do_work(input_data)
        
        await self.logging_manager.log_agent_response(
            agent_name=self.__class__.__name__,
            agent_role="통합 멀티모달 처리 에이전트",
            task_description="콘텐츠 구조화, 템플릿 선택, JSX 생성 통합 처리",
            response_data=result,
            metadata={"integrated_processing": True, "clip_shared": True}
        )
        
        return result

    async def process_magazine_unified(self, magazine_content: Dict, image_analysis: List[Dict], 
                                     user_id: str = "unknown_user") -> Dict:
        """✅ 하이브리드 방식: 요약 없는 CrewAI 분석 + 이미지 배치 강화"""
        
        self.logger.info("=== 하이브리드 멀티모달 매거진 처리 시작 ===")
        
        try:
            # ✅ 원본 데이터 및 이미지 데이터 보존
            self.current_magazine_content = magazine_content
            self.current_image_analysis = image_analysis
            
            self.logger.info(f"✅ 이미지 데이터 수신: {len(image_analysis)}개")
            
            # === 데이터 준비 단계 ===
            if "magazine_id" in magazine_content and magazine_content["magazine_id"]:
                self.logger.info(f"기존 매거진 ID로 데이터 조회: {magazine_content['magazine_id']}")
                magazine_data = await MagazineDBUtils.get_magazine_by_id(magazine_content["magazine_id"])
                if magazine_data:
                    magazine_content = magazine_data.get("content", magazine_content)
                    self.current_magazine_content = magazine_content
                    # 기존 이미지 데이터가 있으면 병합
                    existing_images = await MagazineDBUtils.get_images_by_magazine_id(magazine_content["magazine_id"])
                    if existing_images:
                        image_analysis.extend(existing_images)
                        self.current_image_analysis = image_analysis
            
            # 1. 의미 분석 (공유 CLIP 세션 사용)
            self.logger.info("1단계: 공유 CLIP 기반 의미 분석 실행")
            texts_for_analysis = self._extract_texts_from_sections(magazine_content.get('sections', []))
            similarity_data = await self.semantic_engine.calculate_semantic_similarity(
                texts_for_analysis, image_analysis
            )
            
            # 2. 통합 벡터 패턴 수집
            self.logger.info("2단계: 통합 벡터 패턴 수집")
            unified_patterns = await self._collect_unified_vector_patterns(
                magazine_content, image_analysis
            )
            
            # ✅ 3. 이미지 다양성 최적화 및 섹션별 할당
            self.logger.info("3단계: 이미지 다양성 최적화 및 섹션별 할당")
            optimization_result = await self._execute_image_allocation(
                image_analysis, magazine_content.get('sections', []), unified_patterns
            )
            
            self.image_allocation_result = optimization_result
            
            # 4. 요약 없는 CrewAI 분석
            self.logger.info("4단계: 요약 없는 CrewAI 구조 분석")
            structured_content = await self._execute_crew_analysis_without_summary(
                magazine_content, image_analysis, similarity_data, unified_patterns, user_id
            )
            
            # ✅ 5. 이미지가 포함된 JSX 생성
            self.logger.info("5단계: 이미지 포함 JSX 생성")
            final_sections = await self._process_sections_with_images(
                structured_content, unified_patterns, optimization_result
            )
            
            # 6. 최종 결과 구성
            result = {
                "content_sections": final_sections,
                "processing_metadata": {
                    "unified_processing": True,
                    "crew_ai_enhanced": True,
                    "structured_processing": True,
                    "total_sections": len(final_sections),
                    "original_content_preserved": True,
                    "hybrid_approach": True,
                    "images_allocated": len(image_analysis),
                    "sections_with_images": self._count_sections_with_images(final_sections)
                }
            }
            
            self.logger.info("=== 하이브리드 멀티모달 매거진 처리 완료 ===")
            return result
            
        except Exception as e:
            self.logger.error(f"하이브리드 매거진 처리 실패: {e}")


    async def _execute_image_allocation(self, image_analysis: List[Dict], sections: List[Dict], 
                                      unified_patterns: Dict) -> Dict:
        """✅ 이미지 할당 실행 (ImageDiversityManager 활용)"""
        try:
            # ImageDiversityManager를 통한 이미지 최적화
            optimization_result = await self.image_diversity_manager.optimize_image_distribution(
                image_analysis, sections, unified_patterns
            )
            
            # ✅ 결과 구조 검증 및 표준화
            if optimization_result and "allocation_plan" in optimization_result:
                # allocation_plan을 allocation_details로 복사 (호환성)
                if "allocation_details" not in optimization_result:
                    optimization_result["allocation_details"] = optimization_result["allocation_plan"]
                
                self.logger.info(f"✅ ImageDiversityManager 이미지 할당 성공")
                return optimization_result
            else:
                self.logger.warning("ImageDiversityManager 할당 결과가 비어있음, 직접 할당 수행")
                return self._direct_image_allocation(image_analysis, sections)
            
        except Exception as e:
            self.logger.error(f"이미지 할당 실패: {e}")
            return self._direct_image_allocation(image_analysis, sections)


    async def _execute_crew_analysis_without_summary(self, magazine_content: Dict, image_analysis: List[Dict],
                                                   similarity_data: Dict, unified_patterns: Dict, user_id: str) -> Dict:
        """요약 없는 CrewAI 분석 (전체 데이터 활용)"""
        try:
            # ✅ 요약 없는 컨텍스트 생성
            full_context = self._create_full_llm_context(
                magazine_content, image_analysis, similarity_data, unified_patterns
            )
            
            # 콘텐츠 구조 분석
            self.logger.info("CrewAI: 전체 데이터 기반 구조 분석 시작...")
            content_analysis_task = self._create_enhanced_content_analysis_task(full_context, user_id)
            content_crew = Crew(agents=[self.content_structure_agent], tasks=[content_analysis_task], verbose=False)
            content_result = content_crew.kickoff()
            
            return await self._process_crew_results(content_result)
            
        except Exception as e:
            self.logger.error(f"요약 없는 CrewAI 분석 실패: {e}")
            return self._create_fallback_content_result(magazine_content)

    def _create_full_llm_context(self, magazine_content: Dict, image_analysis: List[Dict], 
                               similarity_data: Dict, unified_patterns: Dict) -> str:
        """요약 없는 전체 LLM 컨텍스트 생성"""
        
        # ✅ 섹션별 완전한 데이터 포함 (요약하지 않음)
        section_details = []
        for i, section in enumerate(magazine_content.get("sections", [])[:3]):  # 섹션 수로만 제한
            section_detail = {
                "section_id": section.get("section_id"),
                "title": section.get("title")
            }
            
            sub_sections = section.get("sub_sections", [])
            if sub_sections:
                section_detail["sub_sections"] = [
                    {
                        "sub_section_id": sub.get("sub_section_id"),
                        "title": sub.get("title"),
                        "content_full": sub.get("body", "")  # ✅ 전체 콘텐츠 포함
                    } for sub in sub_sections[:2]  # 하위 섹션 수로만 제한
                ]
            else:
                section_detail["content_full"] = section.get("content", section.get("body", ""))
                
            section_details.append(section_detail)

        # 이미지 및 벡터 패턴 정보는 요약하여 포함
        context = {
            "magazine_title": magazine_content.get("magazine_title", "제목 없음"),
            "sections": section_details,  # ✅ 완전한 섹션 데이터
            "image_count": len(image_analysis),
            "vector_patterns": {
                "total_patterns": len(unified_patterns.get("section_mappings", {})),
                "sample_templates": list(unified_patterns.get("section_mappings", {}).keys())[:3]
            },
            "semantic_analysis": {
                "similarity_exists": similarity_data.get("similarity_matrix", np.array([])).size > 0
            }
        }
        
        return json.dumps(context, ensure_ascii=False, indent=2)

    async def _process_sections_with_images(self, structured_content: Dict, 
                                          unified_patterns: Dict, 
                                          optimization_result: Dict) -> List[Dict]:
        """✅ 이미지가 포함된 섹션 처리"""
        
        try:
            # CrewAI 구조화 결과 파싱
            if isinstance(structured_content, str):
                try:
                    structured_content = json.loads(structured_content)
                except json.JSONDecodeError:
                    self.logger.warning("구조화된 콘텐츠 JSON 파싱 실패, 직접 처리로 폴백")
                    return await self._process_sections_directly_with_images(
                        self.current_magazine_content, unified_patterns, optimization_result
                    )
            
            # ✅ 구조화된 데이터와 원본 데이터 결합
            original_sections = self.current_magazine_content.get("sections", [])
            structured_sections = structured_content.get("sections", [])
            
            final_sections = []
            section_counter = 0  # 전체 섹션 카운터
            
            for i, section in enumerate(original_sections):
                # 구조화된 정보 찾기
                structured_info = {}
                if i < len(structured_sections):
                    structured_info = structured_sections[i]
                
                sub_sections = section.get("sub_sections", [])
                
                if sub_sections:
                    # 하위 섹션 처리
                    for j, sub_section in enumerate(sub_sections):
                        enhanced_section_data = self._create_enhanced_section_data_with_images(
                            section, sub_section, structured_info, optimization_result, section_counter, j
                        )
                        
                        # 템플릿 선택 및 JSX 생성
                        template_code = await self._select_template_with_structured_info(
                            enhanced_section_data, structured_info, unified_patterns
                        )
                        
                        jsx_result = await self.jsx_generator.generate_jsx_from_template(
                            enhanced_section_data, template_code
                        )
                        
                        final_sections.append({
                            "title": enhanced_section_data["title"],
                            "jsx_code": jsx_result.get("jsx_code", ""),
                            "metadata": {
                                **jsx_result.get("metadata", {}),
                                "structured_analysis_used": True,
                                "processing_method": "structured_data_with_images",
                                "images_included": len(enhanced_section_data.get("images", []))
                            }
                        })
                        
                        section_counter += 1
                else:
                    # 단일 섹션 처리
                    enhanced_section_data = self._create_enhanced_single_section_data_with_images(
                        section, structured_info, optimization_result, section_counter
                    )
                    
                    template_code = await self._select_template_with_structured_info(
                        enhanced_section_data, structured_info, unified_patterns
                    )
                    
                    jsx_result = await self.jsx_generator.generate_jsx_from_template(
                        enhanced_section_data, template_code
                    )
                    
                    final_sections.append({
                        "title": enhanced_section_data["title"],
                        "jsx_code": jsx_result.get("jsx_code", ""),
                        "metadata": {
                            **jsx_result.get("metadata", {}),
                            "structured_analysis_used": True,
                            "processing_method": "structured_data_with_images",
                            "images_included": len(enhanced_section_data.get("images", []))
                        }
                    })
                    
                    section_counter += 1
            
            self.logger.info(f"✅ 이미지 포함 JSX 생성 완료: {len(final_sections)}개 컴포넌트")
            return final_sections
            
        except Exception as e:
            self.logger.error(f"이미지 포함 섹션 처리 실패, 직접 처리로 폴백: {e}")
            return await self._process_sections_directly_with_images(
                self.current_magazine_content, unified_patterns, optimization_result
            )

    def _create_enhanced_section_data_with_images(self, parent_section: Dict, sub_section: Dict, 
                                                structured_info: Dict, optimization_result: Dict, 
                                                section_index: int, subsection_index: int) -> Dict:
        """✅ 이미지가 포함된 향상된 섹션 데이터 생성"""
        
        # 기본 데이터
        sub_section_id = sub_section.get("sub_section_id", f"{section_index+1}-{subsection_index+1}")
        title = sub_section.get("title", f"하위 섹션 {sub_section_id}")
        subtitle = sub_section.get("subtitle", "")
        content = sub_section.get("body", "")
        
        parent_title = parent_section.get("title", "")
        combined_title = f"{parent_title}: {title}" if parent_title else title
        
        # ✅ 이미지 할당 (여러 경로에서 확인)
        assigned_images = self._get_images_for_section_index(section_index, optimization_result)
        
        self.logger.info(f"섹션 {section_index} 이미지 할당: {len(assigned_images)}개")
        
        # 구조화된 정보 추가
        structured_sub_sections = structured_info.get("sub_sections", [])
        structured_sub_info = {}
        if subsection_index < len(structured_sub_sections):
            structured_sub_info = structured_sub_sections[subsection_index]
        
        return {
            "section_id": sub_section_id,
            "title": combined_title,
            "subtitle": subtitle,
            "content": content,
            "images": self._format_images_for_jsx(assigned_images),
            "layout_type": "subsection",
            "metadata": {
                "is_subsection": True,
                "parent_section_id": parent_section.get("section_id"),
                "parent_section_title": parent_title,
                "image_count": len(assigned_images),
                "structured_recommendations": structured_sub_info.get("recommendations", {}),
                "style_hints": structured_sub_info.get("style_hints", [])
            }
        }

    def _create_enhanced_single_section_data_with_images(self, section: Dict, structured_info: Dict, 
                                                       optimization_result: Dict, section_index: int) -> Dict:
        """✅ 이미지가 포함된 단일 섹션 데이터 생성"""
        
        section_id = section.get("section_id", str(section_index + 1))
        title = section.get("title", f"섹션 {section_id}")
        subtitle = section.get("subtitle", "")
        content = section.get("content", section.get("body", ""))
        
        # ✅ 이미지 할당 (여러 경로에서 확인)
        assigned_images = self._get_images_for_section_index(section_index, optimization_result)
        
        self.logger.info(f"섹션 {section_index} 이미지 할당: {len(assigned_images)}개")
        
        return {
            "section_id": section_id,
            "title": title,
            "subtitle": subtitle,
            "content": content,
            "images": self._format_images_for_jsx(assigned_images),
            "layout_type": "standard",
            "metadata": {
                "is_subsection": False,
                "image_count": len(assigned_images),
                "structured_recommendations": structured_info.get("recommendations", {}),
                "style_hints": structured_info.get("style_hints", [])
            }
        }

    def _get_images_for_section_index(self, section_index: int, optimization_result: Dict) -> List[Dict]:
        """✅ 섹션 인덱스별 이미지 가져오기 (보장된 할당)"""
        
        section_key = f"section_{section_index}"
        assigned_images = []
        
        # 1차: allocation_details에서 확인
        if optimization_result and "allocation_details" in optimization_result:
            allocation_details = optimization_result["allocation_details"]
            if section_key in allocation_details:
                assigned_images = allocation_details[section_key].get("images", [])
        
        # 2차: allocation_plan에서 확인
        if not assigned_images and optimization_result and "allocation_plan" in optimization_result:
            allocation_plan = optimization_result["allocation_plan"]
            if section_key in allocation_plan:
                assigned_images = allocation_plan[section_key].get("images", [])
        
        # ✅ 3차: 전체 이미지에서 순환 할당 (보장)
        if not assigned_images and self.current_image_analysis:
            image_count = len(self.current_image_analysis)
            if image_count > 0:
                # 순환 방식으로 이미지 할당
                image_index = section_index % image_count
                assigned_images = [self.current_image_analysis[image_index]]
                
                self.logger.info(f"섹션 {section_index}에 순환 이미지 할당: 인덱스 {image_index}")
        
        # ✅ 4차: 최후의 수단 - 기본 이미지 생성
        if not assigned_images:
            assigned_images = [{
                "image_url": f"https://via.placeholder.com/600x400?text=Section+{section_index+1}",
                "image_name": f"placeholder_section_{section_index+1}",
                "description": f"섹션 {section_index+1} 기본 이미지",
                "width": 600,
                "height": 400
            }]
            
            self.logger.warning(f"섹션 {section_index}에 기본 이미지 할당")
        
        return assigned_images

    async def _process_sections_directly_with_images(self, magazine_content: Dict, 
                                                   unified_patterns: Dict, 
                                                   optimization_result: Dict) -> List[Dict]:
        """✅ 이미지 포함 직접 섹션 처리 (폴백용)"""
        
        sections = magazine_content.get("sections", [])
        final_sections = []
        
        section_counter = 0
        
        for i, section in enumerate(sections):
            sub_sections = section.get("sub_sections", [])
            
            if sub_sections:
                # 하위 섹션이 있는 경우: 각 sub_section을 개별 처리
                for j, sub_section in enumerate(sub_sections):
                    # 이미지 할당
                    assigned_images = self._get_images_for_section_index(section_counter, optimization_result)
                    
                    enhanced_section_data = {
                        "section_id": sub_section.get("sub_section_id", f"{i+1}-{j+1}"),
                        "title": f"{section.get('title', '')}: {sub_section.get('title', '')}",
                        "subtitle": sub_section.get("subtitle", ""),
                        "content": sub_section.get("body", ""),
                        "images": self._format_images_for_jsx(assigned_images),
                        "layout_type": "subsection"
                    }
                    
                    # 템플릿 선택 및 JSX 생성
                    template_code = await self.template_selector.analyze_and_select_template(enhanced_section_data)
                    jsx_result = await self.jsx_generator.generate_jsx_from_template(enhanced_section_data, template_code)
                    
                    final_sections.append({
                        "title": enhanced_section_data["title"],
                        "jsx_code": jsx_result.get("jsx_code", ""),
                        "metadata": {
                            **jsx_result.get("metadata", {}),
                            "content_length": len(enhanced_section_data["content"]),
                            "processing_method": "direct_with_images",
                            "images_included": len(assigned_images)
                        }
                    })
                    
                    section_counter += 1
            else:
                # 하위 섹션이 없는 경우: 단일 섹션 처리
                assigned_images = self._get_images_for_section_index(section_counter, optimization_result)
                
                enhanced_section_data = {
                    "section_id": section.get("section_id", str(i + 1)),
                    "title": section.get("title", f"섹션 {i+1}"),
                    "subtitle": section.get("subtitle", ""),
                    "content": section.get("content", section.get("body", "")),
                    "images": self._format_images_for_jsx(assigned_images),
                    "layout_type": "standard"
                }
                
                template_code = await self.template_selector.analyze_and_select_template(enhanced_section_data)
                jsx_result = await self.jsx_generator.generate_jsx_from_template(enhanced_section_data, template_code)
                
                final_sections.append({
                    "title": enhanced_section_data["title"],
                    "jsx_code": jsx_result.get("jsx_code", ""),
                    "metadata": {
                        **jsx_result.get("metadata", {}),
                        "content_length": len(enhanced_section_data["content"]),
                        "processing_method": "direct_with_images",
                        "images_included": len(assigned_images)
                    }
                })
                
                section_counter += 1
        
        self.logger.info(f"✅ 이미지 포함 직접 JSX 생성 완료: {len(final_sections)}개 컴포넌트")
        return final_sections

    async def _select_template_with_structured_info(self, section_data: Dict, 
                                                  structured_info: Dict, 
                                                  unified_patterns: Dict) -> str:
        """구조화된 정보를 활용한 향상된 템플릿 선택"""
        
        # ✅ 구조화된 정보를 섹션 데이터에 추가
        enhanced_section_data = {
            **section_data,
            "crew_recommendations": structured_info.get("template_recommendations", {}),
            "style_preferences": structured_info.get("style_preferences", []),
            "layout_suggestions": structured_info.get("layout_suggestions", [])
        }
        
        # 향상된 데이터로 템플릿 선택
        template_code = await self.template_selector.analyze_and_select_template(enhanced_section_data)
        
        # 템플릿 검증
        if not template_code or len(template_code.strip()) < 50:
            self.logger.warning(f"템플릿 선택 실패, 기본 템플릿 사용")
            template_code = self._get_default_template_code()
        
        return template_code

    def _format_images_for_jsx(self, images: List[Dict]) -> List[Dict]:
        """✅ 중복 제거가 포함된 JSX 포맷팅"""
        formatted_images = []
        seen_urls = set()  # ✅ URL 기반 중복 방지
        seen_hashes = set()  # ✅ 해시 기반 중복 방지
        
        for i, img in enumerate(images):
            # ✅ 실제 데이터 구조에 맞는 키 사용
            image_url = img.get("image_url", "")
            
            # ✅ URL 기반 1차 중복 검사
            if image_url in seen_urls:
                self.logger.debug(f"URL 중복 이미지 제거: {image_url}")
                continue
            
            # ✅ 해시 기반 2차 중복 검사 (있는 경우)
            image_hash = img.get("perceptual_hash", "")
            if image_hash and image_hash in seen_hashes:
                self.logger.debug(f"해시 중복 이미지 제거: {image_hash}")
                continue
            
            # ✅ 중복이 아닌 경우만 포함
            if image_url:
                seen_urls.add(image_url)
                if image_hash:
                    seen_hashes.add(image_hash)
            
            # ✅ 안전한 alt 텍스트 생성
            city = img.get("city", "")
            location = img.get("location", "")
            
            if city and location:
                clean_location = re.sub(r'["\'\(\)]', '', location[:30])
                simple_alt = f"{city} {clean_location}"
            else:
                simple_alt = f"베네치아 여행 이미지 {i+1}"
            
            safe_alt = re.sub(r'["\']', '', simple_alt)
            
            formatted_img = {
                "image_url": image_url,
                "url": image_url,
                "description": safe_alt,
                "alt": safe_alt,
                "width": 800,
                "height": 600,
                "caption": safe_alt,
                "perceptual_hash": image_hash  # ✅ 해시 정보 보존
            }
            formatted_images.append(formatted_img)
        
        self.logger.info(f"중복 제거 완료: {len(images)} → {len(formatted_images)}개 이미지")
        return formatted_images

    def _direct_image_allocation(self, image_analysis: List[Dict], sections: List[Dict]) -> Dict:
        """✅ 모든 섹션 이미지 배치 보장 직접 할당"""
        
        # ✅ 1. 실제 섹션 수 정확히 계산
        actual_sections = []
        for i, section in enumerate(sections):
            sub_sections = section.get("sub_sections", [])
            if sub_sections:
                for j, sub_section in enumerate(sub_sections):
                    actual_sections.append({
                        "section_index": len(actual_sections),
                        "original_index": i,
                        "sub_index": j,
                        "title": f"{section.get('title', '')}: {sub_section.get('title', '')}",
                        "is_subsection": True
                    })
            else:
                actual_sections.append({
                    "section_index": len(actual_sections),
                    "original_index": i,
                    "sub_index": None,
                    "title": section.get('title', f'섹션 {i+1}'),
                    "is_subsection": False
                })
        
        total_sections = len(actual_sections)
        
        # ✅ 2. 중복 제거된 이미지 준비
        unique_images = self._remove_duplicates_from_image_list(image_analysis)
        
        # ✅ 3. 이미지 부족 시 확장
        if len(unique_images) < total_sections:
            expanded_images = []
            for i in range(total_sections):
                if i < len(unique_images):
                    expanded_images.append(unique_images[i])
                else:
                    # 순환하여 이미지 재사용
                    cycle_index = i % len(unique_images)
                    expanded_images.append(unique_images[cycle_index])
            unique_images = expanded_images
        
        # ✅ 4. 모든 섹션에 균등 분배
        images_per_section = max(1, len(unique_images) // total_sections)
        allocation_details = {}
        
        for i, section_info in enumerate(actual_sections):
            section_key = f"section_{section_info['section_index']}"
            
            # 이미지 할당
            start_idx = i * images_per_section
            end_idx = start_idx + images_per_section
            
            # 마지막 섹션은 남은 모든 이미지 할당
            if i == total_sections - 1:
                end_idx = len(unique_images)
            
            allocated_images = unique_images[start_idx:end_idx]
            
            # 최소 1개 이미지 보장
            if not allocated_images and unique_images:
                allocated_images = [unique_images[i % len(unique_images)]]
            
            allocation_details[section_key] = {
                "images": allocated_images,
                "section_title": section_info["title"],
                "image_count": len(allocated_images),
                "is_subsection": section_info["is_subsection"],
                "original_section_index": section_info["original_index"],
                "guaranteed_allocation": True
            }
        
        self.logger.info(f"✅ 모든 섹션 이미지 할당 완료: {total_sections}개 섹션에 {len(unique_images)}개 이미지 분배")
        
        return {
            "allocation_plan": allocation_details,
            "allocation_details": allocation_details,
            "total_sections": total_sections,
            "total_images": len(unique_images),
            "original_image_count": len(image_analysis),
            "all_sections_covered": True,
            "method": "guaranteed_allocation"
        }

    def _remove_duplicates_from_image_list(self, images: List[Dict]) -> List[Dict]:
        """이미지 리스트에서 중복 제거"""
        unique_images = []
        seen_urls = set()
        seen_hashes = set()
        
        for img in images:
            image_url = img.get("image_url", "")
            image_hash = img.get("perceptual_hash", "")
            
            # URL 기반 중복 검사
            if image_url in seen_urls:
                continue
            
            # 해시 기반 중복 검사 (있는 경우)
            if image_hash and image_hash in seen_hashes:
                continue
            
            # 중복이 아닌 경우 추가
            if image_url:
                seen_urls.add(image_url)
                if image_hash:
                    seen_hashes.add(image_hash)
                unique_images.append(img)
        
        return unique_images

    def _count_sections_with_images(self, final_sections: List[Dict]) -> int:
        """이미지가 포함된 섹션 수 계산"""
        count = 0
        for section in final_sections:
            jsx_code = section.get("jsx_code", "")
            if "<img" in jsx_code and "src=" in jsx_code:
                count += 1
        return count

    def _extract_texts_from_sections(self, sections: List[Dict]) -> List[str]:
        """✅ 중첩된 sub_sections 구조에서 텍스트 추출"""
        texts = []
        
        for section in sections:
            sub_sections = section.get("sub_sections", [])
            
            if sub_sections:
                # sub_sections가 있는 경우, 각 sub_section의 body에서 텍스트 추출
                for sub_section in sub_sections:
                    body_text = sub_section.get("body", "")
                    if body_text and len(body_text.strip()) > 0:
                        texts.append(body_text)
            else:
                # sub_sections가 없는 경우, 최상위 content 또는 body에서 추출
                content_text = section.get("content", section.get("body", ""))
                if content_text and len(content_text.strip()) > 0:
                    texts.append(content_text)
        
        return texts

    async def _collect_unified_vector_patterns(self, magazine_content: Dict,
                                             image_analysis: List[Dict]) -> Dict:
        """✅ AI Search + JSX 템플릿 벡터 패턴 통합 수집"""
        
        sections = magazine_content.get("sections", [])
        self.logger.info(f"통합 벡터 패턴 수집: {len(sections)}개 섹션, {len(image_analysis)}개 이미지")
        
        try:
            unified_patterns = {
                "ai_search_patterns": [],
                "jsx_template_patterns": [],
                "integration_patterns": [],
                "section_mappings": {}
            }
            
            section_index = 0  # 전체 섹션 인덱스 (하위 섹션 포함)
            
            for i, section in enumerate(sections):
                # 하위 섹션이 있는지 확인
                sub_sections = section.get("sub_sections", [])
                
                # 하위 섹션이 없는 경우 - 단일 섹션으로 처리
                if not sub_sections:
                    section_key = f"section_{section_index}"
                    
                    # 패턴 수집 및 융합
                    patterns = await self._collect_section_patterns(section)
                    unified_patterns["section_mappings"][section_key] = patterns
                    
                    section_index += 1
                else:
                    # 하위 섹션이 있는 경우 - 각 하위 섹션을 개별적으로 처리
                    section_title = section.get("title", f"섹션 {i+1}")
                    
                    for j, sub_section in enumerate(sub_sections):
                        section_key = f"section_{section_index}"
                        
                        # 상위 섹션의 제목을 하위 섹션 제목에 추가
                        combined_title = f"{section_title}, {sub_section.get('title', '')}"
                        
                        # 하위 섹션 데이터 구성 (상위 섹션 정보 포함)
                        enhanced_sub_section = {
                            "title": combined_title,
                            "subtitle": sub_section.get("subtitle", ""),
                            "content": sub_section.get("body", ""),
                            "section_id": sub_section.get("sub_section_id", ""),
                            "parent_section_id": section.get("section_id", ""),
                            "parent_section_title": section_title
                        }
                        
                        # 패턴 수집 및 융합
                        patterns = await self._collect_section_patterns(enhanced_sub_section)
                        patterns.update({
                            "is_subsection": True,
                            "parent_section_id": section.get("section_id", ""),
                            "parent_section_title": section_title
                        })
                        
                        unified_patterns["section_mappings"][section_key] = patterns
                        section_index += 1
            
            return unified_patterns
            
        except Exception as e:
            self.logger.error(f"통합 벡터 패턴 수집 실패: {e}")
            return {"section_mappings": {}}

    async def _collect_section_patterns(self, section: Dict) -> Dict:
        """개별 섹션의 패턴 수집"""
        # 1. AI Search 패턴 수집
        ai_search_query = self._build_ai_search_query(section)
        ai_patterns = await self._search_ai_patterns(ai_search_query)
        
        # 2. JSX 템플릿 패턴 수집
        jsx_query = self._build_jsx_template_query(section, ai_patterns)
        jsx_patterns = await self._search_jsx_templates(jsx_query)
        
        # 3. 매거진 레이아웃 패턴 수집
        magazine_query = self._build_magazine_layout_query(section, ai_patterns)
        magazine_patterns = await self._search_magazine_patterns(magazine_query)
        
        # 4. 패턴 융합
        return {
            "ai_search_patterns": ai_patterns,
            "jsx_template_patterns": jsx_patterns,
            "magazine_layout_patterns": magazine_patterns,
            "semantic_score": self._calculate_semantic_alignment(ai_patterns, jsx_patterns),
            "recommended_templates": self._rank_templates(jsx_patterns, ai_patterns)
        }

    def _build_ai_search_query(self, section: Dict) -> str:
        """AI Search 쿼리 생성"""
        title = section.get("title", "")
        content = section.get("content", section.get("body", ""))[:200]
        
        if section.get("parent_section_id") or section.get("parent_section_title"):
            parent_title = section.get("parent_section_title", "")
            return f"magazine layout section subsection {parent_title} {title} {content}"
        else:
            return f"magazine layout section {title} {content}"

    def _build_jsx_template_query(self, section: Dict, ai_patterns: List[Dict]) -> str:
        """JSX 템플릿 쿼리 생성"""
        title = section.get("title", "")
        style_hints = [p.get("style", "") for p in ai_patterns[:2]]
        
        if section.get("parent_section_id") or section.get("parent_section_title"):
            parent_title = section.get("parent_section_title", "")
            return f"jsx component subsection {parent_title} {title} {' '.join(style_hints)}"
        else:
            return f"jsx component section {title} {' '.join(style_hints)}"

    def _build_magazine_layout_query(self, section: Dict, ai_patterns: List[Dict]) -> str:
        """매거진 레이아웃 쿼리 생성"""
        title = section.get("title", "")
        layout_hints = [p.get("layout_type", "") for p in ai_patterns[:2]]
        
        if section.get("parent_section_id") or section.get("parent_section_title"):
            parent_title = section.get("parent_section_title", "")
            subsection_id = section.get("section_id", "")
            return f"magazine layout design subsection {parent_title} {title} {subsection_id} {' '.join(layout_hints)}"
        else:
            section_id = section.get("section_id", "")
            return f"magazine layout design section {title} {section_id} {' '.join(layout_hints)}"

    async def _search_ai_patterns(self, query: str) -> List[Dict]:
        """AI Search 패턴 검색"""
        try:
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(query)
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "text-semantic-patterns-index", top_k=5
                )
            )
            return self.isolation_manager.filter_contaminated_data(results, f"ai_patterns_{hash(query)}")
        except Exception as e:
            self.logger.error(f"AI Search 패턴 검색 실패: {e}")
            return []

    async def _search_jsx_templates(self, query: str) -> List[Dict]:
        """JSX 템플릿 벡터 검색"""
        try:
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(query)
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "jsx-component-vector-index", top_k=5
                )
            )
            return self.isolation_manager.filter_contaminated_data(results, f"jsx_templates_{hash(query)}")
        except Exception as e:
            self.logger.error(f"JSX 템플릿 검색 실패: {e}")
            return []

    async def _search_magazine_patterns(self, query: str) -> List[Dict]:
        """매거진 레이아웃 벡터 검색"""
        try:
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(query)
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "magazine-vector-index", top_k=5
                )
            )
            return self.isolation_manager.filter_contaminated_data(results, f"magazine_patterns_{hash(query)}")
        except Exception as e:
            self.logger.error(f"매거진 패턴 검색 실패: {e}")
            return []

    def _calculate_semantic_alignment(self, ai_patterns: List[Dict], jsx_patterns: List[Dict]) -> float:
        """AI Search 패턴과 JSX 패턴 간 의미적 정렬도 계산"""
        if not ai_patterns or not jsx_patterns:
            return 0.0
        return 0.8  # 임시 점수

    def _rank_templates(self, jsx_patterns: List[Dict], ai_patterns: List[Dict]) -> List[str]:
        """템플릿 순위 매기기"""
        return [p.get("jsx_code", "DefaultTemplate.jsx") for p in jsx_patterns[:3]]

    async def _process_crew_results(self, crew_result):
        """CrewAI 결과 안전하게 처리 (JSON 파싱 강화)"""
        try:
            if crew_result is None:
                self.logger.warning("CrewAI 결과가 None입니다.")
                return self._create_fallback_content_result({})
            
            # CrewAI 결과에서 텍스트 추출
            result_text = ""
            if hasattr(crew_result, 'raw') and crew_result.raw:
                result_text = str(crew_result.raw)
            elif hasattr(crew_result, 'result') and crew_result.result:
                result_text = str(crew_result.result)
            elif hasattr(crew_result, 'output') and crew_result.output:
                result_text = str(crew_result.output)
            else:
                result_text = str(crew_result)
            
            # ✅ 빈 문자열 및 공백 체크
            if not result_text or result_text.strip() == "":
                self.logger.warning("CrewAI 결과가 빈 문자열입니다.")
                return self._create_fallback_content_result({})
            
            # ✅ JSON 형식 검증 및 추출
            if isinstance(result_text, str):
                # 마크다운 코드 블록 제거
                json_match = re.search(r'``````', result_text, re.DOTALL | re.IGNORECASE)
                if json_match:
                    result_text = json_match.group(1).strip()
                
                # 일반 코드 블록 제거
                elif '```' in result_text:
                    json_match = re.search(r'```\s*(.*?)\s*```', result_text, re.DOTALL)
                    if json_match:
                        result_text = json_match.group(1).strip()
                
                # ✅ JSON 객체 패턴 추출
                json_pattern = r'\{.*\}'
                json_match = re.search(json_pattern, result_text, re.DOTALL)
                if json_match:
                    result_text = json_match.group(0)
                
                # ✅ JSON 파싱 시도
                try:
                    # 문자열 정리
                    cleaned_text = result_text.strip()
                    
                    # 잘못된 따옴표 수정
                    cleaned_text = cleaned_text.replace("'", '"')
                    
                    # JSON 파싱
                    parsed_result = json.loads(cleaned_text)
                    
                    # ✅ 필수 필드 검증
                    if self._validate_crew_result_structure(parsed_result):
                        return parsed_result
                    else:
                        self.logger.warning("CrewAI 결과 구조가 올바르지 않습니다.")
                        return self._create_fallback_content_result({})
                        
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON 파싱 실패: {e}")
                    self.logger.warning(f"파싱 시도한 텍스트: {result_text[:200]}...")
                    
                    # ✅ 텍스트에서 정보 추출 시도
                    extracted_data = self._extract_data_from_text(result_text)
                    if extracted_data:
                        return extracted_data
                    else:
                        return self._create_fallback_content_result({})
            
            return self._create_fallback_content_result({})

        except Exception as e:
            self.logger.error(f"CrewAI 결과 처리 실패: {e}")
            return self._create_fallback_content_result({})
        
    def _extract_data_from_text(self, text: str) -> Optional[Dict]:
        """텍스트에서 구조화된 데이터 추출 시도"""
        try:
            # 섹션 제목 추출
            title_pattern = r'(?:제목|title)[:：]\s*([^\n]+)'
            titles = re.findall(title_pattern, text, re.IGNORECASE)
            
            if titles:
                sections = []
                for i, title in enumerate(titles[:3]):  # 최대 3개
                    section = {
                        "title": title.strip(),
                        "template_recommendations": {},
                        "style_preferences": [],
                        "layout_suggestions": []
                    }
                    sections.append(section)
                
                return {
                    "analysis": "text_extraction",
                    "sections": sections,
                    "status": "extracted_from_text"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"텍스트 데이터 추출 실패: {e}")
            return None
            
    def _validate_crew_result_structure(self, data: Dict) -> bool:
        """CrewAI 결과 구조 검증"""
        try:
            # 기본 구조 확인
            if not isinstance(data, dict):
                return False
            
            # sections 필드 확인
            if "sections" not in data:
                return False
            
            sections = data["sections"]
            if not isinstance(sections, list):
                return False
            
            # 각 섹션의 필수 필드 확인
            for section in sections:
                if not isinstance(section, dict):
                    return False
                
                required_fields = ["title"]
                for field in required_fields:
                    if field not in section:
                        return False
            
            return True
            
        except Exception:
            return False
        

    def _create_enhanced_content_analysis_task(self, full_context: str, user_id: str) -> Task:
        """향상된 콘텐츠 분석 태스크 생성 (JSON 형식 강화)"""
        return Task(
            description=f"""
    다음 완전한 매거진 데이터를 분석하여 최적의 구조화 계획을 수립하세요:

    **완전한 매거진 데이터:**
    {full_context}

    **분석 요구사항:**
    1. 각 섹션과 하위 섹션의 특성 분석
    2. 템플릿 선택을 위한 스타일 추천
    3. 이미지 배치를 위한 레이아웃 제안
    4. JSX 생성을 위한 메타데이터 준비

    **중요: 반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.**

    ```json
    {{
    "analysis": "content_structure_analysis",
    "sections": [
        {{
        "title": "섹션 제목",
        "template_recommendations": {{
            "primary": "추천 템플릿 이름",
            "alternative": "대안 템플릿 이름",
            "reason": "추천 이유"
        }},
        "style_preferences": ["스타일1", "스타일2"],
        "layout_suggestions": ["레이아웃1", "레이아웃2"]
        }}
    ],
    "status": "completed"
    }}
    주의사항:

    위 JSON 형식을 정확히 따르세요

    마크다운 코드 블록(```)은 사용하지 마세요

    추가 설명이나 텍스트는 포함하지 마세요

    유효한 JSON만 반환하세요
    """,
    expected_output="구조화된 섹션별 분석 및 추천 결과 (순수 JSON 형식)",
    agent=self.content_structure_agent
    )

    def _create_fallback_content_result(self, magazine_content: Dict) -> Dict:
        """콘텐츠 구조 분석 실패 시 기본 결과 생성 (JSON 파싱 실패 대응)"""
        sections = magazine_content.get("sections", [])
        
        fallback_sections = []
        for i, section in enumerate(sections[:3]):  # 최대 3개만 처리
            # 하위 섹션 확인
            sub_sections = section.get("sub_sections", [])
            
            if sub_sections:
                # 하위 섹션이 있는 경우 각각 처리
                for j, sub_section in enumerate(sub_sections[:2]):  # 하위 섹션도 최대 2개
                    fallback_section = {
                        "title": f"{section.get('title', f'섹션 {i+1}')}: {sub_section.get('title', f'하위섹션 {j+1}')}",
                        "template_recommendations": {
                            "primary": "MixedMagazine07",
                            "alternative": "MixedMagazine08",
                            "reason": "기본 텍스트-이미지 혼합 레이아웃"
                        },
                        "style_preferences": ["modern", "clean", "readable"],
                        "layout_suggestions": ["flexbox", "responsive", "image-text-balance"]
                    }
                    fallback_sections.append(fallback_section)
            else:
                # 단일 섹션 처리
                fallback_section = {
                    "title": section.get("title", f"섹션 {i+1}"),
                    "template_recommendations": {
                        "primary": "MixedMagazine07",
                        "alternative": "MixedMagazine08", 
                        "reason": "기본 텍스트-이미지 혼합 레이아웃"
                    },
                    "style_preferences": ["modern", "clean", "readable"],
                    "layout_suggestions": ["flexbox", "responsive", "image-text-balance"]
                }
                fallback_sections.append(fallback_section)
        
        result = {
            "analysis": "fallback_content_analysis",
            "sections": fallback_sections,
            "status": "fallback_used_due_to_json_parsing_failure"
        }
        
        self.logger.info(f"폴백 결과 생성 완료: {len(fallback_sections)}개 섹션")
        return result

    # CrewAI 에이전트 생성 메서드들
    def _create_content_structure_agent_with_ai_search(self):
        """AI Search 통합 콘텐츠 구조 에이전트"""
        return Agent(
            role="AI Search 벡터 데이터 기반 텍스트 구조 설계자",
            goal="AI Search의 PDF 벡터 데이터를 활용하여 매거진 콘텐츠의 최적 구조를 설계하고 텍스트를 배치",
            backstory="""당신은 15년간 National Geographic, Condé Nast Traveler 등 세계 최고 수준의 매거진에서 편집장으로 활동해온 전문가입니다.

**전문 경력:**
- 저널리즘 및 창작문학 복수 학위 보유
- 퓰리처상 여행 기사 부문 심사위원 3회 역임
- 80개국 이상의 여행 경험과 현지 문화 전문 지식
- AI Search 벡터 데이터 기반 레이아웃 분석 전문성

**AI Search 데이터 활용 전문성:**
당신은 AI Search의 PDF 벡터 데이터를 활용하여 다음과 같이 처리합니다:

1. **벡터 검색 기반 레이아웃 분석**: 3000+ 매거진 레이아웃 패턴을 검색하여 최적의 구조 설계
2. **콘텐츠 구조 최적화**: 벡터화된 텍스트 패턴을 분석하여 가독성과 임팩트를 극대화하는 배치
3. **매거진 스타일 적용**: AI Search 데이터에서 추출한 전문 매거진 수준의 편집 기준 적용
4. **글의 형태 및 문장 길이 최적화**: 벡터 데이터에서 참조한 글의 맺음, 문장 길이, 글의 형태 적용
5. **섹션별 구조 결정**: AI Search 패턴을 기반으로 제목, 부제목, 본문의 최적 구조 결정

**편집 철학:**
"AI Search의 방대한 매거진 데이터를 활용하여 모든 텍스트가 독자의 여행 욕구를 자극하고 감정적 연결을 만드는 강력한 스토리텔링 도구가 되도록 합니다."

**멀티모달 접근:**
- AI Search 벡터 데이터와 실시간 텍스트 분석의 결합
- 텍스트와 이미지의 의미적 연관성을 벡터 검색으로 최적화
- 독자 경험 최적화를 위한 AI Search 패턴 기반 통합적 설계""",
            verbose=True,
            llm=self.llm,
            multimodal=True
        )
        
    def _create_image_layout_agent_with_ai_search(self):
        """AI Search 통합 이미지 레이아웃 에이전트"""
        return Agent(
            role="AI Search 벡터 데이터 기반 이미지 배치 전문가",
            goal="AI Search의 PDF 벡터 데이터와 이미지 분석 결과를 활용하여 최적의 이미지 배치 전략을 수립",
            backstory="""당신은 12년간 Vogue, Harper's Bazaar, National Geographic 등에서 비주얼 디렉터로 활동해온 전문가입니다.

**전문 경력:**
- 시각 디자인 및 사진학 석사 학위 보유
- 국제 사진 전시회 큐레이터 경험
- 매거진 레이아웃에서 이미지-텍스트 조화의 심리학 연구
- AI Search 벡터 데이터 기반 시각적 패턴 분석 전문성

**AI Search 이미지 배치 전문성:**
당신은 AI Search의 벡터 데이터를 활용하여 다음과 같이 이미지를 배치합니다:

1. **벡터 검색 기반 레이아웃 추천**: 3000+ 매거진에서 유사한 이미지 배치 패턴을 검색하여 최적 배치 결정
2. **시각적 균형 최적화**: AI Search 데이터의 이미지 크기, 위치, 간격 패턴을 참조하여 완벽한 시각적 균형 구현
3. **의미적 연관성 강화**: 벡터 검색으로 텍스트 내용과 이미지의 의미적 연결성을 극대화
4. **이미지 개수 최적화**: AI Search 패턴을 기반으로 섹션별 최적 이미지 개수 결정
5. **배치 간격 및 크기 결정**: 벡터 데이터에서 추출한 이미지 크기 비율과 배치 간격 적용

**배치 철학:**
"AI Search의 방대한 시각 데이터를 활용하여 모든 이미지가 단순한 장식이 아니라 스토리를 강화하고 독자의 감정을 움직이는 핵심 요소가 되도록 합니다."

**멀티모달 접근:**
- AI Search 벡터 검색과 실시간 이미지 분석의 결합
- 텍스트 맥락을 고려한 벡터 기반 이미지 선택 및 배치
- AI Search 패턴을 활용한 전체적인 시각적 내러티브 구성""",
            verbose=True,
            llm=self.llm,
            multimodal=True
        )
        
    def _create_semantic_coordinator_agent_with_ai_search(self):
        """AI Search 통합 의미적 조율 에이전트"""
        return Agent(
            role="AI Search 기반 멀티모달 의미적 조율 전문가",
            goal="AI Search 벡터 데이터를 활용하여 텍스트와 이미지의 의미적 연관성을 분석하고 최적의 매거진 구조로 통합 조율",
            backstory="""당신은 20년간 복잡한 멀티미디어 프로젝트의 총괄 디렉터로 활동해온 전문가입니다.

**전문 경력:**
- 멀티미디어 커뮤니케이션 박사 학위 보유
- 국제 매거진 어워드 심사위원장 5회 역임
- AI Search 기반 텍스트-이미지 의미적 연관성 연구 전문가
- 벡터 검색 기반 콘텐츠 분석 시스템 설계 경험

**AI Search 조율 전문성:**
당신은 AI Search의 벡터 데이터를 활용하여 다음과 같이 조율합니다:

1. **벡터 기반 의미적 매칭**: AI Search 데이터를 활용하여 텍스트와 이미지 간의 의미적 연관성을 분석하고 최적 조합 도출
2. **AI Search 패턴 기반 일관성 보장**: 벡터 검색으로 매거진 전체의 톤, 스타일, 메시지 일관성 확보
3. **벡터 데이터 기반 독자 경험 최적화**: AI Search 패턴을 참조하여 독자의 인지적 흐름을 고려한 통합적 설계
4. **격리 시스템 기반 품질 보장**: AI Search 데이터 오염 방지하면서 최고 품질의 벡터 데이터만 활용
5. **실시간 분석과 벡터 검색의 융합**: 현재 콘텐츠와 AI Search 패턴의 완벽한 조화

**조율 철학:**
"AI Search의 방대한 매거진 데이터와 실시간 분석을 융합하여 텍스트와 이미지가 완벽하게 조화를 이루는 혁신적인 매거진을 만들어냅니다."

**통합 접근:**
- AI Search 벡터 검색과 실시간 콘텐츠 분석의 완벽한 융합
- 격리 시스템을 통한 데이터 무결성 검증 및 최적화
- 벡터 패턴 기반 최종 독자 경험의 완성도 극대화""",
            verbose=True,
            llm=self.llm,
            multimodal=True
        )


    def _get_default_template_code(self) -> str:
        """기본 템플릿 코드 반환 (템플릿 선택 실패 시 사용)"""
        return """import React from "react";

    const DefaultMagazineSection = () => {
    return (
        <div style={{ 
        backgroundColor: "white", 
        color: "black", 
        padding: "20px",
        fontFamily: "'Noto Sans KR', sans-serif",
        minHeight: "400px"
        }}>
        <h1 style={{ 
            fontSize: "2.5rem", 
            marginBottom: "16px",
            fontWeight: "bold",
            lineHeight: "1.2"
        }}>
            매거진 섹션 제목
        </h1>
        <h2 style={{ 
            fontSize: "1.25rem", 
            marginBottom: "12px", 
            color: "#606060",
            fontWeight: "500"
        }}>
            부제목
        </h2>
        <div style={{ 
            fontSize: "1rem", 
            lineHeight: "1.7", 
            marginBottom: "20px",
            textAlign: "justify"
        }}>
            <p>
            매거진 콘텐츠가 여기에 표시됩니다. 이 기본 템플릿은 템플릿 선택에 실패했을 때 
            사용되는 안전한 폴백 템플릿입니다. 텍스트와 이미지가 적절히 배치되어 
            읽기 쉬운 레이아웃을 제공합니다.
            </p>
        </div>
        <div style={{ 
            display: "grid", 
            gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", 
            gap: "16px", 
            marginTop: "20px" 
        }}>
            <img 
            src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600" 
            alt="기본 이미지" 
            style={{ 
                width: "100%", 
                height: "250px", 
                objectFit: "cover", 
                borderRadius: "8px",
                boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)"
            }}
            />
        </div>
        </div>
    );
    };

    export default DefaultMagazineSection;"""