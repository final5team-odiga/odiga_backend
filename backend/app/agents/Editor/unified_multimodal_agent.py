import re
import asyncio
import numpy as np
import json
from typing import Dict, List, Any
from crewai import Agent, Task, Crew
from custom_llm import get_azure_llm

from agents.Editor.semantic_analysis_engine import SemanticAnalysisEngine
from agents.Editor.image_diversity_manager import ImageDiversityManager
from agents.jsx.template_selector import SectionStyleAnalyzer
from agents.jsx.unified_jsx_generator import UnifiedJSXGenerator

from utils.isolation.ai_search_isolation import AISearchIsolationManager
from utils.data.pdf_vector_manager import PDFVectorManager
from utils.isolation.session_isolation import SessionAwareMixin
from utils.isolation.agent_communication_isolation import InterAgentCommunicationMixin
from utils.log.logging_manager import LoggingManager

from db.magazine_db_utils import MagazineDBUtils

class UnifiedMultimodalAgent(SessionAwareMixin, InterAgentCommunicationMixin):
    """통합 멀티모달 에이전트 - 템플릿 선택 및 JSX 생성 통합, CLIP 공유 및 벡터 통합"""
    
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
        
        # ✅ 새로 통합되는 컴포넌트
        self.template_selector = SectionStyleAnalyzer()
        self.jsx_generator = UnifiedJSXGenerator()
        self.jsx_generator.set_logger(logger)
        
        self.logging_manager = LoggingManager(self.logger)
        self.__init_session_awareness__()
        self.__init_inter_agent_communication__()
        
        # CrewAI 에이전트들
        self.content_structure_agent = self._create_content_structure_agent_with_ai_search()
        self.image_layout_agent = self._create_image_layout_agent_with_ai_search()
        self.semantic_coordinator_agent = self._create_semantic_coordinator_agent_with_ai_search()

        # ✅ 원본 데이터 보존용 변수
        self.current_magazine_content = None

    def _initialize_shared_clip_session(self):
        """✅ 공유 CLIP 세션 초기화 (중복 방지)"""
        try:
            import open_clip
            import onnxruntime as ort
            import os
            
            onnx_model_path = "models/clip_onnx/clip_visual.quant.onnx"
            
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
        """✅ 완전 통합된 매거진 처리 - 메인 진입점"""
        
        self.logger.info("=== 완전 통합 멀티모달 매거진 처리 시작 ===")
        
        try:
            # ✅ 원본 데이터 보존 (JSX 생성 시 사용)
            self.current_magazine_content = magazine_content
            
            # === 데이터 준비 단계 ===
            if "magazine_id" in magazine_content and magazine_content["magazine_id"]:
                self.logger.info(f"기존 매거진 ID로 데이터 조회: {magazine_content['magazine_id']}")
                magazine_data = await MagazineDBUtils.get_magazine_by_id(magazine_content["magazine_id"])
                if magazine_data:
                    magazine_content = magazine_data.get("content", magazine_content)
                    self.current_magazine_content = magazine_content
                    image_analysis = await MagazineDBUtils.get_images_by_magazine_id(magazine_content["magazine_id"])
            
            # ✅ 중첩된 sub_sections 구조에서 텍스트 추출
            texts_for_analysis = self._extract_texts_from_sections(magazine_content.get('sections', []))
            
            # 1. 의미 분석 (공유 CLIP 세션 사용)
            self.logger.info("1단계: 공유 CLIP 기반 의미 분석 실행")
            similarity_data = await self.semantic_engine.calculate_semantic_similarity(
                texts_for_analysis, image_analysis
            )
            
            # 2. 통합 벡터 패턴 수집 (AI Search + JSX 템플릿)
            self.logger.info("2단계: 통합 벡터 패턴 수집")
            unified_patterns = await self._collect_unified_vector_patterns(
                magazine_content, image_analysis
            )
            
            # 3. 이미지 다양성 최적화 (벡터 통합 + 공유 CLIP)
            self.logger.info("3단계: 벡터 통합 이미지 다양성 최적화")
            optimization_result = await self.image_diversity_manager.optimize_image_distribution(
                image_analysis, magazine_content.get('sections', []), unified_patterns
            )
            
            # 4. CrewAI 기반 콘텐츠 구조화
            self.logger.info("4단계: CrewAI 기반 콘텐츠 구조화")
            structured_content = await self._execute_crew_analysis(
                magazine_content, image_analysis, similarity_data, unified_patterns, user_id
            )
            
            # ✅ 5. 통합된 템플릿 선택 및 JSX 생성
            self.logger.info("5단계: 통합 템플릿 선택 및 JSX 생성")
            final_sections = await self._process_sections_with_templates_and_jsx(
                structured_content, unified_patterns, optimization_result
            )
            
            # 6. 최종 결과 구성
            result = {
                "content_sections": final_sections,
                "processing_metadata": {
                    "unified_processing": True,
                    "template_selection_integrated": True,
                    "jsx_generation_integrated": True,
                    "total_sections": len(final_sections),
                    "ai_search_enhanced": True,
                    "diversity_optimized": True,
                    "clip_shared": True,
                    "vector_integrated": True,
                    "subsection_processing": True
                }
            }
            
            # 7. 결과 로깅
            if "magazine_id" in magazine_content and magazine_content["magazine_id"]:
                self.logger.info(f"멀티모달 처리 결과 로깅: {magazine_content['magazine_id']}")
                await self.logging_manager.log_multimodal_processing_completion(result)
            
            self.logger.info("=== 완전 통합 멀티모달 매거진 처리 완료 ===")
            return result
            
        except Exception as e:
            self.logger.error(f"통합 매거진 처리 실패: {e}")
            return self._create_fallback_result(magazine_content, image_analysis)

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

    def _create_llm_context(self, magazine_content: Dict, image_analysis: List[Dict], 
                        similarity_data: Dict, unified_patterns: Dict) -> str:
        """
        LLM(CrewAI) 에이전트를 위한 지능적이고 간결한 컨텍스트를 생성합니다.
        중첩된 sub_sections 구조를 올바르게 처리하여 데이터 손실을 방지합니다.
        """
        
        # 1. 매거진 콘텐츠 요약 (sub_sections 처리 기능 추가)
        section_overviews = []
        for section in magazine_content.get("sections", [])[:5]: # 최대 5개 섹션
            overview = {
                "section_id": section.get("section_id"),
                "title": section.get("title")
            }
            
            sub_sections = section.get("sub_sections")
            if sub_sections:
                # ✅ sub_sections가 있는 경우, 각 sub_section을 요약
                overview["sub_section_previews"] = [
                    {
                        "sub_section_id": sub.get("sub_section_id"),
                        "title": sub.get("title"),
                        # ✅ body 필드에서 내용 미리보기 추출
                        "content_preview": sub.get("body", "")[:150] + "..." if sub.get("body") else ""
                    } for sub in sub_sections[:3] # 섹션당 최대 3개 하위 섹션
                ]
            else:
                # ✅ sub_sections가 없는 경우, 최상위 content/body 필드에서 추출
                overview["content_preview"] = section.get("content", section.get("body", ""))[:150] + "..." if section.get("content") or section.get("body") else ""
                
            section_overviews.append(overview)

        magazine_summary = {
            "magazine_title": magazine_content.get("magazine_title", "제목 없음"),
            "total_sections": len(magazine_content.get("sections", [])),
            "section_overviews": section_overviews
        }

        # 2. 이미지 분석 요약
        image_summary = {
            "total_images": len(image_analysis),
            "image_previews": [
                {
                    "description": img.get("description", "설명 없음")[:100],
                    "quality_score": round(img.get("overall_quality", 0), 2),
                    "location": f"{img.get('city', '')}, {img.get('country', '')}"
                } for img in image_analysis[:5] # 최대 5개 이미지
            ]
        }
        
        # 3. 의미 분석 요약 (JSON 직렬화 오류 수정 포함)
        similarity_matrix = similarity_data.get("similarity_matrix", np.array([]))
        similarity_summary = {
            "text_image_similarity_exists": similarity_matrix.size > 0,
            "average_similarity": float(np.mean(similarity_matrix)) if similarity_matrix.size > 0 else "N/A"
        }

        # 4. 통합 벡터 패턴 요약
        pattern_summary = {
            "total_pattern_sections": len(unified_patterns.get("section_mappings", {})),
            "sample_pattern": next(iter(unified_patterns.get("section_mappings", {}).values()), {}).get("recommended_templates", [])[:1]
        }

        # 5. 최종 컨텍스트 조합
        llm_context = {
            "magazine_summary": magazine_summary,
            "image_summary": image_summary,
            "semantic_analysis_summary": similarity_summary,
            "vector_pattern_summary": pattern_summary
        }

        return json.dumps(llm_context, ensure_ascii=False, indent=2)

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

    async def _execute_crew_analysis(self, magazine_content: Dict, image_analysis: List[Dict],
                                   similarity_data: Dict, unified_patterns: Dict, user_id: str) -> Dict:
        """CrewAI 기반 콘텐츠 구조화"""
        try:
            # 1. LLM에 전달할 초기 통합 컨텍스트 생성
            llm_context = self._create_llm_context(
                magazine_content, image_analysis, similarity_data, unified_patterns
            )
            
            # 2. 콘텐츠 구조 분석 태스크 실행
            self.logger.info("CrewAI: 콘텐츠 구조 분석 시작...")
            content_analysis_task = self._create_content_analysis_task(llm_context, user_id)
            content_crew = Crew(agents=[self.content_structure_agent], tasks=[content_analysis_task], verbose=False)
            content_result = content_crew.kickoff()
            processed_content_result = await self._process_crew_results(content_result)
            
            # 3. 이미지 레이아웃 분석 태스크 실행
            self.logger.info("CrewAI: 이미지 레이아웃 분석 시작...")
            image_layout_task = self._create_image_layout_task(llm_context)
            image_crew = Crew(agents=[self.image_layout_agent], tasks=[image_layout_task], verbose=False)
            image_result = image_crew.kickoff()
            processed_image_result = await self._process_crew_results(image_result)
            
            # 4. 의미적 조율 태스크 실행
            self.logger.info("CrewAI: 최종 의미적 조율 시작...")
            coordination_context = f"""
# 조율 작업 지시

다음은 텍스트 구조 분석 결과와 이미지 배치 전략입니다. 두 결과를 종합하여 최종 매거진 구조를 완성하세요.

## 1. 텍스트 구조 분석 결과
{self._summarize_data_for_prompt(processed_content_result)}

## 2. 이미지 배치 전략
{self._summarize_data_for_prompt(processed_image_result)}

## 조율 목표
- 텍스트 구조와 이미지 배치가 의미적으로, 시각적으로 완벽하게 조화를 이루도록 하세요.
- 전체적인 완성도를 높여 독자에게 최고의 경험을 제공하는 최종 구조를 JSON 형식으로 출력하세요.
"""
            coordination_task = self._create_coordination_task(coordination_context)
            coordination_crew = Crew(agents=[self.semantic_coordinator_agent], tasks=[coordination_task], verbose=False)
            final_result = coordination_crew.kickoff()
            
            return await self._process_crew_results(final_result)
            
        except Exception as e:
            self.logger.error(f"CrewAI 분석 실패: {e}")
            return self._create_fallback_content_result(magazine_content)

    def _summarize_data_for_prompt(self, data: Any, max_len: int = 3000) -> str:
        """
        LLM 프롬프트를 위해 데이터를 안전하게 요약하고 직렬화합니다.
        Numpy 데이터 타입을 처리하고, 재귀적으로 중첩된 구조를 요약하여 429 오류를 방지합니다.
        """
        try:
            # 1. Numpy 데이터 타입 처리 (JSON 직렬화 오류 방지)
            if isinstance(data, np.ndarray):
                if data.size > 10:
                    return f"Numpy array with shape {data.shape}. Sample: {data.flatten()[:5].tolist()}"
                return data.tolist()
            if isinstance(data, (np.float32, np.float64)):
                return float(data)
            if isinstance(data, (np.int32, np.int64)):
                return int(data)

            # 2. 딕셔너리 데이터 타입 처리
            if isinstance(data, dict):
                summary_dict = {}
                for key, value in data.items():
                    summary_dict[key] = self._summarize_data_for_prompt(value, max_len=500)
                
                json_str = json.dumps(summary_dict, ensure_ascii=False, indent=2)

            # 3. 리스트 데이터 타입 처리
            elif isinstance(data, list):
                if len(data) > 3:
                    summary_list = [f"List with {len(data)} items. First 2 items summarized:"]
                    for item in data[:2]:
                        summary_list.append(self._summarize_data_for_prompt(item, max_len=500))
                    json_str = "\n".join(summary_list)
                else:
                    json_str = json.dumps([self._summarize_data_for_prompt(item, max_len=500) for item in data], ensure_ascii=False, indent=2)

            # 4. 그 외 데이터 타입 (문자열, 숫자 등) 처리
            else:
                json_str = str(data)

            # 5. 최종 길이 제한 (안전장치)
            if len(json_str) > max_len:
                return json_str[:max_len] + f"... (내용이 너무 길어 {len(json_str)}자에서 잘림)"
                
            return json_str
        except Exception as e:
            self.logger.error(f"데이터 요약 중 오류 발생: {e}")
            return f"Error summarizing data: {str(data)[:200]}..."

    async def _process_sections_with_templates_and_jsx(self, structured_content: Dict, 
                                                     unified_patterns: Dict, 
                                                     optimization_result: Dict) -> List[Dict]:
        """✅ 중첩된 sub_sections 구조를 올바르게 처리하는 통합 버전"""
        
        try:
            # CrewAI 구조화 결과에서 섹션 정보 추출
            if isinstance(structured_content, str):
                try:
                    structured_content = json.loads(structured_content)
                except json.JSONDecodeError:
                    self.logger.warning("구조화된 콘텐츠 JSON 파싱 실패, 원본 데이터 사용")
                    structured_content = {}
            
            # ✅ 원본 매거진 데이터 가져오기
            original_magazine_content = self.current_magazine_content
            sections = original_magazine_content.get("sections", [])
            
            final_sections = []
            
            for i, section in enumerate(sections):
                section_id = section.get("section_id", str(i + 1))
                
                # ✅ 핵심: sub_sections 구조 인식 및 처리
                sub_sections = section.get("sub_sections", [])
                
                if sub_sections:
                    # sub_sections가 있는 경우: 각 sub_section을 개별 JSX로 생성
                    for j, sub_section in enumerate(sub_sections):
                        sub_section_data = await self._process_single_subsection(
                            section, sub_section, unified_patterns, optimization_result, i, j
                        )
                        if sub_section_data:
                            final_sections.append(sub_section_data)
                else:
                    # sub_sections가 없는 경우: 기존 방식으로 처리
                    section_data = await self._process_single_section_legacy(
                        section, unified_patterns, optimization_result, i
                    )
                    if section_data:
                        final_sections.append(section_data)
            
            self.logger.info(f"✅ 최종 JSX 생성 완료: {len(final_sections)}개 컴포넌트")
            return final_sections
            
        except Exception as e:
            self.logger.error(f"섹션 처리 실패: {e}")
            return self._create_fallback_sections()

    async def _process_single_subsection(self, parent_section: Dict, sub_section: Dict, 
                                       unified_patterns: Dict, optimization_result: Dict, 
                                       section_index: int, subsection_index: int) -> Dict:
        """개별 하위 섹션을 JSX로 변환"""
        
        try:
            # ✅ 하위 섹션의 실제 콘텐츠 추출
            sub_section_id = sub_section.get("sub_section_id", f"{section_index+1}-{subsection_index+1}")
            title = sub_section.get("title", f"하위 섹션 {sub_section_id}")
            subtitle = sub_section.get("subtitle", "")
            content = sub_section.get("body", "")  # ✅ 실제 콘텐츠는 'body'에 있음
            
            # 상위 섹션 정보 포함
            parent_title = parent_section.get("title", "")
            combined_title = f"{parent_title}: {title}" if parent_title else title
            
            # 이미지 할당 (섹션 인덱스 기반)
            section_key = f"section_{section_index}"
            assigned_images = optimization_result.get("allocation_details", {}).get(section_key, {}).get("images", [])
            
            # JSX 생성을 위한 데이터 구성
            enhanced_section_data = {
                "section_id": sub_section_id,
                "title": combined_title,
                "subtitle": subtitle,
                "content": content,  # ✅ 실제 콘텐츠 사용
                "images": assigned_images[:5],  # 최대 5개 이미지
                "layout_type": "subsection",
                "metadata": {
                    "is_subsection": True,
                    "parent_section_id": parent_section.get("section_id"),
                    "parent_section_title": parent_title
                }
            }
            
            # 템플릿 선택
            template_code = await self.template_selector.analyze_and_select_template(enhanced_section_data)
            
            # JSX 생성
            jsx_result = await self.jsx_generator.generate_jsx_from_template(
                enhanced_section_data, template_code
            )
            
            self.logger.info(f"✅ 하위 섹션 '{title}' JSX 생성 완료 (콘텐츠 길이: {len(content)}자)")
            
            return {
                "title": combined_title,
                "jsx_code": jsx_result.get("jsx_code", ""),
                "metadata": jsx_result.get("metadata", {})
            }
            
        except Exception as e:
            self.logger.error(f"하위 섹션 처리 실패: {e}")
            return self._create_fallback_section_data(sub_section.get("title", "섹션"))

    async def _process_single_section_legacy(self, section: Dict, unified_patterns: Dict, 
                                           optimization_result: Dict, section_index: int) -> Dict:
        """기존 방식의 단일 섹션 처리 (sub_sections가 없는 경우)"""
        
        try:
            section_id = section.get("section_id", str(section_index + 1))
            title = section.get("title", f"섹션 {section_id}")
            subtitle = section.get("subtitle", "")
            content = section.get("content", section.get("body", ""))  # ✅ content 또는 body
            
            # 이미지 할당
            section_key = f"section_{section_index}"
            assigned_images = optimization_result.get("allocation_details", {}).get(section_key, {}).get("images", [])
            
            enhanced_section_data = {
                "section_id": section_id,
                "title": title,
                "subtitle": subtitle,
                "content": content,
                "images": assigned_images[:5],
                "layout_type": "standard",
                "metadata": {
                    "is_subsection": False
                }
            }
            
            # 템플릿 선택 및 JSX 생성
            template_code = await self.template_selector.analyze_and_select_template(enhanced_section_data)
            jsx_result = await self.jsx_generator.generate_jsx_from_template(enhanced_section_data, template_code)
            
            return {
                "title": title,
                "jsx_code": jsx_result.get("jsx_code", ""),
                "metadata": jsx_result.get("metadata", {})
            }
            
        except Exception as e:
            self.logger.error(f"레거시 섹션 처리 실패: {e}")
            return self._create_fallback_section_data(section.get("title", "섹션"))

    async def _process_crew_results(self, crew_result):
        """CrewAI 결과 안전하게 처리 (JSON 파싱 강화)"""
        try:
            if crew_result is None:
                return "결과 없음"
            
            # CrewAI 결과에서 텍스트 추출
            result_text = ""
            if hasattr(crew_result, 'raw') and crew_result.raw:
                result_text = crew_result.raw
            elif hasattr(crew_result, 'result') and crew_result.result:
                result_text = crew_result.result
            elif hasattr(crew_result, 'output') and crew_result.output:
                result_text = crew_result.output
            else:
                result_text = str(crew_result)
            
            # ✅ JSON 파싱 강화: 마크다운 코드 블록 제거
            if isinstance(result_text, str):
                # 마크다운 코드 블록 제거
                json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', result_text)
                if json_match:
                    result_text = json_match.group(1).strip()
                
                # JSON 파싱 시도
                try:
                    parsed_result = json.loads(result_text)
                    return parsed_result
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON 파싱 실패, 원본 텍스트 반환: {e}")
                    return result_text
            
            return result_text if result_text else "분석 결과 없음"

        except Exception as e:
            self.logger.error(f"CrewAI 결과 처리 실패: {e}")
            return f"결과 처리 실패: {e}"

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

    # CrewAI 태스크 생성 메서드들
    def _create_content_analysis_task(self, llm_context, user_id):
        return Task(
            description=f"""
다음 매거진 콘텐츠와 이미지 분석 결과를 통합 벡터 데이터와 함께 분석하여 최적의 텍스트 구조를 설계하세요:

**종합 분석 요약:**
{llm_context}

**사용자 정보:**
- 사용자 ID: {user_id}

**통합 분석 요구사항:**
1. AI Search + JSX 템플릿 + 매거진 레이아웃 패턴 기반 구조 분석
2. 벡터 검색 결과를 활용한 최적 텍스트 배치
3. 템플릿 선택을 위한 메타데이터 준비
4. JSX 생성을 위한 구조화된 데이터 제공
5. 공유 CLIP 세션 기반 의미적 연관성 고려

**출력 형식:**
- 구조화된 섹션별 분석 결과 (통합 패턴 반영)
- 템플릿 선택을 위한 메타데이터 포함
- JSX 생성 준비된 데이터 구조
""",
            expected_output="통합 벡터 패턴 기반 구조화된 텍스트 분석 결과 (JSON 형식)",
            agent=self.content_structure_agent
        )

    def _create_image_layout_task(self, llm_context):
        return Task(
            description=f"""
다음 이미지 분석 결과와 텍스트 내용을 통합 벡터 데이터와 함께 고려하여 최적의 이미지 배치 전략을 수립하세요:

**종합 분석 요약:**
{llm_context}

**통합 배치 요구사항:**
1. AI Search + JSX 템플릿 + 매거진 레이아웃 패턴 기반 이미지 배치
2. 벡터 검색 결과를 활용한 시각적 균형 최적화
3. 템플릿별 이미지 할당 전략 수립
4. JSX 컴포넌트 생성을 위한 배치 데이터 준비
5. 공유 CLIP 세션 기반 이미지-텍스트 의미적 매칭

**출력 형식:**
- 통합 패턴 기반 이미지 배치 전략
- 템플릿별 최적화된 이미지 할당
- JSX 생성을 위한 배치 메타데이터
            """,
            expected_output="통합 벡터 패턴 기반 구조화된 이미지 배치 전략 (JSON 형식)",
            agent=self.image_layout_agent
        )

    def _create_coordination_task(self, coordination_context: str) -> Task:
        """의미적 조율 태스크 생성"""
        return Task(
            description=coordination_context,
            expected_output="텍스트와 이미지가 완벽하게 조율된 최종 매거진 구조 (JSON 형식)",
            agent=self.semantic_coordinator_agent
        )

    # 폴백 메서드들
    def _create_fallback_section_data(self, title: str) -> Dict:
        """폴백 섹션 데이터 생성"""
        return {
            "title": title,
            "jsx_code": f"""
            export default function DefaultSection(props) {{
              return (
                <div className="section-container p-4 my-8">
                  <h2 className="text-2xl font-bold mb-2">{title}</h2>
                  <div className="content" dangerouslySetInnerHTML={{{{ __html: props.content }}}} />
                </div>
              );
            }}
            """,
            "metadata": {
                "template_applied": False,
                "generation_method": "fallback",
                "error": "섹션 처리 중 오류 발생"
            }
        }

    def _create_fallback_sections(self) -> List[Dict]:
        """폴백 섹션들 생성"""
        return [
            {
                "title": "기본 섹션",
                "jsx_code": """
                export default function DefaultSection(props) {
                  return (
                    <div className="section-container p-4 my-8">
                      <h2 className="text-2xl font-bold mb-2">기본 섹션</h2>
                      <div className="content" dangerouslySetInnerHTML={{ __html: props.content }} />
                    </div>
                  );
                }
                """,
                "metadata": {
                    "template_applied": False,
                    "generation_method": "fallback",
                    "error": "전체 섹션 처리 실패"
                }
            }
        ]

    def _create_fallback_result(self, magazine_content: Dict, image_analysis: List[Dict]) -> Dict:
        """전체 처리 실패 시 폴백 결과 생성"""
        return {
            "content_sections": self._create_fallback_sections(),
            "processing_metadata": {
                "unified_processing": False,
                "template_selection_integrated": False,
                "jsx_generation_integrated": False,
                "total_sections": 1,
                "error": "통합 매거진 처리 실패",
                "fallback_used": True
            }
        }

    def _create_fallback_content_result(self, magazine_content: Dict) -> str:
        """콘텐츠 구조 분석 실패 시 기본 결과 생성"""
        sections = magazine_content.get("sections", [])
        result = {
            "analysis": "fallback_content_analysis",
            "sections": [
                {
                    "title": section.get("title", f"섹션 {i+1}"),
                    "subtitle": section.get("subtitle", ""),
                    "content_summary": section.get("content", "")[:100] + "...",
                    "recommended_template": "DefaultTemplate.jsx"
                }
                for i, section in enumerate(sections)
            ],
            "status": "fallback"
        }
        return json.dumps(result)
