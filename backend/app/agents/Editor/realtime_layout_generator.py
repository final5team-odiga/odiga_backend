import asyncio
import json
import time
from typing import Dict, List, Any
from custom_llm import get_azure_llm
from utils.isolation.ai_search_isolation import AISearchIsolationManager
from utils.data.pdf_vector_manager import PDFVectorManager
from utils.isolation.session_isolation import SessionAwareMixin
from utils.isolation.agent_communication_isolation import InterAgentCommunicationMixin
from utils.log.logging_manager import LoggingManager
from collections import Counter

class RealtimeLayoutGenerator(SessionAwareMixin, InterAgentCommunicationMixin):
    """실시간 레이아웃 생성기 - AI Search 벡터 데이터 기반 레이아웃 생성"""
    

    def __init__(self, vector_manager: PDFVectorManager, logger: Any):
        self.llm = get_azure_llm()
        self.logger = logger 
        self._setup_logging_system()
        self.logging_manager = LoggingManager(self.logger)

        self.isolation_manager = AISearchIsolationManager()
  
        self.vector_manager = vector_manager 

        self.__init_session_awareness__()
        self.__init_inter_agent_communication__()

    def _setup_logging_system(self):
        """로그 저장 시스템 설정"""
        self.log_enabled = True
        self.response_counter = 0
        
    async def process_data(self, input_data):
        # 에이전트 작업 수행
        result = await self._do_work(input_data)
        
        # ✅ 응답 로그 저장
        await self.logging_manager.log_agent_response(
            agent_name=self.__class__.__name__,
            agent_role="에이전트 역할 설명",
            task_description="수행한 작업 설명",
            response_data=result,  # 실제 응답 데이터만
            metadata={"additional": "info"}
        )
        
        return result
    async def _log_layout_generation_response(self, layout_result: Dict) -> str:
        """레이아웃 생성 결과 로그 저장"""
        if not self.log_enabled:
            return "logging_disabled"
        
        try:
            response_data = {
                "agent_name": "RealtimeLayoutGenerator",
                "generation_type": "optimized_layouts",
                "total_layouts": len(layout_result.get("optimized_layouts", [])),
                "optimization_level": "ai_search_enhanced",
                "responsive_design": layout_result.get("generation_metadata", {}).get("responsive_design", False),
                "vector_patterns_used": layout_result.get("generation_metadata", {}).get("vector_patterns_used", False),
                "timestamp": time.time(),
                "session_id": self.current_session_id
            }
            
            response_id = f"LayoutGeneration_{int(time.time() * 1000000)}"
            
            # 세션별 저장
            self.store_result(response_data)
            
            self.logger.info(f"📦 RealtimeLayoutGenerator 응답 저장: {response_id}")
            return response_id
            
        except Exception as e:
            self.logger.error(f"로그 저장 실패: {e}")
            return "log_save_failed"
            
    async def generate_optimized_layouts(self, semantic_analysis: Dict, 
                                       available_templates: List[str]) -> Dict:
        """AI Search 벡터 데이터를 활용한 최적화된 레이아웃 생성"""
        
        self.logger.info("=== 실시간 레이아웃 생성 시작 (AI Search 통합) ===")
        
        # 1. AI Search 기반 템플릿별 레이아웃 전략 수립
        layout_strategies = await self._develop_layout_strategies_with_ai_search(
            semantic_analysis, available_templates
        )
        
        # 2. 벡터 데이터 기반 콘텐츠 배치 최적화
        optimized_layouts = await self._optimize_content_placement_with_vectors(
            layout_strategies, semantic_analysis
        )
        
        # 3. AI Search 패턴 기반 시각적 균형 검증
        balanced_layouts = await self._ensure_visual_balance_with_patterns(optimized_layouts)
        
        # 4. 벡터 데이터 기반 반응형 레이아웃 적용
        responsive_layouts = await self._apply_responsive_design_with_ai_search(balanced_layouts)
        
        result = {
            "layout_strategies": layout_strategies,
            "optimized_layouts": responsive_layouts,
            "generation_metadata": {
                "total_layouts": len(responsive_layouts),
                "optimization_level": "ai_search_enhanced",
                "responsive_design": True,
                "vector_patterns_used": True,
                "isolation_applied": True
            }
        }
        
        #  로그 저장 추가
        response_id = await self._log_layout_generation_response(result)
        result["response_id"] = response_id
        
        # ✅ 세션별 결과 저장
        self.store_result(result)
        
        return result
    
    async def _develop_layout_strategies_with_ai_search(self, semantic_analysis: Dict, 
                                                      available_templates: List[str]) -> Dict:
        """AI Search 벡터 데이터 기반 템플릿별 레이아웃 전략 수립"""
        
        strategies = {}
        optimal_combinations = semantic_analysis.get("optimal_combinations", [])
        
        for i, template in enumerate(available_templates):
            if i < len(optimal_combinations):
                combination = optimal_combinations[i]
                
                # AI Search에서 해당 템플릿과 유사한 레이아웃 패턴 검색
                layout_patterns = await self._search_template_layout_patterns(template, combination)
                
                strategy = await self._create_template_strategy_with_patterns(
                    template, combination, semantic_analysis, layout_patterns
                )
                strategies[template] = strategy
        
        return strategies
    
    async def _search_template_layout_patterns(self, template: str, combination: Dict) -> List[Dict]:
        """특정 템플릿에 대한 AI Search 레이아웃 패턴 검색"""
        
        try:
            # 템플릿과 이미지 정보 기반 검색 쿼리 생성
            assigned_images = combination.get("assigned_images", [])
            image_count = len(assigned_images)
            
            # 검색 쿼리 구성
            if image_count <= 1:
                base_query = "single image layout minimal clean design"
            elif image_count <= 3:
                base_query = "multiple images grid layout balanced composition"
            else:
                base_query = "gallery layout many images organized design"
            
            # AI Search 키워드 필터링
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(base_query)
            
            # 벡터 검색 실행
            layout_patterns = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "magazine-vector-index", top_k=8
                )
            )
            
            # 격리된 패턴만 반환
            isolated_patterns = self.isolation_manager.filter_contaminated_data(
                layout_patterns, f"template_patterns_{template}"
            )
            
            self.logger.debug(f"템플릿 {template} 패턴 검색: {len(layout_patterns)} → {len(isolated_patterns)}개")
            return isolated_patterns
            
        except Exception as e:
            self.logger.error(f"템플릿 {template} 패턴 검색 실패: {e}")
            return []
    
    async def _create_template_strategy_with_patterns(self, template: str, combination: Dict, 
                                                    semantic_analysis: Dict, patterns: List[Dict]) -> Dict:
        """AI Search 패턴을 활용한 템플릿 전략 생성"""
        
        section_index = combination.get("section_index", 0)
        assigned_images = combination.get("assigned_images", [])
        
        # 텍스트 의미 정보 추출
        text_semantics = semantic_analysis.get("text_semantics", [])
        current_text = next((t for t in text_semantics if t["section_index"] == section_index), {})
        
        # AI Search 패턴 정보 구성
        pattern_context = ""
        if patterns:
            pattern_info = []
            for pattern in patterns[:3]: 
                pattern_info.append({
                    "레이아웃_타입": pattern.get("layout_type", "균형형"),
                    "이미지_배치": pattern.get("image_placement", "상단"),
                    "텍스트_흐름": pattern.get("text_flow", "단일컬럼"),
                    "시각적_계층": pattern.get("visual_hierarchy", ["제목", "이미지", "본문"]),
                    "간격_설정": pattern.get("spacing_config", "기본"),
                    "이미지_크기_비율": pattern.get("image_size_ratio", "중간")
                })
            pattern_context = f"AI Search 참조 패턴: {json.dumps(pattern_info, ensure_ascii=False)}"
        
        strategy_prompt = f"""
다음 정보를 바탕으로 {template} 템플릿의 최적 레이아웃 전략을 수립하세요:

**텍스트 정보:**
- 제목: {current_text.get("title", "")}
- 내용 미리보기: {current_text.get("content_preview", "")}
- 의미 분석: {current_text.get("semantic_analysis", {})}

**할당된 이미지:**
{json.dumps(assigned_images, ensure_ascii=False, indent=2)}

{pattern_context}

**전략 수립 요구사항:**
1. AI Search 패턴을 참조한 레이아웃 타입 결정
2. 이미지 개수와 특성에 맞는 배치 방식
3. 텍스트와 이미지의 시각적 계층 구조
4. 독자의 시선 흐름 최적화
5. 감정적 임팩트 강화 방안
6. 반응형 디자인 고려사항

JSON 형식으로 출력하세요:
{{
    "layout_type": "텍스트 중심/이미지 중심/균형형",
    "visual_hierarchy": ["요소1", "요소2", "요소3"],
    "image_placement": "상단/하단/좌측/우측/분산",
    "text_flow": "단일 컬럼/다중 컬럼/자유형",
    "emotional_focus": "강조할 감정적 요소",
    "responsive_breakpoints": ["모바일", "태블릿", "데스크톱"],
    "spacing_config": "간격 설정",
    "image_size_ratio": "이미지 크기 비율"
}}
"""
        
        try:
            response = await self.llm.ainvoke(strategy_prompt)
            strategy = json.loads(response)
            
            # 메타데이터 추가
            strategy["template_name"] = template
            strategy["section_index"] = section_index
            strategy["assigned_image_count"] = len(assigned_images)
            strategy["semantic_score"] = combination.get("total_similarity_score", 0.0)
            strategy["ai_search_patterns_used"] = len(patterns)
            strategy["pattern_enhanced"] = len(patterns) > 0
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"템플릿 {template} 전략 생성 실패: {e}")
            
            # 기본 전략 반환
            return self._get_fallback_strategy(template, section_index, assigned_images, patterns)
    
    def _get_fallback_strategy(self, template: str, section_index: int, 
                             assigned_images: List[Dict], patterns: List[Dict]) -> Dict:
        """폴백 전략 생성"""
        
        image_count = len(assigned_images)
        
        # 이미지 개수에 따른 기본 전략
        if image_count == 0:
            layout_type = "텍스트 중심"
            image_placement = "없음"
        elif image_count == 1:
            layout_type = "균형형"
            image_placement = "상단"
        else:
            layout_type = "이미지 중심"
            image_placement = "분산"
        
        return {
            "template_name": template,
            "layout_type": layout_type,
            "visual_hierarchy": ["제목", "이미지", "본문"],
            "image_placement": image_placement,
            "text_flow": "단일 컬럼",
            "emotional_focus": "여행의 즐거움",
            "responsive_breakpoints": ["모바일", "태블릿", "데스크톱"],
            "spacing_config": "기본",
            "image_size_ratio": "중간",
            "section_index": section_index,
            "assigned_image_count": image_count,
            "semantic_score": 0.5,
            "ai_search_patterns_used": len(patterns),
            "pattern_enhanced": False,
            "fallback_used": True
        }
    
    async def _optimize_content_placement_with_vectors(self, layout_strategies: Dict, 
                                                     semantic_analysis: Dict) -> List[Dict]:
        """벡터 데이터 기반 동적 콘텐츠 배치 최적화"""
        
        optimized_layouts = []
        text_semantics = semantic_analysis.get("text_semantics", [])
        optimal_combinations = semantic_analysis.get("optimal_combinations", [])
        
        for template, strategy in layout_strategies.items():
            section_index = strategy.get("section_index", 0)
            
            # 해당 섹션의 텍스트 정보 찾기
            current_text = next((t for t in text_semantics if t["section_index"] == section_index), {})
            current_combination = next((c for c in optimal_combinations if c["section_index"] == section_index), {})
            
            # AI Search 기반 콘텐츠 배치 패턴 검색
            placement_patterns = await self._search_content_placement_patterns(strategy, current_text)
            
            optimized_layout = await self._create_optimized_layout_with_patterns(
                template, strategy, current_text, current_combination, placement_patterns
            )
            
            optimized_layouts.append(optimized_layout)
        
        return optimized_layouts
    
    async def _search_content_placement_patterns(self, strategy: Dict, text_info: Dict) -> List[Dict]:
        """콘텐츠 배치를 위한 AI Search 패턴 검색 (text_info 활용)"""
        
        try:
            # strategy 정보 추출
            layout_type = strategy.get("layout_type", "균형형")
            image_placement = strategy.get("image_placement", "상단")
            text_flow = strategy.get("text_flow", "단일 컬럼")
            

            text_content = text_info.get("content_preview", "")
            semantic_analysis = text_info.get("semantic_analysis", {})
            
            # text_info의 의미적 분석 정보 활용
            main_topics = semantic_analysis.get("주요_주제", [])
            emotional_tone = semantic_analysis.get("감정적_톤", "")
            visual_keywords = semantic_analysis.get("시각적_키워드", [])
            text_structure = semantic_analysis.get("글의_형태", "")
            
            # 텍스트 길이 기반 배치 전략
            text_length = len(text_content)
            if text_length > 300:
                text_density = "긴글"
            elif text_length > 150:
                text_density = "중간글"
            else:
                text_density = "짧은글"
            
            # 통합된 검색 쿼리 생성 (strategy + text_info)
            base_query_parts = [
                layout_type,
                image_placement, 
                text_flow,
                text_density,
                "콘텐츠 배치"
            ]
            
            # 텍스트 특성 기반 쿼리 강화
            if main_topics:
                # 주요 주제 중 첫 번째 추가
                primary_topic = main_topics[0] if isinstance(main_topics, list) else str(main_topics)
                base_query_parts.append(primary_topic)
            
            if emotional_tone:
                base_query_parts.append(emotional_tone)
            
            if text_structure:
                base_query_parts.append(text_structure)
            
            # 시각적 키워드 기반 쿼리 보강
            if visual_keywords:
                # 상위 2개 시각적 키워드 추가
                visual_elements = visual_keywords[:2] if isinstance(visual_keywords, list) else [str(visual_keywords)]
                base_query_parts.extend(visual_elements)
            
            search_query = " ".join(base_query_parts)
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(search_query)
            
            self.logger.debug(f"콘텐츠 배치 패턴 검색 쿼리: {clean_query}")
            
            # 벡터 검색 실행
            placement_patterns = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "magazine-vector-index", top_k=8
                )
            )
            
            # 텍스트 특성 기반 패턴 필터링
            filtered_patterns = self._filter_patterns_by_text_characteristics(
                placement_patterns, text_info
            )
            
            # 격리된 패턴만 반환
            isolated_patterns = self.isolation_manager.filter_contaminated_data(
                filtered_patterns, "content_placement_patterns"
            )
            
            self.logger.debug(f"콘텐츠 배치 패턴 검색 결과: {len(placement_patterns)} → {len(isolated_patterns)}개")
            
            return isolated_patterns
            
        except Exception as e:
            self.logger.error(f"콘텐츠 배치 패턴 검색 실패: {e}")
            return []

    def _filter_patterns_by_text_characteristics(self, patterns: List[Dict], text_info: Dict) -> List[Dict]:
        """텍스트 특성에 맞는 패턴 필터링"""
        
        if not patterns:
            return patterns
        
        semantic_analysis = text_info.get("semantic_analysis", {})
        text_length = len(text_info.get("content_preview", ""))
        
        filtered_patterns = []
        
        for pattern in patterns:
            # 텍스트 길이와 패턴 호환성 확인
            pattern_text_length = pattern.get("recommended_text_length", "medium")
            
            if text_length > 300 and pattern_text_length in ["long", "medium"]:
                # 긴 텍스트에 적합한 패턴
                filtered_patterns.append(pattern)
            elif text_length <= 150 and pattern_text_length in ["short", "medium"]:
                # 짧은 텍스트에 적합한 패턴
                filtered_patterns.append(pattern)
            elif 150 < text_length <= 300:
                # 중간 길이 텍스트는 모든 패턴 허용
                filtered_patterns.append(pattern)
            
            # 감정적 톤 호환성 확인
            emotional_tone = semantic_analysis.get("감정적_톤", "")
            pattern_tone = pattern.get("emotional_compatibility", [])
            
            if emotional_tone and pattern_tone:
                if isinstance(pattern_tone, list) and emotional_tone in pattern_tone:
                    # 감정적 톤이 일치하는 패턴 우선
                    pattern["tone_match_bonus"] = 0.2
                elif isinstance(pattern_tone, str) and emotional_tone == pattern_tone:
                    pattern["tone_match_bonus"] = 0.2
        
        # 필터링된 패턴이 없으면 원본 반환
        return filtered_patterns if filtered_patterns else patterns

    
    async def _create_optimized_layout_with_patterns(self, template: str, strategy: Dict,
                                               text_info: Dict, combination: Dict,
                                               patterns: List[Dict]) -> Dict:
        """AI Search 패턴을 활용한 최적화된 레이아웃 생성"""
        
        pattern_context = ""
        if patterns:
            placement_info = []
            for pattern in patterns[:3]:
                placement_info.append({
                    "제목_위치": pattern.get("title_position", "상단중앙"),
                    "이미지_그리드": pattern.get("image_grid", "단일"),
                    "텍스트_정렬": pattern.get("text_alignment", "좌측"),
                    "간격_설정": pattern.get("spacing", "기본"),
                    "여백_비율": pattern.get("margin_ratio", "표준"),
                    "반응형_조정": pattern.get("responsive_adjustment", "자동")
                })
            pattern_context = f"배치 패턴 참조: {json.dumps(placement_info, ensure_ascii=False)}"

        # ✅ 기존 프롬프트에 JSON 강제 지시만 추가
        layout_prompt = f"""
        다음 전략과 정보를 바탕으로 {template}의 구체적인 레이아웃을 생성하세요:

        **레이아웃 전략:**
        {json.dumps(strategy, ensure_ascii=False, indent=2)}

        **텍스트 정보:**
        {json.dumps(text_info, ensure_ascii=False, indent=2)}

        **이미지 조합:**
        {json.dumps(combination, ensure_ascii=False, indent=2)}

        {pattern_context}

        **레이아웃 생성 요구사항:**
        1. AI Search 패턴을 참조한 제목, 부제목, 본문의 구체적인 배치
        2. 이미지 크기 및 위치 결정 (패턴 기반)
        3. 여백과 간격 최적화 (벡터 데이터 참조)
        4. 타이포그래피 설정 (매거진 표준)
        5. 색상 및 스타일 가이드

        **⚠️ 중요: JSX 코드가 아닌 순수 JSON 데이터만 출력하세요. import문, HTML태그, 컴포넌트 코드는 절대 포함하지 마세요.**

        **출력 형식 (이 JSON 구조만 출력):**
        {{
            "template": "{template}",
            "title": "최적화된 제목",
            "subtitle": "최적화된 부제목", 
            "body": "최적화된 본문",
            "tagline": "태그라인",
            "images": ["이미지 URL 목록"],
            "layout_config": {{
                "title_position": "위치",
                "image_grid": "배치 방식",
                "text_alignment": "정렬",
                "spacing": "간격 설정",
                "margin_ratio": "여백 비율",
                "typography": "폰트 설정"
            }}
        }}

        **다시 한 번 강조: 위의 JSON 형식만 출력하고, JSX 코드나 다른 텍스트는 포함하지 마세요.**
        """
        
        try:
            response = await self.llm.ainvoke(layout_prompt)
            
            #  빈 응답 체크 추가
            if not response or not response.strip():
                self.logger.warning(f"레이아웃 {template} 생성에서 빈 응답 수신")

            
            #  JSON 파싱 안전 처리
            try:
                layout = json.loads(response.strip())
            except json.JSONDecodeError as json_error:
                self.logger.error(f"레이아웃 {template} JSON 파싱 실패: {json_error}")
                self.logger.debug(f"응답 내용: {response[:200]}...")

            
            # 이미지 URL 추가
            assigned_images = combination.get("assigned_images", [])
            if assigned_images and not layout.get("images"):
                layout["images"] = [img.get("image_name", "") for img in assigned_images[:3]]
            
            # 메타데이터 추가
            layout["optimization_metadata"] = {
                "strategy_applied": True,
                "semantic_score": combination.get("total_similarity_score", 0.0),
                "image_count": len(layout.get("images", [])),
                "optimization_level": "ai_search_enhanced",
                "patterns_referenced": len(patterns),
                "vector_enhanced": True
            }
            
            return layout
            
        except Exception as e:
            self.logger.error(f"레이아웃 {template} 최적화 실패: {e}")
            #  올바른 파라미터 개수로 호출
    
    async def _ensure_visual_balance_with_patterns(self, layouts: List[Dict]) -> List[Dict]:
        """AI Search 패턴 기반 시각적 균형 검증 및 조정"""
        
        balanced_layouts = []
        
        for layout in layouts:
            # AI Search에서 시각적 균형 패턴 검색
            balance_patterns = await self._search_visual_balance_patterns(layout)
            
            balanced_layout = await self._balance_single_layout_with_patterns(layout, balance_patterns)
            balanced_layouts.append(balanced_layout)
        
        # 전체 매거진의 시각적 일관성 확인 (AI Search 패턴 기반)
        balanced_layouts = await self._ensure_overall_consistency_with_ai_search(balanced_layouts)
        
        return balanced_layouts
    
    async def _search_visual_balance_patterns(self, layout: Dict) -> List[Dict]:
        """시각적 균형을 위한 AI Search 패턴 검색 (text_length 활용)"""
        
        try:
            image_count = len(layout.get("images", []))
            text_length = len(layout.get("body", ""))
            
            # ✅ text_length 기반 텍스트 밀도 분류
            if text_length <= 100:
                text_density = "짧은텍스트"
            elif text_length <= 300:
                text_density = "중간텍스트"
            elif text_length <= 600:
                text_density = "긴텍스트"
            else:
                text_density = "매우긴텍스트"
            
            #  image_count와 text_length를 모두 고려한 균형 패턴 검색 쿼리 생성
            base_query_parts = ["visual balance layout"]
            
            # 이미지 개수 기반 쿼리 구성
            if image_count == 0:
                base_query_parts.extend(["text only", text_density, "typography spacing"])
            elif image_count == 1:
                base_query_parts.extend(["single image", text_density, "text balance"])
            elif image_count <= 3:
                base_query_parts.extend(["multiple images", text_density, "grid balance"])
            else:
                base_query_parts.extend(["gallery layout", text_density, "image text harmony"])
            
            #  텍스트 길이에 따른 추가 균형 요소
            if text_length > 500:
                # 긴 텍스트의 경우 가독성 중심
                base_query_parts.extend(["readability", "line spacing", "text flow"])
            elif text_length < 150:
                # 짧은 텍스트의 경우 임팩트 중심
                base_query_parts.extend(["impact", "bold typography", "visual emphasis"])
            else:
                # 중간 길이 텍스트의 경우 균형 중심
                base_query_parts.extend(["balanced composition", "harmony"])
            
            #  이미지-텍스트 비율 기반 균형 전략
            if image_count > 0:
                image_text_ratio = image_count / max(text_length / 100, 1)  # 이미지 수 대비 텍스트 블록 수
                
                if image_text_ratio > 2:
                    # 이미지가 많은 경우
                    base_query_parts.append("image dominant layout")
                elif image_text_ratio < 0.5:
                    # 텍스트가 많은 경우
                    base_query_parts.append("text dominant layout")
                else:
                    # 균형잡힌 경우
                    base_query_parts.append("balanced image text ratio")
            
            search_query = " ".join(base_query_parts)
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(search_query)
            
            self.logger.debug(f"시각적 균형 패턴 검색 쿼리: {clean_query}")
            
            # 벡터 검색 실행
            balance_patterns = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "magazine-vector-index", top_k=8  #  더 많은 패턴 검색
                )
            )
            
            #  텍스트 길이와 이미지 수를 고려한 패턴 필터링
            filtered_patterns = self._filter_balance_patterns_by_content_ratio(
                balance_patterns, image_count, text_length
            )
            
            # 격리된 패턴만 반환
            isolated_patterns = self.isolation_manager.filter_contaminated_data(
                filtered_patterns, "visual_balance_patterns"
            )
            
            self.logger.debug(f"시각적 균형 패턴 검색 결과: {len(balance_patterns)} → {len(isolated_patterns)}개")
            
            return isolated_patterns
            
        except Exception as e:
            self.logger.error(f"시각적 균형 패턴 검색 실패: {e}")
            return []

    def _filter_balance_patterns_by_content_ratio(self, patterns: List[Dict], 
                                                image_count: int, text_length: int) -> List[Dict]:
        """콘텐츠 비율을 고려한 균형 패턴 필터링"""
        
        if not patterns:
            return patterns
        
        filtered_patterns = []
        
        for pattern in patterns:
            pattern_image_count = pattern.get("recommended_image_count", 1)
            pattern_text_length = pattern.get("recommended_text_length", "medium")
            
            # 이미지 개수 호환성 확인
            image_compatibility = False
            if image_count == 0 and pattern_image_count == 0:
                image_compatibility = True
            elif image_count == 1 and pattern_image_count <= 2:
                image_compatibility = True
            elif image_count > 1 and pattern_image_count > 1:
                image_compatibility = True
            
            # 텍스트 길이 호환성 확인
            text_compatibility = False
            if text_length <= 150 and pattern_text_length in ["short", "medium"]:
                text_compatibility = True
            elif 150 < text_length <= 400 and pattern_text_length in ["medium", "long"]:
                text_compatibility = True
            elif text_length > 400 and pattern_text_length in ["long", "very_long"]:
                text_compatibility = True
            
            # 호환성이 있는 패턴만 포함
            if image_compatibility or text_compatibility:
                #  호환성 점수 추가
                compatibility_score = 0
                if image_compatibility:
                    compatibility_score += 0.5
                if text_compatibility:
                    compatibility_score += 0.5
                
                pattern["content_compatibility_score"] = compatibility_score
                filtered_patterns.append(pattern)
        
        # 호환성 점수 순으로 정렬
        filtered_patterns.sort(key=lambda x: x.get("content_compatibility_score", 0), reverse=True)
        
        # 필터링된 패턴이 없으면 원본 반환
        return filtered_patterns if filtered_patterns else patterns

    
    async def _balance_single_layout_with_patterns(self, layout: Dict, patterns: List[Dict]) -> Dict:
        """AI Search 패턴을 활용한 개별 레이아웃 시각적 균형 조정"""
        
        image_count = len(layout.get("images", []))
        text_length = len(layout.get("body", ""))
        
        # 패턴 기반 균형 조정
        if patterns:
            # 가장 유사한 패턴의 설정 적용
            best_pattern = patterns[0]
            
            if image_count == 0 and text_length > 500:
                # 텍스트 전용 레이아웃 패턴 적용
                layout["layout_config"]["text_columns"] = best_pattern.get("text_columns", 2)
                layout["layout_config"]["text_spacing"] = best_pattern.get("text_spacing", "넓음")
            elif image_count > 2 and text_length < 200:
                # 이미지 중심 레이아웃 패턴 적용
                layout["layout_config"]["image_grid"] = best_pattern.get("image_grid", "갤러리")
                layout["layout_config"]["text_emphasis"] = best_pattern.get("text_emphasis", "강화")
        else:
            # 기본 균형 조정
            if image_count == 0 and text_length > 500:
                layout["layout_config"]["text_columns"] = 2
                layout["layout_config"]["text_spacing"] = "넓음"
            elif image_count > 2 and text_length < 200:
                layout["layout_config"]["image_grid"] = "갤러리"
                layout["layout_config"]["text_emphasis"] = "강화"
        
        # 시각적 균형 점수 계산
        balance_score = self._calculate_visual_balance_score_with_patterns(layout, patterns)
        layout["optimization_metadata"]["visual_balance_score"] = balance_score
        layout["optimization_metadata"]["pattern_balance_applied"] = len(patterns) > 0
        
        return layout
    
    def _calculate_visual_balance_score_with_patterns(self, layout: Dict, patterns: List[Dict]) -> float:
        """AI Search 패턴을 고려한 시각적 균형 점수 계산"""
        
        image_count = len(layout.get("images", []))
        text_length = len(layout.get("body", ""))
        
        # 기본 균형 점수
        ideal_image_ratio = min(image_count / 2.0, 1.0)
        ideal_text_ratio = min(text_length / 500.0, 1.0)
        base_score = (ideal_image_ratio + ideal_text_ratio) / 2.0
        
        # 패턴 기반 보정
        if patterns:
            pattern_bonus = 0.2  # 패턴이 있으면 20% 보너스
            base_score = min(base_score + pattern_bonus, 1.0)
        
        return base_score
    
    async def _ensure_overall_consistency_with_ai_search(self, layouts: List[Dict]) -> List[Dict]:
        """AI Search 패턴 기반 전체 매거진 시각적 일관성 확보"""
        
        # AI Search에서 매거진 일관성 패턴 검색
        consistency_patterns = await self._search_magazine_consistency_patterns(layouts)
        
        # 전체 스타일 가이드 생성 (AI Search 패턴 기반)
        style_guide = self._generate_style_guide_with_patterns(layouts, consistency_patterns)
        
        # 각 레이아웃에 스타일 가이드 적용
        consistent_layouts = []
        for layout in layouts:
            consistent_layout = self._apply_style_guide_with_patterns(layout, style_guide)
            consistent_layouts.append(consistent_layout)
        
        return consistent_layouts
    
    async def _search_magazine_consistency_patterns(self, layouts: List[Dict]) -> List[Dict]:
        """매거진 일관성을 위한 AI Search 패턴 검색"""
        
        try:
            # 전체 레이아웃 특성 분석
            total_images = sum(len(layout.get("images", [])) for layout in layouts)
            total_sections = len(layouts)
            
            search_query = f"magazine consistency {total_sections} sections {total_images} images layout"
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(search_query)
            
            # 벡터 검색 실행
            consistency_patterns = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "magazine-vector-index", top_k=5
                )
            )
            
            # 격리된 패턴만 반환
            isolated_patterns = self.isolation_manager.filter_contaminated_data(
                consistency_patterns, "magazine_consistency_patterns"
            )
            
            return isolated_patterns
            
        except Exception as e:
            self.logger.error(f"매거진 일관성 패턴 검색 실패: {e}")
            return []
    
    def _generate_style_guide_with_patterns(self, layouts: List[Dict], patterns: List[Dict]) -> Dict:
        """AI Search 패턴을 활용한 스타일 가이드 생성 (layouts 활용)"""
        
        # layouts에서 실제 사용된 스타일 분석
        layout_analysis = self._analyze_current_layouts(layouts)
        
        # 기본 스타일 가이드
        style_guide = {
            "title_position": "상단",
            "text_alignment": "좌측", 
            "color_scheme": "warm",
            "typography": {
                "title_font": "bold",
                "body_font": "regular",
                "line_height": 1.6
            },
            "spacing": {
                "section_margin": "2rem",
                "element_padding": "1rem"
            }
        }
        
        #  layouts에서 추출한 실제 사용 패턴 반영
        if layout_analysis:
            # 가장 많이 사용된 설정들을 기본값으로 적용
            style_guide.update({
                "title_position": layout_analysis.get("most_common_title_position", "상단"),
                "text_alignment": layout_analysis.get("most_common_text_alignment", "좌측"),
                "color_scheme": layout_analysis.get("dominant_color_scheme", "warm")
            })
            
            #  layouts의 이미지-텍스트 비율 기반 스타일 조정
            avg_image_count = layout_analysis.get("average_image_count", 1)
            avg_text_length = layout_analysis.get("average_text_length", 300)
            
            if avg_image_count > 2:
                # 이미지가 많은 매거진: 시각적 임팩트 중심
                style_guide["typography"]["title_font"] = "extra_bold"
                style_guide["spacing"]["section_margin"] = "3rem"
            elif avg_text_length > 500:
                # 텍스트가 많은 매거진: 가독성 중심
                style_guide["typography"]["line_height"] = 1.8
                style_guide["spacing"]["element_padding"] = "1.5rem"
        
        #  AI Search 패턴 기반 스타일 개선 (기존 + layouts 정보 결합)
        if patterns:
            best_pattern = patterns[0]
            
            #  layouts 분석 결과와 AI Search 패턴을 결합
            enhanced_style = self._merge_layout_analysis_with_patterns(
                layout_analysis, best_pattern
            )
            
            style_guide.update({
                "color_scheme": enhanced_style.get("color_scheme", style_guide["color_scheme"]),
                "typography": {
                    "title_font": enhanced_style.get("title_font", style_guide["typography"]["title_font"]),
                    "body_font": enhanced_style.get("body_font", style_guide["typography"]["body_font"]),
                    "line_height": enhanced_style.get("line_height", style_guide["typography"]["line_height"])
                },
                "spacing": {
                    "section_margin": enhanced_style.get("section_margin", style_guide["spacing"]["section_margin"]),
                    "element_padding": enhanced_style.get("element_padding", style_guide["spacing"]["element_padding"])
                }
            })
            
            style_guide["pattern_enhanced"] = True
            style_guide["pattern_source"] = best_pattern.get("pdf_name", "ai_search")
            style_guide["layouts_analyzed"] = True  #  layouts 분석 적용 표시
        
        #  layouts 기반 메타데이터 추가
        style_guide["layout_statistics"] = {
            "total_layouts": len(layouts),
            "average_image_count": layout_analysis.get("average_image_count", 0) if layout_analysis else 0,
            "average_text_length": layout_analysis.get("average_text_length", 0) if layout_analysis else 0,
            "style_consistency_score": layout_analysis.get("consistency_score", 0.5) if layout_analysis else 0.5
        }
        
        return style_guide

    def _analyze_current_layouts(self, layouts: List[Dict]) -> Dict:
        """현재 레이아웃들의 스타일 패턴 분석"""
        
        if not layouts:
            return {}
        
        # 각 레이아웃에서 스타일 정보 수집
        title_positions = []
        text_alignments = []
        color_schemes = []
        image_counts = []
        text_lengths = []
        
        for layout in layouts:
            layout_config = layout.get("layout_config", {})
            
            # 스타일 설정 수집
            title_positions.append(layout_config.get("title_position", "상단"))
            text_alignments.append(layout_config.get("text_alignment", "좌측"))
            
            # 색상 스키마 추출 (style_guide에서)
            style_guide = layout_config.get("style_guide", {})
            color_schemes.append(style_guide.get("color_scheme", "warm"))
            
            # 콘텐츠 특성 수집
            image_counts.append(len(layout.get("images", [])))
            text_lengths.append(len(layout.get("body", "")))
        
        #  통계 분석
        analysis = {
            "most_common_title_position": self._get_most_common(title_positions),
            "most_common_text_alignment": self._get_most_common(text_alignments),
            "dominant_color_scheme": self._get_most_common(color_schemes),
            "average_image_count": sum(image_counts) / len(image_counts) if image_counts else 0,
            "average_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            "consistency_score": self._calculate_style_consistency(layouts)
        }
        
        return analysis

    def _get_most_common(self, items: List[str]) -> str:
        """리스트에서 가장 빈번한 항목 반환"""
        if not items:
            return ""
        
        counter = Counter(items)
        return counter.most_common(1)[0][0]

    def _calculate_style_consistency(self, layouts: List[Dict]) -> float:
        """스타일 일관성 점수 계산"""
        
        if len(layouts) <= 1:
            return 1.0
        
        # 각 스타일 요소의 일관성 확인
        consistency_scores = []
        
        # 제목 위치 일관성
        title_positions = [layout.get("layout_config", {}).get("title_position", "상단") for layout in layouts]
        title_consistency = len(set(title_positions)) / len(title_positions)
        consistency_scores.append(1.0 - title_consistency + 0.1)  # 역수로 변환
        
        # 텍스트 정렬 일관성
        text_alignments = [layout.get("layout_config", {}).get("text_alignment", "좌측") for layout in layouts]
        text_consistency = len(set(text_alignments)) / len(text_alignments)
        consistency_scores.append(1.0 - text_consistency + 0.1)
        
        # 이미지 개수 일관성 (비슷한 범위인지 확인)
        image_counts = [len(layout.get("images", [])) for layout in layouts]
        image_variance = self._calculate_variance(image_counts)
        image_consistency = 1.0 / (1.0 + image_variance)  # 분산이 낮을수록 높은 점수
        consistency_scores.append(image_consistency)
        
        return sum(consistency_scores) / len(consistency_scores)

    def _calculate_variance(self, numbers: List[float]) -> float:
        """숫자 리스트의 분산 계산"""
        if not numbers:
            return 0.0
        
        mean = sum(numbers) / len(numbers)
        variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
        return variance

    def _merge_layout_analysis_with_patterns(self, layout_analysis: Dict, pattern: Dict) -> Dict:
        """레이아웃 분석 결과와 AI Search 패턴 결합"""
        
        enhanced_style = {}
        
        #  레이아웃 분석 우선, 패턴으로 보완
        avg_image_count = layout_analysis.get("average_image_count", 1)
        avg_text_length = layout_analysis.get("average_text_length", 300)
        
        # 색상 스키마 결정
        if avg_image_count > 2:
            # 이미지가 많으면 패턴의 색상 사용
            enhanced_style["color_scheme"] = pattern.get("color_scheme", "vibrant")
        else:
            # 텍스트 중심이면 현재 분석 결과 유지
            enhanced_style["color_scheme"] = layout_analysis.get("dominant_color_scheme", "warm")
        
        # 타이포그래피 결정
        if avg_text_length > 500:
            # 긴 텍스트: 가독성 중심
            enhanced_style["title_font"] = pattern.get("readable_title_font", "medium_bold")
            enhanced_style["body_font"] = pattern.get("readable_body_font", "regular")
            enhanced_style["line_height"] = pattern.get("readable_line_height", 1.8)
        else:
            # 짧은 텍스트: 임팩트 중심
            enhanced_style["title_font"] = pattern.get("impact_title_font", "extra_bold")
            enhanced_style["body_font"] = pattern.get("impact_body_font", "medium")
            enhanced_style["line_height"] = pattern.get("impact_line_height", 1.4)
        
        # 간격 설정
        consistency_score = layout_analysis.get("consistency_score", 0.5)
        if consistency_score > 0.8:
            # 일관성이 높으면 현재 설정 유지
            enhanced_style["section_margin"] = "2rem"
            enhanced_style["element_padding"] = "1rem"
        else:
            # 일관성이 낮으면 패턴의 설정 적용
            enhanced_style["section_margin"] = pattern.get("section_margin", "2.5rem")
            enhanced_style["element_padding"] = pattern.get("element_padding", "1.2rem")
        
        return enhanced_style

    
    def _apply_style_guide_with_patterns(self, layout: Dict, style_guide: Dict) -> Dict:
        """AI Search 패턴 기반 스타일 가이드 적용"""
        
        if "layout_config" not in layout:
            layout["layout_config"] = {}
        
        # 스타일 가이드 적용
        layout["layout_config"].update({
            "style_guide": style_guide,
            "consistency_applied": True,
            "pattern_enhanced": style_guide.get("pattern_enhanced", False)
        })
        
        # 메타데이터 업데이트
        layout["optimization_metadata"]["style_consistency"] = True
        layout["optimization_metadata"]["ai_search_style_applied"] = style_guide.get("pattern_enhanced", False)
        
        return layout
    
    async def _apply_responsive_design_with_ai_search(self, layouts: List[Dict]) -> List[Dict]:
        """AI Search 패턴 기반 반응형 레이아웃 적용"""
        
        responsive_layouts = []
        
        for layout in layouts:
            # AI Search에서 반응형 패턴 검색
            responsive_patterns = await self._search_responsive_patterns(layout)
            
            responsive_layout = await self._make_layout_responsive_with_patterns(layout, responsive_patterns)
            responsive_layouts.append(responsive_layout)
        
        return responsive_layouts
    
    async def _search_responsive_patterns(self, layout: Dict) -> List[Dict]:
        """반응형 디자인을 위한 AI Search 패턴 검색"""
        
        try:
            image_count = len(layout.get("images", []))
            layout_type = layout.get("optimization_metadata", {}).get("optimization_level", "basic")
            
            search_query = f"responsive design {image_count} images {layout_type} mobile tablet desktop"
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(search_query)
            
            # 벡터 검색 실행
            responsive_patterns = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "magazine-vector-index", top_k=5
                )
            )
            
            # 격리된 패턴만 반환
            isolated_patterns = self.isolation_manager.filter_contaminated_data(
                responsive_patterns, "responsive_patterns"
            )
            
            return isolated_patterns
            
        except Exception as e:
            self.logger.error(f"반응형 패턴 검색 실패: {e}")
            return []
    
    async def _make_layout_responsive_with_patterns(self, layout: Dict, patterns: List[Dict]) -> Dict:
        """AI Search 패턴을 활용한 반응형 레이아웃 변환"""
        
        # 기본 반응형 브레이크포인트
        breakpoints = {
            "mobile": {"max_width": "768px", "columns": 1, "image_size": "full"},
            "tablet": {"max_width": "1024px", "columns": 2, "image_size": "large"},
            "desktop": {"min_width": "1025px", "columns": 3, "image_size": "original"}
        }
        
        # AI Search 패턴 기반 반응형 설정 개선
        if patterns:
            best_pattern = patterns[0]
            
            # 패턴에서 반응형 설정 추출
            pattern_responsive = best_pattern.get("responsive_config", {})
            if pattern_responsive:
                breakpoints.update(pattern_responsive)
        
        # 이미지 개수에 따른 반응형 조정
        image_count = len(layout.get("images", []))
        if image_count > 2:
            breakpoints["mobile"]["image_grid"] = "carousel"
            breakpoints["tablet"]["image_grid"] = "grid_2x1"
            breakpoints["desktop"]["image_grid"] = "grid_3x1"
        
        # 레이아웃에 반응형 설정 추가
        layout["responsive_config"] = breakpoints
        
        # 메타데이터 업데이트
        layout["optimization_metadata"]["responsive_design"] = True
        layout["optimization_metadata"]["breakpoints_count"] = len(breakpoints)
        layout["optimization_metadata"]["pattern_responsive_applied"] = len(patterns) > 0
        
        return layout

    async def generate_layout_strategy_for_section(self, section_data: Dict) -> Dict:
        """
        주어진 단일 섹션의 의미 분석 정보를 바탕으로 템플릿에 독립적인 이상적인 레이아웃 전략을 생성합니다.
        이 메서드는 템플릿 목록 없이 작동하며, 콘텐츠의 본질적인 특성에 집중합니다.
        
        Args:
            section_data (Dict): 섹션 데이터 및 메타데이터
                {
                    'title': '섹션 제목',
                    'subtitle': '부제목',
                    'final_content': '최종 콘텐츠',
                    'metadata': {
                        'style': '스타일',
                        'emotion': '감정 톤',
                        'keywords': ['키워드1', '키워드2'],
                        'image_count': 이미지 수
                    }
                }
        
        Returns:
            Dict: 레이아웃 전략 JSON
                {
                    'layout_type': '레이아웃 타입',
                    'visual_hierarchy': ['요소1', '요소2', ...],
                    'image_placement': '이미지 배치',
                    'text_flow': '텍스트 흐름',
                    'emotional_focus': '감정적 초점',
                    'key_features': ['특징1', '특징2', ...]
                }
        """
        # 로그 시작
        title = section_data.get('title', '제목 없음')
        self.logger.info(f"섹션 '{title}'에 대한 이상적인 레이아웃 전략 생성 시작")
        
        try:
            # 메타데이터 추출
            metadata = section_data.get('metadata', {})
            
            # 1. AI Search에서 일반적인 레이아웃 패턴 검색
            query_text = self._create_general_layout_query(metadata)
            layout_patterns = await self._search_layout_patterns(query_text, title)
            
            # 2. LLM 프롬프트 생성 (전략 생성을 위해)
            prompt = self._create_strategy_generation_prompt(section_data, layout_patterns)
            
            # 3. LLM을 통해 전략 생성
            response = await self.llm.ainvoke(prompt)
            
            # JSON 파싱
            try:
                strategy = json.loads(response)
            except json.JSONDecodeError:
                # JSON 형식이 아닌 경우, 정규식으로 추출 시도
                import re
                json_pattern = r'\{[\s\S]*\}'
                match = re.search(json_pattern, response)
                if match:
                    strategy = json.loads(match.group(0))
                else:
                    raise ValueError("응답에서 JSON을 추출할 수 없습니다")
            
            # 메타데이터 보강
            strategy["section_title"] = title
            strategy["pattern_enhanced"] = len(layout_patterns) > 0
            strategy["ai_search_patterns_used"] = len(layout_patterns)
            
            self.logger.info(f"섹션 '{title}' 레이아웃 전략 생성 성공 (패턴 {len(layout_patterns)}개 사용)")
            return strategy
            
        except Exception as e:
            self.logger.error(f"레이아웃 전략 생성 실패 (섹션 '{title}'): {str(e)}")
            # 오류 발생 시, 기본 폴백 전략 반환
            return self._get_fallback_strategy_for_section(section_data)
    
    def _create_general_layout_query(self, metadata: Dict) -> str:
        """전략 생성을 위한 AI Search 검색 쿼리를 생성합니다."""
        image_count = metadata.get('image_count', 0)
        style = metadata.get('style', 'modern')
        emotion = metadata.get('emotion', 'neutral')
        
        if image_count == 0:
            query = f"{style} {emotion} text-focused article layout"
        elif image_count == 1:
            query = f"{style} {emotion} single featured image layout"
        elif image_count <= 3:
            query = f"{style} {emotion} balanced grid layout with {image_count} images"
        else:
            query = f"{style} {emotion} dynamic gallery for multiple images"
        
        return query

    async def _search_layout_patterns(self, query: str, section_identifier: str) -> List[Dict]:
        """AI Search를 통해 일반적인 레이아웃 패턴을 검색합니다."""
        try:
            # AI Search 키워드 필터링 (격리)
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(query)
            
            # 벡터 검색 실행 (magazine-vector-index에서 레이아웃 패턴 검색)
            layout_patterns = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "magazine-vector-index", top_k=5
                )
            )
            
            # 격리된 패턴만 반환
            isolated_patterns = self.isolation_manager.filter_contaminated_data(
                layout_patterns, f"layout_strategy_patterns_{section_identifier}"
            )
            
            self.logger.debug(f"레이아웃 패턴 검색: '{query}' -> {len(isolated_patterns)}개 패턴 확보")
            return isolated_patterns
        except Exception as e:
            self.logger.error(f"레이아웃 패턴 검색 실패: {str(e)}")
            return []
            
    def _create_strategy_generation_prompt(self, section_data: Dict, patterns: List[Dict]) -> str:
        """레이아웃 전략 생성을 위한 LLM 프롬프트를 구성합니다."""
        title = section_data.get('title', '')
        content_preview = section_data.get('final_content', '')[:200] + '...' if len(section_data.get('final_content', '')) > 200 else section_data.get('final_content', '')
        metadata = section_data.get('metadata', {})
        
        # 메타데이터 추출
        style = metadata.get('style', 'modern')
        emotion = metadata.get('emotion', 'neutral')
        keywords = metadata.get('keywords', [])
        image_count = metadata.get('image_count', 0)
        
        # 키워드를 문자열로 변환
        if isinstance(keywords, list):
            keywords_str = ', '.join(keywords)
        else:
            keywords_str = str(keywords)
        
        # 패턴 정보 구성
        pattern_context = ""
        if patterns:
            pattern_info = []
            for p in patterns[:3]:  # 최대 3개 패턴만 참조
                pattern_info.append(f"- {p.get('layout_type', '균형형')} (이미지: {p.get('image_placement', '상단')}, 텍스트: {p.get('text_flow', '단일컬럼')})")
            pattern_context = "참고할 수 있는 AI Search 레이아웃 패턴:\n" + "\n".join(pattern_info)

        return f"""
다음 콘텐츠 정보를 바탕으로, 이 섹션에 가장 이상적인 레이아웃 '전략'을 JSON 형식으로 상세히 설계해주세요. 
이 전략은 나중에 이 설계도에 가장 잘 맞는 실제 JSX 템플릿을 찾는 데 사용됩니다.

### 섹션 정보:
- 제목: {title}
- 내용 미리보기: {content_preview}

### 콘텐츠 특성:
- 스타일: {style}
- 감정 톤: {emotion}
- 키워드: {keywords_str}
- 이미지 수: {image_count}

{pattern_context}

### 설계 요구사항:
1. 콘텐츠의 특성(스타일, 감정, 이미지 수)을 종합하여 레이아웃 타입을 결정하세요.
2. 텍스트, 이미지 등 핵심 요소들의 시각적 계층 구조를 정의하세요.
3. 독자의 시선 흐름, 이미지 배치, 텍스트 흐름 등을 구체적으로 명시하세요.

### 출력 형식 (JSON):
{{
    "layout_type": "텍스트 중심 | 이미지 중심 | 균형형 | 그리드 | 갤러리",
    "visual_hierarchy": ["주요 요소1", "중간 요소2", "보조 요소3"],
    "image_placement": "상단 | 하단 | 좌측 | 우측 | 분산형 | 없음",
    "text_flow": "단일 컬럼 | 다중 컬럼 | 자유형",
    "emotional_focus": "콘텐츠의 감정을 극대화하기 위한 시각적 강조점",
    "key_features": ["콘텐츠의 핵심 특징을 나타내는 키워드 배열"]
}}

JSON 형식으로만 응답하세요. 설명이나 다른 텍스트는 포함하지 마세요.
"""

    def _get_fallback_strategy_for_section(self, section_data: Dict) -> Dict:
        """LLM 호출 실패 시 사용할 기본 레이아웃 전략을 반환합니다."""
        metadata = section_data.get('metadata', {})
        image_count = metadata.get('image_count', 0)
        title = section_data.get('title', '제목 없음')
        
        self.logger.warning(f"섹션 '{title}'에 대한 폴백 레이아웃 전략을 사용합니다.")
        
        if image_count > 0:
            return {
                "layout_type": "균형형",
                "visual_hierarchy": ["이미지", "제목", "본문"],
                "image_placement": "상단",
                "text_flow": "단일 컬럼",
                "emotional_focus": "이미지 강조",
                "key_features": ["default", "balanced", "image-focused"],
                "section_title": title,
                "pattern_enhanced": False,
                "ai_search_patterns_used": 0
            }
        else:
            return {
                "layout_type": "텍스트 중심",
                "visual_hierarchy": ["제목", "부제목", "본문"],
                "image_placement": "없음",
                "text_flow": "단일 컬럼",
                "emotional_focus": "가독성",
                "key_features": ["default", "text-centric", "readability"],
                "section_title": title,
                "pattern_enhanced": False,
                "ai_search_patterns_used": 0
            }
