import asyncio
import numpy as np
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from utils.data.pdf_vector_manager import PDFVectorManager
from utils.isolation.ai_search_isolation import AISearchIsolationManager
from utils.isolation.session_isolation import SessionAwareMixin
from utils.log.logging_manager import LoggingManager

class RealtimeLayoutGenerator(SessionAwareMixin):
    """실시간 레이아웃 생성기 - AI Search 벡터 패턴 기반 고급 레이아웃 전략 수립"""
    
    def __init__(self, vector_manager: PDFVectorManager, logger: Any):
        super().__init__()
        self.vector_manager = vector_manager
        self.logger = logger
        self.isolation_manager = AISearchIsolationManager()
        self.logging_manager = LoggingManager(self.logger)
        
        # 레이아웃 전략 캐시
        self.layout_strategy_cache = {}
        self.pattern_cache = {}
        
        # 레이아웃 분석 메트릭
        self.layout_metrics = {
            "visual_balance": 0.0,
            "content_hierarchy": 0.0,
            "responsive_score": 0.0,
            "aesthetic_score": 0.0
        }
        
        self.__init_session_awareness__()
        self.logger.info("✅ RealtimeLayoutGenerator 초기화 완료")

    async def generate_layout_strategy_for_section(self, section_data: Dict) -> Dict:
        """✅ 섹션별 레이아웃 전략 생성 (AI Search 벡터 패턴 기반)"""
        
        try:
            section_id = section_data.get("section_id", "unknown")
            title = section_data.get("title", "제목 없음")
            content = section_data.get("content", "")
            images = section_data.get("images", [])
            
            self.logger.info(f"레이아웃 전략 생성 시작: {title} (이미지: {len(images)}개)")
            
            # 1. 콘텐츠 특성 분석
            content_analysis = await self._analyze_content_characteristics(section_data)
            
            # 2. AI Search 벡터 패턴 수집
            vector_patterns = await self._collect_layout_vector_patterns(section_data)
            
            # 3. 레이아웃 전략 수립
            layout_strategy = await self._develop_layout_strategy_with_ai_search(
                content_analysis, vector_patterns, section_data
            )
            
            # 4. 시각적 균형 최적화
            balanced_strategy = await self._optimize_visual_balance_with_patterns(
                layout_strategy, vector_patterns
            )
            
            # 5. 반응형 디자인 적용
            responsive_strategy = await self._apply_responsive_design_with_ai_search(
                balanced_strategy, vector_patterns
            )
            
            # 6. 최종 전략 검증 및 보완
            final_strategy = await self._validate_and_enhance_strategy(
                responsive_strategy, section_data, vector_patterns
            )
            
            self.logger.info(f"레이아웃 전략 생성 완료: {final_strategy.get('layout_type', 'unknown')}")
            return final_strategy
            
        except Exception as e:
            self.logger.error(f"레이아웃 전략 생성 실패: {e}")
            return self._create_fallback_strategy(section_data)

    async def _analyze_content_characteristics(self, section_data: Dict) -> Dict:
        """콘텐츠 특성 분석"""
        
        content = section_data.get("content", "")
        images = section_data.get("images", [])
        title = section_data.get("title", "")
        
        # 텍스트 분석
        content_length = len(content)
        word_count = len(content.split()) if content else 0
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        
        # 이미지 분석
        image_count = len(images)
        has_images = image_count > 0
        
        # 제목 분석
        title_length = len(title)
        title_complexity = len(title.split(':')) > 1  # 콜론으로 구분된 복합 제목
        
        analysis = {
            "content_metrics": {
                "length": content_length,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "reading_time": max(1, word_count // 200),  # 분 단위
                "complexity": "high" if word_count > 300 else "medium" if word_count > 100 else "low"
            },
            "image_metrics": {
                "count": image_count,
                "has_images": has_images,
                "image_density": image_count / max(1, word_count // 100),
                "visual_weight": "high" if image_count > 2 else "medium" if image_count > 0 else "low"
            },
            "title_metrics": {
                "length": title_length,
                "complexity": title_complexity,
                "style": "complex" if title_complexity else "simple"
            },
            "overall_characteristics": {
                "content_type": self._determine_content_type(content, images),
                "layout_preference": self._suggest_layout_preference(content_length, image_count),
                "visual_hierarchy": self._analyze_visual_hierarchy(title, content, images)
            }
        }
        
        return analysis

    async def _collect_layout_vector_patterns(self, section_data: Dict) -> Dict:
        """✅ AI Search 벡터 패턴 수집 (레이아웃 특화)"""
        
        try:
            title = section_data.get("title", "")
            content = section_data.get("content", "")[:200]  # 처음 200자만
            image_count = len(section_data.get("images", []))
            
            # 1. 템플릿 레이아웃 패턴 검색
            template_query = f"magazine template layout {title} images:{image_count}"
            template_patterns = await self._search_template_layout_patterns(template_query)
            
            # 2. 콘텐츠 배치 패턴 검색
            placement_query = f"content placement text image balance {content}"
            placement_patterns = await self._search_content_placement_patterns(placement_query)
            
            # 3. 시각적 균형 패턴 검색
            balance_query = f"visual balance design layout hierarchy {title}"
            balance_patterns = await self._search_visual_balance_patterns(balance_query)
            
            # 4. 반응형 디자인 패턴 검색
            responsive_query = f"responsive design mobile tablet desktop layout"
            responsive_patterns = await self._search_responsive_design_patterns(responsive_query)
            
            vector_patterns = {
                "template_patterns": template_patterns,
                "placement_patterns": placement_patterns,
                "balance_patterns": balance_patterns,
                "responsive_patterns": responsive_patterns,
                "pattern_scores": self._calculate_pattern_relevance_scores(
                    template_patterns, placement_patterns, balance_patterns, responsive_patterns
                )
            }
            
            return vector_patterns
            
        except Exception as e:
            self.logger.error(f"벡터 패턴 수집 실패: {e}")
            return {"template_patterns": [], "placement_patterns": [], "balance_patterns": [], "responsive_patterns": []}

    async def _search_template_layout_patterns(self, query: str) -> List[Dict]:
        """템플릿 레이아웃 패턴 검색"""
        try:
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(query)
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "jsx-component-vector-index", top_k=5
                )
            )
            return self.isolation_manager.filter_contaminated_data(results, f"template_patterns_{hash(query)}")
        except Exception as e:
            self.logger.error(f"템플릿 패턴 검색 실패: {e}")
            return []

    async def _search_content_placement_patterns(self, query: str) -> List[Dict]:
        """콘텐츠 배치 패턴 검색"""
        try:
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(query)
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "magazine-vector-index", top_k=5
                )
            )
            return self.isolation_manager.filter_contaminated_data(results, f"placement_patterns_{hash(query)}")
        except Exception as e:
            self.logger.error(f"배치 패턴 검색 실패: {e}")
            return []

    async def _search_visual_balance_patterns(self, query: str) -> List[Dict]:
        """시각적 균형 패턴 검색"""
        try:
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(query)
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "text-semantic-patterns-index", top_k=5
                )
            )
            return self.isolation_manager.filter_contaminated_data(results, f"balance_patterns_{hash(query)}")
        except Exception as e:
            self.logger.error(f"균형 패턴 검색 실패: {e}")
            return []

    async def _search_responsive_design_patterns(self, query: str) -> List[Dict]:
        """반응형 디자인 패턴 검색"""
        try:
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(query)
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "jsx-component-vector-index", top_k=3
                )
            )
            return self.isolation_manager.filter_contaminated_data(results, f"responsive_patterns_{hash(query)}")
        except Exception as e:
            self.logger.error(f"반응형 패턴 검색 실패: {e}")
            return []

    async def _develop_layout_strategy_with_ai_search(self, content_analysis: Dict, 
                                                    vector_patterns: Dict, 
                                                    section_data: Dict) -> Dict:
        """✅ AI Search 패턴 기반 레이아웃 전략 수립"""
        
        # 콘텐츠 특성 추출
        content_metrics = content_analysis.get("content_metrics", {})
        image_metrics = content_analysis.get("image_metrics", {})
        title_metrics = content_analysis.get("title_metrics", {})
        
        # 벡터 패턴 분석
        template_patterns = vector_patterns.get("template_patterns", [])
        placement_patterns = vector_patterns.get("placement_patterns", [])
        pattern_scores = vector_patterns.get("pattern_scores", {})
        
        # 기본 레이아웃 타입 결정
        layout_type = self._determine_layout_type_from_patterns(
            content_metrics, image_metrics, template_patterns
        )
        
        # 시각적 계층 구조 설계
        visual_hierarchy = self._design_visual_hierarchy_from_patterns(
            title_metrics, content_metrics, image_metrics, placement_patterns
        )
        
        # 이미지 배치 전략
        image_placement = self._determine_image_placement_from_patterns(
            image_metrics, template_patterns, placement_patterns
        )
        
        # 텍스트 흐름 설계
        text_flow = self._design_text_flow_from_patterns(
            content_metrics, placement_patterns
        )
        
        # 감정적 포커스 결정
        emotional_focus = self._determine_emotional_focus_from_patterns(
            section_data, template_patterns
        )
        
        strategy = {
            "layout_type": layout_type,
            "visual_hierarchy": visual_hierarchy,
            "image_placement": image_placement,
            "text_flow": text_flow,
            "emotional_focus": emotional_focus,
            "key_features": self._extract_key_features_from_patterns(template_patterns),
            "section_title": section_data.get("title", ""),
            "pattern_enhanced": True,
            "ai_search_patterns_used": len(template_patterns) + len(placement_patterns),
            "confidence_score": pattern_scores.get("overall_confidence", 0.7),
            "optimization_metadata": {
                "content_complexity": content_metrics.get("complexity", "medium"),
                "visual_weight": image_metrics.get("visual_weight", "medium"),
                "pattern_alignment": pattern_scores.get("pattern_alignment", 0.6)
            }
        }
        
        return strategy

    async def _optimize_visual_balance_with_patterns(self, layout_strategy: Dict, 
                                                   vector_patterns: Dict) -> Dict:
        """벡터 패턴 기반 시각적 균형 최적화"""
        
        balance_patterns = vector_patterns.get("balance_patterns", [])
        
        # 균형 점수 계산
        balance_score = self._calculate_visual_balance_score(layout_strategy, balance_patterns)
        
        # 균형 개선 제안
        if balance_score < 0.7:
            optimized_strategy = self._apply_balance_improvements(layout_strategy, balance_patterns)
        else:
            optimized_strategy = layout_strategy.copy()
        
        optimized_strategy["visual_balance"] = {
            "score": balance_score,
            "optimized": balance_score < 0.7,
            "balance_patterns_used": len(balance_patterns)
        }
        
        return optimized_strategy

    async def _apply_responsive_design_with_ai_search(self, layout_strategy: Dict, 
                                                    vector_patterns: Dict) -> Dict:
        """AI Search 패턴 기반 반응형 디자인 적용"""
        
        responsive_patterns = vector_patterns.get("responsive_patterns", [])
        
        # 반응형 브레이크포인트 설정
        breakpoints = self._determine_responsive_breakpoints(responsive_patterns)
        
        # 디바이스별 레이아웃 조정
        device_adaptations = self._create_device_adaptations(layout_strategy, responsive_patterns)
        
        responsive_strategy = layout_strategy.copy()
        responsive_strategy["responsive_design"] = {
            "breakpoints": breakpoints,
            "device_adaptations": device_adaptations,
            "responsive_patterns_used": len(responsive_patterns),
            "mobile_optimized": True,
            "tablet_optimized": True,
            "desktop_optimized": True
        }
        
        return responsive_strategy

    async def _validate_and_enhance_strategy(self, strategy: Dict, section_data: Dict, 
                                           vector_patterns: Dict) -> Dict:
        """전략 검증 및 보완"""
        
        # 전략 일관성 검증
        consistency_score = self._validate_strategy_consistency(strategy)
        
        # 콘텐츠 적합성 검증
        content_fit_score = self._validate_content_fit(strategy, section_data)
        
        # 패턴 정렬도 검증
        pattern_alignment_score = self._validate_pattern_alignment(strategy, vector_patterns)
        
        # 종합 점수 계산
        overall_score = (consistency_score + content_fit_score + pattern_alignment_score) / 3
        
        enhanced_strategy = strategy.copy()
        enhanced_strategy["validation"] = {
            "consistency_score": consistency_score,
            "content_fit_score": content_fit_score,
            "pattern_alignment_score": pattern_alignment_score,
            "overall_score": overall_score,
            "validated": overall_score > 0.6
        }
        
        # 점수가 낮으면 보완 적용
        if overall_score < 0.6:
            enhanced_strategy = self._apply_strategy_enhancements(enhanced_strategy, section_data)
        
        return enhanced_strategy

    def _create_fallback_strategy(self, section_data: Dict) -> Dict:
        """폴백 전략 생성"""
        
        image_count = len(section_data.get("images", []))
        content_length = len(section_data.get("content", ""))
        title = section_data.get("title", "제목 없음")
        
        if image_count > 0 and content_length > 200:
            layout_type = "균형형"
            visual_hierarchy = ["이미지", "제목", "본문"]
            image_placement = "상단"
        elif image_count > 0:
            layout_type = "이미지 중심"
            visual_hierarchy = ["이미지", "제목", "본문"]
            image_placement = "중앙"
        else:
            layout_type = "텍스트 중심"
            visual_hierarchy = ["제목", "부제목", "본문"]
            image_placement = "없음"
        
        return {
            "layout_type": layout_type,
            "visual_hierarchy": visual_hierarchy,
            "image_placement": image_placement,
            "text_flow": "단일 컬럼",
            "emotional_focus": "가독성",
            "key_features": ["default", "fallback", "safe"],
            "section_title": title,
            "pattern_enhanced": False,
            "ai_search_patterns_used": 0,
            "confidence_score": 0.5,
            "fallback_used": True
        }

    # ✅ 유틸리티 메서드들
    def _determine_content_type(self, content: str, images: List[Dict]) -> str:
        """콘텐츠 타입 결정"""
        if not content and not images:
            return "empty"
        elif len(images) > len(content.split()) // 50:
            return "visual_heavy"
        elif len(content.split()) > 300:
            return "text_heavy"
        else:
            return "balanced"

    def _suggest_layout_preference(self, content_length: int, image_count: int) -> str:
        """레이아웃 선호도 제안"""
        if image_count > 2:
            return "grid"
        elif content_length > 500:
            return "column"
        else:
            return "flexbox"

    def _analyze_visual_hierarchy(self, title: str, content: str, images: List[Dict]) -> List[str]:
        """시각적 계층 구조 분석"""
        hierarchy = []
        
        if images:
            hierarchy.append("이미지")
        
        hierarchy.append("제목")
        
        if len(content) > 200:
            hierarchy.append("본문")
        
        return hierarchy

    def _determine_layout_type_from_patterns(self, content_metrics: Dict, 
                                           image_metrics: Dict, 
                                           template_patterns: List[Dict]) -> str:
        """패턴 기반 레이아웃 타입 결정"""
        
        # 패턴에서 레이아웃 타입 추출
        pattern_types = []
        for pattern in template_patterns[:3]:
            layout_method = pattern.get("layout_method", "")
            if layout_method:
                pattern_types.append(layout_method)
        
        # 콘텐츠 특성과 패턴 조합
        image_count = image_metrics.get("count", 0)
        content_complexity = content_metrics.get("complexity", "medium")
        
        if "grid" in pattern_types and image_count > 1:
            return "그리드형"
        elif "flexbox" in pattern_types and content_complexity == "high":
            return "플렉스형"
        elif image_count > 0:
            return "균형형"
        else:
            return "텍스트 중심"

    def _design_visual_hierarchy_from_patterns(self, title_metrics: Dict, 
                                             content_metrics: Dict, 
                                             image_metrics: Dict, 
                                             placement_patterns: List[Dict]) -> List[str]:
        """패턴 기반 시각적 계층 구조 설계"""
        
        hierarchy = []
        
        # 패턴에서 계층 구조 힌트 추출
        pattern_hierarchies = []
        for pattern in placement_patterns[:3]:
            if "hierarchy" in pattern.get("description", "").lower():
                pattern_hierarchies.append(pattern)
        
        # 이미지 우선순위 결정
        if image_metrics.get("visual_weight") == "high":
            hierarchy.append("이미지")
        
        # 제목 배치
        if title_metrics.get("complexity"):
            hierarchy.extend(["제목", "부제목"])
        else:
            hierarchy.append("제목")
        
        # 본문 배치
        if content_metrics.get("complexity") != "low":
            hierarchy.append("본문")
        
        # 이미지가 낮은 우선순위인 경우
        if image_metrics.get("visual_weight") in ["low", "medium"] and "이미지" not in hierarchy:
            hierarchy.append("이미지")
        
        return hierarchy

    def _determine_image_placement_from_patterns(self, image_metrics: Dict, 
                                               template_patterns: List[Dict], 
                                               placement_patterns: List[Dict]) -> str:
        """패턴 기반 이미지 배치 결정"""
        
        image_count = image_metrics.get("count", 0)
        
        if image_count == 0:
            return "없음"
        
        # 패턴에서 이미지 배치 힌트 추출
        placement_hints = []
        for pattern in template_patterns + placement_patterns:
            component_name = pattern.get("component_name", "").lower()
            if "top" in component_name or "header" in component_name:
                placement_hints.append("상단")
            elif "side" in component_name or "left" in component_name or "right" in component_name:
                placement_hints.append("측면")
            elif "center" in component_name or "middle" in component_name:
                placement_hints.append("중앙")
        
        # 가장 많이 나타나는 배치 방식 선택
        if placement_hints:
            return max(set(placement_hints), key=placement_hints.count)
        
        # 기본 배치 로직
        if image_count == 1:
            return "상단"
        elif image_count <= 3:
            return "측면"
        else:
            return "그리드"

    def _design_text_flow_from_patterns(self, content_metrics: Dict, 
                                      placement_patterns: List[Dict]) -> str:
        """패턴 기반 텍스트 흐름 설계"""
        
        content_length = content_metrics.get("length", 0)
        word_count = content_metrics.get("word_count", 0)
        
        # 패턴에서 텍스트 흐름 힌트 추출
        flow_hints = []
        for pattern in placement_patterns:
            description = pattern.get("description", "").lower()
            if "column" in description:
                flow_hints.append("다중 컬럼")
            elif "single" in description:
                flow_hints.append("단일 컬럼")
        
        if flow_hints:
            return max(set(flow_hints), key=flow_hints.count)
        
        # 기본 로직
        if word_count > 400:
            return "다중 컬럼"
        else:
            return "단일 컬럼"

    def _determine_emotional_focus_from_patterns(self, section_data: Dict, 
                                               template_patterns: List[Dict]) -> str:
        """패턴 기반 감정적 포커스 결정"""
        
        title = section_data.get("title", "").lower()
        
        # 제목에서 감정 키워드 추출
        emotional_keywords = {
            "excitement": ["모험", "흥미", "신나는", "역동"],
            "calm": ["평온", "고요", "차분", "휴식"],
            "elegance": ["우아", "세련", "고급", "품격"],
            "energy": ["활기", "생동", "열정", "에너지"]
        }
        
        for emotion, keywords in emotional_keywords.items():
            if any(keyword in title for keyword in keywords):
                return emotion
        
        # 패턴에서 스타일 힌트 추출
        for pattern in template_patterns[:3]:
            component_name = pattern.get("component_name", "").lower()
            if "elegant" in component_name:
                return "elegance"
            elif "modern" in component_name:
                return "energy"
        
        return "balanced"

    def _extract_key_features_from_patterns(self, template_patterns: List[Dict]) -> List[str]:
        """패턴에서 주요 특징 추출"""
        features = []
        
        for pattern in template_patterns[:5]:
            component_name = pattern.get("component_name", "")
            layout_method = pattern.get("layout_method", "")
            
            if component_name:
                features.append(component_name.lower())
            if layout_method:
                features.append(layout_method.lower())
        
        # 중복 제거 및 상위 5개만 선택
        unique_features = list(set(features))[:5]
        
        if not unique_features:
            unique_features = ["default", "responsive", "modern"]
        
        return unique_features

    def _calculate_pattern_relevance_scores(self, template_patterns: List[Dict], 
                                          placement_patterns: List[Dict], 
                                          balance_patterns: List[Dict], 
                                          responsive_patterns: List[Dict]) -> Dict:
        """패턴 관련성 점수 계산"""
        
        total_patterns = len(template_patterns) + len(placement_patterns) + len(balance_patterns) + len(responsive_patterns)
        
        if total_patterns == 0:
            return {"overall_confidence": 0.3, "pattern_alignment": 0.3}
        
        # 각 패턴 타입별 가중치
        template_weight = len(template_patterns) * 0.4
        placement_weight = len(placement_patterns) * 0.3
        balance_weight = len(balance_patterns) * 0.2
        responsive_weight = len(responsive_patterns) * 0.1
        
        overall_confidence = min(1.0, (template_weight + placement_weight + balance_weight + responsive_weight) / 10)
        pattern_alignment = min(1.0, total_patterns / 15)
        
        return {
            "overall_confidence": overall_confidence,
            "pattern_alignment": pattern_alignment,
            "template_strength": min(1.0, len(template_patterns) / 5),
            "placement_strength": min(1.0, len(placement_patterns) / 5),
            "balance_strength": min(1.0, len(balance_patterns) / 5),
            "responsive_strength": min(1.0, len(responsive_patterns) / 3)
        }

    def _calculate_visual_balance_score(self, layout_strategy: Dict, 
                                      balance_patterns: List[Dict]) -> float:
        """시각적 균형 점수 계산"""
        
        base_score = 0.6
        
        # 레이아웃 타입별 기본 점수
        layout_type = layout_strategy.get("layout_type", "")
        if "균형" in layout_type:
            base_score += 0.2
        elif "그리드" in layout_type:
            base_score += 0.1
        
        # 패턴 매칭 보너스
        pattern_bonus = min(0.2, len(balance_patterns) * 0.04)
        
        return min(1.0, base_score + pattern_bonus)

    def _apply_balance_improvements(self, layout_strategy: Dict, 
                                  balance_patterns: List[Dict]) -> Dict:
        """균형 개선 적용"""
        
        improved_strategy = layout_strategy.copy()
        
        # 이미지 배치 조정
        if layout_strategy.get("image_placement") == "상단":
            improved_strategy["image_placement"] = "측면"
        
        # 시각적 계층 구조 조정
        hierarchy = layout_strategy.get("visual_hierarchy", [])
        if len(hierarchy) > 2 and "이미지" in hierarchy:
            # 이미지를 중간으로 이동
            hierarchy_copy = hierarchy.copy()
            if "이미지" in hierarchy_copy:
                hierarchy_copy.remove("이미지")
                hierarchy_copy.insert(1, "이미지")
            improved_strategy["visual_hierarchy"] = hierarchy_copy
        
        return improved_strategy

    def _determine_responsive_breakpoints(self, responsive_patterns: List[Dict]) -> Dict:
        """반응형 브레이크포인트 결정"""
        
        # 기본 브레이크포인트
        breakpoints = {
            "mobile": "768px",
            "tablet": "1024px",
            "desktop": "1200px"
        }
        
        # 패턴에서 브레이크포인트 힌트 추출
        for pattern in responsive_patterns:
            description = pattern.get("description", "").lower()
            if "mobile" in description and "480px" in description:
                breakpoints["mobile"] = "480px"
            elif "tablet" in description and "768px" in description:
                breakpoints["tablet"] = "768px"
        
        return breakpoints

    def _create_device_adaptations(self, layout_strategy: Dict, 
                                 responsive_patterns: List[Dict]) -> Dict:
        """디바이스별 레이아웃 적응"""
        
        base_layout = layout_strategy.get("layout_type", "균형형")
        
        adaptations = {
            "mobile": {
                "layout": "단일 컬럼",
                "image_size": "full-width",
                "font_scale": 0.9,
                "spacing_scale": 0.8
            },
            "tablet": {
                "layout": base_layout,
                "image_size": "responsive",
                "font_scale": 1.0,
                "spacing_scale": 1.0
            },
            "desktop": {
                "layout": base_layout,
                "image_size": "optimized",
                "font_scale": 1.1,
                "spacing_scale": 1.2
            }
        }
        
        return adaptations

    def _validate_strategy_consistency(self, strategy: Dict) -> float:
        """전략 일관성 검증"""
        
        required_fields = ["layout_type", "visual_hierarchy", "image_placement", "text_flow"]
        present_fields = sum(1 for field in required_fields if strategy.get(field))
        
        consistency_score = present_fields / len(required_fields)
        
        # 논리적 일관성 검사
        layout_type = strategy.get("layout_type", "")
        image_placement = strategy.get("image_placement", "")
        
        if "이미지 중심" in layout_type and image_placement == "없음":
            consistency_score -= 0.3
        
        return max(0.0, consistency_score)

    def _validate_content_fit(self, strategy: Dict, section_data: Dict) -> float:
        """콘텐츠 적합성 검증"""
        
        content_length = len(section_data.get("content", ""))
        image_count = len(section_data.get("images", []))
        
        layout_type = strategy.get("layout_type", "")
        image_placement = strategy.get("image_placement", "")
        
        fit_score = 0.7  # 기본 점수
        
        # 콘텐츠 길이와 레이아웃 매칭
        if content_length > 500 and "텍스트 중심" in layout_type:
            fit_score += 0.2
        elif content_length < 200 and "이미지 중심" in layout_type:
            fit_score += 0.2
        
        # 이미지 수와 배치 매칭
        if image_count > 2 and "그리드" in image_placement:
            fit_score += 0.1
        elif image_count == 0 and image_placement == "없음":
            fit_score += 0.1
        
        return min(1.0, fit_score)

    def _validate_pattern_alignment(self, strategy: Dict, vector_patterns: Dict) -> float:
        """패턴 정렬도 검증"""
        
        pattern_scores = vector_patterns.get("pattern_scores", {})
        overall_confidence = pattern_scores.get("overall_confidence", 0.5)
        pattern_alignment = pattern_scores.get("pattern_alignment", 0.5)
        
        # 전략과 패턴의 일치도
        alignment_score = (overall_confidence + pattern_alignment) / 2
        
        # 패턴 사용 여부에 따른 보정
        if strategy.get("pattern_enhanced", False):
            alignment_score += 0.1
        
        return min(1.0, alignment_score)

    def _apply_strategy_enhancements(self, strategy: Dict, section_data: Dict) -> Dict:
        """전략 보완 적용"""
        
        enhanced_strategy = strategy.copy()
        
        # 신뢰도가 낮은 경우 보수적 접근
        enhanced_strategy["layout_type"] = "균형형"
        enhanced_strategy["text_flow"] = "단일 컬럼"
        
        # 이미지 배치 안전화
        image_count = len(section_data.get("images", []))
        if image_count > 0:
            enhanced_strategy["image_placement"] = "상단"
        else:
            enhanced_strategy["image_placement"] = "없음"
        
        enhanced_strategy["enhanced"] = True
        enhanced_strategy["enhancement_reason"] = "low_validation_score"
        
        return enhanced_strategy

    async def process_data(self, input_data):
        """SessionAwareMixin 호환성을 위한 메서드"""
        result = await self._do_work(input_data)
        
        await self.logging_manager.log_agent_response(
            agent_name=self.__class__.__name__,
            agent_role="실시간 레이아웃 생성 에이전트",
            task_description="AI Search 벡터 패턴 기반 레이아웃 전략 수립",
            response_data=result,
            metadata={"vector_enhanced": True, "realtime_generation": True}
        )
        
        return result

    async def _do_work(self, input_data):
        """실제 작업 수행"""
        if isinstance(input_data, dict) and "section_data" in input_data:
            return await self.generate_layout_strategy_for_section(input_data["section_data"])
        else:
            return {"error": "Invalid input data for RealtimeLayoutGenerator"}