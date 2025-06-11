from typing import Dict, List, Optional
from ...utils.log.hybridlogging import get_hybrid_logger
from ...utils.data.pdf_vector_manager import PDFVectorManager

class SectionStyleAnalyzer:
    """
    편집이 완료된 최종 콘텐츠와 메타데이터를 분석하여
    각 섹션에 가장 적합한 JSX 템플릿을 찾는 지능형 선택자.
    ✅ 통합 벡터 검색 지원 추가
    """
    def __init__(self):
        self.logger = get_hybrid_logger(self.__class__.__name__)
        try:
            self.vector_manager = PDFVectorManager(default_index="jsx-component-vector-index")
            self.logger.info("PDFVectorManager 초기화 성공 (JSX 컴포넌트 인덱스)")
        except Exception as e:
            self.logger.error(f"PDFVectorManager 초기화 실패: {e}")
            self.vector_manager = None

    async def analyze_and_select_template(self, section_data: Dict, layout_strategy: Optional[Dict] = None) -> str:
        """
        주어진 섹션 데이터와 레이아웃 전략을 분석하여 최적의 템플릿 코드를 반환합니다.
        ✅ 통합 벡터 패턴 지원 추가
        """
        title = section_data.get('title', '제목 없음')
        self.logger.info(f"섹션 분석 및 템플릿 선택 시작: '{title}'")

        if not self.vector_manager:
            self.logger.warning("PDFVectorManager가 초기화되지 않았습니다. 기본 템플릿을 반환합니다.")
            return self._get_default_template()

        # ✅ 통합 벡터 패턴 활용
        metadata = section_data.get('metadata', {})
        ai_search_patterns = metadata.get('ai_search_patterns', [])
        jsx_patterns = metadata.get('jsx_patterns', [])

        # 콘텐츠 추출
        final_content = section_data.get('final_content', '')
        if not final_content and 'content' in section_data:
            final_content = section_data.get('content', '')

        # 1. 검색 쿼리 텍스트 생성 (통합 패턴 활용)
        if ai_search_patterns or jsx_patterns:
            self.logger.info("통합 벡터 패턴 기반 검색 쿼리 생성")
            query_text = self._create_query_from_unified_patterns(section_data, ai_search_patterns, jsx_patterns)
        elif layout_strategy:
            self.logger.info(f"레이아웃 전략 기반 검색 쿼리 생성: {layout_strategy.get('layout_type', '알 수 없음')} 타입")
            query_text = self._create_query_from_layout_strategy(section_data, layout_strategy)
        else:
            self.logger.info("기본 검색 쿼리 생성")
            query_text = self._create_query_text(final_content, metadata)
                
        # 2. 이미지 수 추출
        image_count = metadata.get('image_count')
        if image_count is None:
            if layout_strategy and "image_placement" in layout_strategy:
                if layout_strategy["image_placement"].lower() == "없음":
                    image_count = 0
                else:
                    image_count = 1
            elif "image" in final_content.lower() or "photo" in final_content.lower():
                image_count = 2
            else:
                image_count = 0
        
        self.logger.info(f"생성된 검색 쿼리: '{query_text}', 이미지 수: {image_count}")

        # ✅ 3. PDFVectorManager를 통해 템플릿 검색 (results 변수 정의)
        results = []  # ✅ 초기화 추가
        try:
            results = self.vector_manager.search_similar_layouts(
                query_text=query_text,
                index_name="jsx-component-vector-index",
                top_k=5
            )
            
            if not results:
                self.logger.info("직접 검색 결과 없음, 레이아웃 추천 시도")
                results = self.vector_manager.get_layout_recommendations(
                    content_description=query_text,
                    image_count=image_count,
                    index_type="jsx-component-vector-index"
                )
        except Exception as e:
            self.logger.error(f"템플릿 검색 중 오류 발생: {e}")
            results = []

        if not results:
            self.logger.warning("검색된 템플릿이 없습니다. 기본 템플릿을 사용합니다.")
            return self._get_default_template()

        # ✅ 4. 콘텐츠 길이 기반 필터링 (results 정의 후 실행)
        content_length = len(section_data.get("content", ""))
        if content_length > 100:  # 콘텐츠가 긴 경우
            filtered_results = self._filter_by_content_requirements(results, content_length)
            if filtered_results:  # 필터링 결과가 있으면 사용
                results = filtered_results

        # ✅ 5. 통합 패턴 기반 필터링 우선 적용
        if len(results) > 1:
            filtered_results = self._filter_by_unified_patterns(results, ai_search_patterns, jsx_patterns)
            
            if not filtered_results and layout_strategy:
                filtered_results = self._filter_by_layout_strategy(results, layout_strategy)
            
            if not filtered_results and image_count is not None:
                filtered_results = self._filter_by_image_count(results, image_count)
            
            if filtered_results:
                results = filtered_results

        # 6. 최적 템플릿 코드 반환
        best_template = results[0]
        template_name = best_template.get('component_name', best_template.get('id', 'unknown'))
        
        self.logger.info(f"최적 템플릿 선택 완료: {template_name} (점수: {best_template.get('score', 0):.3f})")
        
        jsx_code = best_template.get('jsx_code')
        if not jsx_code:
            self.logger.warning(f"선택된 템플릿 '{template_name}'에 JSX 코드가 없습니다. 기본 템플릿을 사용합니다.")
            return self._get_default_template()
            
        return jsx_code

    def _create_query_from_unified_patterns(self, section_data: Dict, ai_search_patterns: List[Dict], jsx_patterns: List[Dict]) -> str:
        """✅ 통합 벡터 패턴 기반 검색 쿼리 생성"""
        title = section_data.get('title', '')
        metadata = section_data.get('metadata', {})
        
        query_parts = []
        
        # AI Search 패턴에서 키워드 추출
        for pattern in ai_search_patterns[:2]:
            style = pattern.get('style', '')
            layout_type = pattern.get('layout_type', '')
            if style:
                query_parts.append(f"{style} style")
            if layout_type:
                query_parts.append(f"{layout_type} layout")
        
        # JSX 패턴에서 컴포넌트 특성 추출
        for pattern in jsx_patterns[:2]:
            component_type = pattern.get('component_type', '')
            features = pattern.get('features', [])
            if component_type:
                query_parts.append(f"{component_type} component")
            if features:
                query_parts.extend(features[:2])
        
        # 기본 메타데이터 추가
        style = metadata.get('style', '')
        emotion = metadata.get('emotion', '')
        if style:
            query_parts.append(f"{style} style")
        if emotion:
            query_parts.append(f"{emotion} mood")
        
        # 제목 추가
        if title:
            query_parts.append(f"for {title}")
        
        return " ".join(filter(None, query_parts))

    def _filter_by_unified_patterns(self, results: List[Dict], ai_search_patterns: List[Dict], jsx_patterns: List[Dict]) -> List[Dict]:
        """✅ 통합 벡터 패턴 기반 필터링"""
        filtered_results = []
        
        # AI Search 패턴 기반 필터링
        for result in results:
            result_score = 0
            
            # AI Search 패턴 매칭
            for pattern in ai_search_patterns:
                style = pattern.get('style', '')
                layout_type = pattern.get('layout_type', '')
                
                result_keywords = result.get('search_keywords', [])
                if isinstance(result_keywords, str):
                    result_keywords = [kw.strip().lower() for kw in result_keywords.split(',')]
                
                if style and any(style.lower() in kw for kw in result_keywords):
                    result_score += 2
                if layout_type and any(layout_type.lower() in kw for kw in result_keywords):
                    result_score += 2
            
            # JSX 패턴 매칭
            for pattern in jsx_patterns:
                component_type = pattern.get('component_type', '')
                features = pattern.get('features', [])
                
                if component_type and component_type.lower() in result.get('component_name', '').lower():
                    result_score += 3
                
                for feature in features:
                    if feature.lower() in str(result.get('jsx_code', '')).lower():
                        result_score += 1
            
            if result_score > 0:
                result['unified_pattern_score'] = result_score
                filtered_results.append(result)
        
        # 점수 순으로 정렬
        filtered_results.sort(key=lambda x: x.get('unified_pattern_score', 0), reverse=True)
        return filtered_results
    
    def _filter_by_content_requirements(self, results: List[Dict], content_length: int) -> List[Dict]:
        """콘텐츠 길이에 따른 템플릿 필터링"""
        
        if content_length > 500:  # 긴 콘텐츠인 경우
            # 텍스트 중심 또는 혼합 템플릿 우선
            preferred_templates = []
            for result in results:
                template_name = result.get("component_name", "").lower()
                if any(keyword in template_name for keyword in ["text", "mixed", "magazine"]):
                    preferred_templates.append(result)
            
            if preferred_templates:
                return preferred_templates
        
        return results

    def _filter_by_layout_strategy(self, results: List[Dict], layout_strategy: Dict) -> List[Dict]:
        """레이아웃 전략 기반 필터링"""
        filtered_results = []
        layout_type = layout_strategy.get("layout_type", "").lower()
        
        for result in results:
            result_layout_type = result.get('layout_type', '').lower()
            result_keywords = result.get('search_keywords', [])
            
            if isinstance(result_keywords, str):
                result_keywords = [kw.strip().lower() for kw in result_keywords.split(',')]
            
            if (layout_type in result_layout_type or 
                any(layout_type in kw for kw in result_keywords)):
                filtered_results.append(result)
        
        return filtered_results

    def _filter_by_image_count(self, results: List[Dict], image_count: int) -> List[Dict]:
        """이미지 수 기반 필터링"""
        filtered_results = []
        
        for result in results:
            result_img_count = result.get('image_count', 0)
            
            if (image_count == 0 and result_img_count == 0) or \
               (image_count == 1 and result_img_count == 1) or \
               (image_count > 1 and result_img_count > 1):
                filtered_results.append(result)
        
        return filtered_results

    def _create_query_from_layout_strategy(self, section_data: Dict, layout_strategy: Dict) -> str:
        """레이아웃 전략을 기반으로 풍부한 검색 쿼리를 생성합니다."""
        title = section_data.get('title', '')
        metadata = section_data.get('metadata', {})
        style = metadata.get('style', '')
        emotion = metadata.get('emotion', '')
        
        layout_type = layout_strategy.get('layout_type', '')
        visual_hierarchy = layout_strategy.get('visual_hierarchy', [])
        image_placement = layout_strategy.get('image_placement', '')
        text_flow = layout_strategy.get('text_flow', '')
        emotional_focus = layout_strategy.get('emotional_focus', '')
        key_features = layout_strategy.get('key_features', [])
        
        hierarchy_str = ' to '.join(visual_hierarchy) if visual_hierarchy else ''
        features_str = ' '.join(key_features) if isinstance(key_features, list) else str(key_features)
        
        query_parts = [
            f"{layout_type} layout",
            f"with {image_placement} image placement" if image_placement and image_placement.lower() != '없음' else "without images",
            f"{text_flow} text flow" if text_flow else "",
            f"visual hierarchy from {hierarchy_str}" if hierarchy_str else "",
            f"emphasizing {emotional_focus}" if emotional_focus else "",
            f"{style} style" if style else "",
            f"{emotion} mood" if emotion else "",
            features_str
        ]
        
        query = " ".join(filter(None, query_parts))
        
        if title:
            query += f" for content about {title}"
            
        return query

    def _create_query_text(self, content: str, metadata: Dict) -> str:
        """✅ 전체 콘텐츠를 활용한 쿼리 텍스트 생성"""
        style = metadata.get('style', '')
        emotion = metadata.get('emotion', '')
        keywords = metadata.get('keywords', [])
        
        summary = content
        
        query_parts = [style, emotion]
        
        if isinstance(keywords, list):
            query_parts.extend(keywords)
        elif isinstance(keywords, str):
            query_parts.append(keywords)
            
        query_parts.append(summary)
        
        return " ".join(filter(None, query_parts))
        
    def _get_default_template(self) -> str:
        """기본 템플릿 반환"""
        return """
        export default function DefaultTemplate(props) {
          return (
            <div className="p-4 max-w-4xl mx-auto">
              <h1 className="text-3xl font-bold mb-4">{props.title}</h1>
              <div className="prose max-w-none" dangerouslySetInnerHTML={{ __html: props.body }} />
            </div>
          );
        }
        """

