import json
from typing import Dict, List, Any, Optional
from utils.log.hybridlogging import get_hybrid_logger
from utils.data.pdf_vector_manager import PDFVectorManager

class SectionStyleAnalyzer:
    """
    편집이 완료된 최종 콘텐츠와 메타데이터를 분석하여
    각 섹션에 가장 적합한 JSX 템플릿을 찾는 지능형 선택자.
    """
    def __init__(self):
        self.logger = get_hybrid_logger(self.__class__.__name__)
        # PDFVectorManager를 초기화하고 JSX 컴포넌트 인덱스 사용
        try:
            self.vector_manager = PDFVectorManager(default_index="jsx-component-vector-index")
            self.logger.info("PDFVectorManager 초기화 성공 (JSX 컴포넌트 인덱스)")
        except Exception as e:
            self.logger.error(f"PDFVectorManager 초기화 실패: {e}")
            self.vector_manager = None

    async def analyze_and_select_template(self, section_data: Dict, layout_strategy: Optional[Dict] = None) -> str:
        """
        주어진 섹션 데이터와 레이아웃 전략을 분석하여 최적의 템플릿 코드를 반환합니다.

        Args:
            section_data (Dict): 최종 편집된 섹션 데이터.
                                 {'title': '...', 'final_content': '...', 'metadata': {'style': '...', 'image_count': ...}}
            layout_strategy (Dict, optional): RealtimeLayoutGenerator가 생성한 레이아웃 전략.
                                            {'layout_type': '...', 'visual_hierarchy': [...], ...}
        
        Returns:
            str: 선택된 JSX 템플릿의 실제 코드.
        """
        title = section_data.get('title', '제목 없음')
        self.logger.info(f"섹션 분석 및 템플릿 선택 시작: '{title}'")

        if not self.vector_manager:
            self.logger.warning("PDFVectorManager가 초기화되지 않았습니다. 기본 템플릿을 반환합니다.")
            return self._get_default_template()

        # 콘텐츠 추출
        final_content = section_data.get('final_content', '')
        if not final_content and 'content' in section_data:
            final_content = section_data.get('content', '')  # 대체 필드 확인
            
        metadata = section_data.get('metadata', {})

        # 1. 검색 쿼리 텍스트 생성 (레이아웃 전략 활용)
        if layout_strategy:
            self.logger.info(f"레이아웃 전략 기반 검색 쿼리 생성: {layout_strategy.get('layout_type', '알 수 없음')} 타입")
            query_text = self._create_query_from_layout_strategy(section_data, layout_strategy)
        else:
            self.logger.info("레이아웃 전략 없음, 기본 검색 쿼리 생성")
            query_text = self._create_query_text(final_content, metadata)
        
        # 2. 이미지 수 추출
        image_count = metadata.get('image_count')
        if image_count is None:
            # 이미지 수를 추론
            if layout_strategy and "image_placement" in layout_strategy:
                # 레이아웃 전략에서 이미지 배치 정보 확인
                if layout_strategy["image_placement"].lower() == "없음":
                    image_count = 0
                else:
                    image_count = 1  # 최소 1개 이상
            elif "image" in final_content.lower() or "photo" in final_content.lower():
                image_count = 2  # 이미지 관련 단어가 있으면 기본적으로 이미지가 있다고 가정
            else:
                image_count = 0  # 없으면 텍스트 중심으로 가정
        
        self.logger.info(f"생성된 검색 쿼리: '{query_text}', 이미지 수: {image_count}")

        # 3. PDFVectorManager를 통해 템플릿 검색
        try:
            # search_similar_layouts 메서드 사용
            results = self.vector_manager.search_similar_layouts(
                query_text=query_text,
                index_name="jsx-component-vector-index",
                top_k=5  # 여러 결과 중 최적의 것 선택
            )
            
            # 결과가 없으면 get_layout_recommendations 시도
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

        # 4. 이미지 수와 레이아웃 타입에 따른 필터링 (결과가 여러 개일 경우)
        if len(results) > 1:
            filtered_results = []
            
            # 레이아웃 전략이 있는 경우, 레이아웃 타입 기반 필터링 우선
            if layout_strategy and "layout_type" in layout_strategy:
                layout_type = layout_strategy["layout_type"].lower()
                
                for result in results:
                    # 레이아웃 타입 일치 여부 확인
                    result_layout_type = result.get('layout_type', '').lower()
                    result_keywords = result.get('search_keywords', [])
                    
                    # 키워드가 문자열인 경우 리스트로 변환
                    if isinstance(result_keywords, str):
                        result_keywords = [kw.strip().lower() for kw in result_keywords.split(',')]
                    
                    # 레이아웃 타입이 직접 일치하거나 키워드에 포함된 경우
                    if (layout_type in result_layout_type or 
                        any(layout_type in kw for kw in result_keywords)):
                        filtered_results.append(result)
            
            # 레이아웃 타입 필터링 결과가 없거나 레이아웃 전략이 없는 경우, 이미지 수 기반 필터링
            if not filtered_results and image_count is not None:
                for result in results:
                    result_img_count = result.get('image_count', 0)
                    # 이미지 수가 비슷한 템플릿 우선
                    if (image_count == 0 and result_img_count == 0) or \
                       (image_count == 1 and result_img_count == 1) or \
                       (image_count > 1 and result_img_count > 1):
                        filtered_results.append(result)
            
            # 필터링 결과가 있으면 사용, 없으면 원래 결과 사용
            if filtered_results:
                results = filtered_results

        # 5. 최적 템플릿 코드 반환
        best_template = results[0]  # 가장 높은 점수의 템플릿
        template_name = best_template.get('component_name', best_template.get('id', 'unknown'))
        
        self.logger.info(f"최적 템플릿 선택 완료: {template_name} (점수: {best_template.get('score', 0):.3f})")
        
        # JSX 코드 추출
        jsx_code = best_template.get('jsx_code')
        if not jsx_code:
            self.logger.warning(f"선택된 템플릿 '{template_name}'에 JSX 코드가 없습니다. 기본 템플릿을 사용합니다.")
            return self._get_default_template()
            
        return jsx_code

    def _create_query_from_layout_strategy(self, section_data: Dict, layout_strategy: Dict) -> str:
        """레이아웃 전략을 기반으로 풍부한 검색 쿼리를 생성합니다."""
        # 섹션 데이터에서 기본 정보 추출
        title = section_data.get('title', '')
        metadata = section_data.get('metadata', {})
        style = metadata.get('style', '')
        emotion = metadata.get('emotion', '')
        
        # 레이아웃 전략에서 중요 정보 추출
        layout_type = layout_strategy.get('layout_type', '')
        visual_hierarchy = layout_strategy.get('visual_hierarchy', [])
        image_placement = layout_strategy.get('image_placement', '')
        text_flow = layout_strategy.get('text_flow', '')
        emotional_focus = layout_strategy.get('emotional_focus', '')
        key_features = layout_strategy.get('key_features', [])
        
        # 시각적 계층 구조를 문자열로 변환
        hierarchy_str = ' to '.join(visual_hierarchy) if visual_hierarchy else ''
        
        # 키 특징을 문자열로 변환
        features_str = ' '.join(key_features) if isinstance(key_features, list) else str(key_features)
        
        # 모든 요소를 조합하여 풍부한 쿼리 생성
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
        
        # 빈 문자열 제거 후 조합
        query = " ".join(filter(None, query_parts))
        
        # 제목을 추가하여 콘텐츠 특성 반영
        if title:
            query += f" for content about {title}"
            
        return query

    def _create_query_text(self, content: str, metadata: Dict) -> str:
        """벡터 검색을 위한 쿼리 텍스트 생성 (레이아웃 전략이 없을 때 사용)"""
        style = metadata.get('style', '')
        emotion = metadata.get('emotion', '')
        keywords = metadata.get('keywords', [])
        
        # 콘텐츠 요약 (간단한 버전)
        summary = (content[:100] + '..') if len(content) > 100 else content
        
        query_parts = [style, emotion]
        
        # 키워드가 문자열이면 그대로 사용, 리스트면 합치기
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