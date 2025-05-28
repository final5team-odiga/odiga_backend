import re
import asyncio
import json
from typing import Dict, List
from crewai import Agent, Task
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager


class JSXContentAnalyzer:
    """콘텐츠 분석 전문 에이전트 (벡터 데이터 통합)"""

    def __init__(self):
        self.llm = get_azure_llm()
        self.vector_manager = PDFVectorManager()

    async def analyze_content_for_jsx(self, content: Dict, section_index: int, total_sections: int) -> Dict:
        """JSX 생성을 위한 콘텐츠 분석 (벡터 데이터 활용)"""

        # 기본 분석
        basic_analysis = self._create_default_analysis(content, section_index)

        # 벡터 데이터로 분석 강화
        vector_enhanced_analysis = await self._enhance_analysis_with_vectors(
            content, basic_analysis)

        print(f"✅ 콘텐츠 분석 완료: {vector_enhanced_analysis.get('recommended_layout', '기본')} 레이아웃 권장 (벡터 강화: {vector_enhanced_analysis.get('vector_enhanced', False)})")

        return vector_enhanced_analysis

    async def _enhance_analysis_with_vectors(self, content: Dict, basic_analysis: Dict) -> Dict:
        """벡터 데이터로 분석 강화"""

        try:
            # 콘텐츠를 벡터 검색 쿼리로 변환
            content_query = f"{content.get('title', '')} {content.get('body', '')[:300]}"

            # 유사한 매거진 레이아웃 검색
            similar_layouts = await self.vector_manager.search_similar_layouts(
                content_query,
                "magazine_layout",
                top_k=5
            )

            if similar_layouts:
                # 벡터 데이터 기반 분석 보강
                enhanced_analysis = basic_analysis.copy()
                enhanced_analysis['vector_enhanced'] = True
                enhanced_analysis['similar_layouts'] = similar_layouts

                # 벡터 데이터 기반 레이아웃 추천 재조정
                vector_layout_recommendation = self._get_vector_layout_recommendation(
                    similar_layouts)
                if vector_layout_recommendation:
                    enhanced_analysis['recommended_layout'] = vector_layout_recommendation
                    enhanced_analysis['layout_confidence'] = self._calculate_vector_confidence(
                        similar_layouts)

                # 벡터 기반 색상 팔레트 추천
                enhanced_analysis['vector_color_palette'] = self._get_vector_color_palette(
                    similar_layouts)

                # 벡터 기반 타이포그래피 스타일
                enhanced_analysis['vector_typography'] = self._get_vector_typography_style(
                    similar_layouts)

                return enhanced_analysis
            else:
                basic_analysis['vector_enhanced'] = False
                return basic_analysis

        except Exception as e:
            print(f"⚠️ 벡터 데이터 분석 강화 실패: {e}")
            basic_analysis['vector_enhanced'] = False
            return basic_analysis

    def _get_vector_layout_recommendation(self, similar_layouts: List[Dict]) -> str:
        """벡터 데이터 기반 레이아웃 추천"""

        # PDF 소스 분석
        layout_types = []
        for layout in similar_layouts:
            layout_info = layout.get('layout_info', {})

            # 이미지와 텍스트 블록 비율로 레이아웃 타입 추론
            text_blocks = len(layout_info.get('text_blocks', []))
            images = len(layout_info.get('images', []))

            if images == 0:
                layout_types.append('minimal')
            elif images == 1 and text_blocks <= 3:
                layout_types.append('hero')
            elif images <= 3 and text_blocks <= 6:
                layout_types.append('grid')
            elif images > 3:
                layout_types.append('gallery')
            else:
                layout_types.append('magazine')

        # 가장 많이 추천된 타입 반환
        if layout_types:
            return max(set(layout_types), key=layout_types.count)

        return None

    def _calculate_vector_confidence(self, similar_layouts: List[Dict]) -> float:
        """벡터 기반 신뢰도 계산"""

        if not similar_layouts:
            return 0.5

        # 유사도 점수 평균
        scores = [layout.get('score', 0) for layout in similar_layouts]
        avg_score = sum(scores) / len(scores)

        # 레이아웃 일관성 보너스
        layout_consistency = len(set(self._get_vector_layout_recommendation(
            [layout]) for layout in similar_layouts))
        consistency_bonus = 0.2 if layout_consistency <= 2 else 0.1

        return min(avg_score + consistency_bonus, 1.0)

    def _get_vector_color_palette(self, similar_layouts: List[Dict]) -> str:
        """벡터 데이터 기반 색상 팔레트"""

        # PDF 소스 기반 색상 팔레트 매핑
        pdf_sources = [layout.get('pdf_name', '').lower()
                       for layout in similar_layouts]

        if any('travel' in source for source in pdf_sources):
            return "여행 블루 팔레트"
        elif any('culture' in source for source in pdf_sources):
            return "문화 브라운 팔레트"
        elif any('lifestyle' in source for source in pdf_sources):
            return "라이프스타일 핑크 팔레트"
        elif any('nature' in source for source in pdf_sources):
            return "자연 그린 팔레트"
        else:
            return "클래식 그레이 팔레트"

    def _get_vector_typography_style(self, similar_layouts: List[Dict]) -> str:
        """벡터 데이터 기반 타이포그래피 스타일"""

        # 텍스트 블록 분석
        total_text_blocks = sum(len(layout.get('layout_info', {}).get(
            'text_blocks', [])) for layout in similar_layouts)
        avg_text_blocks = total_text_blocks / \
            len(similar_layouts) if similar_layouts else 0

        if avg_text_blocks > 8:
            return "정보 집약형"
        elif avg_text_blocks > 5:
            return "균형잡힌 편집형"
        elif avg_text_blocks > 2:
            return "미니멀 모던"
        else:
            return "대형 타이틀 중심"

    def _create_default_analysis(self, content: Dict, section_index: int) -> Dict:
        """기본 분석 결과 생성 (기존 메서드 유지)"""

        body_length = len(content.get('body', ''))
        image_count = len(content.get('images', []))

        # 텍스트 길이 기반 레이아웃 결정
        if body_length < 300:
            recommended_layout = "minimal"
        elif image_count == 0:
            recommended_layout = "minimal"
        elif image_count == 1:
            recommended_layout = "hero"
        elif image_count <= 4:
            recommended_layout = "grid"
        else:
            recommended_layout = "magazine"

        return {
            "text_length": "보통" if body_length < 500 else "긺",
            "emotion_tone": "peaceful",
            "image_strategy": "그리드" if image_count > 1 else "단일",
            "layout_complexity": "보통",
            "recommended_layout": recommended_layout,
            "color_palette": "차분한 블루",
            "typography_style": "모던"
        }
