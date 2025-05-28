import asyncio
from typing import Dict, List
from crewai import Agent, Task
from custom_llm import get_azure_llm


class JSXLayoutDesigner:
    """레이아웃 설계 전문 에이전트"""

    def __init__(self):
        self.llm = get_azure_llm()

    def create_agent(self):
        return Agent(
            role="JSX Layout Design Specialist",
            goal="분석된 콘텐츠 특성에 맞는 최적의 JSX 레이아웃 구조를 설계",
            backstory="""당신은 React JSX 레이아웃 설계 전문가입니다.
            다양한 콘텐츠 특성에 맞는 레이아웃 구조를 설계하고,
            Styled Components를 활용한 효과적인 CSS 구조를 계획하는 전문가입니다.""",
            verbose=True,
            llm=self.llm
        )

    async def design_layout_structure(self, content: Dict, analysis: Dict, component_name: str) -> Dict:
        """레이아웃 구조 설계"""

        agent = self.create_agent()

        design_task = Task(
            description=f"""
            분석된 콘텐츠 특성에 맞는 JSX 레이아웃 구조를 설계하세요:
            
            **콘텐츠 정보:**
            - 제목: {content.get('title', '')}
            - 이미지 수: {len(content.get('images', []))}개
            - 텍스트 길이: {len(content.get('body', ''))}자
            
            **분석 결과:**
            - 권장 레이아웃: {analysis.get('recommended_layout', 'grid')}
            - 감정 톤: {analysis.get('emotion_tone', 'neutral')}
            - 이미지 전략: {analysis.get('image_strategy', 'grid')}
            - 복잡도: {analysis.get('layout_complexity', 'normal')}
            
            **설계 요구사항:**
            1. 컴포넌트 이름: {component_name}
            2. Styled Components 구조 계획
            3. 그리드 시스템 설계
            4. 이미지 배치 전략
            5. 텍스트 흐름 계획
            6. 색상 팔레트 선정
            
            **출력 형식:**
            {{
                "layout_type": "grid",
                "grid_structure": "repeat(2, 1fr)",
                "styled_components": ["Container", "Title", "ImageGrid"],
                "color_scheme": {{"primary": "#2c3e50", "secondary": "#ecf0f1"}},
                "image_layout": "grid_2x2",
                "text_sections": ["header", "content", "footer"]
            }}
            """,
            agent=agent,
            expected_output="레이아웃 구조 설계 JSON"
        )

        try:
            result = await agent.execute_task(design_task)
            design_result = self._parse_design_result(str(result), analysis)

            print(f"✅ 레이아웃 설계 완료: {design_result.get('layout_type', '기본')} 구조")
            return design_result

        except Exception as e:
            print(f"⚠️ 레이아웃 설계 실패: {e}")
            return self._create_default_design(analysis, component_name)

    def _parse_design_result(self, result_text: str, analysis: Dict) -> Dict:
        """설계 결과 파싱"""
        try:
            import json
            import re

            json_match = re.search(r'\{[^}]*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        return self._create_default_design(analysis, "DefaultComponent")

    def _create_default_design(self, analysis: Dict, component_name: str) -> Dict:
        """기본 설계 생성"""

        layout_type = analysis.get('recommended_layout', 'grid')

        designs = {
            "hero": {
                "layout_type": "hero",
                "grid_structure": "1fr",
                "styled_components": ["Container", "HeroImage", "Title", "Subtitle", "Content"],
                "color_scheme": {"primary": "#667eea", "secondary": "#764ba2"},
                "image_layout": "hero_single",
                "text_sections": ["hero_content"]
            },
            "grid": {
                "layout_type": "grid",
                "grid_structure": "1fr 1fr",
                "styled_components": ["Container", "Header", "MainGrid", "TextSection", "ImageSection"],
                "color_scheme": {"primary": "#2c3e50", "secondary": "#f8f9fa"},
                "image_layout": "grid_2x2",
                "text_sections": ["header", "main_content"]
            },
            "magazine": {
                "layout_type": "magazine",
                "grid_structure": "2fr 1fr",
                "styled_components": ["Container", "MagazineGrid", "FeaturedImage", "ContentSection"],
                "color_scheme": {"primary": "#2c3e50", "secondary": "#e74c3c"},
                "image_layout": "featured_plus_thumbnails",
                "text_sections": ["title_section", "content_section", "bottom_section"]
            },
            "minimal": {
                "layout_type": "minimal",
                "grid_structure": "1fr",
                "styled_components": ["Container", "Title", "Subtitle", "Content", "Divider"],
                "color_scheme": {"primary": "#2c3e50", "secondary": "#ecf0f1"},
                "image_layout": "none",
                "text_sections": ["centered_content"]
            },
            "card": {
                "layout_type": "card",
                "grid_structure": "1fr",
                "styled_components": ["Container", "Card", "CardImage", "CardContent"],
                "color_scheme": {"primary": "#2c3e50", "secondary": "#ffffff"},
                "image_layout": "card_top",
                "text_sections": ["card_content"]
            }
        }

        return designs.get(layout_type, designs["grid"])
