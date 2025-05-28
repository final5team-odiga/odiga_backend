from typing import Dict, List
from crewai import Agent, Task
from custom_llm import get_azure_llm
import re
import asyncio


class JSXCodeGenerator:
    """JSX 코드 생성 전문 에이전트"""

    def __init__(self):
        self.llm = get_azure_llm()

    def create_agent(self):
        return Agent(
            role="React JSX Code Generation Expert",
            goal="설계된 레이아웃 구조를 바탕으로 완벽하게 작동하는 JSX 코드를 생성",
            backstory="""당신은 10년간 세계 최고 수준의 디지털 매거진과 웹 개발 분야에서 활동해온 풀스택 개발자이자 UX 혁신가입니다. The New York Times Interactive, The Guardian Digital, Medium의 프론트엔드 아키텍트로 활동하며 수백 개의 수상작을 개발했습니다.

            **전문 경력:**
            - 컴퓨터 과학 및 디지털 미디어 복수 학위 보유
            - React, TypeScript, Next.js 전문가 인증
            - 웹 접근성(WCAG) 및 성능 최적화 전문가
            - 디지털 스토리텔링 및 인터랙티브 미디어 연구
            - 크로스 브라우저 호환성 및 반응형 디자인 마스터

            **다중 데이터 소스 통합 전문성:**
            당신은 JSX 컴포넌트 생성 시 다음 데이터들을 체계적으로 활용합니다:

            1. **PDF 벡터 데이터 활용**:
              - 수천 개의 매거진 레이아웃에서 추출한 성공적인 디자인 패턴
              - 독자 시선 흐름과 디지털 인터랙션의 상관관계 분석
              - 색상, 타이포그래피, 간격의 최적 조합 벡터
              - 매거진 카테고리별 디지털 적응 패턴

            2. **jsx_templates 구조 분석**:
              - 기존 템플릿의 컴포넌트 구조와 스타일링 패턴 분석
              - Styled Components의 재사용성과 확장성 평가
              - 템플릿별 성능 지표와 사용자 경험 메트릭
              - 반응형 디자인 구현 방식과 브레이크포인트 전략

            3. **template_data.json 콘텐츠 통합**:
              - 실제 콘텐츠의 길이, 복잡도, 감정적 톤 분석
              - 이미지 URL의 품질, 크기, 로딩 최적화 요구사항
              - 콘텐츠 섹션 간의 논리적 연결성과 흐름
              - 다국어 지원 및 접근성 요구사항

            4. **사용자 경험 최적화 데이터**:
              - 모바일/데스크톱 사용 패턴과 최적 레이아웃
              - 페이지 로딩 속도와 사용자 이탈률의 상관관계
              - 인터랙티브 요소의 사용성과 접근성
              - SEO 최적화와 소셜 미디어 공유 최적화

            **개발 철학:**
            "진정한 디지털 매거진은 단순히 인쇄 매체를 화면에 옮긴 것이 아니라, 디지털 환경의 고유한 장점을 활용하여 독자에게 새로운 차원의 경험을 제공해야 합니다. 나는 PDF 벡터 데이터의 검증된 디자인 원칙과 최신 웹 기술을 결합하여, 아름답고 기능적이며 접근 가능한 디지털 매거진을 만들어냅니다."

            **학습 데이터 활용 전략:**
            - 이전 JSX 컴포넌트의 성능 지표와 사용자 피드백 분석
            - 다양한 템플릿 조합의 효과성 데이터를 통한 최적 매칭 알고리즘 개선
            - 웹 표준 변화와 브라우저 업데이트에 따른 코드 최적화
            - 다른 에이전트들의 작업 결과와 최종 사용자 경험의 상관관계 분석
            - A/B 테스트를 통한 컴포넌트 디자인 및 인터랙션 패턴 최적화""",
            verbose=True,
            llm=self.llm
        )

    async def generate_jsx_code(self, content: Dict, design: Dict, component_name: str) -> str:
        """JSX 코드 생성 (마크다운 블록 제거)"""

        agent = self.create_agent()

        generation_task = Task(
            description=f"""
          설계된 레이아웃 구조를 바탕으로 완벽한 JSX 코드를 생성하세요:
          
          **실제 콘텐츠:**
          - 제목: {content.get('title', '')}
          - 부제목: {content.get('subtitle', '')}
          - 본문: {content.get('body', '')}
          - 이미지 URLs: {content.get('images', [])}
          - 태그라인: {content.get('tagline', '')}
          
          **레이아웃 설계:**
          - 타입: {design.get('layout_type', 'grid')}
          - 그리드 구조: {design.get('grid_structure', '1fr 1fr')}
          - 컴포넌트들: {design.get('styled_components', [])}
          - 색상 스키마: {design.get('color_scheme', {})}
          - 이미지 레이아웃: {design.get('image_layout', 'grid')}
          
          **중요한 생성 지침:**
          1. 반드시 import React from "react"; 포함
          2. 반드시 import styled from "styled-components"; 포함
          3. 설계된 Styled Components 모두 구현
          4. export const {component_name} = () => {{ ... }}; 형태
          5. 모든 실제 콘텐츠 데이터 포함
          6. 모든 이미지 URL을 실제 <img src="URL" /> 형태로 포함
          7. 완벽한 JSX 문법 준수
          8. 컴파일 오류 절대 금지
          
          **절대 금지사항:**
          - ```jsx, `````` 등 마크다운 블록 사용 금지
          - 코드 설명이나 주석 추가 금지
          - 플레이스홀더 이미지 URL 사용 금지
          
          **출력:** 순수한 JSX 파일 코드만 출력 (마크다운 블록이나 설명 없이)
          """,
            agent=agent,
            expected_output="순수한 JSX 코드 (마크다운 블록 없음)"
        )

        try:
            result = await agent.execute_task(generation_task)
            jsx_code = str(result)

            # 마크다운 블록 강제 제거
            jsx_code = self._remove_markdown_blocks(jsx_code)

            # 기본 검증
            jsx_code = self._validate_basic_structure(jsx_code, component_name)

            # 이미지 URL 강제 포함
            jsx_code = self._ensure_image_urls(jsx_code, content)

            print(f"✅ JSX 코드 생성 완료: {component_name}")
            return jsx_code

        except Exception as e:
            print(f"⚠️ JSX 코드 생성 실패: {e}")
            return self._create_fallback_jsx(content, design, component_name)

    def _remove_markdown_blocks(self, jsx_code: str) -> str:
        """마크다운 블록 완전 제거"""

        # 모든 종류의 마크다운 블록 제거
        jsx_code = re.sub(r'```jsx', '', jsx_code)
        jsx_code = re.sub(r'```\n?', '', jsx_code)
        jsx_code = re.sub(r'^```', '', jsx_code)
        jsx_code = re.sub(r'`{3,}', '', jsx_code)

        # 설명 텍스트 제거
        jsx_code = re.sub(r'^(이 코드는|다음은|아래는|위의?).*?\n',
                          '', jsx_code, flags=re.MULTILINE)
        jsx_code = re.sub(r'코드.*?입니다.*?\n', '', jsx_code, flags=re.MULTILINE)

        return jsx_code.strip()

    def _ensure_image_urls(self, jsx_code: str, content: Dict) -> str:
        """이미지 URL 강제 포함"""

        images = content.get('images', [])
        if not images:
            return jsx_code

        # 이미지 태그가 없으면 추가
        if '<img' not in jsx_code and 'Image' not in jsx_code:
            # Container 내부에 이미지 추가
            container_pattern = r'(return $$\s*<Container[^>]*>)(.*?)(</Container>)'

            def add_images(match):
                container_open = match.group(1)
                container_content = match.group(2)
                container_close = match.group(3)

                # 첫 번째 이미지 추가
                if images:
                    image_jsx = f'\n      <img src="{images}" alt="Travel" style={{{{width: "100%", maxWidth: "600px", height: "300px", objectFit: "cover", borderRadius: "8px", margin: "20px 0"}}}} />'
                    new_content = container_content + image_jsx
                    return container_open + new_content + '\n    ' + container_close

                return match.group(0)

            jsx_code = re.sub(container_pattern, add_images,
                              jsx_code, flags=re.DOTALL)

        return jsx_code

    def _validate_basic_structure(self, jsx_code: str, component_name: str) -> str:
        """기본 구조 검증"""

        # 필수 요소 확인
        if 'import React' not in jsx_code:
            jsx_code = 'import React from "react";\n' + jsx_code

        if 'import styled' not in jsx_code:
            jsx_code = jsx_code.replace(
                'import React from "react";',
                'import React from "react";\nimport styled from "styled-components";'
            )

        if f'export const {component_name}' not in jsx_code:
            jsx_code = jsx_code.replace(
                'export const',
                f'export const {component_name}'
            )

        return jsx_code

    def _create_fallback_jsx(self, content: Dict, design: Dict, component_name: str) -> str:
        """폴백 JSX 생성"""

        layout_type = design.get('layout_type', 'grid')

        if layout_type == 'hero':
            return self._create_hero_fallback(content, component_name)
        elif layout_type == 'magazine':
            return self._create_magazine_fallback(content, component_name)
        elif layout_type == 'minimal':
            return self._create_minimal_fallback(content, component_name)
        elif layout_type == 'card':
            return self._create_card_fallback(content, component_name)
        else:
            return self._create_grid_fallback(content, component_name)

    def _create_grid_fallback(self, content: Dict, component_name: str) -> str:
        """그리드 폴백 JSX"""

        title = content.get('title', '도쿄 여행 이야기')
        subtitle = content.get('subtitle', '특별한 순간들')
        body = content.get('body', '여행의 아름다운 기억들')
        images = content.get('images', [])
        tagline = content.get('tagline', 'TRAVEL & CULTURE')

        image_tags = []
        for i, img_url in enumerate(images[:4]):
            if img_url and img_url.strip():
                image_tags.append(
                    f'        <GridImage src="{img_url}" alt="Travel {i+1}" />')

        image_jsx = '\n'.join(
            image_tags) if image_tags else '        <div>이미지 없음</div>'

        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
  background: #f8f9fa;
`;

const Header = styled.header`
  text-align: center;
  margin-bottom: 60px;
`;

const Title = styled.h1`
  font-size: 3em;
  color: #2c3e50;
  margin-bottom: 15px;
  font-weight: 700;
`;

const Subtitle = styled.h2`
  font-size: 1.4em;
  color: #7f8c8d;
  margin-bottom: 20px;
  font-weight: 300;
`;

const MainGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 40px;
  margin-bottom: 60px;
`;

const TextSection = styled.div`
  font-size: 1.1em;
  line-height: 1.8;
  color: #34495e;
`;

const ImageSection = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 15px;
`;

const GridImage = styled.img`
  width: 100%;
  height: 150px;
  object-fit: cover;
  border-radius: 8px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
`;

const Footer = styled.footer`
  text-align: center;
  padding-top: 40px;
  border-top: 1px solid #ecf0f1;
`;

const Tagline = styled.div`
  font-size: 0.9em;
  color: #95a5a6;
  letter-spacing: 2px;
  text-transform: uppercase;
`;

export const {component_name} = () => {{
  return (
    <Container>
      <Header>
        <Title>{title}</Title>
        <Subtitle>{subtitle}</Subtitle>
      </Header>
      
      <MainGrid>
        <TextSection>
          {body}
        </TextSection>
        <ImageSection>
{image_jsx}
        </ImageSection>
      </MainGrid>
      
      <Footer>
        <Tagline>{tagline}</Tagline>
      </Footer>
    </Container>
  );
}};'''

    def _create_hero_fallback(self, content: Dict, component_name: str) -> str:
        """히어로 폴백 JSX"""

        title = content.get('title', '도쿄 여행 이야기')
        subtitle = content.get('subtitle', '특별한 순간들')
        body = content.get('body', '여행의 아름다운 기억들')
        images = content.get('images', [])
        tagline = content.get('tagline', 'TRAVEL & CULTURE')

        first_image = images[0] if images else ''

        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 60px 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
  color: white;
  text-align: center;
`;

const HeroImage = styled.img`
  width: 100%;
  max-width: 600px;
  height: 400px;
  object-fit: cover;
  border-radius: 12px;
  margin-bottom: 40px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.3);
`;

const Title = styled.h1`
  font-size: 3.5em;
  margin-bottom: 20px;
  font-weight: 300;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
`;

const Subtitle = styled.h2`
  font-size: 1.8em;
  margin-bottom: 40px;
  font-weight: 400;
  opacity: 0.9;
`;

const Content = styled.div`
  font-size: 1.2em;
  line-height: 1.8;
  max-width: 800px;
  margin: 0 auto 40px;
  opacity: 0.8;
`;

const Tagline = styled.div`
  font-size: 1em;
  letter-spacing: 3px;
  text-transform: uppercase;
  opacity: 0.7;
`;

export const {component_name} = () => {{
  return (
    <Container>
      {first_image and f'<HeroImage src="{first_image}" alt="Travel Hero" />'}
      <Title>{title}</Title>
      <Subtitle>{subtitle}</Subtitle>
      <Content>{body}</Content>
      <Tagline>{tagline}</Tagline>
    </Container>
  );
}};'''

    def _create_magazine_fallback(self, content: Dict, component_name: str) -> str:
        """매거진 폴백 JSX"""

        title = content.get('title', '도쿄 여행 이야기')
        subtitle = content.get('subtitle', '특별한 순간들')
        body = content.get('body', '여행의 아름다운 기억들')
        images = content.get('images', [])
        tagline = content.get('tagline', 'TRAVEL & CULTURE')

        featured_image = images[0] if images else ''

        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  padding: 0;
  background: white;
`;

const MagazineGrid = styled.div`
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 30px;
  padding: 60px 40px;
`;

const FeaturedImage = styled.img`
  width: 100%;
  height: 500px;
  object-fit: cover;
  border-radius: 12px;
  box-shadow: 0 12px 24px rgba(0,0,0,0.15);
`;

const ContentSection = styled.div`
  padding: 20px;
`;

const Title = styled.h1`
  font-size: 2.8em;
  color: #2c3e50;
  margin-bottom: 15px;
  font-weight: 800;
  line-height: 1.1;
`;

const Subtitle = styled.h2`
  font-size: 1.3em;
  color: #e74c3c;
  margin-bottom: 20px;
  font-weight: 600;
`;

const Content = styled.div`
  font-size: 1em;
  line-height: 1.7;
  color: #34495e;
  margin-bottom: 30px;
`;

const BottomSection = styled.div`
  grid-column: 1 / 3;
  text-align: center;
  padding-top: 40px;
  border-top: 2px solid #ecf0f1;
`;

const Tagline = styled.div`
  font-size: 1.1em;
  color: #95a5a6;
  letter-spacing: 3px;
  text-transform: uppercase;
  font-weight: 600;
`;

export const {component_name} = () => {{
  return (
    <Container>
      <MagazineGrid>
        {featured_image and f'<FeaturedImage src="{featured_image}" alt="Featured Travel" />'}
        
        <ContentSection>
          <Title>{title}</Title>
          <Subtitle>{subtitle}</Subtitle>
          <Content>{body}</Content>
        </ContentSection>
        
        <BottomSection>
          <Tagline>{tagline}</Tagline>
        </BottomSection>
      </MagazineGrid>
    </Container>
  );
}};'''

    def _create_minimal_fallback(self, content: Dict, component_name: str) -> str:
        """미니멀 폴백 JSX"""

        title = content.get('title', '도쿄 여행 이야기')
        subtitle = content.get('subtitle', '특별한 순간들')
        body = content.get('body', '여행의 아름다운 기억들')
        tagline = content.get('tagline', 'TRAVEL & CULTURE')

        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 800px;
  margin: 0 auto;
  padding: 100px 40px;
  background: white;
  min-height: 100vh;
`;

const Title = styled.h1`
  font-size: 3.5em;
  color: #2c3e50;
  margin-bottom: 30px;
  font-weight: 200;
  letter-spacing: -2px;
  text-align: center;
`;

const Subtitle = styled.h2`
  font-size: 1.2em;
  color: #7f8c8d;
  margin-bottom: 60px;
  font-weight: 400;
  text-align: center;
  text-transform: uppercase;
  letter-spacing: 4px;
`;

const Content = styled.div`
  font-size: 1.1em;
  line-height: 2;
  color: #34495e;
  margin-bottom: 80px;
  text-align: justify;
`;

const Divider = styled.div`
  width: 100px;
  height: 2px;
  background: #ecf0f1;
  margin: 60px auto;
`;

const Tagline = styled.div`
  font-size: 0.8em;
  color: #bdc3c7;
  letter-spacing: 3px;
  text-transform: uppercase;
  text-align: center;
`;

export const {component_name} = () => {{
  return (
    <Container>
      <Title>{title}</Title>
      <Subtitle>{subtitle}</Subtitle>
      <Content>{body}</Content>
      <Divider />
      <Tagline>{tagline}</Tagline>
    </Container>
  );
}};'''

    def _create_card_fallback(self, content: Dict, component_name: str) -> str:
        """카드 폴백 JSX"""

        title = content.get('title', '도쿄 여행 이야기')
        subtitle = content.get('subtitle', '특별한 순간들')
        body = content.get('body', '여행의 아름다운 기억들')
        images = content.get('images', [])
        tagline = content.get('tagline', 'TRAVEL & CULTURE')

        first_image = images[0] if images else ''

        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 1000px;
  margin: 0 auto;
  padding: 60px 20px;
  background: #f5f5f5;
  min-height: 100vh;
`;

const Card = styled.div`
  background: white;
  border-radius: 20px;
  box-shadow: 0 20px 40px rgba(0,0,0,0.1);
  overflow: hidden;
  margin-bottom: 40px;
`;

const CardImage = styled.img`
  width: 100%;
  height: 300px;
  object-fit: cover;
`;

const CardContent = styled.div`
  padding: 40px;
`;

const Title = styled.h1`
  font-size: 2.5em;
  color: #2c3e50;
  margin-bottom: 15px;
  font-weight: 700;
`;

const Subtitle = styled.h2`
  font-size: 1.2em;
  color: #e74c3c;
  margin-bottom: 25px;
  font-weight: 500;
`;

const Content = styled.div`
  font-size: 1.1em;
  line-height: 1.8;
  color: #34495e;
  margin-bottom: 30px;
`;

const TaglineCard = styled.div`
  background: #2c3e50;
  color: white;
  padding: 20px;
  text-align: center;
  border-radius: 15px;
  font-size: 0.9em;
  letter-spacing: 2px;
  text-transform: uppercase;
`;

export const {component_name} = () => {{
  return (
    <Container>
      <Card>
        {first_image and f'<CardImage src="{first_image}" alt="Travel Card" />'}
        <CardContent>
          <Title>{title}</Title>
          <Subtitle>{subtitle}</Subtitle>
          <Content>{body}</Content>
        </CardContent>
      </Card>
      <TaglineCard>{tagline}</TaglineCard>
    </Container>
  );
}};'''
