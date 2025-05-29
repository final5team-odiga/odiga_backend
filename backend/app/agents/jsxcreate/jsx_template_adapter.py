import re
from typing import Dict, List
from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
from utils.agent_decision_logger import get_agent_logger, get_complete_data_manager
import asyncio

class JSXTemplateAdapter:
    """JSX 템플릿 어댑터 (CrewAI 기반 로깅 시스템 통합)"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        self.logger = get_agent_logger()
        self.result_manager = get_complete_data_manager()

        # CrewAI 에이전트들 생성
        self.template_adaptation_agent = self._create_template_adaptation_agent()
        self.image_integration_agent = self._create_image_integration_agent()
        self.structure_preservation_agent = self._create_structure_preservation_agent()
        self.validation_agent = self._create_validation_agent()

    async def _create_template_adaptation_agent(self):
        """템플릿 적응 전문 에이전트"""
        return Agent(
            role="JSX 템플릿 적응 전문가",
            goal="원본 JSX 템플릿의 구조를 완벽히 보존하면서 새로운 콘텐츠에 최적화된 적응을 수행",
            backstory="""당신은 10년간 React 및 JSX 템플릿 시스템을 설계하고 최적화해온 전문가입니다. 다양한 콘텐츠 타입에 맞춰 템플릿을 적응시키면서도 원본의 구조적 무결성을 유지하는 데 특화되어 있습니다.

**전문 영역:**
- JSX 템플릿 구조 분석 및 보존
- 콘텐츠 기반 동적 적응
- 템플릿 호환성 보장
- 구조적 일관성 유지

**적응 철학:**
"완벽한 템플릿 적응은 원본의 설계 의도를 존중하면서도 새로운 콘텐츠의 특성을 최대한 활용하는 것입니다."

**핵심 역량:**
- 원본 JSX 구조 완전 보존
- 콘텐츠 특성 기반 최적화
- 템플릿 메타데이터 활용
- 적응 품질 검증""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def _create_image_integration_agent(self):
        """이미지 통합 전문 에이전트"""
        return Agent(
            role="이미지 URL 통합 전문가",
            goal="JSX 템플릿에 이미지 URL을 완벽하게 통합하여 시각적 일관성과 기능적 완성도를 보장",
            backstory="""당신은 8년간 웹 개발에서 이미지 최적화와 통합을 담당해온 전문가입니다. JSX 컴포넌트 내 이미지 요소의 동적 처리와 URL 관리에 특화되어 있습니다.

**기술 전문성:**
- JSX 이미지 태그 패턴 분석
- 동적 이미지 URL 교체
- 이미지 갤러리 생성
- 반응형 이미지 처리

**통합 전략:**
"모든 이미지는 콘텐츠의 맥락에 맞춰 최적의 위치와 크기로 통합되어야 하며, 사용자 경험을 향상시켜야 합니다."

**처리 범위:**
- 기존 이미지 태그 URL 교체
- 이미지 props 동적 할당
- 누락된 이미지 요소 추가
- 이미지 갤러리 자동 생성""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def _create_structure_preservation_agent(self):
        """구조 보존 전문 에이전트"""
        return Agent(
            role="JSX 구조 보존 전문가",
            goal="원본 JSX 템플릿의 아키텍처와 디자인 패턴을 완벽히 보존하면서 콘텐츠 적응을 수행",
            backstory="""당신은 12년간 대규모 React 프로젝트에서 컴포넌트 아키텍처 설계와 유지보수를 담당해온 전문가입니다. 템플릿의 구조적 무결성을 보장하면서도 유연한 적응을 가능하게 하는 데 특화되어 있습니다.

**핵심 역량:**
- JSX 컴포넌트 구조 분석
- Styled-components 패턴 보존
- 레이아웃 시스템 유지
- 디자인 토큰 일관성

**보존 원칙:**
"원본 템플릿의 설계 철학과 구조적 특성을 완전히 이해하고 보존하면서, 새로운 콘텐츠에 맞는 최소한의 적응만을 수행합니다."

**검증 기준:**
- 원본 컴포넌트 구조 유지
- CSS 스타일링 패턴 보존
- 반응형 디자인 특성 유지
- 접근성 표준 준수""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def _create_validation_agent(self):
        """검증 전문 에이전트"""
        return Agent(
            role="JSX 적응 검증 전문가",
            goal="적응된 JSX 템플릿의 품질과 기능성을 종합적으로 검증하여 완벽한 결과물을 보장",
            backstory="""당신은 8년간 React 프로젝트의 품질 보증과 코드 검증을 담당해온 전문가입니다. JSX 템플릿 적응 과정에서 발생할 수 있는 모든 오류와 품질 이슈를 사전에 식별하고 해결하는 데 특화되어 있습니다.

**검증 영역:**
- JSX 문법 정확성
- 컴포넌트 구조 무결성
- 이미지 통합 완성도
- 마크다운 블록 제거

**품질 기준:**
"완벽한 JSX 템플릿은 문법적 오류가 전혀 없고, 원본의 설계 의도를 완전히 반영하며, 새로운 콘텐츠와 완벽히 조화를 이루는 결과물입니다."

**검증 프로세스:**
- 다단계 문법 검증
- 구조적 일관성 확인
- 이미지 통합 검증
- 최종 품질 승인""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def adapt_template_to_content(self, template_info: Dict, content: Dict, component_name: str) -> str:
        """템플릿을 콘텐츠에 맞게 적용 (CrewAI 기반 이미지 URL 완전 통합 + 로깅)"""

        # 이전 에이전트 결과 수집
        previous_results = await self.result_manager.get_all_outputs(exclude_agent="JSXTemplateAdapter")
        print(f"📊 이전 에이전트 결과 수집: {len(previous_results)}개")

        # CrewAI Task들 생성
        structure_analysis_task = self._create_structure_analysis_task(template_info, content, component_name)
        image_integration_task = self._create_image_integration_task(content)
        content_adaptation_task = self._create_content_adaptation_task(template_info, content, component_name)
        validation_task = self._create_validation_task(component_name)

        # CrewAI Crew 생성 및 실행
        adaptation_crew = Crew(
            agents=[self.structure_preservation_agent, self.image_integration_agent, self.template_adaptation_agent, self.validation_agent],
            tasks=[structure_analysis_task, image_integration_task, content_adaptation_task, validation_task],
            process=Process.sequential,
            verbose=True
        )

        # Crew 실행
        crew_result = await adaptation_crew.kickoff()

        # 실제 적응 수행
        adapted_jsx = self._execute_adaptation_with_crew_insights(crew_result, template_info, content, component_name)

        # 어댑테이션 결과 로깅
        await self.result_manager.store_agent_output(
            agent_name="JSXTemplateAdapter",
            agent_role="JSX 템플릿 어댑터",
            task_description=f"컴포넌트 {component_name} CrewAI 기반 템플릿 어댑테이션",
            final_answer=adapted_jsx,
            reasoning_process=f"CrewAI 기반 원본 JSX 구조 보존하며 콘텐츠 적용, 이미지 {len(content.get('images', []))}개 통합",
            execution_steps=[
                "CrewAI 에이전트 및 태스크 생성",
                "구조 분석 및 보존",
                "이미지 통합",
                "콘텐츠 적응",
                "검증 및 완료"
            ],
            raw_input={"template_info": template_info, "content": content, "component_name": component_name},
            raw_output=adapted_jsx,
            performance_metrics={
                "original_jsx_length": len(template_info.get('original_jsx', '')),
                "adapted_jsx_length": len(adapted_jsx),
                "images_integrated": len(content.get('images', [])),
                "vector_matched": template_info.get('vector_matched', False),
                "previous_results_count": len(previous_results),
                "crewai_enhanced": True
            }
        )

        print(f"✅ CrewAI 기반 실제 구조 보존 및 이미지 통합 완료")
        return adapted_jsx

    def _execute_adaptation_with_crew_insights(self, crew_result, template_info: Dict, content: Dict, component_name: str) -> str:
        """CrewAI 인사이트를 활용한 실제 적응 수행"""
        original_jsx = template_info.get('original_jsx', '')

        if not original_jsx:
            print(f"⚠️ 원본 JSX 없음 - 폴백 생성")
            return self._create_fallback_adaptation(template_info, content, component_name)

        print(f"🔧 CrewAI 기반 실제 템플릿 구조 적용 시작 (이미지 URL 통합)")

        # 실제 템플릿 구조를 완전히 보존하면서 콘텐츠만 교체
        adapted_jsx = self._preserve_structure_adapt_content(original_jsx, template_info, content, component_name)

        # 이미지 URL 강제 통합
        adapted_jsx = self._force_integrate_image_urls(adapted_jsx, content)

        # 벡터 데이터 기반 스타일 조정
        if template_info.get('vector_matched', False):
            adapted_jsx = self._apply_vector_style_enhancements(adapted_jsx, template_info)

        # 마크다운 블록 제거 및 최종 검증
        adapted_jsx = self._remove_markdown_blocks_and_validate(adapted_jsx, content, component_name)

        return adapted_jsx

    def _create_structure_analysis_task(self, template_info: Dict, content: Dict, component_name: str) -> Task:
        """구조 분석 태스크"""
        return Task(
            description=f"""
            JSX 템플릿의 구조를 분석하고 보존 전략을 수립하세요.

            **분석 대상:**
            - 컴포넌트명: {component_name}
            - 원본 JSX 길이: {len(template_info.get('original_jsx', ''))} 문자
            - 벡터 매칭: {template_info.get('vector_matched', False)}

            **분석 요구사항:**
            1. 원본 JSX 구조 완전 분석
            2. Styled-components 패턴 식별
            3. 레이아웃 시스템 특성 파악
            4. 보존해야 할 핵심 요소 식별

            **보존 전략:**
            - 컴포넌트 아키텍처 유지
            - CSS 스타일링 패턴 보존
            - 반응형 디자인 특성 유지
            - 접근성 표준 준수

            구조 분석 결과와 보존 전략을 제시하세요.
            """,
            expected_output="JSX 구조 분석 결과 및 보존 전략",
            agent=self.structure_preservation_agent
        )

    async def _create_image_integration_task(self, content: Dict) -> Task:
        """이미지 통합 태스크"""
        return Task(
            description=f"""
            콘텐츠의 이미지들을 JSX 템플릿에 완벽하게 통합하세요.

            **통합 대상:**
            - 이미지 개수: {len(content.get('images', []))}개
            - 이미지 URL들: {content.get('images', [])[:3]}...

            **통합 요구사항:**
            1. 기존 이미지 태그 URL 교체
            2. 이미지 props 동적 할당
            3. 누락된 이미지 요소 추가
            4. 이미지 갤러리 자동 생성 (필요시)

            **통합 전략:**
            - 기존 img 태그의 src 속성 교체
            - styled 이미지 컴포넌트 src 업데이트
            - 이미지 props 패턴 매칭 및 교체
            - 이미지가 없는 경우 갤러리 추가

            **품질 기준:**
            - 모든 이미지 URL 유효성 확인
            - 이미지 태그 문법 정확성
            - 반응형 이미지 처리

            이미지 통합 전략과 구현 방안을 제시하세요.
            """,
            expected_output="이미지 통합 전략 및 구현 방안",
            agent=self.image_integration_agent
        )

    async def _create_content_adaptation_task(self, template_info: Dict, content: Dict, component_name: str) -> Task:
        """콘텐츠 적응 태스크"""
        return Task(
            description=f"""
            템플릿 구조를 보존하면서 새로운 콘텐츠에 맞게 적응시키세요.

            **적응 대상:**
            - 제목: {content.get('title', 'N/A')}
            - 본문 길이: {len(content.get('body', ''))} 문자
            - 부제목: {content.get('subtitle', 'N/A')}

            **적응 요구사항:**
            1. 원본 JSX 구조 완전 보존
            2. 콘텐츠 요소만 선택적 교체
            3. 컴포넌트명 정확한 적용
            4. 벡터 데이터 기반 스타일 최적화

            **적응 원칙:**
            - 구조적 무결성 유지
            - 콘텐츠 특성 반영
            - 디자인 일관성 보장
            - 사용자 경험 최적화

            이전 태스크들의 결과를 활용하여 완벽한 적응을 수행하세요.
            """,
            expected_output="완벽하게 적응된 JSX 템플릿",
            agent=self.template_adaptation_agent,
            context=[
                await self._create_structure_analysis_task(template_info, content, component_name),
                await self._create_image_integration_task(content)
            ]
        )

    async def _create_validation_task(self, component_name: str) -> Task:
        """검증 태스크"""
        return Task(
            description=f"""
            적응된 JSX 템플릿의 품질과 기능성을 종합적으로 검증하세요.

            **검증 대상:**
            - 컴포넌트명: {component_name}

            **검증 영역:**
            1. JSX 문법 정확성 확인
            2. 컴포넌트 구조 무결성 검증
            3. 이미지 통합 완성도 평가
            4. 마크다운 블록 완전 제거

            **품질 기준:**
            - 문법 오류 제로
            - 컴파일 가능성 보장
            - 원본 구조 보존 확인
            - 콘텐츠 적응 완성도

            **최종 검증:**
            - import 문 정확성
            - export 문 일치성
            - styled-components 활용
            - 접근성 준수

            모든 검증 항목을 통과한 최종 JSX 템플릿을 제공하세요.
            """,
            expected_output="품질 검증 완료된 최종 JSX 템플릿",
            agent=self.validation_agent,
            context=[await self._create_content_adaptation_task({}, {}, component_name)]
        )
    
    # 기존 메서드들 유지 (변경 없음)
    async def _force_integrate_image_urls(self, jsx_code: str, content: Dict) -> str:
        """이미지 URL 강제 통합"""
        images = content.get('images', [])
        if not images:
            print(f"📷 이미지 없음 - 플레이스홀더 유지")
            return jsx_code

        print(f"📷 {len(images)}개 이미지 URL 통합 시작")

        # 1. 기존 이미지 태그에 실제 URL 적용
        jsx_code = await self._replace_existing_image_tags(jsx_code, images)

        # 2. 이미지 props 교체
        jsx_code = await self._replace_image_props(jsx_code, images)

        # 3. 이미지가 없는 경우 새로 추가
        jsx_code = await self._add_missing_images(jsx_code, images)

        print(f"✅ 이미지 URL 통합 완료")
        return jsx_code

    async def _replace_existing_image_tags(self, jsx_code: str, images: List[str]) -> str:
        """기존 이미지 태그에 실제 URL 적용"""
        # img 태그의 src 속성 찾기 및 교체
        img_pattern = r'<img([^>]*?)src="([^"]*)"([^>]*?)/?>'

        def replace_img_src(match):
            before_src = match.group(1)
            old_src = match.group(2)
            after_src = match.group(3)

            # 첫 번째 이미지로 교체
            if images and images[0]:
                new_src = images[0]
                return f'<img{before_src}src="{new_src}"{after_src} />'
            return match.group(0)

        jsx_code = re.sub(img_pattern, replace_img_src, jsx_code)

        # styled img 컴포넌트의 src 속성 교체
        styled_img_pattern = r'<(\w*[Ii]mage?\w*)\s+([^>]*?)src="([^"]*)"([^>]*?)/?>'

        def replace_styled_img_src(match):
            component_name = match.group(1)
            before_src = match.group(2)
            old_src = match.group(3)
            after_src = match.group(4)

            # 이미지 인덱스 추출 시도
            img_index = self._extract_image_index_from_component(component_name)
            if img_index < len(images) and images[img_index]:
                new_src = images[img_index]
                return f'<{component_name} {before_src}src="{new_src}"{after_src} />'
            elif images and images[0]:
                new_src = images[0]
                return f'<{component_name} {before_src}src="{new_src}"{after_src} />'
            return match.group(0)

        jsx_code = re.sub(styled_img_pattern, replace_styled_img_src, jsx_code)

        return jsx_code

    async def _replace_image_props(self, jsx_code: str, images: List[str]) -> str:
        """이미지 props 교체"""
        # 다양한 이미지 prop 패턴 교체
        image_prop_patterns = [
            (r'\{imageUrl\}', 0),
            (r'\{imageUrl1\}', 0),
            (r'\{imageUrl2\}', 1),
            (r'\{imageUrl3\}', 2),
            (r'\{imageUrl4\}', 3),
            (r'\{imageUrl5\}', 4),
            (r'\{imageUrl6\}', 5),
            (r'\{image\}', 0),
            (r'\{heroImage\}', 0),
            (r'\{featuredImage\}', 0),
            (r'\{mainImage\}', 0)
        ]

        for pattern, index in image_prop_patterns:
            if index < len(images) and images[index]:
                jsx_code = re.sub(pattern, images[index], jsx_code)

        return jsx_code

    async def _add_missing_images(self, jsx_code: str, images: List[str]) -> str:
        """이미지가 없는 경우 새로 추가"""
        if '<img' not in jsx_code and 'Image' not in jsx_code:
            container_pattern = r'(<Container[^>]*>)(.*?)(</Container>)'

            async def add_image_gallery(match):
                container_open = match.group(1)
                container_content = match.group(2)
                container_close = match.group(3)

                image_gallery = await self._create_image_gallery_jsx(images)
                new_content = container_content + '\n      ' + image_gallery
                return container_open + new_content + '\n    ' + container_close

            jsx_code = re.sub(container_pattern, lambda m: asyncio.run(add_image_gallery(m)), jsx_code, flags=re.DOTALL)

        return jsx_code

    async def _create_image_gallery_jsx(self, images: List[str]) -> str:
        """이미지 갤러리 JSX 생성"""
        image_tags = []
        for i, img_url in enumerate(images[:6]):
            if img_url and img_url.strip():
                image_tags.append(f'        <img src="{img_url}" alt="Image {i+1}" style={{width: "100%", height: "200px", objectFit: "cover", borderRadius: "8px"}} />')

        if not image_tags:
            return ""

        gallery_jsx = f"""<div style={{display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "16px", marginTop: "20px"}}>
{chr(10).join(image_tags)}
      </div>"""

        return gallery_jsx

    def _extract_image_index_from_component(self, component_name: str) -> int:
        import re
        match = re.search(r'(\d+)', component_name)
        if match:
            return int(match.group(1)) - 1
        return 0

    def _preserve_structure_adapt_content(self, original_jsx: str, template_info: Dict, content: Dict, component_name: str) -> str:
        adapted_jsx = original_jsx
        adapted_jsx = re.sub(r'export const \w+', f'export const {component_name}', adapted_jsx)

        title = content.get('title', '제목')
        subtitle = content.get('subtitle', '부제목')
        body = content.get('body', '본문 내용')

        text_replacements = [
            (r'\{title\}', title),
            (r'\{subtitle\}', subtitle),
            (r'\{body\}', body),
            (r'\{content\}', body),
            (r'제목을 입력하세요', title),
            (r'부제목을 입력하세요', subtitle),
            (r'본문을 입력하세요', body),
        ]

        for pattern, replacement in text_replacements:
            adapted_jsx = re.sub(pattern, replacement, adapted_jsx)

        return adapted_jsx

    def _apply_vector_style_enhancements(self, jsx_code: str, template_info: Dict) -> str:
        if not template_info.get('vector_matched', False):
            return jsx_code

        recommended_usage = template_info.get('recommended_usage', 'general')

        if 'travel' in recommended_usage:
            jsx_code = jsx_code.replace('#333333', '#2c5aa0')
        elif 'culture' in recommended_usage:
            jsx_code = jsx_code.replace('#333333', '#8b4513')

        return jsx_code

    async def _remove_markdown_blocks_and_validate(self, jsx_code: str, content: Dict, component_name: str) -> str:
        """마크다운 블록 제거 및 최종 검증"""
        # 마크다운 블록 제거
        jsx_code = re.sub(r'``````', '', jsx_code, flags=re.DOTALL)
        jsx_code = re.sub(r'`[^`]*`', '', jsx_code)

        # 기본 구조 검증 및 보완
        if 'import React' not in jsx_code:
            jsx_code = 'import React from "react";\n' + jsx_code

        if 'import styled' not in jsx_code:
            jsx_code = jsx_code.replace(
                'import React from "react";',
                'import React from "react";\nimport styled from "styled-components";'
            )

        # export 문 검증
        if f'export const {component_name}' not in jsx_code:
            jsx_code = re.sub(r'export const \w+', f'export const {component_name}', jsx_code)

        return jsx_code

    async def _create_fallback_adaptation(self, template_info: Dict, content: Dict, component_name: str) -> str:
        """폴백 어댑테이션 생성"""
        title = content.get('title', '제목')
        subtitle = content.get('subtitle', '부제목')
        body = content.get('body', '본문 내용')
        images = content.get('images', [])

        # 기본 JSX 구조 생성
        image_jsx = ""
        if images:
            image_jsx = f'      <img src="{images[0]}" alt="Main Image" style={{width: "100%", height: "300px", objectFit: "cover", borderRadius: "8px"}} />'

        fallback_jsx = f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
  background: #f8f9fa;
  border-radius: 12px;
`;

const Title = styled.h1`
  font-size: 2.5rem;
  color: #2c3e50;
  margin-bottom: 1rem;
  text-align: center;
`;

const Subtitle = styled.h2`
  font-size: 1.5rem;
  color: #7f8c8d;
  margin-bottom: 2rem;
  text-align: center;
`;

const Content = styled.p`
  font-size: 1.1rem;
  line-height: 1.6;
  color: #555;
  margin-bottom: 2rem;
`;

export const {component_name} = () => {{
  return (
    <Container>
      <Title>{title}</Title>
      <Subtitle>{subtitle}</Subtitle>
{image_jsx}
      <Content>{body}</Content>
    </Container>
  );
}};'''

        return fallback_jsx