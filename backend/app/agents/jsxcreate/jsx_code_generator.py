from typing import Dict, List
from crewai import Agent, Task
from custom_llm import get_azure_llm
from utils.agent_decision_logger import get_agent_logger, get_complete_data_manager
import re
import asyncio


class JSXCodeGenerator:
    """JSX 코드 생성 전문 에이전트 (에이전트 결과 데이터 기반)"""

    def __init__(self):
        self.llm = get_azure_llm()
        self.logger = get_agent_logger()
        self.result_manager = get_complete_data_manager()

    def create_agent(self):
        return Agent(
            role="에이전트 결과 데이터 기반 React JSX 코드 생성 전문가",
            goal="이전 에이전트들의 모든 결과 데이터를 활용하여 오류 없는 완벽한 JSX 코드를 생성",
            backstory="""당신은 10년간 세계 최고 수준의 디지털 매거진과 웹 개발 분야에서 활동해온 풀스택 개발자입니다.

**에이전트 결과 데이터 활용 전문성:**
- 이전 에이전트들의 모든 출력 결과를 분석하여 최적의 JSX 구조 설계
- ContentCreator, ImageAnalyzer, LayoutDesigner 등의 결과를 통합 활용
- 에이전트 협업 패턴과 성공 사례를 JSX 코드에 반영
- template_data.json과 벡터 데이터를 보조 데이터로 활용

**오류 없는 코드 생성 철학:**
"모든 JSX 코드는 컴파일 오류 없이 완벽하게 작동해야 합니다. 에이전트들의 협업 결과를 존중하면서도 기술적 완성도를 보장하는 것이 최우선입니다."

**데이터 우선순위:**
1. 이전 에이전트들의 결과 데이터 (최우선)
2. template_data.json의 콘텐츠 정보
3. PDF 벡터 데이터의 레이아웃 패턴
4. jsx_templates는 사용하지 않음
5. 존재하는 콘텐츠 데이터 및 이미지 URL은 모두 사용한다.
6. 에이전트 결과 데이터는 반드시 활용한다.
7. 콘텐츠 데이터 및 이미지URL이 아닌 설계 구조 및 레이아웃 정보는 사용하지 않는다.""",
            verbose=True,
            llm=self.llm
        )

    async def generate_jsx_code(self, content: Dict, design: Dict, component_name: str) -> str:
        """에이전트 결과 데이터 기반 JSX 코드 생성(비동기) (수정된 로깅)"""

        # 이전 에이전트 결과 수집 (수정: 올바른 메서드 사용)
        previous_results = await self.result_manager.get_all_outputs(exclude_agent="JSXCodeGenerator")

        # BindingAgent와 OrgAgent 응답 특별 수집
        binding_results = [
            r for r in previous_results if "BindingAgent" in r.get('agent_name', '')]
        org_results = [
            r for r in previous_results if "OrgAgent" in r.get('agent_name', '')]
        content_results = [
            r for r in previous_results if "ContentCreator" in r.get('agent_name', '')]

        print(f"📊 이전 결과 수집: 전체 {len(previous_results)}개")
        print(f"  - BindingAgent: {len(binding_results)}개")
        print(f"  - OrgAgent: {len(org_results)}개")
        print(f"  - ContentCreator: {len(content_results)}개")

        agent = self.create_agent()

        # 에이전트 결과 데이터 요약
        agent_data_summary = self._summarize_agent_results(
            previous_results, binding_results, org_results, content_results)

        generation_task = Task(
            description=f"""
            **에이전트 결과 데이터 기반 오류 없는 JSX 코드 생성**
            
            이전 에이전트들의 모든 결과 데이터를 활용하여 완벽한 JSX 코드를 생성하세요:

            **이전 에이전트 결과 데이터 ({len(previous_results)}개):**
            {agent_data_summary}

            **BindingAgent 이미지 배치 인사이트 ({len(binding_results)}개):**
            {self._extract_binding_insights(binding_results)}

            **OrgAgent 텍스트 구조 인사이트 ({len(org_results)}개):**
            {self._extract_org_insights(org_results)}

            **ContentCreator 콘텐츠 인사이트 ({len(content_results)}개):**
            {self._extract_content_insights(content_results)}

            **실제 콘텐츠 (template_data.json 기반):**
            - 제목: {content.get('title', '')}
            - 부제목: {content.get('subtitle', '')}
            - 본문: {content.get('body', '')}
            - 이미지 URLs: {content.get('images', [])}
            - 태그라인: {content.get('tagline', '')}

            **레이아웃 설계 (LayoutDesigner 결과):**
            - 타입: {design.get('layout_type', 'grid')}
            - 그리드 구조: {design.get('grid_structure', '1fr 1fr')}
            - 컴포넌트들: {design.get('styled_components', [])}
            - 색상 스키마: {design.get('color_scheme', {})}

            **오류 없는 JSX 생성 지침:**
            1. 반드시 import React from "react"; 포함
            2. 반드시 import styled from "styled-components"; 포함
            3. export const {component_name} = () => {{ ... }}; 형태 준수
            4. 모든 중괄호, 괄호 정확히 매칭
            5. 모든 이미지 URL을 실제 <img src="URL" /> 형태로 포함
            6. className 사용 (class 아님)
            7. JSX 문법 완벽 준수

            **절대 금지사항:**
            - `````` 마크다운 블록
            - 문법 오류 절대 금지
            - 불완전한 return 문 금지
            - jsx_templates 참조 금지

            **에이전트 결과 데이터 활용 방법:**
            - BindingAgent의 이미지 배치 전략을 JSX 이미지 태그에 반영
            - OrgAgent의 텍스트 구조를 JSX 컴포넌트 구조에 반영
            - ContentCreator의 콘텐츠 품질을 JSX 스타일링에 반영
            - 이전 성공적인 JSX 패턴 재사용
            - 협업 에이전트들의 품질 지표 고려

            **출력:** 순수한 JSX 파일 코드만 출력 (설명이나 마크다운 없이)
            """,
            agent=agent,
            expected_output="에이전트 결과 데이터 기반 오류 없는 순수 JSX 코드"
        )

        try:
            # 비동기 태스크 실행 (agent.execute_task가 비동기 지원해야 함)
            result = await agent.execute_task(generation_task)
            jsx_code = str(result)

            # 에이전트 결과 기반 후처리
            jsx_code = self._post_process_with_agent_results(
                jsx_code, previous_results, binding_results, org_results, content_results, content, component_name)

            # 결과 저장 (수정: 올바른 메서드 사용)
            await self.result_manager.store_agent_output(
                agent_name="JSXCodeGenerator",
                agent_role="JSX 코드 생성 전문가",
                task_description=f"컴포넌트 {component_name} JSX 코드 생성",
                final_answer=jsx_code,
                reasoning_process=f"이전 {len(previous_results)}개 에이전트 결과 활용하여 JSX 생성",
                execution_steps=[
                    "에이전트 결과 수집 및 분석",
                    "BindingAgent/OrgAgent/ContentCreator 인사이트 추출",
                    "JSX 코드 생성",
                    "후처리 및 검증"
                ],
                raw_input={"content": content, "design": design,
                           "component_name": component_name},
                raw_output=jsx_code,
                performance_metrics={
                    "agent_results_utilized": len(previous_results),
                    "binding_results_count": len(binding_results),
                    "org_results_count": len(org_results),
                    "content_results_count": len(content_results),
                    "jsx_templates_ignored": True,
                    "error_free_validated": self._validate_jsx_syntax(jsx_code),
                    "code_length": len(jsx_code)
                }
            )

            print(f"✅ 에이전트 데이터 기반 JSX 코드 생성 완료: {component_name}")
            return jsx_code

        except Exception as e:
            print(f"⚠️ JSX 코드 생성 실패: {e}")

            # 에러 로깅(비동기)
            await self.result_manager.store_agent_output(
                agent_name="JSXCodeGenerator_Error",
                agent_role="에러 처리",
                task_description=f"컴포넌트 {component_name} 생성 중 에러 발생",
                final_answer=f"ERROR: {str(e)}",
                reasoning_process="JSX 코드 생성 중 예외 발생",
                error_logs=[{"error": str(e), "component": component_name}]
            )

            return self._create_agent_based_fallback_jsx(content, design, component_name, previous_results)

    def _summarize_agent_results(self, previous_results: List[Dict], binding_results: List[Dict], org_results: List[Dict], content_results: List[Dict]) -> str:
        """에이전트 결과 데이터 요약 (모든 에이전트 포함)"""

        if not previous_results:
            return "이전 에이전트 결과 없음 - 기본 패턴 사용"

        summary_parts = []

        # 에이전트별 결과 분류
        agent_groups = {}
        for result in previous_results:
            agent_name = result.get('agent_name', 'unknown')
            if agent_name not in agent_groups:
                agent_groups[agent_name] = []
            agent_groups[agent_name].append(result)

        # 각 에이전트 그룹 요약
        for agent_name, results in agent_groups.items():
            latest_result = results[-1]  # 최신 결과
            answer_length = len(latest_result.get('final_answer', ''))

            summary_parts.append(
                f"- {agent_name}: {len(results)}개 결과, 최신 답변 길이: {answer_length}자")

        # 특별 요약
        summary_parts.append(f"- BindingAgent 특별 수집: {len(binding_results)}개")
        summary_parts.append(f"- OrgAgent 특별 수집: {len(org_results)}개")
        summary_parts.append(
            f"- ContentCreator 특별 수집: {len(content_results)}개")

        return "\n".join(summary_parts)

    def _extract_binding_insights(self, binding_results: List[Dict]) -> str:
        """BindingAgent 인사이트 추출"""

        if not binding_results:
            return "BindingAgent 결과 없음"

        insights = []
        for result in binding_results:
            answer = result.get('final_answer', '')
            if '그리드' in answer or 'grid' in answer.lower():
                insights.append("- 그리드 기반 이미지 배치 전략")
            if '갤러리' in answer or 'gallery' in answer.lower():
                insights.append("- 갤러리 스타일 이미지 배치")
            if '배치' in answer:
                insights.append("- 전문적 이미지 배치 분석 완료")

        return "\n".join(insights) if insights else "BindingAgent 일반적 이미지 처리"

    def _extract_org_insights(self, org_results: List[Dict]) -> str:
        """OrgAgent 인사이트 추출"""

        if not org_results:
            return "OrgAgent 결과 없음"

        insights = []
        for result in org_results:
            answer = result.get('final_answer', '')
            if '구조' in answer or 'structure' in answer.lower():
                insights.append("- 체계적 텍스트 구조 설계")
            if '레이아웃' in answer or 'layout' in answer.lower():
                insights.append("- 전문적 레이아웃 구조 분석")
            if '매거진' in answer or 'magazine' in answer.lower():
                insights.append("- 매거진 스타일 텍스트 편집")

        return "\n".join(insights) if insights else "OrgAgent 일반적 텍스트 처리"

    def _extract_content_insights(self, content_results: List[Dict]) -> str:
        """ContentCreator 인사이트 추출"""

        if not content_results:
            return "ContentCreator 결과 없음"

        insights = []
        for result in content_results:
            answer = result.get('final_answer', '')
            performance = result.get('performance_metrics', {})

            if len(answer) > 2000:
                insights.append("- 풍부한 콘텐츠 생성 완료")
            if '여행' in answer and '매거진' in answer:
                insights.append("- 여행 매거진 스타일 콘텐츠")
            if performance.get('content_richness', 0) > 1.5:
                insights.append("- 고품질 콘텐츠 확장 성공")

        return "\n".join(insights) if insights else "ContentCreator 일반적 콘텐츠 처리"

    def _post_process_with_agent_results(self, jsx_code: str, previous_results: List[Dict],
                                         binding_results: List[Dict], org_results: List[Dict],
                                         content_results: List[Dict], content: Dict, component_name: str) -> str:
        """에이전트 결과로 JSX 후처리 (모든 에이전트 포함)"""

        # 1. 마크다운 블록 제거
        jsx_code = self._remove_markdown_blocks(jsx_code)

        # 2. 기본 구조 검증
        jsx_code = self._validate_basic_structure(jsx_code, component_name)

        # 3. BindingAgent 결과 기반 이미지 강화
        jsx_code = self._enhance_with_binding_results(
            jsx_code, binding_results, content)

        # 4. OrgAgent 결과 기반 텍스트 구조 강화
        jsx_code = self._enhance_with_org_results(
            jsx_code, org_results, content)

        # 5. ContentCreator 결과 기반 콘텐츠 품질 강화
        jsx_code = self._enhance_with_content_results(
            jsx_code, content_results, content)

        # 6. 이미지 URL 강제 포함
        jsx_code = self._ensure_image_urls(jsx_code, content)

        # 7. 최종 오류 검사 및 수정
        jsx_code = self._final_error_check_and_fix(jsx_code, component_name)

        return jsx_code

    def _enhance_with_content_results(self, jsx_code: str, content_results: List[Dict], content: Dict) -> str:
        """ContentCreator 결과로 콘텐츠 품질 강화"""

        if not content_results:
            return jsx_code

        latest_content = content_results[-1]
        content_answer = latest_content.get('final_answer', '')
        performance = latest_content.get('performance_metrics', {})

        # 콘텐츠 품질에 따른 스타일 강화
        if len(content_answer) > 2000 or performance.get('content_richness', 0) > 1.5:
            # 고품질 콘텐츠일 때 프리미엄 스타일 적용
            jsx_code = jsx_code.replace(
                'background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);',
                'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'
            )
            jsx_code = jsx_code.replace(
                'color: #2c3e50;',
                'color: #ffffff;'
            )

        if '여행' in content_answer and '매거진' in content_answer:
            # 여행 매거진 스타일 강화
            jsx_code = jsx_code.replace(
                'border-radius: 12px;',
                'border-radius: 16px;\n  box-shadow: 0 12px 24px rgba(0,0,0,0.15);'
            )

        return jsx_code

    # 기존 유틸리티 메서드들 유지
    def _remove_markdown_blocks(self, jsx_code: str) -> str:
        """마크다운 블록 완전 제거"""
        jsx_code = re.sub(r'``````', '', jsx_code)
        jsx_code = re.sub(r'``````', '', jsx_code)
        jsx_code = re.sub(r'^(이 코드는|다음은|아래는).*?\n', '',
                          jsx_code, flags=re.MULTILINE)
        return jsx_code.strip()

    def _validate_basic_structure(self, jsx_code: str, component_name: str) -> str:
        """기본 구조 검증"""
        if 'import React' not in jsx_code:
            jsx_code = 'import React from "react";\n' + jsx_code

        if 'import styled' not in jsx_code:
            jsx_code = jsx_code.replace(
                'import React from "react";',
                'import React from "react";\nimport styled from "styled-components";'
            )

        if f'export const {component_name}' not in jsx_code:
            jsx_code = re.sub(r'export const \w+',
                              f'export const {component_name}', jsx_code)

        return jsx_code

    def _enhance_with_binding_results(self, jsx_code: str, binding_results: List[Dict], content: Dict) -> str:
        """BindingAgent 결과로 이미지 강화"""

        if not binding_results:
            return jsx_code

        latest_binding = binding_results[-1]
        binding_answer = latest_binding.get('final_answer', '')

        # 이미지 배치 전략 반영
        if '그리드' in binding_answer or 'grid' in binding_answer.lower():
            # 그리드 스타일 이미지 갤러리로 교체
            jsx_code = jsx_code.replace(
                'display: flex;',
                'display: grid;\n  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));'
            )

        if '갤러리' in binding_answer or 'gallery' in binding_answer.lower():
            # 갤러리 스타일 강화
            jsx_code = jsx_code.replace(
                'gap: 20px;',
                'gap: 15px;\n  padding: 20px;'
            )

        return jsx_code

    def _enhance_with_org_results(self, jsx_code: str, org_results: List[Dict], content: Dict) -> str:
        """OrgAgent 결과로 텍스트 구조 강화"""

        if not org_results:
            return jsx_code

        latest_org = org_results[-1]
        org_answer = latest_org.get('final_answer', '')

        # 텍스트 구조 개선
        if '매거진' in org_answer or 'magazine' in org_answer.lower():
            # 매거진 스타일 타이포그래피 강화
            jsx_code = jsx_code.replace(
                'font-size: 3em;',
                'font-size: 3.5em;\n  font-weight: 300;\n  letter-spacing: -1px;'
            )

        if '구조' in org_answer or 'structure' in org_answer.lower():
            # 구조적 여백 개선
            jsx_code = jsx_code.replace(
                'margin-bottom: 50px;',
                'margin-bottom: 60px;\n  padding-bottom: 30px;\n  border-bottom: 1px solid #f0f0f0;'
            )

        return jsx_code

    def _ensure_image_urls(self, jsx_code: str, content: Dict) -> str:
        """이미지 URL 강제 포함"""
        images = content.get('images', [])
        if not images:
            return jsx_code

        if '<img' not in jsx_code and 'Image' not in jsx_code:
            first_image = images[0] if images else ''
            image_jsx = f'\n      <img src="{first_image}" alt="Travel" style={{{{width: "100%", maxWidth: "600px", height: "300px", objectFit: "cover", borderRadius: "8px", margin: "20px 0"}}}} />'
            jsx_code = jsx_code.replace(
                '<Container>', f'<Container>{image_jsx}')

        return jsx_code

    def _final_error_check_and_fix(self, jsx_code: str, component_name: str) -> str:
        """최종 오류 검사 및 수정"""
        # 중괄호 매칭
        open_braces = jsx_code.count('{')
        close_braces = jsx_code.count('}')
        if open_braces != close_braces:
            if open_braces > close_braces:
                jsx_code += '}' * (open_braces - close_braces)

        # 문법 오류 수정
        jsx_code = re.sub(r'\{\{([^}]+)\}\}', r'{\1}', jsx_code)
        jsx_code = jsx_code.replace('class=', 'className=')
        jsx_code = re.sub(r'\{\s*\}', '', jsx_code)

        # 마지막 세미콜론 확인
        if not jsx_code.rstrip().endswith('};'):
            jsx_code = jsx_code.rstrip() + '\n};'

        return jsx_code

    def _validate_jsx_syntax(self, jsx_code: str) -> bool:
        """JSX 문법 검증"""
        try:
            has_import_react = 'import React' in jsx_code
            has_import_styled = 'import styled' in jsx_code
            has_export = 'export const' in jsx_code
            has_return = 'return (' in jsx_code
            has_closing = jsx_code.rstrip().endswith('};')

            open_braces = jsx_code.count('{')
            close_braces = jsx_code.count('}')
            braces_matched = open_braces == close_braces

            return all([has_import_react, has_import_styled, has_export, has_return, has_closing, braces_matched])
        except Exception:
            return False

    def _create_agent_based_fallback_jsx(self, content: Dict, design: Dict, component_name: str, previous_results: List[Dict]) -> str:
        """에이전트 데이터 기반 폴백 JSX"""

        title = content.get('title', '에이전트 협업 여행 이야기')
        subtitle = content.get('subtitle', '특별한 순간들')
        body = content.get('body', '다양한 AI 에이전트들이 협업하여 생성한 여행 콘텐츠입니다.')
        images = content.get('images', [])
        tagline = content.get('tagline', 'AI AGENTS COLLABORATION')

        # 에이전트 결과 반영
        if previous_results:
            agent_count = len(set(r.get('agent_name')
                              for r in previous_results))
            body = f"{body}\n\n이 콘텐츠는 {agent_count}개의 전문 AI 에이전트가 협업하여 생성했습니다."

        image_tags = []
        for i, img_url in enumerate(images[:4]):
            if img_url and img_url.strip():
                image_tags.append(
                    f'        <TravelImage src="{img_url}" alt="Travel {i+1}" />')

        image_jsx = '\n'.join(
            image_tags) if image_tags else '        <PlaceholderDiv>에이전트 기반 콘텐츠</PlaceholderDiv>'

        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 60px 20px;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  min-height: 100vh;
`;

const Header = styled.header`
  text-align: center;
  margin-bottom: 50px;
`;

const Title = styled.h1`
  font-size: 3em;
  color: #2c3e50;
  margin-bottom: 20px;
  font-weight: 300;
`;

const Subtitle = styled.h2`
  font-size: 1.4em;
  color: #7f8c8d;
  margin-bottom: 30px;
`;

const Content = styled.div`
  font-size: 1.2em;
  line-height: 1.8;
  color: #34495e;
  margin-bottom: 40px;
  max-width: 800px;
  margin-left: auto;
  margin-right: auto;
  white-space: pre-line;
`;

const ImageGallery = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin: 40px 0;
`;

const TravelImage = styled.img`
  width: 100%;
  height: 200px;
  object-fit: cover;
  border-radius: 12px;
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
  transition: transform 0.3s ease;
  
  &:hover {{
    transform: translateY(-5px);
  }}
`;

const PlaceholderDiv = styled.div`
  width: 100%;
  height: 200px;
  background: #e9ecef;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6c757d;
  font-size: 1.1em;
`;

const Footer = styled.footer`
  text-align: center;
  margin-top: 50px;
  padding-top: 30px;
  border-top: 1px solid #ecf0f1;
`;

const Tagline = styled.div`
  font-size: 0.9em;
  color: #95a5a6;
  letter-spacing: 3px;
  text-transform: uppercase;
  font-weight: 600;
`;

export const {component_name} = () => {{
  return (
    <Container>
      <Header>
        <Title>{title}</Title>
        <Subtitle>{subtitle}</Subtitle>
      </Header>
      
      <Content>{body}</Content>
      
      <ImageGallery>
{image_jsx}
      </ImageGallery>
      
      <Footer>
        <Tagline>{tagline}</Tagline>
      </Footer>
    </Container>
  );
}};'''
