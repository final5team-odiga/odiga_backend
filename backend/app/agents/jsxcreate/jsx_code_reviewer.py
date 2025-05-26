import os
import re
import ast
import asyncio
from typing import Dict, List, Tuple, Optional
from crewai import Agent, Task, Crew
from custom_llm import get_azure_llm

class JSXCodeReviewer:
    """비동기 JSX 코드 리뷰 및 수정 에이전트"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        
    def create_syntax_analyzer_agent(self):
        """JSX 구문 분석 에이전트"""
        return Agent(
            role="JSX Syntax Analysis & Error Detection Specialist",
            goal="JSX 코드의 구문 오류, 문법 문제, React 규칙 위반을 정확히 감지하고 분석",
            backstory="""당신은 React와 JSX 구문 분석의 최고 전문가입니다.
            10년 이상 React 개발 경험을 가지고 있으며, 수천 개의 JSX 컴포넌트를 
            리뷰하고 수정해온 베테랑 개발자입니다.
            
            당신의 전문 분야:
            - JSX 구문 오류 및 문법 문제 정확한 감지
            - React Hooks 규칙 및 생명주기 검증
            - Styled Components 문법 및 CSS-in-JS 오류 분석
            - Import/Export 문제 및 의존성 오류 감지
            - 컴포넌트 구조 및 Props 타입 검증
            - 접근성(a11y) 및 SEO 최적화 검증
            
            당신은 코드의 모든 라인을 꼼꼼히 분석하여 컴파일 오류, 런타임 오류,
            그리고 잠재적 문제까지 모두 찾아내는 것이 사명입니다.""",
            verbose=True,
            llm=self.llm
        )
    
    def create_code_fixer_agent(self):
        """JSX 코드 수정 에이전트"""
        return Agent(
            role="JSX Code Correction & Optimization Expert",
            goal="감지된 JSX 오류를 정확하고 효율적으로 수정하여 완벽한 React 컴포넌트 생성",
            backstory="""당신은 JSX 코드 수정과 최적화의 마스터입니다.
            복잡한 React 애플리케이션의 버그 수정과 성능 최적화를 전문으로 하며,
            어떤 JSX 오류든 깔끔하고 효율적으로 해결할 수 있습니다.
            
            당신의 수정 철학:
            - 최소한의 변경으로 최대한의 효과
            - React 모범 사례와 최신 패턴 적용
            - 성능과 가독성을 모두 고려한 코드 작성
            - 접근성과 사용자 경험을 향상시키는 수정
            - 미래의 유지보수를 고려한 확장 가능한 구조
            
            당신은 단순히 오류를 수정하는 것이 아니라, 
            코드를 더 나은 방향으로 개선하는 것이 목표입니다.""",
            verbose=True,
            llm=self.llm
        )
    
    def create_quality_validator_agent(self):
        """코드 품질 검증 에이전트"""
        return Agent(
            role="React Code Quality & Best Practices Validator",
            goal="수정된 JSX 코드의 품질, 성능, 보안을 검증하고 React 모범 사례 준수 확인",
            backstory="""당신은 React 코드 품질과 모범 사례의 최종 검증자입니다.
            Facebook(Meta) React 팀의 가이드라인과 최신 React 생태계의 
            모범 사례를 완벽하게 숙지하고 있습니다.
            
            당신의 검증 기준:
            - React 18+ 최신 기능과 패턴 활용도
            - 성능 최적화 (메모이제이션, 지연 로딩 등)
            - 접근성(WCAG 2.1 AA) 준수
            - 보안 취약점 (XSS, CSRF 등) 검사
            - 코드 가독성과 유지보수성
            - 테스트 가능성과 모듈화 수준
            
            당신은 완벽한 React 컴포넌트만을 승인하며,
            조금이라도 개선 여지가 있다면 추가 수정을 요구합니다.""",
            verbose=True,
            llm=self.llm
        )
    
    async def review_and_fix_jsx_async(self, jsx_code: str, content: Dict, component_name: str) -> Dict:
        """비동기 JSX 코드 리뷰 및 수정"""
        
        print(f"🔍 비동기 JSX 코드 리뷰 시작: {component_name}")
        
        # 1단계: 초기 구문 분석 (비동기)
        syntax_issues = await self._analyze_syntax_async(jsx_code)
        
        if not syntax_issues:
            print(f"✅ {component_name}: 구문 오류 없음")
            return {
                'fixed_code': jsx_code,
                'issues_found': [],
                'fixes_applied': [],
                'quality_score': 95
            }
        
        print(f"⚠️ {component_name}: {len(syntax_issues)}개 이슈 발견")
        
        # 2단계: 에이전트 기반 수정 (비동기)
        fixed_result = await self._fix_jsx_with_agents_async(
            jsx_code, content, component_name, syntax_issues
        )
        
        return fixed_result
    
    async def _analyze_syntax_async(self, jsx_code: str) -> List[Dict]:
        """비동기 구문 분석"""
        
        issues = []
        
        # 기본 구문 검사들을 비동기로 실행
        tasks = [
            self._check_react_import(jsx_code),
            self._check_jsx_syntax(jsx_code),
            self._check_styled_components(jsx_code),
            self._check_component_structure(jsx_code),
            self._check_props_usage(jsx_code)
        ]
        
        # 모든 검사를 병렬로 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                issues.extend(result)
            elif isinstance(result, Exception):
                print(f"구문 분석 중 오류: {result}")
        
        return issues
    
    async def _check_react_import(self, jsx_code: str) -> List[Dict]:
        """React import 검사"""
        issues = []
        
        # React import 확인
        if not re.search(r'import\s+React\s+from\s+["\']react["\']', jsx_code):
            issues.append({
                'type': 'missing_import',
                'severity': 'error',
                'message': 'React import가 누락되었습니다',
                'line': 1,
                'fix_suggestion': 'import React from "react";'
            })
        
        # styled-components import 확인
        if 'styled.' in jsx_code and not re.search(r'import\s+styled\s+from\s+["\']styled-components["\']', jsx_code):
            issues.append({
                'type': 'missing_import',
                'severity': 'error',
                'message': 'styled-components import가 누락되었습니다',
                'line': 2,
                'fix_suggestion': 'import styled from "styled-components";'
            })
        
        return issues
    
    async def _check_jsx_syntax(self, jsx_code: str) -> List[Dict]:
        """JSX 구문 검사"""
        issues = []
        
        # JSX 태그 매칭 검사
        open_tags = re.findall(r'<(\w+)[^>]*(?<!/)>', jsx_code)
        close_tags = re.findall(r'</(\w+)>', jsx_code)
        
        for tag in open_tags:
            if tag not in close_tags and tag not in ['img', 'br', 'hr', 'input', 'meta', 'link']:
                issues.append({
                    'type': 'unclosed_tag',
                    'severity': 'error',
                    'message': f'태그 <{tag}>가 닫히지 않았습니다',
                    'tag': tag,
                    'fix_suggestion': f'</{tag}> 태그를 추가하세요'
                })
        
        # JSX 표현식 검사
        jsx_expressions = re.findall(r'\{([^}]*)\}', jsx_code)
        for expr in jsx_expressions:
            if expr.strip() == '':
                issues.append({
                    'type': 'empty_expression',
                    'severity': 'warning',
                    'message': '빈 JSX 표현식이 있습니다',
                    'expression': expr,
                    'fix_suggestion': '빈 표현식을 제거하거나 적절한 값을 넣으세요'
                })
        
        # 잘못된 속성명 검사
        invalid_props = re.findall(r'(\w+)=\{[^}]*\}', jsx_code)
        html_props = ['class', 'for', 'tabindex']
        for prop in invalid_props:
            if prop in html_props:
                react_prop = {'class': 'className', 'for': 'htmlFor', 'tabindex': 'tabIndex'}.get(prop)
                issues.append({
                    'type': 'invalid_prop',
                    'severity': 'error',
                    'message': f'HTML 속성 "{prop}"은 React에서 "{react_prop}"으로 사용해야 합니다',
                    'prop': prop,
                    'fix_suggestion': f'{prop}을 {react_prop}으로 변경하세요'
                })
        
        return issues
    
    async def _check_styled_components(self, jsx_code: str) -> List[Dict]:
        """Styled Components 검사"""
        issues = []
        
        # styled 컴포넌트 정의 검사
        styled_components = re.findall(r'const\s+(\w+)\s*=\s*styled\.(\w+)`([^`]*)`', jsx_code)
        
        for comp_name, element, css in styled_components:
            # CSS 구문 검사
            if not css.strip():
                issues.append({
                    'type': 'empty_styled_component',
                    'severity': 'warning',
                    'message': f'Styled component {comp_name}이 비어있습니다',
                    'component': comp_name,
                    'fix_suggestion': 'CSS 스타일을 추가하거나 컴포넌트를 제거하세요'
                })
            
            # 잘못된 CSS 속성 검사
            css_lines = css.split('\n')
            for line in css_lines:
                line = line.strip()
                if line and ':' in line:
                    prop = line.split(':')[0].strip()
                    if prop.endswith('-'):
                        issues.append({
                            'type': 'invalid_css_property',
                            'severity': 'error',
                            'message': f'잘못된 CSS 속성: {prop}',
                            'component': comp_name,
                            'fix_suggestion': f'CSS 속성 {prop}을 올바르게 수정하세요'
                        })
        
        return issues
    
    async def _check_component_structure(self, jsx_code: str) -> List[Dict]:
        """컴포넌트 구조 검사"""
        issues = []
        
        # export 문 검사
        export_match = re.search(r'export\s+const\s+(\w+)', jsx_code)
        if not export_match:
            issues.append({
                'type': 'missing_export',
                'severity': 'error',
                'message': '컴포넌트 export가 누락되었습니다',
                'fix_suggestion': 'export const ComponentName = () => { ... }; 형태로 컴포넌트를 export하세요'
            })
        
        # return 문 검사
        if 'return (' not in jsx_code and 'return(' not in jsx_code:
            issues.append({
                'type': 'missing_return',
                'severity': 'error',
                'message': 'return 문이 누락되었습니다',
                'fix_suggestion': 'return ( ... ); 형태로 JSX를 반환하세요'
            })
        
        # Fragment 사용 검사
        jsx_elements = re.findall(r'return\s*\(\s*([^)]+)\)', jsx_code, re.DOTALL)
        for jsx in jsx_elements:
            jsx_clean = jsx.strip()
            if jsx_clean.count('<') > 1 and not jsx_clean.startswith('<>') and not jsx_clean.startswith('<React.Fragment'):
                # 여러 최상위 요소가 있는 경우
                top_level_tags = re.findall(r'<(\w+)', jsx_clean)
                if len(set(top_level_tags)) > 1:
                    issues.append({
                        'type': 'multiple_root_elements',
                        'severity': 'error',
                        'message': '여러 최상위 요소가 있습니다. Fragment로 감싸야 합니다',
                        'fix_suggestion': '<> ... </> 또는 <React.Fragment> ... </React.Fragment>로 감싸세요'
                    })
        
        return issues
    
    async def _check_props_usage(self, jsx_code: str) -> List[Dict]:
        """Props 사용 검사"""
        issues = []
        
        # 사용되지 않는 props 검사
        function_match = re.search(r'=\s*\(\s*\{([^}]*)\}\s*\)', jsx_code)
        if function_match:
            props = [p.strip() for p in function_match.group(1).split(',') if p.strip()]
            jsx_body = jsx_code[function_match.end():]
            
            for prop in props:
                if prop and prop not in jsx_body:
                    issues.append({
                        'type': 'unused_prop',
                        'severity': 'warning',
                        'message': f'Props "{prop}"이 사용되지 않습니다',
                        'prop': prop,
                        'fix_suggestion': f'사용되지 않는 props {prop}을 제거하세요'
                    })
        
        return issues
    
    async def _fix_jsx_with_agents_async(self, jsx_code: str, content: Dict, component_name: str, issues: List[Dict]) -> Dict:
        """에이전트 기반 비동기 JSX 수정"""
        
        # 에이전트 생성
        syntax_analyzer = self.create_syntax_analyzer_agent()
        code_fixer = self.create_code_fixer_agent()
        quality_validator = self.create_quality_validator_agent()
        
        # 1단계: 상세 분석
        analysis_task = Task(
            description=f"""
            다음 JSX 코드를 상세히 분석하고 모든 오류와 개선점을 찾아내세요:
            
            **컴포넌트 이름:** {component_name}
            
            **JSX 코드:**
            ```
            {jsx_code}
            ```
            
            **발견된 기본 이슈들:**
            {self._format_issues_for_analysis(issues)}
            
            **실제 콘텐츠 데이터:**
            - 제목: {content.get('title', '')}
            - 부제목: {content.get('subtitle', '')}
            - 본문: {content.get('body', '')[:200]}...
            - 이미지 수: {len(content.get('images', []))}개
            - 태그라인: {content.get('tagline', '')}
            
            **분석 요구사항:**
            
            1. **구문 오류 분석**
               - Import/Export 문제
               - JSX 태그 매칭 오류
               - JavaScript 문법 오류
               - Styled Components 구문 문제
            
            2. **React 규칙 위반 검사**
               - Hooks 사용 규칙
               - 컴포넌트 구조 문제
               - Props 타입 및 사용법
               - 생명주기 관련 이슈
            
            3. **성능 및 최적화 이슈**
               - 불필요한 리렌더링 요소
               - 메모리 누수 가능성
               - 비효율적인 CSS 구조
               - 접근성 문제
            
            4. **코드 품질 문제**
               - 가독성 저하 요소
               - 유지보수성 문제
               - 확장성 제약
               - 보안 취약점
            
            **출력 형식:**
            오류 분류: [critical/major/minor]
            구체적 문제점: [상세한 문제 설명]
            영향도: [컴파일/런타임/성능/보안]
            수정 우선순위: [1-10]
            권장 해결방안: [구체적 수정 방법]
            """,
            agent=syntax_analyzer,
            expected_output="JSX 코드의 모든 오류와 개선점에 대한 상세 분석"
        )
        
        # 2단계: 코드 수정
        fixing_task = Task(
            description=f"""
            분석된 JSX 오류들을 모두 수정하여 완벽하게 작동하는 React 컴포넌트를 생성하세요:
            
            **수정 지침:**
            
            1. **모든 구문 오류 완전 수정**
               - Import 문 추가/수정
               - JSX 태그 매칭 수정
               - JavaScript 문법 오류 해결
               - Styled Components 구문 수정
            
            2. **React 모범 사례 적용**
               - 최신 React 패턴 사용
               - 효율적인 컴포넌트 구조
               - 적절한 Props 활용
               - 성능 최적화 적용
            
            3. **실제 콘텐츠 완벽 적용**
               - 제목: {content.get('title', '')}
               - 부제목: {content.get('subtitle', '')}
               - 본문: {content.get('body', '')}
               - 이미지들: {content.get('images', [])}
               - 태그라인: {content.get('tagline', '')}
            
            4. **코드 품질 향상**
               - 가독성 개선
               - 접근성 향상
               - 보안 강화
               - 확장성 고려
            
            **중요 요구사항:**
            - 컴포넌트 이름: {component_name}
            - 모든 실제 콘텐츠 데이터 포함
            - 완벽한 JSX 문법 준수
            - React 18+ 호환성
            - 반응형 디자인 지원
            
            **출력:** 완전히 수정된 JSX 코드만 출력 (설명 없이)
            """,
            agent=code_fixer,
            expected_output="모든 오류가 수정된 완벽한 JSX 컴포넌트",
            context=[analysis_task]
        )
        
        # 3단계: 품질 검증
        validation_task = Task(
            description=f"""
            수정된 JSX 코드의 품질을 검증하고 최종 승인 여부를 결정하세요:
            
            **검증 기준:**
            
            1. **기능적 완성도 (30점)**
               - 컴파일 오류 없음
               - 런타임 오류 없음
               - 모든 기능 정상 작동
            
            2. **React 모범 사례 (25점)**
               - 최신 React 패턴 사용
               - 성능 최적화 적용
               - Hooks 올바른 사용
            
            3. **코드 품질 (25점)**
               - 가독성과 유지보수성
               - 확장성과 재사용성
               - 일관된 코딩 스타일
            
            4. **사용자 경험 (20점)**
               - 접근성 준수
               - 반응형 디자인
               - 성능 최적화
            
            **출력 형식:**
            기능적 완성도: [점수/30]
            React 모범 사례: [점수/25]
            코드 품질: [점수/25]
            사용자 경험: [점수/20]
            총점: [점수/100]
            승인 여부: [승인/재수정 필요]
            개선 권장사항: [구체적 개선 방안]
            """,
            agent=quality_validator,
            expected_output="수정된 JSX 코드의 품질 평가 및 승인 여부",
            context=[analysis_task, fixing_task]
        )
        
        # Crew 실행
        crew = Crew(
            agents=[syntax_analyzer, code_fixer, quality_validator],
            tasks=[analysis_task, fixing_task, validation_task],
            verbose=True
        )
        
        try:
            result = await asyncio.to_thread(crew.kickoff)
            
            # 결과 추출
            fixed_code = str(fixing_task.output) if hasattr(fixing_task, 'output') else jsx_code
            validation_result = str(validation_task.output) if hasattr(validation_task, 'output') else ""
            
            # 최종 정제
            fixed_code = self._final_code_cleanup(fixed_code, content, component_name)
            
            # 품질 점수 추출
            quality_score = self._extract_quality_score(validation_result)
            
            return {
                'fixed_code': fixed_code,
                'issues_found': issues,
                'fixes_applied': self._extract_fixes_applied(str(analysis_task.output) if hasattr(analysis_task, 'output') else ""),
                'quality_score': quality_score,
                'validation_result': validation_result
            }
            
        except Exception as e:
            print(f"⚠️ 에이전트 기반 수정 실패: {e}")
            # 폴백: 기본 수정
            return await self._fallback_fix_async(jsx_code, content, component_name, issues)
    
    def _format_issues_for_analysis(self, issues: List[Dict]) -> str:
        """이슈를 분석용 텍스트로 포맷팅"""
        if not issues:
            return "기본 이슈 없음"
        
        formatted = []
        for i, issue in enumerate(issues):
            formatted.append(f"""
            이슈 {i+1}:
            - 타입: {issue.get('type', 'unknown')}
            - 심각도: {issue.get('severity', 'unknown')}
            - 메시지: {issue.get('message', '')}
            - 수정 제안: {issue.get('fix_suggestion', '')}
            """)
        
        return "\n".join(formatted)
    
    def _extract_quality_score(self, validation_result: str) -> int:
        """검증 결과에서 품질 점수 추출"""
        score_match = re.search(r'총점:\s*(\d+)', validation_result)
        if score_match:
            return int(score_match.group(1))
        
        # 폴백: 기본 점수 계산
        if "승인" in validation_result:
            return 85
        elif "재수정" in validation_result:
            return 65
        else:
            return 75
    
    def _extract_fixes_applied(self, analysis_result: str) -> List[str]:
        """분석 결과에서 적용된 수정사항 추출"""
        fixes = []
        
        # 일반적인 수정사항 패턴 추출
        fix_patterns = [
            r'수정.*?:\s*([^\n]+)',
            r'해결.*?:\s*([^\n]+)',
            r'개선.*?:\s*([^\n]+)'
        ]
        
        for pattern in fix_patterns:
            matches = re.findall(pattern, analysis_result, re.IGNORECASE)
            fixes.extend(matches)
        
        return fixes[:10]  # 최대 10개
    
    def _final_code_cleanup(self, jsx_code: str, content: Dict, component_name: str) -> str:
        """최종 코드 정리"""
        
        # 기본 구조 확인
        if not jsx_code.startswith('import React'):
            jsx_code = 'import React from "react";\nimport styled from "styled-components";\n\n' + jsx_code
        
        # 컴포넌트 이름 확인
        if f"export const {component_name}" not in jsx_code:
            jsx_code = re.sub(r'export const \w+', f'export const {component_name}', jsx_code)
        
        # 실제 콘텐츠 확인
        jsx_code = self._ensure_content_integration(jsx_code, content)
        
        # 문법 오류 최종 수정
        jsx_code = self._fix_common_syntax_errors(jsx_code)
        
        return jsx_code
    
    def _ensure_content_integration(self, jsx_code: str, content: Dict) -> str:
        """실제 콘텐츠 통합 확인"""
        
        title = content.get('title', '여행 이야기')
        subtitle = content.get('subtitle', '특별한 순간들')
        body = content.get('body', '여행의 아름다운 기억들')
        images = content.get('images', [])
        tagline = content.get('tagline', 'TRAVEL & CULTURE')
        
        # 콘텐츠 플레이스홀더 교체
        content_replacements = [
            (r'\{title\}', title),
            (r'\{subtitle\}', subtitle),
            (r'\{body\}', body),
            (r'\{tagline\}', tagline),
            (r'제목.*?입력', title),
            (r'부제목.*?입력', subtitle),
            (r'본문.*?입력', body)
        ]
        
        for pattern, replacement in content_replacements:
            jsx_code = re.sub(pattern, replacement, jsx_code, flags=re.IGNORECASE)
        
        # 이미지 URL 확인 및 수정
        if images:
            # 이미지 태그에 실제 URL 적용
            for i, img_url in enumerate(images[:6]):
                if img_url and img_url.strip():
                    # 플레이스홀더 이미지 URL 교체
                    placeholder_patterns = [
                        r'src="[^"]*placeholder[^"]*"',
                        r'src="[^"]*example[^"]*"',
                        r'src=""'
                    ]
                    
                    for pattern in placeholder_patterns:
                        if re.search(pattern, jsx_code):
                            jsx_code = re.sub(pattern, f'src="{img_url}"', jsx_code, count=1)
                            break
        
        return jsx_code
    
    def _fix_common_syntax_errors(self, jsx_code: str) -> str:
        """일반적인 구문 오류 수정"""
        
        # 1. 이중 중괄호 수정
        jsx_code = re.sub(r'\{\{([^}]+)\}\}', r'{\1}', jsx_code)
        
        # 2. 스타일 객체 수정
        jsx_code = re.sub(r'style=\{([^}]+)\}', r'style={{\1}}', jsx_code)
        
        # 3. className 수정
        jsx_code = re.sub(r'\bclass=', 'className=', jsx_code)
        
        # 4. 닫히지 않은 태그 수정
        if jsx_code.count('<') != jsx_code.count('>'):
            jsx_code += '\n    </Container>\n  );\n};'
        
        # 5. 빈 JSX 표현식 제거
        jsx_code = re.sub(r'\{\s*\}', '', jsx_code)
        
        # 6. 잘못된 주석 수정
        jsx_code = re.sub(r'//[^\n]*\n', '', jsx_code)  # JSX 내 // 주석 제거
        jsx_code = re.sub(r'/\*[^*]*\*/', '', jsx_code)  # /* */ 주석 제거
        
        return jsx_code
    
    async def _fallback_fix_async(self, jsx_code: str, content: Dict, component_name: str, issues: List[Dict]) -> Dict:
        """폴백 비동기 수정"""
        
        print(f"🔧 {component_name}: 폴백 수정 모드")
        
        # 기본 수정 적용
        fixed_code = jsx_code
        
        # 기본 import 추가
        if not fixed_code.startswith('import React'):
            fixed_code = 'import React from "react";\nimport styled from "styled-components";\n\n' + fixed_code
        
        # 컴포넌트 이름 수정
        if f"export const {component_name}" not in fixed_code:
            fixed_code = re.sub(r'export const \w+', f'export const {component_name}', fixed_code)
        
        # 기본 구문 오류 수정
        fixed_code = self._fix_common_syntax_errors(fixed_code)
        
        # 실제 콘텐츠 적용
        fixed_code = self._ensure_content_integration(fixed_code, content)
        
        return {
            'fixed_code': fixed_code,
            'issues_found': issues,
            'fixes_applied': ['기본 import 추가', '컴포넌트 이름 수정', '구문 오류 수정', '콘텐츠 통합'],
            'quality_score': 70,
            'validation_result': '폴백 수정 완료'
        }
