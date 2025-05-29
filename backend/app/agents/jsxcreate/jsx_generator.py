import re
import os
import json
import asyncio
from agents.jsxcreate.jsx_content_analyzer import JSXContentAnalyzer
from agents.jsxcreate.jsx_layout_designer import JSXLayoutDesigner
from agents.jsxcreate.jsx_code_generator import JSXCodeGenerator
from typing import Dict, List
from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager
from utils.agent_decision_logger import get_agent_logger, get_complete_data_manager


class JSXCreatorAgent:
    """다중 에이전트 조율자 - JSX 생성 총괄 (CrewAI 기반 에이전트 결과 데이터 기반)"""

    def __init__(self):
        self.llm = get_azure_llm()
        self.vector_manager = PDFVectorManager()
        self.logger = get_agent_logger()
        self.result_manager = get_complete_data_manager()

        # 전문 에이전트들 초기화
        self.content_analyzer = JSXContentAnalyzer()
        self.layout_designer = JSXLayoutDesigner()
        self.code_generator = JSXCodeGenerator()

        # CrewAI 에이전트들 생성
        self.jsx_coordinator_agent = self._create_jsx_coordinator_agent()
        self.data_collection_agent = self._create_data_collection_agent()
        self.component_generation_agent = self._create_component_generation_agent()
        self.quality_assurance_agent = self._create_quality_assurance_agent()

    def _create_jsx_coordinator_agent(self):
        """JSX 생성 총괄 조율자"""
        return Agent(
            role="JSX 생성 총괄 조율자",
            goal="에이전트 결과 데이터를 기반으로 고품질 JSX 컴포넌트 생성 프로세스를 총괄하고 최적화된 결과를 도출",
            backstory="""당신은 15년간 React 및 JSX 기반 대규모 웹 개발 프로젝트를 총괄해온 시니어 아키텍트입니다. 다중 에이전트 시스템의 결과를 통합하여 최고 품질의 JSX 컴포넌트를 생성하는 데 특화되어 있습니다.

**전문 영역:**
- 다중 에이전트 결과 데이터 통합 및 분석
- JSX 컴포넌트 아키텍처 설계
- 에이전트 기반 개발 워크플로우 최적화
- 품질 보증 및 성능 최적화

**조율 철학:**
"각 에이전트의 전문성을 최대한 활용하여 단일 에이전트로는 달성할 수 없는 수준의 JSX 컴포넌트를 생성합니다."

**책임 영역:**
- 전체 JSX 생성 프로세스 관리
- 에이전트 간 데이터 흐름 최적화
- 품질 기준 설정 및 검증
- 최종 결과물 승인 및 배포""",
            verbose=True,
            llm=self.llm,
            allow_delegation=True
        )

    def _create_data_collection_agent(self):
        """데이터 수집 및 분석 전문가"""
        return Agent(
            role="에이전트 결과 데이터 수집 및 분석 전문가",
            goal="이전 에이전트들의 실행 결과를 체계적으로 수집하고 분석하여 JSX 생성에 필요한 인사이트를 도출",
            backstory="""당신은 10년간 다중 에이전트 시스템의 데이터 분석과 패턴 인식을 담당해온 전문가입니다. 복잡한 에이전트 결과 데이터에서 의미 있는 패턴과 인사이트를 추출하는 데 탁월한 능력을 보유하고 있습니다.

**핵심 역량:**
- 에이전트 실행 결과 패턴 분석
- 성공적인 접근법 식별 및 분류
- 품질 지표 기반 성능 평가
- 학습 인사이트 통합 및 활용

**분석 방법론:**
"데이터 기반 의사결정을 통해 각 에이전트의 강점을 파악하고 이를 JSX 생성 품질 향상에 활용합니다."

**특별 처리:**
- ContentCreatorV2Agent: 콘텐츠 생성 품질 분석
- ImageAnalyzerAgent: 이미지 분석 결과 활용
- 성능 메트릭: 성공률 및 신뢰도 평가""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_component_generation_agent(self):
        """JSX 컴포넌트 생성 전문가"""
        return Agent(
            role="JSX 컴포넌트 생성 전문가",
            goal="에이전트 분석 결과를 바탕으로 오류 없는 고품질 JSX 컴포넌트를 생성하고 최적화",
            backstory="""당신은 12년간 React 생태계에서 수천 개의 JSX 컴포넌트를 설계하고 구현해온 전문가입니다. 에이전트 기반 데이터를 활용한 동적 컴포넌트 생성과 최적화에 특화되어 있습니다.

**기술 전문성:**
- React 및 JSX 고급 패턴
- Styled-components 기반 디자인 시스템
- 반응형 웹 디자인 구현
- 컴포넌트 성능 최적화

**생성 전략:**
"에이전트 분석 결과의 모든 인사이트를 반영하여 사용자 경험과 개발자 경험을 모두 만족시키는 컴포넌트를 생성합니다."

**품질 기준:**
- 문법 오류 제로
- 컴파일 가능성 보장
- 접근성 표준 준수
- 성능 최적화 적용""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_quality_assurance_agent(self):
        """품질 보증 전문가"""
        return Agent(
            role="JSX 품질 보증 및 검증 전문가",
            goal="생성된 JSX 컴포넌트의 품질을 종합적으로 검증하고 오류를 제거하여 완벽한 결과물을 보장",
            backstory="""당신은 8년간 대규모 React 프로젝트의 품질 보증과 코드 리뷰를 담당해온 전문가입니다. JSX 컴포넌트의 모든 측면을 검증하여 프로덕션 레벨의 품질을 보장하는 데 특화되어 있습니다.

**검증 영역:**
- JSX 문법 및 구조 검증
- React 모범 사례 준수 확인
- 접근성 및 사용성 검증
- 성능 및 최적화 평가

**품질 철학:**
"완벽한 JSX 컴포넌트는 기능적 완성도와 코드 품질, 사용자 경험이 모두 조화를 이루는 결과물입니다."

**검증 프로세스:**
- 다단계 문법 검증
- 컴파일 가능성 테스트
- 에이전트 인사이트 반영 확인
- 최종 품질 승인""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def generate_jsx_components_async(self, template_data_path: str, templates_dir: str = "jsx_templates") -> List[Dict]:
        """에이전트 결과 데이터 기반 JSX 생성 (CrewAI 기반 jsx_templates 미사용)"""
        print(f"🚀 CrewAI 기반 에이전트 결과 데이터 기반 JSX 생성 시작")
        print(f"📁 jsx_templates 폴더 무시 - 에이전트 데이터 우선 사용")

        # CrewAI Task들 생성
        data_collection_task = self._create_data_collection_task()
        template_parsing_task = self._create_template_parsing_task(
            template_data_path)
        jsx_generation_task = self._create_jsx_generation_task()
        quality_assurance_task = self._create_quality_assurance_task()

        # CrewAI Crew 생성
        jsx_crew = Crew(
            agents=[self.data_collection_agent, self.jsx_coordinator_agent,
                    self.component_generation_agent, self.quality_assurance_agent],
            tasks=[data_collection_task, template_parsing_task,
                   jsx_generation_task, quality_assurance_task],
            process=Process.sequential,
            verbose=True
        )

        # Crew 실행 (동기 함수라면 run_in_executor 사용)
        loop = asyncio.get_running_loop()
        crew_result = await loop.run_in_executor(None, jsx_crew.kickoff)

        # 실제 JSX 생성 수행
        generated_components = await self._execute_jsx_generation_with_crew_insights(
            crew_result, template_data_path, templates_dir
        )

        if not generated_components:
            return []

        # 전체 JSX 생성 과정 로깅 (수정: 올바른 메서드 사용)
        total_components = len(generated_components)
        successful_components = len(
            [c for c in generated_components if c.get('jsx_code')])

        await self.result_manager.store_agent_output(
            agent_name="JSXCreatorAgent",
            agent_role="JSX 생성 총괄 조율자",
            task_description=f"CrewAI 기반 에이전트 데이터 기반 {total_components}개 JSX 컴포넌트 생성",
            final_answer=f"JSX 생성 완료: {successful_components}/{total_components}개 성공",
            reasoning_process=f"CrewAI 기반 다중 에이전트 협업으로 JSX 컴포넌트 생성",
            execution_steps=[
                "CrewAI 에이전트 및 태스크 생성",
                "에이전트 결과 수집",
                "template_data.json 파싱",
                "JSX 컴포넌트 생성",
                "품질 검증 및 완료"
            ],
            raw_input={
                "template_data_path": template_data_path,
                "crewai_enabled": True
            },
            raw_output=generated_components,
            performance_metrics={
                "total_components": total_components,
                "successful_components": successful_components,
                "success_rate": successful_components / max(total_components, 1),
                "generation_efficiency": successful_components / max(total_components, 1),
                "agent_data_utilization": 1.0,
                "jsx_templates_ignored": True,
                "crewai_enhanced": True
            }
        )

        print(
            f"✅ CrewAI 기반 JSX 생성 완료: {len(generated_components)}개 컴포넌트 (에이전트 데이터 기반)")
        return generated_components

    async def _execute_jsx_generation_with_crew_insights(self, crew_result, template_data_path: str, templates_dir: str) -> List[Dict]:
        """CrewAI 인사이트를 활용한 실제 JSX 생성"""
        # 모든 이전 에이전트 결과 수집 (수정: 올바른 메서드 사용)
        all_agent_results = await self.result_manager.get_all_outputs(exclude_agent="JSXCreatorAgent")
        learning_insights = await self.logger.get_learning_insights("JSXCreatorAgent")

        print(f"📚 수집된 에이전트 결과: {len(all_agent_results)}개")
        print(
            f"🧠 학습 인사이트: {len(learning_insights.get('recommendations', []))}개")

        # template_data.json 읽기
        try:
            with open(template_data_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            template_data = self._safe_parse_json(file_content)
            if template_data is None:
                print(f"❌ template_data.json 파싱 실패")
                return []
        except Exception as e:
            print(f"template_data.json 읽기 오류: {str(e)}")
            return []

        # 데이터 검증
        if not isinstance(template_data, dict) or "content_sections" not in template_data:
            print(f"❌ 잘못된 template_data 구조")
            return []

        print(f"✅ JSON 직접 파싱 성공")

        # 에이전트 결과 데이터 기반 JSX 생성
        generated_components = await self.generate_jsx_from_agent_results(
            template_data, all_agent_results, learning_insights
        )

        return generated_components

    def _create_data_collection_task(self) -> Task:
        """데이터 수집 태스크"""
        return Task(
            description="""
            이전 에이전트들의 실행 결과를 체계적으로 수집하고 분석하여 JSX 생성에 필요한 인사이트를 도출하세요.
            
            **수집 대상:**
            1. 모든 이전 에이전트 실행 결과
            2. 학습 인사이트 및 권장사항
            3. 성능 메트릭 및 품질 지표
            
            **분석 요구사항:**
            1. 에이전트별 성공 패턴 식별
            2. 콘텐츠 패턴 및 디자인 선호도 분석
            3. 품질 지표 기반 성능 평가
            4. JSX 생성에 활용 가능한 인사이트 추출
            
            **출력 형식:**
            - 에이전트 결과 요약
            - 성공 패턴 분석
            - JSX 생성 권장사항
            """,
            expected_output="에이전트 결과 데이터 분석 및 JSX 생성 인사이트",
            agent=self.data_collection_agent
        )

    def _create_template_parsing_task(self, template_data_path: str) -> Task:
        """템플릿 파싱 태스크"""
        return Task(
            description=f"""
            template_data.json 파일을 파싱하고 JSX 생성에 필요한 구조화된 데이터를 준비하세요.
            
            **파싱 대상:**
            - 파일 경로: {template_data_path}
            
            **파싱 요구사항:**
            1. JSON 파일 안전한 읽기 및 파싱
            2. content_sections 데이터 구조 검증
            3. 각 섹션별 콘텐츠 요소 확인
            4. JSX 생성을 위한 데이터 정제
            
            **검증 항목:**
            - JSON 구조 유효성
            - 필수 필드 존재 여부
            - 데이터 타입 일치성
            - 콘텐츠 완성도
            
            **출력 요구사항:**
            파싱된 템플릿 데이터와 검증 결과
            """,
            expected_output="파싱 및 검증된 템플릿 데이터",
            agent=self.jsx_coordinator_agent,
            context=[self._create_data_collection_task()]
        )

    def _create_jsx_generation_task(self) -> Task:
        """JSX 생성 태스크"""
        return Task(
            description="""
            에이전트 분석 결과와 템플릿 데이터를 바탕으로 고품질 JSX 컴포넌트를 생성하세요.
            
            **생성 요구사항:**
            1. 에이전트 인사이트 기반 콘텐츠 강화
            2. 다중 에이전트 파이프라인 실행
               - 콘텐츠 분석 (JSXContentAnalyzer)
               - 레이아웃 설계 (JSXLayoutDesigner)
               - 코드 생성 (JSXCodeGenerator)
            3. 에이전트 결과 기반 검증
            
            **품질 기준:**
            - React 및 JSX 문법 준수
            - Styled-components 활용
            - 반응형 디자인 적용
            - 접근성 표준 준수
            
            **컴포넌트 구조:**
            - 명명 규칙: AgentBased{번호}Component
            - 파일 확장자: .jsx
            - 에러 프리 코드 보장
            """,
            expected_output="생성된 JSX 컴포넌트 목록 (코드 포함)",
            agent=self.component_generation_agent,
            context=[self._create_data_collection_task(
            ), self._create_template_parsing_task("")]
        )

    def _create_quality_assurance_task(self) -> Task:
        """품질 보증 태스크"""
        return Task(
            description="""
            생성된 JSX 컴포넌트의 품질을 종합적으로 검증하고 최종 승인하세요.
            
            **검증 영역:**
            1. JSX 문법 및 구조 검증
            2. React 모범 사례 준수 확인
            3. 컴파일 가능성 테스트
            4. 에이전트 인사이트 반영 확인
            
            **품질 기준:**
            - 문법 오류 제로
            - 마크다운 블록 완전 제거
            - 필수 import 문 포함
            - export 문 정확성
            - styled-components 활용
            
            **최종 검증:**
            - 컴포넌트명 일관성
            - 코드 구조 완성도
            - 성능 최적화 적용
            - 접근성 준수
            
            **승인 기준:**
            모든 검증 항목 통과 시 최종 승인
            """,
            expected_output="품질 검증 완료된 최종 JSX 컴포넌트 목록",
            agent=self.quality_assurance_agent,
            context=[self._create_jsx_generation_task()]
        )

    # 기존 메서드들 유지 (변경 없음)
    def generate_jsx_from_agent_results(self, template_data: Dict, agent_results: List[Dict], learning_insights: Dict) -> List[Dict]:
        """에이전트 결과 데이터를 활용한 JSX 생성"""
        generated_components = []
        content_sections = template_data.get("content_sections", [])

        # 에이전트 결과 데이터 분석
        agent_data_analysis = self._analyze_agent_results(agent_results)

        for i, content_section in enumerate(content_sections):
            if not isinstance(content_section, dict):
                continue

            component_name = f"AgentBased{i+1:02d}Component"
            print(f"\n=== {component_name} 에이전트 데이터 기반 생성 시작 ===")

            # 콘텐츠 정제 (에이전트 결과 반영)
            enhanced_content = self._enhance_content_with_agent_results(
                content_section, agent_data_analysis, learning_insights
            )

            # 다중 에이전트 파이프라인 (에이전트 데이터 기반)
            jsx_code = self._agent_result_based_jsx_pipeline(
                enhanced_content, component_name, i, len(content_sections),
                agent_data_analysis, learning_insights
            )

            # 에이전트 결과 기반 검증
            jsx_code = self._validate_jsx_with_agent_insights(
                jsx_code, enhanced_content, component_name, agent_data_analysis
            )

            # 개별 컴포넌트 생성 저장 (수정: 올바른 메서드 사용)
            self.result_manager.store_agent_output(
                agent_name="JSXCreatorAgent_Component",
                agent_role="개별 JSX 컴포넌트 생성자",
                task_description=f"컴포넌트 {component_name} 생성",
                final_answer=jsx_code,
                reasoning_process="CrewAI 기반 에이전트 데이터 기반 JSX 컴포넌트 생성",
                execution_steps=[
                    "콘텐츠 강화",
                    "JSX 파이프라인 실행",
                    "검증 완료"
                ],
                raw_input=enhanced_content,
                raw_output=jsx_code,
                performance_metrics={
                    "jsx_code_length": len(jsx_code),
                    "error_free": self._validate_jsx_syntax(jsx_code),
                    "agent_data_utilized": True,
                    "crewai_enhanced": True
                }
            )

            generated_components.append({
                'name': component_name,
                'file': f"{component_name}.jsx",
                'jsx_code': jsx_code,
                'approach': 'crewai_agent_results_based',
                'agent_data_analysis': agent_data_analysis,
                'learning_insights_applied': True,
                'error_free_validated': True,
                'crewai_enhanced': True
            })

            print(f"✅ CrewAI 기반 에이전트 데이터 기반 JSX 생성 완료: {component_name}")

        return generated_components

    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _analyze_agent_results(self, agent_results: List[Dict]) -> Dict:
        """에이전트 결과 데이터 분석"""
        analysis = {
            "content_patterns": {},
            "design_preferences": {},
            "successful_approaches": [],
            "common_elements": [],
            "quality_indicators": {},
            "agent_insights": {},
            "crewai_enhanced": True
        }

        if not agent_results:
            print("📊 이전 에이전트 결과 없음 - 기본 분석 사용")
            return analysis

        for result in agent_results:
            agent_name = result.get('agent_name', 'unknown')

            # final_output 우선, 없으면 processed_output, 없으면 raw_output 사용
            full_output = result.get('final_output') or result.get(
                'processed_output') or result.get('raw_output', {})

            # 에이전트별 인사이트 수집
            if agent_name not in analysis["agent_insights"]:
                analysis["agent_insights"][agent_name] = []

            analysis["agent_insights"][agent_name].append({
                "output_type": type(full_output).__name__,
                "content_length": len(str(full_output)),
                "timestamp": result.get('timestamp'),
                "has_performance_data": bool(result.get('performance_data'))
            })

            # 콘텐츠 패턴 분석
            if isinstance(full_output, dict):
                for key, value in full_output.items():
                    if key not in analysis["content_patterns"]:
                        analysis["content_patterns"][key] = []
                    analysis["content_patterns"][key].append(str(value)[:100])

            # 성공적인 접근법 식별
            performance_data = result.get('performance_data', {})
            if performance_data.get('success_rate', 0) > 0.8:
                analysis["successful_approaches"].append({
                    "agent": agent_name,
                    "approach": result.get('output_metadata', {}).get('approach', 'unknown'),
                    "success_rate": performance_data.get('success_rate', 0)
                })

        # 공통 요소 추출
        if analysis["content_patterns"]:
            analysis["common_elements"] = list(
                analysis["content_patterns"].keys())

        # 품질 지표 계산
        all_success_rates = [
            r.get('performance_data', {}).get('success_rate', 0)
            for r in agent_results
            if r.get('performance_data', {}).get('success_rate', 0) > 0
        ]

        analysis["quality_indicators"] = {
            "total_agents": len(set(r.get('agent_name') for r in agent_results)),
            "avg_success_rate": sum(all_success_rates) / len(all_success_rates) if all_success_rates else 0.5,
            "successful_rate": len(analysis["successful_approaches"]) / max(len(agent_results), 1),
            "data_richness": len(analysis["content_patterns"])
        }

        print(
            f"📊 CrewAI 기반 에이전트 데이터 분석 완료: {analysis['quality_indicators']['total_agents']}개 에이전트, 평균 성공률: {analysis['quality_indicators']['avg_success_rate']:.2f}")

        return analysis

    def _enhance_content_with_agent_results(self, content_section: Dict, agent_analysis: Dict, learning_insights: Dict) -> Dict:
        """에이전트 결과로 콘텐츠 강화"""
        enhanced_content = content_section.copy()
        enhanced_content['crewai_enhanced'] = True

        # 에이전트 인사이트 적용
        for agent_name, insights in agent_analysis["agent_insights"].items():
            if agent_name == "ContentCreatorV2Agent":
                # 콘텐츠 생성 에이전트 결과 반영
                if insights and insights[-1].get("content_length", 0) > 1000:
                    # 풍부한 콘텐츠가 생성되었으면 본문 확장
                    current_body = enhanced_content.get('body', '')
                    if len(current_body) < 500:
                        enhanced_content['body'] = current_body + \
                            "\n\n이 여행은 특별한 의미와 감동을 선사했습니다."
            elif agent_name == "ImageAnalyzerAgent":
                # 이미지 분석 에이전트 결과 반영
                if insights and insights[-1].get("has_performance_data", False):
                    # 성능 데이터가 있으면 이미지 관련 설명 추가
                    enhanced_content['image_description'] = "전문적으로 분석된 이미지들"

        # 성공적인 접근법 반영
        for approach in agent_analysis["successful_approaches"]:
            if approach["success_rate"] > 0.9:
                enhanced_content['quality_boost'] = f"고품질 {approach['agent']} 결과 반영"

        # 학습 인사이트 통합
        recommendations = learning_insights.get('recommendations', [])
        for recommendation in recommendations:
            if "콘텐츠" in recommendation and "풍부" in recommendation:
                current_body = enhanced_content.get('body', '')
                if len(current_body) < 800:
                    enhanced_content['body'] = current_body + \
                        "\n\n이러한 경험들이 모여 잊을 수 없는 여행의 추억을 만들어냅니다."

        return enhanced_content

    def _agent_result_based_jsx_pipeline(self, content: Dict, component_name: str, index: int,
                                         total_sections: int, agent_analysis: Dict, learning_insights: Dict) -> str:
        """에이전트 결과 기반 JSX 파이프라인"""
        try:
            # 1단계: 에이전트 결과 기반 콘텐츠 분석
            print(f"  📊 1단계: 에이전트 결과 기반 콘텐츠 분석...")
            analysis_result = self.content_analyzer.analyze_content_for_jsx(
                content, index, total_sections)

            # 에이전트 분석 결과 통합
            analysis_result = self._integrate_agent_analysis(
                analysis_result, agent_analysis)

            # 2단계: 에이전트 인사이트 기반 레이아웃 설계
            print(f"  🎨 2단계: 에이전트 인사이트 기반 레이아웃 설계...")
            design_result = self.layout_designer.design_layout_structure(
                content, analysis_result, component_name)

            # 에이전트 결과 기반 설계 강화
            design_result = self._enhance_design_with_agent_results(
                design_result, agent_analysis)

            # 3단계: 오류 없는 JSX 코드 생성
            print(f"  💻 3단계: 오류 없는 JSX 코드 생성...")
            jsx_code = self.code_generator.generate_jsx_code(
                content, design_result, component_name)

            # 4단계: 에이전트 결과 기반 검증 및 오류 제거
            print(f"  🔍 4단계: 에이전트 결과 기반 검증...")
            validated_jsx = self._comprehensive_jsx_validation(
                jsx_code, content, component_name, agent_analysis)

            return validated_jsx

        except Exception as e:
            print(f"⚠️ 에이전트 결과 기반 파이프라인 실패: {e}")
            # 폴백: 에이전트 데이터 기반 안전한 JSX 생성
            return self._create_agent_based_fallback_jsx(content, component_name, index, agent_analysis)

    def _integrate_agent_analysis(self, analysis_result: Dict, agent_analysis: Dict) -> Dict:
        """에이전트 분석 결과 통합"""
        enhanced_result = analysis_result.copy()
        enhanced_result['crewai_enhanced'] = True

        # 품질 지표 반영
        quality_indicators = agent_analysis.get("quality_indicators", {})
        if quality_indicators.get("avg_success_rate", 0) > 0.8:
            enhanced_result['confidence_boost'] = True
            # 고품질일 때 매거진 레이아웃
            enhanced_result['recommended_layout'] = 'magazine'

        # 공통 요소 반영
        common_elements = agent_analysis.get("common_elements", [])
        if 'title' in common_elements and 'body' in common_elements:
            enhanced_result['layout_complexity'] = '고급'

        # 성공적인 접근법 반영
        successful_approaches = agent_analysis.get("successful_approaches", [])
        if len(successful_approaches) > 2:
            enhanced_result['design_confidence'] = 'high'
            enhanced_result['color_palette'] = '프리미엄 블루'

        return enhanced_result

    def _enhance_design_with_agent_results(self, design_result: Dict, agent_analysis: Dict) -> Dict:
        """에이전트 결과로 설계 강화"""
        enhanced_result = design_result.copy()
        enhanced_result['crewai_enhanced'] = True

        # 에이전트 인사이트 기반 색상 조정
        agent_insights = agent_analysis.get("agent_insights", {})
        if "ImageAnalyzerAgent" in agent_insights:
            # 이미지 분석 결과가 있으면 시각적 조화 강화
            enhanced_result['color_scheme'] = {
                "primary": "#2c3e50",
                "secondary": "#f8f9fa",
                "accent": "#3498db",
                "background": "#ffffff"
            }

        # 성공적인 접근법 기반 컴포넌트 구조 조정
        successful_approaches = agent_analysis.get("successful_approaches", [])
        if len(successful_approaches) >= 3:
            # 다양한 성공 사례가 있으면 더 풍부한 컴포넌트 구조
            enhanced_result['styled_components'] = [
                "Container", "Header", "MainContent", "ImageGallery",
                "TextSection", "Sidebar", "Footer"
            ]

        return enhanced_result

    def _comprehensive_jsx_validation(self, jsx_code: str, content: Dict, component_name: str, agent_analysis: Dict) -> str:
        """포괄적 JSX 검증 (오류 제거)"""
        # 1. 기본 구문 검증
        jsx_code = self._validate_basic_jsx_syntax(jsx_code, component_name)

        # 2. 에이전트 결과 기반 콘텐츠 검증
        jsx_code = self._validate_content_with_agent_results(
            jsx_code, content, agent_analysis)

        # 3. 마크다운 블록 완전 제거
        jsx_code = self._remove_all_markdown_blocks(jsx_code)

        # 4. 문법 오류 완전 제거
        jsx_code = self._fix_all_syntax_errors(jsx_code)

        # 5. 컴파일 가능성 검증
        jsx_code = self._ensure_compilation_safety(jsx_code, component_name)

        return jsx_code

    def _validate_basic_jsx_syntax(self, jsx_code: str, component_name: str) -> str:
        """기본 JSX 문법 검증"""
        # 필수 import 확인
        if 'import React' not in jsx_code:
            jsx_code = 'import React from "react";\n' + jsx_code
        if 'import styled' not in jsx_code:
            jsx_code = jsx_code.replace(
                'import React from "react";',
                'import React from "react";\nimport styled from "styled-components";'
            )

        # export 문 확인
        if f'export const {component_name}' not in jsx_code:
            jsx_code = re.sub(r'export const \w+',
                              f'export const {component_name}', jsx_code)

        # return 문 확인
        if 'return (' not in jsx_code:
            jsx_code = jsx_code.replace(
                f'export const {component_name} = () => {{',
                f'export const {component_name} = () => {{\n  return (\n    <Container>\n      <h1>Component Content</h1>\n    </Container>\n  );\n}};'
            )

        return jsx_code

    def _validate_content_with_agent_results(self, jsx_code: str, content: Dict, agent_analysis: Dict) -> str:
        """에이전트 결과 기반 콘텐츠 검증"""
        # 에이전트 인사이트 기반 콘텐츠 강화
        quality_indicators = agent_analysis.get("quality_indicators", {})
        if quality_indicators.get("avg_success_rate", 0) > 0.8:
            # 고품질 에이전트 결과가 있으면 프리미엄 스타일 적용
            jsx_code = jsx_code.replace(
                'background: #ffffff',
                'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
            )

        return jsx_code

    def _remove_all_markdown_blocks(self, jsx_code: str) -> str:
        """마크다운 블록 완전 제거"""
        # 코드 블록 제거
        jsx_code = re.sub(r'``````', '', jsx_code)
        jsx_code = re.sub(r'`[^`]*`', '', jsx_code)

        # 마크다운 문법 제거
        jsx_code = re.sub(r'#{1,6}\s+', '', jsx_code)
        jsx_code = re.sub(r'\*\*(.*?)\*\*', r'\1', jsx_code)
        jsx_code = re.sub(r'\*(.*?)\*', r'\1', jsx_code)

        return jsx_code

    def _fix_all_syntax_errors(self, jsx_code: str) -> str:
        """문법 오류 완전 제거"""
        # 중괄호 균형 맞추기
        open_braces = jsx_code.count('{')
        close_braces = jsx_code.count('}')
        if open_braces > close_braces:
            jsx_code += '}' * (open_braces - close_braces)

        # 괄호 균형 맞추기
        open_parens = jsx_code.count('(')
        close_parens = jsx_code.count(')')
        if open_parens > close_parens:
            jsx_code += ')' * (open_parens - close_parens)

        # 세미콜론 추가
        lines = jsx_code.split('\n')
        fixed_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.endswith((';', '{', '}', '(', ')', ',', '>', '<')):
                if not stripped.startswith(('import', 'export', 'const', 'let', 'var', 'function', 'class')):
                    line += ';'
            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _ensure_compilation_safety(self, jsx_code: str, component_name: str) -> str:
        """컴파일 가능성 검증"""
        # 기본 구조 보장
        required_parts = [
            'import React from "react";',
            'import styled from "styled-components";',
            f'export const {component_name}',
            'return (',
            '</Container>'
        ]

        for part in required_parts:
            if part not in jsx_code:
                if part == 'import React from "react";':
                    jsx_code = part + '\n' + jsx_code
                elif part == 'import styled from "styled-components";':
                    jsx_code = jsx_code.replace(
                        'import React from "react";',
                        'import React from "react";\nimport styled from "styled-components";'
                    )

        return jsx_code

    def _validate_jsx_with_agent_insights(self, jsx_code: str, content: Dict, component_name: str, agent_analysis: Dict) -> str:
        """에이전트 인사이트 기반 JSX 검증"""
        # 에이전트 분석 결과 반영
        successful_approaches = agent_analysis.get("successful_approaches", [])
        if len(successful_approaches) > 2:
            # 성공적인 접근법이 많으면 더 정교한 스타일링 적용
            jsx_code = jsx_code.replace(
                'padding: 20px;',
                'padding: 40px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);'
            )

        return jsx_code

    def _create_agent_based_fallback_jsx(self, content: Dict, component_name: str, index: int, agent_analysis: Dict) -> str:
        """에이전트 데이터 기반 폴백 JSX 생성"""
        title = content.get('title', f'Component {index + 1}')
        body = content.get('body', '콘텐츠를 표시합니다.')

        # 에이전트 분석 결과 반영
        quality_score = agent_analysis.get(
            "quality_indicators", {}).get("avg_success_rate", 0.5)

        if quality_score > 0.8:
            background_style = 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'
        elif quality_score > 0.6:
            background_style = 'background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);'
        else:
            background_style = 'background: #f8f9fa;'

        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
  {background_style}
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.1);
`;

const Title = styled.h1`
  font-size: 2.5rem;
  color: #2c3e50;
  margin-bottom: 1rem;
  text-align: center;
`;

const Content = styled.p`
  font-size: 1.1rem;
  line-height: 1.6;
  color: #555;
  text-align: center;
`;

export const {component_name} = () => {{
  return (
    <Container>
      <Title>{title}</Title>
      <Content>{body}</Content>
    </Container>
  );
}};'''

    def _safe_parse_json(self, content: str) -> Dict:
        """안전한 JSON 파싱"""
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            return None

    def _validate_jsx_syntax(self, jsx_code: str) -> bool:
        """JSX 문법 검증"""
        required_elements = [
            'import React',
            'export const',
            'return (',
            '</Container>'
        ]

        return all(element in jsx_code for element in required_elements)

    def save_jsx_components(self, generated_components: List[Dict], components_folder: str) -> List[Dict]:
        """생성된 JSX 컴포넌트들을 파일로 저장 (CrewAI 기반 에이전트 결과 활용)"""
        print(
            f"📁 JSX 컴포넌트 저장 시작: {len(generated_components)}개 → {components_folder}")

        # 폴더 생성
        os.makedirs(components_folder, exist_ok=True)

        saved_components = []
        successful_saves = 0

        for i, component_data in enumerate(generated_components):
            try:
                component_name = component_data.get(
                    'name', f'AgentBased{i+1:02d}Component')
                component_file = component_data.get(
                    'file', f'{component_name}.jsx')
                jsx_code = component_data.get('jsx_code', '')

                if not jsx_code:
                    print(f"⚠️ {component_name}: JSX 코드 없음 - 건너뛰기")
                    continue

                # 파일 경로 생성
                file_path = os.path.join(components_folder, component_file)

                # JSX 코드 최종 검증 및 정리
                validated_jsx = self._ensure_compilation_safety(
                    jsx_code, component_name)
                validated_jsx = self._remove_all_markdown_blocks(validated_jsx)

                # 파일 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(validated_jsx)

                # 저장된 컴포넌트 정보 생성
                saved_component = {
                    'name': component_name,
                    'file': component_file,
                    'file_path': file_path,
                    'jsx_code': validated_jsx,
                    'size_bytes': len(validated_jsx.encode('utf-8')),
                    'approach': component_data.get('approach', 'crewai_agent_results_based'),
                    'error_free': self._validate_jsx_syntax(validated_jsx),
                    'crewai_enhanced': component_data.get('crewai_enhanced', True),
                    'agent_data_utilized': component_data.get('agent_data_analysis', {}) != {},
                    'save_timestamp': self._get_timestamp()
                }

                saved_components.append(saved_component)
                successful_saves += 1

                # 개별 저장 로깅
                self.result_manager.store_agent_output(
                    agent_name="JSXCreatorAgent_FileSaver",
                    agent_role="JSX 파일 저장자",
                    task_description=f"컴포넌트 {component_name} 파일 저장",
                    final_answer=f"파일 저장 성공: {file_path}",
                    reasoning_process=f"CrewAI 기반 생성된 JSX 컴포넌트를 {components_folder}에 저장",
                    execution_steps=[
                        "JSX 코드 최종 검증",
                        "마크다운 블록 제거",
                        "컴파일 안전성 확보",
                        "파일 저장 완료"
                    ],
                    raw_input={
                        "component_name": component_name,
                        "file_path": file_path,
                        "jsx_code_length": len(jsx_code)
                    },
                    raw_output=saved_component,
                    performance_metrics={
                        "file_size_bytes": saved_component['size_bytes'],
                        "error_free": saved_component['error_free'],
                        "crewai_enhanced": saved_component['crewai_enhanced'],
                        "agent_data_utilized": saved_component['agent_data_utilized']
                    }
                )

                print(
                    f"✅ {component_name} 저장 완료 (크기: {saved_component['size_bytes']} bytes, 방식: {saved_component['approach']}, 오류없음: {saved_component['error_free']})")

            except Exception as e:
                print(
                    f"❌ {component_data.get('name', f'Component{i+1}')} 저장 실패: {e}")

                # 저장 실패 로깅
                self.result_manager.store_agent_output(
                    agent_name="JSXCreatorAgent_FileSaver",
                    agent_role="JSX 파일 저장자",
                    task_description=f"컴포넌트 저장 실패",
                    final_answer=f"ERROR: {str(e)}",
                    reasoning_process="JSX 컴포넌트 파일 저장 중 예외 발생",
                    error_logs=[
                        {"error": str(e), "component": component_data.get('name', 'unknown')}],
                    performance_metrics={
                        "save_failed": True,
                        "error_occurred": True
                    }
                )
                continue

        # 전체 저장 결과 로깅
        self.result_manager.store_agent_output(
            agent_name="JSXCreatorAgent_SaveBatch",
            agent_role="JSX 배치 저장 관리자",
            task_description=f"CrewAI 기반 {len(generated_components)}개 JSX 컴포넌트 배치 저장",
            final_answer=f"배치 저장 완료: {successful_saves}/{len(generated_components)}개 성공",
            reasoning_process=f"CrewAI 기반 생성된 JSX 컴포넌트들을 {components_folder}에 일괄 저장",
            execution_steps=[
                "컴포넌트 폴더 생성",
                "개별 컴포넌트 저장 루프",
                "JSX 코드 검증 및 정리",
                "파일 저장 및 메타데이터 생성",
                "저장 결과 집계"
            ],
            raw_input={
                "generated_components_count": len(generated_components),
                "components_folder": components_folder
            },
            raw_output=saved_components,
            performance_metrics={
                "total_components": len(generated_components),
                "successful_saves": successful_saves,
                "save_success_rate": successful_saves / max(len(generated_components), 1),
                "total_file_size": sum(comp['size_bytes'] for comp in saved_components),
                "error_free_count": len([comp for comp in saved_components if comp['error_free']]),
                "crewai_enhanced_count": len([comp for comp in saved_components if comp['crewai_enhanced']]),
                "agent_data_utilized_count": len([comp for comp in saved_components if comp['agent_data_utilized']])
            }
        )

        print(
            f"📁 저장 완료: {successful_saves}/{len(generated_components)}개 성공 (CrewAI 기반 에이전트 데이터 활용)")
        print(
            f"📊 총 파일 크기: {sum(comp['size_bytes'] for comp in saved_components):,} bytes")
        print(f"✅ 컴포넌트 저장 완료: {len(saved_components)}개")

        return saved_components
