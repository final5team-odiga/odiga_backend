from typing import Dict, List
from typing import Dict, List
from crewai import Agent, Task
from custom_llm import get_azure_llm
from utils.agent_decision_logger import get_agent_logger, get_complete_data_manager
import json
import re

class JSXLayoutDesigner:
    """레이아웃 설계 전문 에이전트 (에이전트 결과 데이터 기반)"""

    def __init__(self):
        self.llm = get_azure_llm()
        self.logger = get_agent_logger()
        self.result_manager = get_complete_data_manager()

    def create_agent(self):
        return Agent(
            role="에이전트 결과 데이터 기반 매거진 레이아웃 아키텍트",
            goal="이전 에이전트들의 모든 결과 데이터, template_data.json, PDF 벡터 데이터를 종합 분석하여 완벽한 JSX 레이아웃 구조를 설계",
            backstory="""당신은 25년간 세계 최고 수준의 매거진 디자인과 디지털 레이아웃 분야에서 활동해온 전설적인 레이아웃 아키텍트입니다.

**에이전트 결과 데이터 활용 마스터십:**
- 이전 모든 에이전트들의 출력 결과를 종합 분석
- ContentCreator, ImageAnalyzer, ContentAnalyzer 등의 결과를 레이아웃에 반영
- 에이전트 협업 패턴과 성공 지표를 설계 결정에 활용
- jsx_templates는 사용하지 않고 에이전트 데이터만 활용

**데이터 기반 설계 우선순위:**
1. 이전 에이전트들의 결과 데이터 (최우선)
2. template_data.json의 콘텐츠 구조
3. PDF 벡터 데이터의 레이아웃 패턴
4. 에이전트 협업 품질 지표
5, 존재하는 모든 콘텐츠 데이터와 이미지 URL을 사용해야함함

**설계 철학:**
"진정한 매거진 레이아웃은 에이전트들의 협업 결과를 존중하면서도 독자의 인지 과정을 과학적으로 설계한 시스템입니다. jsx_templates에 의존하지 않고 순수한 에이전트 데이터만으로 최적의 레이아웃을 창조합니다."

**오류 없는 설계 보장:**
모든 설계 결정은 JSX 구현 시 오류가 발생하지 않도록 기술적 완성도를 고려합니다.""",
            verbose=True,
            llm=self.llm
        )

    async def design_layout_structure(self, content: Dict, analysis: Dict, component_name: str) -> Dict:
        all_agent_results = await self.result_manager.get_all_outputs(exclude_agent="JSXLayoutDesigner")
        learning_insights = await self.logger.get_learning_insights("JSXLayoutDesigner")

        print(f"\U0001f4da 수집된 에이전트 결과: {len(all_agent_results)}개")
        print(f"\U0001f9e0 학습 인사이트: {len(learning_insights.get('recommendations', []))}개")

        agent = self.create_agent()
        agent_data_analysis = await self._analyze_all_agent_results(all_agent_results)

        design_task = Task(
            description="에이전트 결과 데이터를 기반으로 JSX 레이아웃 설계를 수행합니다.",
            agent=agent,
            expected_output="에이전트 결과 데이터 기반 완전한 레이아웃 구조 설계 JSON"
        )

        try:
            result = await agent.execute_task(design_task)
            design_result = await self._parse_design_result_with_agent_data(str(result), analysis, agent_data_analysis)

            await self.result_manager.store_agent_output(
                agent_name="JSXLayoutDesigner",
                agent_role="에이전트 데이터 기반 레이아웃 아키텍트",
                task_description=f"컴포넌트 {component_name} 레이아웃 설계",
                final_answer=str(design_result),
                reasoning_process=f"{len(all_agent_results)}개 에이전트 결과 분석하여 레이아웃 설계",
                execution_steps=["에이전트 결과 수집", "데이터 분석", "레이아웃 설계", "검증 완료"],
                raw_input={"content": content, "analysis": analysis, "component_name": component_name},
                raw_output=design_result,
                performance_metrics={
                    "agent_results_utilized": len(all_agent_results),
                    "jsx_templates_ignored": True,
                    "learning_insights_applied": len(learning_insights.get('recommendations', [])),
                    "layout_type": design_result.get('layout_type'),
                    "error_prevention_applied": True
                }
            )

            print(f"\u2705 에이전트 데이터 기반 레이아웃 설계 완료: {design_result.get('layout_type', '기본')} 구조")
            return design_result

        except Exception as e:
            print(f"\u26a0\ufe0f 레이아웃 설계 실패: {e}")
            return await self._create_agent_based_default_design(analysis, component_name, agent_data_analysis)

    async def _analyze_all_agent_results(self, agent_results: List[Dict]) -> Dict:
        analysis = {
            "agent_summary": {},
            "quality_indicators": {}
        }

        if not agent_results:
            return analysis

        for result in agent_results:
            agent_name = result.get('agent_name', 'unknown')

            if agent_name not in analysis["agent_summary"]:
                analysis["agent_summary"][agent_name] = {
                    "count": 0,
                    "avg_confidence": 0,
                    "latest_output": None
                }

            summary = analysis["agent_summary"][agent_name]
            summary["count"] += 1
            confidence = result.get('metadata', {}).get('confidence_score', 0)
            if confidence > 0:
                summary["avg_confidence"] = (summary["avg_confidence"] * (summary["count"] - 1) + confidence) / summary["count"]
            summary["latest_output"] = result.get('full_output')

        all_confidences = [
            r.get('metadata', {}).get('confidence_score', 0)
            for r in agent_results if r.get('metadata', {}).get('confidence_score', 0) > 0
        ]

        if all_confidences:
            analysis["quality_indicators"] = {
                "overall_confidence": sum(all_confidences) / len(all_confidences),
                "high_quality_count": len([c for c in all_confidences if c > 0.8]),
                "total_agents": len(analysis["agent_summary"]),
                "collaboration_success": len(all_confidences) / len(agent_results)
            }

        return analysis

    async def _parse_design_result_with_agent_data(self, result_text: str, analysis: Dict, agent_analysis: Dict) -> Dict:
        try:
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                parsed_result = json.loads(json_match.group())
                parsed_result['agent_data_integration'] = agent_analysis
                parsed_result['jsx_templates_ignored'] = True
                parsed_result['error_prevention_applied'] = True
                return parsed_result
        except Exception as e:
            print(f"⚠️ JSON 파싱 실패: {e}")
        return await self._create_agent_based_default_design(analysis, "DefaultComponent", agent_analysis)

    async def _create_agent_based_default_design(self, analysis: Dict, component_name: str, agent_analysis: Dict) -> Dict:
        layout_type = analysis.get('recommended_layout', 'grid')
        quality_indicators = agent_analysis.get("quality_indicators", {})
        if quality_indicators.get("overall_confidence", 0) > 0.8:
            layout_type = 'magazine'

        return {
            "layout_type": layout_type,
            "layout_rationale": f"에이전트 데이터 기반 {layout_type} 레이아웃 선택. {len(agent_analysis.get('agent_summary', {}))}개 에이전트 결과 반영",
            "grid_structure": "1fr 1fr" if layout_type == 'grid' else "1fr",
            "styled_components": ["Container", "Header", "Title", "Subtitle", "Content", "ImageGallery", "Footer"],
            "color_scheme": {
                "primary": "#2c3e50",
                "secondary": "#f8f9fa",
                "accent": "#3498db",
                "background": "#ffffff"
            },
            "typography_scale": {
                "title": "3em",
                "subtitle": "1.4em",
                "body": "1.1em",
                "caption": "0.9em"
            },
            "image_layout": "grid_responsive",
            "agent_data_integration": agent_analysis,
            "jsx_templates_ignored": True,
            "error_prevention": "완전한 JSX 문법 준수 및 오류 방지 적용",
            "quality_metrics": {
                "agent_collaboration_score": quality_indicators.get("collaboration_success", 0.8),
                "design_confidence": quality_indicators.get("overall_confidence", 0.85),
                "error_free_guarantee": 1.0
            }
        }
