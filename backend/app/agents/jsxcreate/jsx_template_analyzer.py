import os
import re
from typing import Dict, List
from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager
from utils.agent_decision_logger import get_agent_logger, get_complete_data_manager
import asyncio

class JSXTemplateAnalyzer:
    """JSX 템플릿 분석기 (CrewAI 기반 로깅 시스템 통합)"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        self.templates_cache = {}
        self.vector_manager = PDFVectorManager()
        self.logger = get_agent_logger()
        self.result_manager = get_complete_data_manager()
        
        # CrewAI 에이전트들 생성
        self.template_analysis_agent = self._create_template_analysis_agent()
        self.vector_enhancement_agent = self._create_vector_enhancement_agent()
        self.agent_result_integrator = self._create_agent_result_integrator()
        self.template_selector_agent = self._create_template_selector_agent()

    async def _create_template_analysis_agent(self):
        """템플릿 분석 전문 에이전트"""
        return Agent(
            role="JSX 템플릿 구조 분석 전문가",
            goal="JSX 템플릿 파일들의 구조적 특성과 레이아웃 패턴을 정밀 분석하여 최적화된 분류 및 특성 정보를 제공",
            backstory="""당신은 12년간 React 및 JSX 생태계에서 컴포넌트 아키텍처 분석과 패턴 인식을 담당해온 전문가입니다. 다양한 JSX 템플릿의 구조적 특성을 분석하여 최적의 사용 시나리오를 도출하는 데 특화되어 있습니다.

**전문 영역:**
- JSX 컴포넌트 구조 분석
- Styled-components 패턴 인식
- 레이아웃 시스템 분류
- 템플릿 복잡도 평가

**분석 방법론:**
"모든 JSX 템플릿은 고유한 설계 철학과 사용 목적을 가지고 있으며, 이를 정확히 분석하여 최적의 콘텐츠 매칭을 가능하게 합니다."

**핵심 역량:**
- 컴포넌트명 및 Props 추출
- Styled-components 패턴 분석
- 레이아웃 타입 분류 (simple/hero/grid/gallery)
- 이미지 전략 및 텍스트 전략 평가
- 복잡도 수준 측정""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def _create_vector_enhancement_agent(self):
        """벡터 데이터 강화 전문가"""
        return Agent(
            role="PDF 벡터 데이터 기반 템플릿 강화 전문가",
            goal="PDF 벡터 데이터베이스와 템플릿 특성을 매칭하여 템플릿 분석 결과를 강화하고 최적화된 사용 권장사항을 제공",
            backstory="""당신은 10년간 벡터 데이터베이스와 유사도 검색 시스템을 활용한 템플릿 최적화를 담당해온 전문가입니다. Azure Cognitive Search와 PDF 벡터 데이터를 활용하여 템플릿의 잠재적 활용도를 극대화하는 데 특화되어 있습니다.

**기술 전문성:**
- 벡터 유사도 검색 및 매칭
- PDF 레이아웃 패턴 분석
- 템플릿-콘텐츠 호환성 평가
- 사용 시나리오 최적화

**강화 전략:**
"벡터 데이터의 풍부한 레이아웃 정보를 활용하여 각 템플릿의 최적 활용 시나리오를 식별하고 신뢰도를 향상시킵니다."

**출력 강화 요소:**
- 벡터 매칭 기반 신뢰도 계산
- 유사 레이아웃 기반 사용 권장
- PDF 소스 기반 용도 분류
- 레이아웃 패턴 최적화""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def _create_agent_result_integrator(self):
        """에이전트 결과 통합 전문가"""
        return Agent(
            role="에이전트 결과 통합 및 템플릿 강화 전문가",
            goal="BindingAgent와 OrgAgent의 실행 결과를 분석하여 템플릿 특성을 강화하고 최적화된 인사이트를 제공",
            backstory="""당신은 8년간 다중 에이전트 시스템의 결과 통합과 패턴 분석을 담당해온 전문가입니다. BindingAgent의 이미지 배치 전략과 OrgAgent의 텍스트 구조 분석 결과를 템플릿 특성 강화에 활용하는 데 특화되어 있습니다.

**통합 전문성:**
- BindingAgent 이미지 배치 인사이트 활용
- OrgAgent 텍스트 구조 분석 통합
- 에이전트 간 시너지 효과 극대화
- 템플릿 신뢰도 향상

**분석 방법론:**
"각 에이전트의 전문성을 템플릿 분석에 반영하여 단일 분석으로는 달성할 수 없는 수준의 정확도와 신뢰도를 확보합니다."

**강화 영역:**
- 그리드/갤러리 레이아웃 최적화
- 이미지 배치 전략 반영
- 텍스트 구조 복잡도 조정
- 매거진 스타일 최적화""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def _create_template_selector_agent(self):
        """템플릿 선택 전문가"""
        return Agent(
            role="콘텐츠 기반 최적 템플릿 선택 전문가",
            goal="콘텐츠 특성과 템플릿 분석 결과를 종합하여 가장 적합한 템플릿을 선택하고 선택 근거를 제공",
            backstory="""당신은 15년간 콘텐츠 관리 시스템과 템플릿 매칭 알고리즘을 설계해온 전문가입니다. 복잡한 콘텐츠 특성과 다양한 템플릿 옵션 중에서 최적의 조합을 찾아내는 데 특화되어 있습니다.

**선택 전문성:**
- 콘텐츠-템플릿 호환성 분석
- 다차원 점수 계산 시스템
- 벡터 데이터 기반 매칭
- 에이전트 인사이트 통합

**선택 철학:**
"완벽한 템플릿 선택은 콘텐츠의 본질적 특성과 템플릿의 구조적 강점이 완벽히 조화를 이루는 지점에서 이루어집니다."

**평가 기준:**
- 이미지 개수 및 전략 매칭
- 텍스트 길이 및 복잡도 적합성
- 벡터 데이터 기반 보너스
- 에이전트 인사이트 반영
- 감정 톤 및 용도 일치성""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def analyze_jsx_templates(self, templates_dir: str = "jsx_templates") -> Dict[str, Dict]:
        """jsx_templates 폴더의 모든 템플릿 분석 (CrewAI 기반 벡터 데이터 활용 + 로깅)"""

        # 이전 에이전트 결과 수집
        previous_results = self.result_manager.get_all_outputs(exclude_agent="JSXTemplateAnalyzer")
        binding_results = [r for r in previous_results if "BindingAgent" in r.get('agent_name', '')]
        org_results = [r for r in previous_results if "OrgAgent" in r.get('agent_name', '')]

        print(f"📊 이전 에이전트 결과 수집: 전체 {len(previous_results)}개, BindingAgent {len(binding_results)}개, OrgAgent {len(org_results)}개")

        if not os.path.exists(templates_dir):
            print(f"❌ 템플릿 폴더 없음: {templates_dir}")
            # 에러 로깅
            self.result_manager.store_agent_output(
                agent_name="JSXTemplateAnalyzer",
                agent_role="JSX 템플릿 분석기",
                task_description=f"템플릿 폴더 {templates_dir} 분석",
                final_answer=f"ERROR: 템플릿 폴더 없음 - {templates_dir}",
                reasoning_process="템플릿 폴더 존재 확인 실패",
                error_logs=[{"error": f"Directory not found: {templates_dir}"}],
                performance_metrics={
                    "previous_results_count": len(previous_results),
                    "binding_results_count": len(binding_results),
                    "org_results_count": len(org_results)
                }
            )
            return {}

        jsx_files = [f for f in os.listdir(templates_dir) if f.endswith('.jsx')]

        if not jsx_files:
            print(f"❌ JSX 템플릿 파일 없음: {templates_dir}")
            # 에러 로깅
            self.result_manager.store_agent_output(
                agent_name="JSXTemplateAnalyzer",
                agent_role="JSX 템플릿 분석기",
                task_description=f"템플릿 파일 검색 in {templates_dir}",
                final_answer=f"ERROR: JSX 파일 없음 - {templates_dir}",
                reasoning_process="JSX 템플릿 파일 검색 실패",
                error_logs=[{"error": f"No .jsx files found in {templates_dir}"}],
                performance_metrics={
                    "previous_results_count": len(previous_results),
                    "binding_results_count": len(binding_results),
                    "org_results_count": len(org_results)
                }
            )
            return {}

        # CrewAI Task들 생성
        template_analysis_task = self._create_template_analysis_task(templates_dir, jsx_files)
        vector_enhancement_task = self._create_vector_enhancement_task()
        agent_integration_task = self._create_agent_integration_task(binding_results, org_results)

        # CrewAI Crew 생성 및 실행
        analysis_crew = Crew(
            agents=[self.template_analysis_agent, self.vector_enhancement_agent, self.agent_result_integrator],
            tasks=[template_analysis_task, vector_enhancement_task, agent_integration_task],
            process=Process.sequential,
            verbose=True
        )

        # Crew 실행
        crew_result = await analysis_crew.kickoff()

        # 실제 분석 수행
        analyzed_templates = await self._execute_template_analysis_with_crew_insights(
            crew_result, templates_dir, jsx_files, binding_results, org_results
        )

        self.templates_cache = analyzed_templates

        # 전체 분석 결과 로깅
        successful_analyses = len([t for t in analyzed_templates.values() if t.get('analysis_success', True)])

        self.result_manager.store_agent_output(
            agent_name="JSXTemplateAnalyzer",
            agent_role="JSX 템플릿 분석기",
            task_description=f"CrewAI 기반 {len(jsx_files)}개 JSX 템플릿 분석",
            final_answer=f"성공적으로 {successful_analyses}/{len(jsx_files)}개 템플릿 분석 완료",
            reasoning_process=f"CrewAI 기반 벡터 데이터와 {len(previous_results)}개 에이전트 결과 통합하여 템플릿별 특성 분석",
            execution_steps=[
                "CrewAI 에이전트 및 태스크 생성",
                "이전 에이전트 결과 수집",
                "템플릿 폴더 검증",
                "JSX 파일 목록 수집",
                "개별 템플릿 분석",
                "벡터 데이터 통합",
                "에이전트 결과 강화",
                "분석 결과 캐싱"
            ],
            raw_input={
                "templates_dir": templates_dir,
                "jsx_files": jsx_files,
                "previous_results_count": len(previous_results)
            },
            raw_output=analyzed_templates,
            performance_metrics={
                "total_templates": len(jsx_files),
                "successful_analyses": successful_analyses,
                "success_rate": successful_analyses / len(jsx_files) if jsx_files else 0,
                "vector_enhanced_count": len([t for t in analyzed_templates.values() if t.get('vector_matched', False)]),
                "agent_enhanced_count": len([t for t in analyzed_templates.values() if t.get('agent_enhanced', False)]),
                "previous_results_utilized": len(previous_results),
                "binding_results_count": len(binding_results),
                "org_results_count": len(org_results),
                "crewai_enhanced": True
            }
        )

        return analyzed_templates

    async def _execute_template_analysis_with_crew_insights(self, crew_result, templates_dir: str, jsx_files: List[str], 
                                                        binding_results: List[Dict], org_results: List[Dict]) -> Dict[str, Dict]:
        """CrewAI 인사이트를 활용한 실제 템플릿 분석 (비동기)"""
        print(f"📁 CrewAI 기반 {len(jsx_files)}개 JSX 템플릿 분석 시작 (벡터 데이터 통합 + 에이전트 결과 활용)")

        analyzed_templates = {}
        successful_analyses = 0

        async def analyze_single(jsx_file: str) -> Tuple[str, Dict]:
            file_path = os.path.join(templates_dir, jsx_file)

            loop = asyncio.get_event_loop()
            template_analysis = await loop.run_in_executor(None, self._analyze_single_template, file_path, jsx_file)
            template_analysis = await loop.run_in_executor(None, self._enhance_with_vector_data, template_analysis, jsx_file)
            template_analysis = await loop.run_in_executor(None, self._enhance_with_agent_results, template_analysis, binding_results, org_results)

            print(f"✅ {jsx_file} 분석 완료: {template_analysis['layout_type']} (벡터 매칭: {template_analysis['vector_matched']}, 에이전트 강화: {template_analysis.get('agent_enhanced', False)})")
            return jsx_file, template_analysis

        tasks = [analyze_single(f) for f in jsx_files]
        results = await asyncio.gather(*tasks)

        for jsx_file, analysis in results:
            analyzed_templates[jsx_file] = analysis
            if analysis.get('analysis_success', True):
                successful_analyses += 1

        return analyzed_templates

    def _create_template_analysis_task(self, templates_dir: str, jsx_files: List[str]) -> Task:
        """템플릿 분석 태스크"""
        return Task(
            description=f"""
            {templates_dir} 폴더의 {len(jsx_files)}개 JSX 템플릿 파일들을 체계적으로 분석하세요.

            **분석 대상 파일들:**
            {', '.join(jsx_files)}

            **분석 요구사항:**
            1. 각 JSX 파일의 구조적 특성 분석
            2. 컴포넌트명 및 Props 추출
            3. Styled-components 패턴 인식
            4. 레이아웃 타입 분류 (simple/hero/grid/gallery/overlay)
            5. 이미지 전략 및 텍스트 전략 평가
            6. 복잡도 수준 측정 (simple/moderate/complex)

            **분석 결과 구조:**
            각 템플릿별로 다음 정보 포함:
            - 기본 정보 (파일명, 컴포넌트명, props)
            - 레이아웃 특성 (타입, 특징, 그리드 구조)
            - 콘텐츠 전략 (이미지, 텍스트)
            - 복잡도 및 사용 권장사항

            모든 템플릿의 상세 분석 결과를 제공하세요.
            """,
            expected_output="JSX 템플릿별 상세 분석 결과",
            agent=self.template_analysis_agent
        )

    def _create_vector_enhancement_task(self) -> Task:
        """벡터 강화 태스크"""
        return Task(
            description="""
            PDF 벡터 데이터베이스를 활용하여 템플릿 분석 결과를 강화하세요.

            **강화 요구사항:**
            1. 각 템플릿의 레이아웃 특성을 벡터 검색 쿼리로 변환
            2. 유사한 매거진 레이아웃 패턴 검색 (top 3)
            3. 벡터 매칭 기반 신뢰도 계산
            4. PDF 소스 기반 사용 용도 분류

            **강화 영역:**
            - 레이아웃 신뢰도 향상
            - 사용 시나리오 최적화
            - 벡터 매칭 상태 표시
            - 유사 레이아웃 정보 제공

            **출력 요구사항:**
            - 벡터 매칭 성공/실패 상태
            - 신뢰도 점수 (0.0-1.0)
            - 권장 사용 용도
            - 유사 레이아웃 목록

            이전 태스크의 분석 결과를 벡터 데이터로 강화하세요.
            """,
            expected_output="벡터 데이터 기반 강화된 템플릿 분석 결과",
            agent=self.vector_enhancement_agent,
            context=[self._create_template_analysis_task("", [])]
        )

    def _create_agent_integration_task(self, binding_results: List[Dict], org_results: List[Dict]) -> Task:
        """에이전트 통합 태스크"""
        return Task(
            description=f"""
            BindingAgent와 OrgAgent의 실행 결과를 분석하여 템플릿 특성을 더욱 강화하세요.

            **통합 대상:**
            - BindingAgent 결과: {len(binding_results)}개
            - OrgAgent 결과: {len(org_results)}개

            **BindingAgent 인사이트 활용:**
            1. 이미지 배치 전략 분석 (그리드/갤러리)
            2. 시각적 일관성 평가 결과 반영
            3. 전문적 이미지 배치 인사이트 통합

            **OrgAgent 인사이트 활용:**
            1. 텍스트 구조 복잡도 분석
            2. 매거진 스타일 최적화 정보
            3. 구조화된 레이아웃 인사이트

            **강화 방법:**
            - 템플릿 신뢰도 점수 향상
            - 레이아웃 타입별 보너스 적용
            - 사용 권장사항 정교화
            - 에이전트 인사이트 메타데이터 추가

            이전 태스크들의 결과에 에이전트 인사이트를 통합하여 최종 강화된 템플릿 분석을 완성하세요.
            """,
            expected_output="에이전트 인사이트가 통합된 최종 템플릿 분석 결과",
            agent=self.agent_result_integrator,
            context=[self._create_template_analysis_task("", []), self._create_vector_enhancement_task()]
        )

    async def get_best_template_for_content(self, content: Dict, analysis: Dict) -> str:
        """콘텐츠에 가장 적합한 템플릿 선택 (CrewAI 기반 벡터 데이터 + 에이전트 결과 활용 + 로깅)"""
        previous_results = self.result_manager.get_all_outputs(exclude_agent="JSXTemplateAnalyzer")
        binding_results = [r for r in previous_results if "BindingAgent" in r.get('agent_name', '')]
        org_results = [r for r in previous_results if "OrgAgent" in r.get('agent_name', '')]

        if not self.templates_cache:
            selected_template = "Section01.jsx"
            self.result_manager.store_agent_output(
                agent_name="JSXTemplateAnalyzer_Selector",
                agent_role="템플릿 선택기",
                task_description="콘텐츠 기반 최적 템플릿 선택",
                final_answer=selected_template,
                reasoning_process="템플릿 캐시 없어 기본 템플릿 선택",
                raw_input={"content": content, "analysis": analysis},
                raw_output=selected_template,
                performance_metrics={
                    "fallback_selection": True,
                    "previous_results_count": len(previous_results)
                }
            )
            return selected_template

        template_selection_task = self._create_template_selection_task(content, analysis, previous_results)
        selection_crew = Crew(
            agents=[self.template_selector_agent],
            tasks=[template_selection_task],
            process=Process.sequential,
            verbose=True
        )

        # 비동기 실행
        crew_result = await asyncio.to_thread(selection_crew.kickoff)
        selected_template = await asyncio.to_thread(
            self._execute_template_selection_with_crew_insights,
            crew_result, content, analysis, previous_results, binding_results, org_results
        )

        return selected_template

    async def _execute_template_selection_with_crew_insights(self, crew_result, content: Dict, analysis: Dict, 
                                                     previous_results: List[Dict], binding_results: List[Dict], 
                                                     org_results: List[Dict]) -> str:
        """CrewAI 인사이트를 활용한 실제 템플릿 선택 (비동기 버전)"""
        image_count = len(content.get('images', []))
        text_length = len(content.get('body', ''))
        content_emotion = analysis.get('emotion_tone', 'neutral')

        # 콘텐츠 기반 벡터 검색 (비동기 처리 가정)
        content_query = f"{content.get('title', '')} {content.get('body', '')[:200]}"
        content_vectors = await self.vector_manager.search_similar_layouts_async(
            content_query,
            "magazine_layout",
            top_k=5
        )

        best_template = None
        best_score = 0
        scoring_details = []

        for template_name, template_info in self.templates_cache.items():
            score = 0
            score_breakdown = {"template": template_name}

            template_images = template_info['image_strategy']
            if image_count == 0 and template_images == 0:
                score += 30
                score_breakdown["image_match"] = 30
            elif image_count == 1 and template_images == 1:
                score += 30
                score_breakdown["image_match"] = 30
            elif image_count > 1 and template_images > 1:
                score += 20
                score_breakdown["image_match"] = 20

            if text_length < 300 and template_info['layout_type'] in ['simple', 'hero']:
                score += 20
                score_breakdown["text_match"] = 20
            elif text_length > 500 and template_info['layout_type'] in ['grid', 'gallery']:
                score += 20
                score_breakdown["text_match"] = 20

            if template_info.get('vector_matched', False):
                vector_bonus = template_info.get('layout_confidence', 0) * 30
                score += vector_bonus
                score_breakdown["vector_bonus"] = vector_bonus

            if template_info.get('agent_enhanced', False):
                agent_bonus = 0
                binding_insights = template_info.get('binding_insights', [])
                if binding_insights:
                    if image_count > 1 and 'grid_layout_optimized' in binding_insights:
                        agent_bonus += 15
                    if image_count > 3 and 'gallery_layout_optimized' in binding_insights:
                        agent_bonus += 15
                    if 'professional_image_placement' in binding_insights:
                        agent_bonus += 10

                org_insights = template_info.get('org_insights', [])
                if org_insights:
                    if text_length > 500 and 'structured_text_layout' in org_insights:
                        agent_bonus += 15
                    if 'magazine_style_optimized' in org_insights:
                        agent_bonus += 20
                    if text_length > 800 and 'complex_content_support' in org_insights:
                        agent_bonus += 10

                score += agent_bonus
                score_breakdown["agent_bonus"] = agent_bonus

            template_vectors = template_info.get('similar_pdf_layouts', [])
            vector_match_bonus = self._calculate_vector_content_match(content_vectors, template_vectors) * 20
            score += vector_match_bonus
            score_breakdown["content_vector_match"] = vector_match_bonus

            recommended_usage = template_info.get('recommended_usage', 'general')
            if content_emotion == 'peaceful' and 'culture' in recommended_usage:
                score += 15
                score_breakdown["emotion_match"] = 15
            elif content_emotion == 'exciting' and 'travel' in recommended_usage:
                score += 15
                score_breakdown["emotion_match"] = 15

            score_breakdown["total_score"] = score
            scoring_details.append(score_breakdown)

            if score > best_score:
                best_score = score
                best_template = template_name

        selected_template = best_template or "Section01.jsx"

        selected_info = self.templates_cache.get(selected_template, {})

        await self.result_manager.store_agent_output_async(
            agent_name="JSXTemplateAnalyzer_Selector",
            agent_role="템플릿 선택기",
            task_description="CrewAI 기반 콘텐츠 기반 최적 템플릿 선택",
            final_answer=selected_template,
            reasoning_process=f"CrewAI 기반 벡터 데이터와 {len(previous_results)}개 에이전트 결과 분석으로 최적 템플릿 선택. 최종 점수: {best_score}",
            execution_steps=[
                "CrewAI 템플릿 선택 태스크 실행",
                "이전 에이전트 결과 수집",
                "콘텐츠 특성 분석",
                "템플릿별 점수 계산",
                "벡터 매칭 보너스 적용",
                "에이전트 인사이트 보너스 적용",
                "최고 점수 템플릿 선택"
            ],
            raw_input={
                "content": content,
                "analysis": analysis,
                "image_count": image_count,
                "text_length": text_length,
                "content_emotion": content_emotion,
                "previous_results_count": len(previous_results)
            },
            raw_output={
                "selected_template": selected_template,
                "best_score": best_score,
                "scoring_details": scoring_details,
                "selected_info": selected_info
            },
            performance_metrics={
                "templates_evaluated": len(self.templates_cache),
                "best_score": best_score,
                "vector_matched": selected_info.get('vector_matched', False),
                "agent_enhanced": selected_info.get('agent_enhanced', False),
                "layout_confidence": selected_info.get('layout_confidence', 0),
                "content_vectors_found": len(content_vectors),
                "previous_results_count": len(previous_results),
                "binding_results_count": len(binding_results),
                "org_results_count": len(org_results),
                "binding_insights_applied": len(selected_info.get('binding_insights', [])),
                "org_insights_applied": len(selected_info.get('org_insights', [])),
                "crewai_enhanced": True
            }
        )

        print(f"🎯 CrewAI 기반 템플릿 선택: {selected_template}")
        print(f"- 점수: {best_score}")
        print(f"- 벡터 매칭: {selected_info.get('vector_matched', False)}")
        print(f"- 에이전트 강화: {selected_info.get('agent_enhanced', False)}")
        print(f"- 신뢰도: {selected_info.get('layout_confidence', 0)}")
        print(f"- 용도: {selected_info.get('recommended_usage', 'general')}")
        print(f"- BindingAgent 인사이트: {len(selected_info.get('binding_insights', []))}개")
        print(f"- OrgAgent 인사이트: {len(selected_info.get('org_insights', []))}개")

        return selected_template

    def _create_template_selection_task(self, content: Dict, analysis: Dict, previous_results: List[Dict]) -> Task:
        """템플릿 선택 태스크"""
        return Task(
            description=f"""
            콘텐츠 특성과 템플릿 분석 결과를 종합하여 가장 적합한 템플릿을 선택하세요.
            
            **콘텐츠 특성:**
            - 이미지 개수: {len(content.get('images', []))}개
            - 텍스트 길이: {len(content.get('body', ''))} 문자
            - 감정 톤: {analysis.get('emotion_tone', 'neutral')}
            - 제목: {content.get('title', 'N/A')}
            
            **이전 에이전트 결과:** {len(previous_results)}개
            
            **선택 기준:**
            1. 이미지 개수 및 전략 매칭 (30점)
            2. 텍스트 길이 및 복잡도 적합성 (20점)
            3. 벡터 데이터 기반 보너스 (최대 30점)
            4. 에이전트 인사이트 보너스 (최대 40점)
            5. 콘텐츠 벡터 매칭 (20점)
            6. 감정 톤 매칭 (15점)
            
            **에이전트 인사이트 활용:**
            - BindingAgent: 이미지 배치 전략 최적화
            - OrgAgent: 텍스트 구조 및 매거진 스타일
            
            **최종 출력:**
            - 선택된 템플릿명
            - 총 점수 및 점수 세부사항
            - 선택 근거 및 신뢰도
            
            모든 템플릿을 평가하여 최고 점수의 템플릿을 선택하세요.
            """,
            expected_output="최적 템플릿 선택 결과 및 상세 점수 분석",
            agent=self.template_selector_agent
        )

    # 기존 메서드들 유지 (변경 없음)
    async def _enhance_with_agent_results(self, template_analysis: Dict, binding_results: List[Dict], org_results: List[Dict]) -> Dict:
        """에이전트 결과 데이터로 템플릿 분석 강화"""
        enhanced_analysis = template_analysis.copy()
        enhanced_analysis['agent_enhanced'] = False
        enhanced_analysis['binding_insights'] = []
        enhanced_analysis['org_insights'] = []

        if not binding_results and not org_results:
            return enhanced_analysis

        enhanced_analysis['agent_enhanced'] = True

        if binding_results:
            latest_binding = binding_results[-1]
            binding_answer = latest_binding.get('agent_final_answer', '')

            if '그리드' in binding_answer or 'grid' in binding_answer.lower():
                enhanced_analysis['binding_insights'].append('grid_layout_optimized')
                if enhanced_analysis['layout_type'] == 'grid':
                    enhanced_analysis['layout_confidence'] = min(enhanced_analysis.get('layout_confidence', 0.5) + 0.2, 1.0)

            if '갤러리' in binding_answer or 'gallery' in binding_answer.lower():
                enhanced_analysis['binding_insights'].append('gallery_layout_optimized')
                if enhanced_analysis['layout_type'] == 'gallery':
                    enhanced_analysis['layout_confidence'] = min(enhanced_analysis.get('layout_confidence', 0.5) + 0.2, 1.0)

            if '배치' in binding_answer or 'placement' in binding_answer.lower():
                enhanced_analysis['binding_insights'].append('professional_image_placement')
                enhanced_analysis['recommended_usage'] = enhanced_analysis.get('recommended_usage', 'general') + '_image_focused'

        if org_results:
            latest_org = org_results[-1]
            org_answer = latest_org.get('agent_final_answer', '')

            if '구조' in org_answer or 'structure' in org_answer.lower():
                enhanced_analysis['org_insights'].append('structured_text_layout')
                if enhanced_analysis['text_strategy'] > 3:
                    enhanced_analysis['layout_confidence'] = min(enhanced_analysis.get('layout_confidence', 0.5) + 0.15, 1.0)

            if '매거진' in org_answer or 'magazine' in org_answer.lower():
                enhanced_analysis['org_insights'].append('magazine_style_optimized')
                enhanced_analysis['recommended_usage'] = 'magazine_editorial'

            if '복잡' in org_answer or 'complex' in org_answer.lower():
                enhanced_analysis['org_insights'].append('complex_content_support')
                if enhanced_analysis['complexity_level'] == 'complex':
                    enhanced_analysis['layout_confidence'] = min(enhanced_analysis.get('layout_confidence', 0.5) + 0.1, 1.0)

        return enhanced_analysis

    async def _enhance_with_vector_data(self, template_analysis: Dict, jsx_file: str) -> Dict:
        """벡터 데이터로 템플릿 분석 강화"""
        try:
            layout_query = self._create_layout_query_from_template(template_analysis)
            similar_layouts = await self.vector_manager.search_similar_layouts(
                layout_query,
                "magazine_layout",
                top_k=3
            )

            if similar_layouts:
                template_analysis['vector_matched'] = True
                template_analysis['similar_pdf_layouts'] = similar_layouts
                template_analysis['layout_confidence'] = self._calculate_layout_confidence(template_analysis, similar_layouts)
                template_analysis['recommended_usage'] = self._determine_usage_from_vectors(similar_layouts)
            else:
                template_analysis['vector_matched'] = False
                template_analysis['similar_pdf_layouts'] = []
                template_analysis['layout_confidence'] = 0.5
                template_analysis['recommended_usage'] = 'general'

        except Exception as e:
            print(f"⚠️ 벡터 데이터 통합 실패 ({jsx_file}): {e}")
            template_analysis['vector_matched'] = False
            template_analysis['similar_pdf_layouts'] = []
            template_analysis['layout_confidence'] = 0.3

        return template_analysis

    async def _create_layout_query_from_template(self, template_analysis: Dict) -> str:
        """템플릿 분석 결과를 벡터 검색 쿼리로 변환"""
        layout_type = template_analysis['layout_type']
        image_count = template_analysis['image_strategy']
        complexity = template_analysis['complexity_level']
        features = template_analysis['layout_features']

        query_parts = [
            f"{layout_type} magazine layout",
            f"{image_count} images" if image_count > 0 else "text focused",
            f"{complexity} complexity design",
            "grid system" if template_analysis['grid_structure'] else "flexible layout"
        ]

        if 'fixed_height' in features:
            query_parts.append("fixed height sections")
        if 'vertical_layout' in features:
            query_parts.append("vertical column layout")
        if 'gap_spacing' in features:
            query_parts.append("spaced elements design")

        return " ".join(query_parts)

    async def _calculate_layout_confidence(self, template_analysis: Dict, similar_layouts: List[Dict]) -> float:
        """벡터 매칭 기반 레이아웃 신뢰도 계산"""
        if not similar_layouts:
            return 0.3

        avg_similarity = sum(layout.get('score', 0) for layout in similar_layouts) / len(similar_layouts)
        complexity_bonus = 0.2 if template_analysis['complexity_level'] == 'moderate' else 0.1
        image_bonus = 0.1 if template_analysis['image_strategy'] > 0 else 0.05

        confidence = min(avg_similarity + complexity_bonus + image_bonus, 1.0)
        return round(confidence, 2)

    async def _determine_usage_from_vectors(self, similar_layouts: List[Dict]) -> str:
        """벡터 데이터 기반 사용 용도 결정"""
        if not similar_layouts:
            return 'general'

        pdf_sources = [layout.get('pdf_name', '') for layout in similar_layouts]

        if any('travel' in source.lower() for source in pdf_sources):
            return 'travel_focused'
        elif any('culture' in source.lower() for source in pdf_sources):
            return 'culture_focused'
        elif any('lifestyle' in source.lower() for source in pdf_sources):
            return 'lifestyle_focused'
        else:
            return 'editorial'

    async def _calculate_vector_content_match(self, content_vectors: List[Dict], template_vectors: List[Dict]) -> float:
        """콘텐츠 벡터와 템플릿 벡터 간 매칭 점수"""
        if not content_vectors or not template_vectors:
            return 0.0

        content_sources = set(v.get('pdf_name', '') for v in content_vectors)
        template_sources = set(v.get('pdf_name', '') for v in template_vectors)

        common_sources = content_sources.intersection(template_sources)
        if content_sources:
            match_ratio = len(common_sources) / len(content_sources)
            return min(match_ratio, 1.0)

        return 0.0

    async def _analyze_single_template(self, file_path: str, file_name: str) -> Dict:
        """개별 JSX 템플릿 분석 (기존 메서드 유지)"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                jsx_content = await f.read()

            component_name = self._extract_component_name(jsx_content)
            props = self._extract_props(jsx_content)
            styled_components = self._extract_styled_components(jsx_content)
            layout_structure = self._analyze_layout_structure(jsx_content)

            return {
                'file_name': file_name,
                'component_name': component_name,
                'props': props,
                'styled_components': styled_components,
                'layout_type': layout_structure['type'],
                'layout_features': layout_structure['features'],
                'grid_structure': layout_structure['grid'],
                'image_strategy': layout_structure['images'],
                'text_strategy': layout_structure['text'],
                'complexity_level': layout_structure['complexity'],
                'original_jsx': jsx_content,
                'analysis_success': True
            }

        except Exception as e:
            print(f"⚠️ {file_name} 분석 실패: {e}")
            self.result_manager.store_agent_output(
                agent_name="JSXTemplateAnalyzer_SingleTemplate",
                agent_role="개별 템플릿 분석기",
                task_description=f"템플릿 {file_name} 분석",
                final_answer=f"ERROR: {str(e)}",
                reasoning_process=f"템플릿 파일 {file_path} 분석 중 예외 발생",
                error_logs=[{"error": str(e), "file": file_name}]
            )

            return self._create_default_template_analysis(file_name)

    # 기존 메서드들 유지 (변경 없음)
    async def _extract_component_name(self, jsx_content: str) -> str:
        """컴포넌트 이름 추출"""
        match = re.search(r'export const (\w+)', jsx_content)
        return match.group(1) if match else "UnknownComponent"

    async def _extract_props(self, jsx_content: str) -> List[str]:
        """Props 추출"""
        props_match = re.search(r'\(\s*\{\s*([^}]+)\s*\}\s*\)', jsx_content)
        if props_match:
            props_str = props_match.group(1)
            props = [prop.strip() for prop in props_str.split(',')]
            return [prop for prop in props if prop]
        return []

    async def _extract_styled_components(self, jsx_content: str) -> List[Dict]:
        """Styled Components 추출"""
        styled_components = []
        pattern = r'const\s+(\w+)\s*=\s*styled\.(\w+)`([^`]*)`'
        matches = re.findall(pattern, jsx_content, re.DOTALL)

        for comp_name, element_type, css_content in matches:
            styled_components.append({
                'name': comp_name,
                'element': element_type,
                'css': css_content.strip(),
                'properties': await self._extract_css_properties(css_content)
            })

        return styled_components

    async def _extract_css_properties(self, css_content: str) -> Dict:
        """CSS 속성 분석"""
        properties = {
            'display': 'block',
            'position': 'static',
            'grid': False,
            'flex': False,
            'absolute': False
        }

        if 'display: flex' in css_content or 'display: inline-flex' in css_content:
            properties['display'] = 'flex'
            properties['flex'] = True

        if 'display: grid' in css_content:
            properties['display'] = 'grid'
            properties['grid'] = True

        if 'position: absolute' in css_content:
            properties['position'] = 'absolute'
            properties['absolute'] = True

        return properties

    async def _analyze_layout_structure(self, jsx_content: str) -> Dict:
        """레이아웃 구조 분석"""
        image_count = jsx_content.count('styled.img')

        if 'position: absolute' in jsx_content:
            layout_type = 'overlay'
        elif 'display: grid' in jsx_content or 'display: inline-flex' in jsx_content:
            if image_count == 0:
                layout_type = 'text_only'
            elif image_count == 1:
                layout_type = 'hero'
            elif image_count <= 3:
                layout_type = 'grid'
            else:
                layout_type = 'gallery'
        else:
            layout_type = 'simple'

        features = []
        if 'height: 800px' in jsx_content:
            features.append('fixed_height')
        if 'max-width: 1000px' in jsx_content:
            features.append('max_width_constrained')
        if 'gap:' in jsx_content:
            features.append('gap_spacing')
        if 'flex-direction: column' in jsx_content:
            features.append('vertical_layout')

        styled_comp_count = jsx_content.count('const Styled')
        if styled_comp_count <= 3:
            complexity = 'simple'
        elif styled_comp_count <= 6:
            complexity = 'moderate'
        else:
            complexity = 'complex'

        return {
            'type': layout_type,
            'features': features,
            'grid': 'display: grid' in jsx_content,
            'images': image_count,
            'text': jsx_content.count('font-size:'),
            'complexity': complexity
        }

    async def _create_default_template_analysis(self, file_name: str) -> Dict:
        """기본 템플릿 분석 결과"""
        return {
            'file_name': file_name,
            'component_name': 'DefaultComponent',
            'props': ['title', 'subtitle', 'body', 'imageUrl'],
            'styled_components': [],
            'layout_type': 'simple',
            'layout_features': [],
            'grid_structure': False,
            'image_strategy': 1,
            'text_strategy': 3,
            'complexity_level': 'simple',
            'original_jsx': '',
            'vector_matched': False,
            'similar_pdf_layouts': [],
            'layout_confidence': 0.3,
            'recommended_usage': 'general',
            'analysis_success': False,
            'agent_enhanced': False,
            'binding_insights': [],
            'org_insights': []
        }
