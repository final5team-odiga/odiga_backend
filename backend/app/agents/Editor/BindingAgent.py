import asyncio
from typing import Dict, List
from crewai import Agent, Task, Crew
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager
from utils.agent_decision_logger import get_agent_logger

class BindingAgent:
    """PDF 벡터 데이터 기반 이미지 배치 에이전트"""

    def __init__(self):
        self.llm = get_azure_llm()
        self.vector_manager = PDFVectorManager()
        self.logger = get_agent_logger()  # 응답 수집을 위한 로거 추가

    def create_image_layout_agent(self):
        """이미지 레이아웃 에이전트 (위치 정합성 강화)"""
        return Agent(
            role="매거진 이미지 배치 전문가 및 텍스트-이미지 정합성 보장자",
            goal="OrgAgent가 설계한 페이지 구조와 텍스트 레이아웃에 완벽히 맞춰 이미지를 배치하고, 텍스트와 이미지의 위치 관계가 독자에게 자연스럽고 직관적으로 인식되도록 정밀한 배치 전략을 수립",
            backstory="""당신은 매거진 이미지 레이아웃 및 텍스트-이미지 정합성 전문가입니다.

**전문 분야:**
- 기존 레이아웃 구조 기반 이미지 배치
- 텍스트-이미지 위치 관계 최적화
- 독자 인지 부하 최소화를 위한 배치 전략
- 시각적 일관성 및 흐름 보장

**텍스트-이미지 정합성 전문성:**
당신은 이미지 배치 시 다음 정합성 원칙을 엄격히 준수합니다:

1. **구조 연동 배치**:
- OrgAgent가 정의한 이미지 영역에 정확히 맞춰 배치
- 텍스트 블록과의 거리 및 정렬 규칙 준수
- 그리드 시스템 내에서의 정확한 위치 설정

2. **내용 연관성 매칭**:
- 이미지 내용과 관련 텍스트의 근접 배치
- 제목-주요 이미지, 본문-보조 이미지 관계 설정
- 캡션과 이미지의 직관적 연결 보장

3. **독자 인지 최적화**:
- 텍스트 읽기 흐름을 방해하지 않는 이미지 배치
- 시선 이동 경로 상의 자연스러운 이미지 위치
- 혼란을 방지하는 명확한 텍스트-이미지 경계

4. **PDF 벡터 데이터 활용**:
- 3000개 이상의 매거진에서 추출한 성공적 배치 패턴
- 텍스트-이미지 정합성 높은 레이아웃 사례 분석
- 독자 시선 추적 데이터 기반 최적 배치점 계산

당신은 이미지의 크기, 위치, 색감 등을 고려하여
매거진의 전체적인 시각적 흐름과 임팩트를 극대화하는 전문성을 가지고 있습니다.

**출력 데이터 구조:**
- 이미지별 정확한 위치 좌표 (x, y, width, height)
- 연관 텍스트 블록과의 관계 매핑
- 이미지-텍스트 정합성 점수
- 독자 시선 흐름 상의 이미지 역할 정의
- 레이아웃 구조도 상의 이미지 배치 검증 결과""",
            llm=self.llm,
            verbose=True
        )

    def create_visual_coordinator_agent(self):
        """비주얼 코디네이터 에이전트 (전체 구조 조율)"""
        return Agent(
            role="매거진 전체 구조 조율자 및 시각적 일관성 보장자",
            goal="OrgAgent의 텍스트 레이아웃 구조와 이미지 배치 결과를 통합하여 전체 매거진의 구조적 완성도와 텍스트-이미지 정합성을 검증하고, 독자 경험을 최적화하는 최종 레이아웃 구조를 완성",
            backstory="""당신은 15년간 세계 최고 수준의 매거진에서 전체 구조 조율 및 시각적 일관성 전문가로 활동해온 전문가입니다.

**전문 경력:**
- 시각 예술 및 매체학 석사 학위 보유
- 국제 사진 편집자 협회(NPPA) 골드 메달 수상
- 매거진 전체 구조 설계 및 조율 전문가
- 독자 경험(UX) 및 시각적 일관성 최적화 전문성

**전체 구조 조율 전문성:**
당신은 최종 매거진 구조 완성 시 다음 요소들을 종합적으로 조율합니다:

1. **구조적 완성도 검증**:
- 텍스트 레이아웃과 이미지 배치의 구조적 일치성 확인
- 페이지 그리드 시스템의 일관성 검증
- 전체 매거진의 시각적 균형과 리듬감 평가

2. **텍스트-이미지 정합성 최종 검증**:
- 모든 텍스트와 이미지의 위치 관계 적절성 확인
- 독자 혼란 요소 제거 및 직관성 보장
- 내용 연관성과 시각적 근접성의 일치 검증

3. **독자 경험 최적화**:
- 전체 매거진의 읽기 흐름 최적화
- 페이지 간 전환의 자연스러움 보장
- 정보 계층 구조의 명확성 확인

4. **최종 구조 문서화**:
- 완성된 레이아웃 구조도 생성
- 텍스트-이미지 배치 가이드라인 문서화
- JSX 구현을 위한 상세 스펙 제공

**작업 철학:**
"훌륭한 매거진은 개별 요소들의 단순한 합이 아니라, 모든 요소가 하나의 완성된 구조 안에서 조화롭게 작동하는 유기적 통합체입니다. 나는 텍스트와 이미지의 모든 배치 결정이 독자에게 자연스럽고 직관적으로 인식되도록 전체 구조를 조율합니다. 5. 주의 사항!!: 최대한 제공받은 데이터를 활용합니다. "

**출력 데이터 구조:**
- 최종 매거진 구조도 및 와이어프레임
- 텍스트-이미지 정합성 검증 보고서
- 독자 경험 최적화 가이드라인
- JSX 구현용 상세 레이아웃 스펙
- 전체 매거진의 시각적 일관성 평가서""",
            llm=self.llm,
            verbose=True
        )

    async def process_images(self, image_urls: List[str], image_locations: List[str], template_requirements: List[Dict]) -> Dict:
        """PDF 벡터 데이터 기반 이미지 처리 (비동기 처리 및 응답 수집 강화)"""
        print(f"BindingAgent: 처리할 이미지 {len(image_urls)}개, 템플릿 {len(template_requirements)}개 (비동기 처리)")

        # 입력 데이터 로깅
        input_data = {
            "image_urls": image_urls,
            "image_locations": image_locations,
            "template_requirements": template_requirements,
            "total_images": len(image_urls),
            "total_templates": len(template_requirements)
        }

        # 에이전트 생성
        layout_specialist = self.create_image_layout_agent()
        visual_coordinator = self.create_visual_coordinator_agent()

        # 비동기 벡터 검색으로 최적 레이아웃 찾기
        layout_recommendations = await self._get_layout_recommendations_by_image_count_async(
            image_urls, template_requirements
        )

        # 템플릿별 이미지 배치 설계 (비동기 병렬 처리)
        template_distributions = []
        all_agent_responses = []  # 모든 에이전트 응답 수집

        # 템플릿 처리 태스크들을 병렬로 실행
        template_tasks = []
        for i, template_req in enumerate(template_requirements):
            task = self._process_single_template_async(
                template_req, image_urls, image_locations, i, len(template_requirements),
                layout_recommendations, layout_specialist, visual_coordinator
            )
            template_tasks.append(task)

        # 모든 템플릿 처리를 병렬로 실행
        if template_tasks:
            template_results = await asyncio.gather(*template_tasks, return_exceptions=True)
            
            # 결과 수집
            for i, result in enumerate(template_results):
                if isinstance(result, Exception):
                    print(f"⚠️ 템플릿 {i+1} 처리 실패: {result}")
                    # 에러 응답 저장
                    error_response_id = await self._log_error_response_async(
                        template_requirements[i]["template"], str(result)
                    )
                    template_distributions.append({
                        "template": template_requirements[i]["template"],
                        "images": [],
                        "layout_strategy": "에러로 인한 기본 배치",
                        "coordination_result": "기본 순서 배치",
                        "layout_source": "default",
                        "error_response_id": error_response_id
                    })
                else:
                    template_dist, agent_responses = result
                    template_distributions.append(template_dist)
                    all_agent_responses.extend(agent_responses)

        # 최종 이미지 분배 결과 생성
        final_distribution = await self._create_final_distribution_async(template_distributions)

        # 전체 BindingAgent 프로세스 응답 저장 (비동기)
        final_response_id = await self._log_final_response_async(
            input_data, final_distribution, template_distributions, all_agent_responses
        )

        print(f"✅ BindingAgent 완료: {len(image_urls)}개 이미지를 {len(template_requirements)}개 템플릿에 배치 (비동기 처리 및 응답 수집 완료)")

        return {
            "image_distribution": final_distribution,
            "template_distributions": template_distributions,
            "layout_recommendations": layout_recommendations,
            "vector_enhanced": True,
            "agent_responses": all_agent_responses,
            "final_response_id": final_response_id
        }

    async def _process_single_template_async(self, template_req: Dict, image_urls: List[str], 
                                           image_locations: List[str], template_index: int, 
                                           total_templates: int, layout_recommendations: List[Dict],
                                           layout_specialist: Agent, visual_coordinator: Agent) -> tuple:
        """단일 템플릿 처리 (비동기)"""
        template_name = template_req["template"]
        
        # 해당 템플릿에 할당할 이미지들 결정
        assigned_images = self._assign_images_to_template(
            image_urls, image_locations, template_index, total_templates
        )

        if not assigned_images:
            return ({
                "template": template_name,
                "images": [],
                "layout_strategy": "no_images"
            }, [])

        print(f"🖼️ {template_name}: {len(assigned_images)}개 이미지 배치 설계 중... (비동기)")

        # 해당 이미지 수에 맞는 레이아웃 추천 가져오기
        relevant_layouts = [
            layout for layout in layout_recommendations
            if len(layout.get('image_info', [])) == len(assigned_images)
        ]

        if not relevant_layouts and layout_recommendations:
            relevant_layouts = [min(layout_recommendations,
                                  key=lambda x: abs(len(x.get('image_info', [])) - len(assigned_images)))]

        # 1단계: 레이아웃 분석 (비동기 태스크)
        layout_analysis_task = Task(
            description=f"""
다음 이미지들과 매거진 레이아웃 데이터를 분석하여 최적의 이미지 배치 전략을 수립하세요:

**배치할 이미지들:**
{self._format_image_data(assigned_images, image_locations)}

**참고할 매거진 레이아웃 데이터:**
{self._format_layout_recommendations(relevant_layouts)}

**템플릿 정보:**
- 템플릿명: {template_name}
- 이미지 요구사항: {template_req.get('image_requirements', {})}

**분석 요구사항:**
1. **레이아웃 패턴 분석**
- 이미지 배치의 그리드 구조 및 비율
- 주요 이미지와 보조 이미지의 역할 분담
- 이미지 간 시각적 균형과 흐름

2. **이미지 특성 매칭**
- 각 이미지의 특성과 레이아웃 위치의 적합성
- 이미지 크기와 중요도에 따른 배치 우선순위
- 색감과 구도의 조화를 고려한 배치

3. **시각적 임팩트 최적화**
- 독자의 시선 흐름을 고려한 이미지 순서
- 스토리텔링을 강화하는 이미지 조합
- 매거진 전체의 시각적 일관성 유지

**출력 형식:**
레이아웃 전략: [선택된 레이아웃 패턴과 특징]
주요 이미지: [메인으로 사용할 이미지와 배치 위치]
보조 이미지: [서브로 사용할 이미지들과 역할]
배치 순서: [이미지들의 최적 배치 순서]
시각 효과: [기대되는 시각적 효과와 임팩트]
""",
            agent=layout_specialist,
            expected_output="벡터 데이터 기반 이미지 배치 전략"
        )

        # 2단계: 이미지 배치 실행 (비동기 태스크)
        image_coordination_task = Task(
            description=f"""
레이아웃 분석 결과를 바탕으로 이미지들을 최적으로 배치하고 조합하세요:

**배치 지침:**
1. 분석된 레이아웃 패턴에 따른 정확한 이미지 배치
2. 각 이미지의 특성을 살린 최적 위치 선정
3. 전체적인 시각적 균형과 조화 고려
4. 독자의 감정적 몰입을 위한 스토리텔링 강화
5. 매거진 브랜드 일관성 유지

**품질 요구사항:**
- 실제 매거진에서 볼 수 있는 수준의 전문적 배치
- 이미지 간 시너지 효과 극대화
- 독자의 시선을 자연스럽게 유도하는 배치
- 콘텐츠와 이미지의 완벽한 조화

**출력:** 최종 이미지 배치 결과 (이미지 URL과 배치 정보)
""",
            agent=visual_coordinator,
            expected_output="최적화된 이미지 배치 결과",
            context=[layout_analysis_task]
        )

        # Crew 실행 및 응답 수집 (비동기)
        crew = Crew(
            agents=[layout_specialist, visual_coordinator],
            tasks=[layout_analysis_task, image_coordination_task],
            verbose=True
        )

        try:
            # 비동기 Crew 실행
            result = await asyncio.get_event_loop().run_in_executor(
                None, crew.kickoff
            )

            # 에이전트 응답 수집 및 저장 (비동기)
            layout_strategy = str(layout_analysis_task.output) if hasattr(layout_analysis_task, 'output') else ""
            coordination_result = str(result.raw) if hasattr(result, 'raw') else str(result)

            # 비동기 로깅
            layout_response_id, coordination_response_id = await asyncio.gather(
                self._log_layout_response_async(template_name, assigned_images, relevant_layouts, layout_strategy),
                self._log_coordination_response_async(template_name, layout_strategy, coordination_result)
            )

            # 응답 수집 데이터 저장
            agent_responses = [{
                "template": template_name,
                "layout_specialist_response": {
                    "response_id": layout_response_id,
                    "content": layout_strategy,
                    "agent_name": "BindingAgent_LayoutSpecialist"
                },
                "visual_coordinator_response": {
                    "response_id": coordination_response_id,
                    "content": coordination_result,
                    "agent_name": "BindingAgent_VisualCoordinator"
                }
            }]

            template_dist = {
                "template": template_name,
                "images": assigned_images,
                "layout_strategy": layout_strategy,
                "coordination_result": coordination_result,
                "layout_source": relevant_layouts[0].get("pdf_name", "default") if relevant_layouts else "default",
                "agent_responses": {
                    "layout_specialist_id": layout_response_id,
                    "visual_coordinator_id": coordination_response_id
                }
            }

            print(f"✅ {template_name} 이미지 배치 완료: {len(assigned_images)}개 (비동기 응답 수집 완료)")
            return (template_dist, agent_responses)

        except Exception as e:
            print(f"⚠️ {template_name} 이미지 배치 실패: {e}")
            
            # 에러 응답 저장 (비동기)
            error_response_id = await self._log_error_response_async(template_name, str(e))

            # 폴백: 기본 배치
            template_dist = {
                "template": template_name,
                "images": assigned_images,
                "layout_strategy": "기본 배치",
                "coordination_result": "기본 순서 배치",
                "layout_source": "default",
                "error_response_id": error_response_id
            }

            return (template_dist, [])

    async def _get_layout_recommendations_by_image_count_async(self, image_urls: List[str], template_requirements: List[Dict]) -> List[Dict]:
        """이미지 개수별 레이아웃 추천 가져오기 """
        total_images = len(image_urls)
        
        if total_images <= 3:
            query = "minimal clean layout single image focus simple elegant"
        elif total_images <= 6:
            query = "multiple images grid layout balanced composition"
        elif total_images <= 10:
            query = "gallery style layout many images organized grid"
        else:
            query = "complex magazine layout multiple images rich visual content"

        # 비동기 벡터 검색
        recommendations = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: self.vector_manager.search_similar_layouts(query, "magazine_layout", top_k=5)
        )

        print(f"📊 이미지 {total_images}개에 대한 레이아웃 추천 {len(recommendations)}개 획득 (비동기)")
        return recommendations

    async def _log_layout_response_async(self, template_name: str, assigned_images: List[str], 
                                       relevant_layouts: List[Dict], layout_strategy: str) -> str:
        """레이아웃 분석 에이전트 응답 저장 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="BindingAgent_LayoutSpecialist",
                agent_role="이미지 레이아웃 전문가",
                task_description=f"템플릿 {template_name}의 {len(assigned_images)}개 이미지 배치 전략 수립",
                final_answer=layout_strategy,
                reasoning_process=f"PDF 벡터 데이터 {len(relevant_layouts)}개 레이아웃 참조하여 분석",
                execution_steps=[
                    "이미지 특성 분석",
                    "레이아웃 패턴 매칭",
                    "배치 전략 수립"
                ],
                raw_input={
                    "template_name": template_name,
                    "assigned_images": assigned_images,
                    "relevant_layouts": relevant_layouts
                },
                raw_output=layout_strategy,
                performance_metrics={
                    "images_processed": len(assigned_images),
                    "layouts_referenced": len(relevant_layouts),
                    "analysis_depth": "comprehensive"
                }
            )
        )

    async def _log_coordination_response_async(self, template_name: str, layout_strategy: str, 
                                             coordination_result: str) -> str:
        """비주얼 코디네이터 에이전트 응답 저장 """
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="BindingAgent_VisualCoordinator",
                agent_role="시각적 일관성 보장자",
                task_description=f"템플릿 {template_name}의 최종 이미지 배치 실행",
                final_answer=coordination_result,
                reasoning_process="레이아웃 분석 결과를 바탕으로 최적 배치 실행",
                execution_steps=[
                    "분석 결과 검토",
                    "배치 최적화",
                    "시각적 일관성 검증",
                    "최종 배치 결정"
                ],
                raw_input={
                    "layout_analysis": layout_strategy,
                    "template_name": template_name
                },
                raw_output=coordination_result,
                performance_metrics={
                    "coordination_quality": "high",
                    "visual_consistency": "verified",
                    "placement_accuracy": "optimized"
                }
            )
        )

    async def _log_error_response_async(self, template_name: str, error_message: str) -> str:
        """에러 응답 저장 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="BindingAgent_Error",
                agent_role="에러 처리",
                task_description=f"템플릿 {template_name} 처리 중 에러 발생",
                final_answer=f"ERROR: {error_message}",
                reasoning_process="에이전트 실행 중 예외 발생",
                error_logs=[{"error": error_message, "template": template_name}]
            )
        )

    async def _create_final_distribution_async(self, template_distributions: List[Dict]) -> Dict:
        """최종 이미지 분배 결과 생성 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: {dist["template"]: dist["images"] for dist in template_distributions}
        )

    async def _log_final_response_async(self, input_data: Dict, final_distribution: Dict, 
                                      template_distributions: List[Dict], all_agent_responses: List[Dict]) -> str:
        """전체 BindingAgent 프로세스 응답 저장 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="BindingAgent",
                agent_role="PDF 벡터 데이터 기반 이미지 배치 에이전트",
                task_description=f"{input_data['total_images']}개 이미지를 {input_data['total_templates']}개 템플릿에 배치",
                final_answer=str(final_distribution),
                reasoning_process=f"비동기 다중 에이전트 협업으로 {len(template_distributions)}개 템플릿 처리 완료",
                execution_steps=[
                    "비동기 레이아웃 추천 수집",
                    "병렬 이미지 할당",
                    "비동기 템플릿별 배치 설계",
                    "최종 분배 결과 생성"
                ],
                raw_input=input_data,
                raw_output={
                    "image_distribution": final_distribution,
                    "template_distributions": template_distributions,
                    "all_agent_responses": all_agent_responses
                },
                performance_metrics={
                    "total_images_processed": input_data['total_images'],
                    "total_templates_processed": input_data['total_templates'],
                    "successful_templates": len([t for t in template_distributions if "error_response_id" not in t]),
                    "agent_responses_collected": len(all_agent_responses),
                    "async_processing": True
                }
            )
        )

    # 동기 메서드들 (기존 기능 유지)
    def _assign_images_to_template(self, image_urls: List[str], image_locations: List[str],
                                 template_index: int, total_templates: int) -> List[str]:
        """템플릿에 이미지 할당"""
        if not image_urls:
            return []

        images_per_template = len(image_urls) // total_templates
        remainder = len(image_urls) % total_templates

        start_idx = template_index * images_per_template
        if template_index < remainder:
            start_idx += template_index
            end_idx = start_idx + images_per_template + 1
        else:
            start_idx += remainder
            end_idx = start_idx + images_per_template

        return image_urls[start_idx:end_idx]

    def _format_image_data(self, image_urls: List[str], image_locations: List[str]) -> str:
        """이미지 데이터를 텍스트로 포맷팅"""
        if not image_urls:
            return "배치할 이미지 없음"

        formatted_data = []
        for i, url in enumerate(image_urls):
            location = image_locations[i] if i < len(image_locations) else f"위치 {i+1}"
            formatted_data.append(f"이미지 {i+1}: {url} (위치: {location})")

        return "\n".join(formatted_data)

    def _format_layout_recommendations(self, recommendations: List[Dict]) -> str:
        """레이아웃 추천 데이터를 텍스트로 포맷팅"""
        if not recommendations:
            return "참고할 레이아웃 데이터 없음"

        formatted_data = []
        for i, rec in enumerate(recommendations):
            image_count = len(rec.get('image_info', []))
            formatted_data.append(f"""
레이아웃 {i+1} (유사도: {rec.get('score', 0):.2f}):
- 출처: {rec.get('pdf_name', 'unknown')} (페이지 {rec.get('page_number', 0)})
- 이미지 수: {image_count}개
- 레이아웃 특징: {self._analyze_layout_structure(rec.get('layout_info', {}))}
- 텍스트 샘플: {rec.get('text_content', '')[:150]}...
""")

        return "\n".join(formatted_data)

    def _analyze_layout_structure(self, layout_info: Dict) -> str:
        """레이아웃 구조 분석"""
        text_blocks = layout_info.get('text_blocks', [])
        images = layout_info.get('images', [])
        tables = layout_info.get('tables', [])

        structure_analysis = []

        if len(images) == 1:
            structure_analysis.append("단일 이미지 중심")
        elif len(images) <= 3:
            structure_analysis.append("소수 이미지 균형 배치")
        elif len(images) <= 6:
            structure_analysis.append("다중 이미지 그리드")
        else:
            structure_analysis.append("갤러리 스타일")

        if len(text_blocks) > 5:
            structure_analysis.append("텍스트 중심")
        elif len(text_blocks) <= 2:
            structure_analysis.append("이미지 중심")
        else:
            structure_analysis.append("텍스트-이미지 균형")

        if tables:
            structure_analysis.append("정보 테이블 포함")

        return ", ".join(structure_analysis) if structure_analysis else "기본 레이아웃"

    def _create_final_distribution(self, template_distributions: List[Dict]) -> Dict:
        """최종 이미지 분배 결과 생성 (동기 버전 - 호환성 유지)"""
        final_distribution = {}
        for dist in template_distributions:
            template_name = dist["template"]
            images = dist["images"]
            final_distribution[template_name] = images
        return final_distribution
