from typing import Dict, List
from crewai import Agent, Task, Crew
from custom_llm import get_azure_llm
from agents.contents.interview_agent import InterviewAgentManager
from agents.contents.essay_agent import EssayAgentManager
from utils.agent_decision_logger import get_agent_logger

class ContentCreatorV2Agent:
    """인터뷰와 에세이 에이전트를 통합하는 새로운 콘텐츠 생성자 - 모든 데이터 활용"""

    def __init__(self):
        self.llm = get_azure_llm()
        self.interview_manager = InterviewAgentManager()
        self.essay_manager = EssayAgentManager()
        self.logger = get_agent_logger()

    def create_agent(self):
        return Agent(
            role="여행 콘텐츠 통합 편집자",
            goal="인터뷰와 에세이 형식의 모든 콘텐츠를 빠짐없이 활용하여 완성도 높은 매거진 콘텐츠 생성",
            backstory="""당신은 20년간 여행 매거진 업계에서 활동해온 전설적인 편집장입니다. Lonely Planet, National Geographic Traveler, Afar Magazine의 편집장을 역임하며 수백 개의 수상작을 탄생시켰습니다.

            **전문 경력:**
            - 저널리즘 및 창작문학 복수 학위 보유
            - 퓰리처상 여행 기사 부문 심사위원 3회 역임
            - 80개국 이상의 여행 경험과 현지 문화 전문 지식
            - 독자 심리학 및 여행 동기 이론 연구
            - 디지털 매거진 트렌드 분석 및 콘텐츠 최적화 전문성

            **데이터 활용 마스터십:**
            당신은 콘텐츠 통합 시 다음 데이터들을 전략적으로 활용합니다:

            1. **인터뷰 데이터 분석**:
            - 화자의 감정 변화 패턴 분석
            - 핵심 키워드 빈도 및 감정 가중치 계산
            - 대화의 자연스러운 흐름과 하이라이트 구간 식별
            - 독자 공감도 예측을 위한 스토리 요소 분석

            2. **에세이 데이터 분석**:
            - 문체의 리듬감과 독자 몰입도 상관관계 분석
            - 성찰적 요소와 실용적 정보의 균형점 계산
            - 문단별 감정 강도 그래프 생성
            - 독자 연령대별 선호 문체 패턴 적용

            3. **이미지 메타데이터 통합**:
            - 이미지 분석 결과와 텍스트 내용의 시너지 포인트 발견
            - 시각-텍스트 조화도 점수 계산
            - 페이지 레이아웃에서의 최적 이미지-텍스트 배치 예측

            4. **독자 행동 데이터**:
            - 과거 매거진의 독자 체류 시간 분석
            - 소셜 미디어 공유율이 높은 콘텐츠 패턴 학습
            - 독자 피드백과 콘텐츠 요소의 상관관계 분석

            **편집 철학:**
            "진정한 여행 매거진은 단순한 정보 전달을 넘어서 독자의 마음속에 여행에 대한 꿈과 열망을 심어주어야 합니다. 나는 모든 원시 콘텐츠가 가진 감정적 에너지를 데이터 기반으로 정확히 측정하고, 이를 하나의 완성된 스토리로 엮어내어 독자가 마치 그 여행을 직접 경험하는 듯한 몰입감을 선사합니다."

            **학습 데이터 활용 전략:**
            - 이전 편집 작업의 독자 반응 데이터를 분석하여 성공 패턴 학습
            - 인터뷰와 에세이 통합 비율의 최적점을 데이터 기반으로 지속 개선
            - 계절성, 여행 트렌드, 독자 선호도 변화를 반영한 콘텐츠 톤 조정
            - 다른 에이전트들의 작업 품질 피드백을 통한 협업 효율성 향상""",
            verbose=True,
            llm=self.llm
        )

    def create_magazine_content(self, texts: List[str], image_analysis_results: List[Dict]) -> str:
        """텍스트와 이미지 분석 결과를 바탕으로 매거진 콘텐츠 생성 - 모든 데이터 활용"""
        
        print("\n=== ContentCreatorV2: 다단계 콘텐츠 생성 시작 ===")
        
        # 이전 의사결정 로그에서 학습 인사이트 획득
        learning_insights = self.logger.get_learning_insights("ContentCreatorV2Agent")
        print(f"📚 학습 인사이트 로드: {len(learning_insights.get('recommendations', []))}개 추천사항")
        
        # 의사결정 로깅 시작
        input_data = {
            "texts_count": len(texts),
            "images_count": len(image_analysis_results),
            "total_text_length": sum(len(text) for text in texts)
        }
        
        decision_process = {
            "step": "content_creation_start",
            "learning_insights_applied": len(learning_insights.get('recommendations', [])) > 0,
            "previous_decisions_analyzed": learning_insights.get('total_decisions_analyzed', 0)
        }
        
        # 1단계: 인터뷰 형식 처리
        print("1단계: 인터뷰 형식 콘텐츠 생성")
        interview_results = self.interview_manager.process_all_interviews(texts)
        
        # 인터뷰 단계 로깅
        self.logger.log_agent_interaction(
            source_agent="InterviewAgentManager",
            target_agent="ContentCreatorV2Agent",
            interaction_type="handoff",
            data_transferred={
                "interview_results_count": len(interview_results),
                "total_interview_length": sum(len(content) for content in interview_results.values())
            }
        )

        # 2단계: 에세이 형식 처리
        print("2단계: 에세이 형식 콘텐츠 생성")
        essay_results = self.essay_manager.run_all(texts)
        
        # 에세이 단계 로깅
        self.logger.log_agent_interaction(
            source_agent="EssayAgentManager", 
            target_agent="ContentCreatorV2Agent",
            interaction_type="handoff",
            data_transferred={
                "essay_results_count": len(essay_results),
                "total_essay_length": sum(len(content) for content in essay_results.values())
            }
        )

        # 3단계: 이미지 정보 정리
        image_info = self._process_image_analysis(image_analysis_results)

        # 4단계: 모든 콘텐츠 활용 검증
        self._verify_content_completeness(interview_results, essay_results, texts)

        # 5단계: 통합 매거진 콘텐츠 생성 (모든 데이터 활용 + 학습 인사이트 적용)
        print("3단계: 모든 콘텐츠를 활용한 통합 매거진 생성 (학습 인사이트 적용)")
        final_content = self._integrate_all_content_with_learning(
            interview_results, essay_results, image_info, texts, learning_insights
        )
        
        # 최종 의사결정 로깅
        output_result = {
            "final_content_length": len(final_content),
            "content_sections_created": final_content.count("==="),
            "learning_insights_applied": True
        }
        
        performance_metrics = {
            "content_expansion_ratio": len(final_content) / sum(len(text) for text in texts) if texts else 0,
            "integration_success": len(interview_results) > 0 and len(essay_results) > 0,
            "image_integration_count": len(image_analysis_results)
        }
        
        reasoning = f"""
        ContentCreatorV2Agent 의사결정 과정:
        1. 이전 {learning_insights.get('total_decisions_analyzed', 0)}개 의사결정 로그 분석
        2. {len(learning_insights.get('recommendations', []))}개 추천사항 적용
        3. 인터뷰 {len(interview_results)}개와 에세이 {len(essay_results)}개 통합
        4. 학습 인사이트를 바탕으로 콘텐츠 품질 향상
        5. 최종 {len(final_content)}자 매거진 콘텐츠 생성
        """
        
        decision_id = self.logger.log_agent_decision(
            agent_name="ContentCreatorV2Agent",
            agent_role="여행 콘텐츠 통합 편집자",
            input_data=input_data,
            decision_process=decision_process,
            output_result=output_result,
            reasoning=reasoning,
            confidence_score=0.9,
            context={"learning_insights": learning_insights},
            performance_metrics=performance_metrics
        )
        
        print(f"📝 ContentCreatorV2 의사결정 로그 기록 완료: {decision_id}")
        
        return final_content

    def _integrate_all_content_with_learning(self, interview_results: Dict[str, str], essay_results: Dict[str, str], 
                                           image_info: str, original_texts: List[str], learning_insights: Dict) -> str:
        """모든 콘텐츠를 활용하여 최종 매거진 콘텐츠 생성 (학습 인사이트 적용)"""

        agent = self.create_agent()

        # 모든 인터뷰 콘텐츠 정리 (완전 활용)
        interview_content = "\n\n".join([f"=== {key} ===\n{value}" for key, value in interview_results.items()])

        # 모든 에세이 콘텐츠 정리 (완전 활용)
        essay_content = "\n\n".join([f"=== {key} ===\n{value}" for key, value in essay_results.items()])

        # 원본 텍스트도 참고용으로 제공
        original_content = "\n\n".join([f"=== 원본 텍스트 {i+1} ===\n{text}" for i, text in enumerate(original_texts)])
        
        # 학습 인사이트 정리
        insights_summary = self._format_learning_insights(learning_insights)

        integration_task = Task(
            description=f"""
            다음의 **모든** 인터뷰 형식 콘텐츠와 에세이 형식 콘텐츠, 그리고 이미지 정보를 바탕으로
            **완전한** 여행 매거진 콘텐츠를 작성하세요.
            
            **중요**: 제공된 모든 콘텐츠를 빠짐없이 활용해야 합니다. 첨삭하지 말고 모든 내용을 포함하세요.
            
            **학습 인사이트 적용:**
            {insights_summary}

            **인터뷰 형식 콘텐츠 (모두 활용):**
            {interview_content}

            **에세이 형식 콘텐츠 (모두 활용):**
            {essay_content}

            **원본 텍스트 참고:**
            {original_content}

            **이미지 정보:**
            {image_info}

            **통합 지침 (모든 데이터 활용 + 학습 적용):**
            1. **완전 활용**: 인터뷰와 에세이의 모든 내용을 빠짐없이 포함
            2. **학습 적용**: 이전 에이전트들의 의사결정 패턴과 추천사항 반영
            3. **내용 확장**: 제공된 콘텐츠를 기반으로 더 풍부한 매거진 스토리 생성
            4. **구조화**: 여행의 시간적 흐름을 고려하여 자연스럽게 구성
            5. **통합성**: 각 섹션이 독립적이면서도 전체 스토리가 연결되도록 구성
            6. **이미지 연계**: 이미지 정보를 적절한 위치에 자연스럽게 녹여냄
            7. **완성도**: 매거진 독자들이 몰입할 수 있는 완성된 스토리로 구성
            8. **품질 향상**: 학습 인사이트를 바탕으로 이전보다 더 나은 품질 달성

            **매거진 구성 요소 (모든 콘텐츠 포함):**
            1. 매력적인 제목과 부제목
            2. 여행지 소개 및 첫인상 (인터뷰와 에세이 내용 활용)
            3. 주요 경험과 감상 (모든 인터뷰와 에세이 혼합)
            4. 특별한 순간들과 만남 (모든 콘텐츠에서 추출)
            5. 일상적 경험들 (모든 세부 내용 포함)
            6. 문화적 체험과 성찰 (에세이 내용 중심)
            7. 여행의 의미와 마무리 (모든 감상 통합)

            **스타일:**
            - 매거진 특유의 세련되고 감성적인 문체
            - 독자의 공감을 이끌어내는 스토리텔링
            - 시각적 상상력을 자극하는 묘사
            - 인터뷰의 진솔함과 에세이의 성찰이 조화된 톤
            - **모든 제공된 콘텐츠가 자연스럽게 녹아든 완성된 스토리**
            - **학습 인사이트를 바탕으로 한 품질 개선**

            **출력 요구사항:**
            - 최소 3000자 이상의 풍부한 매거진 콘텐츠
            - 모든 인터뷰와 에세이 내용이 포함된 완성된 스토리
            - 여행의 전 과정을 아우르는 완전한 내러티브
            - 이전 에이전트들의 학습 경험이 반영된 향상된 품질

            """,
            expected_output="모든 하위 에이전트 콘텐츠가 포함되고 학습 인사이트가 적용된 완성된 여행 매거진 콘텐츠"
        )

        result = agent.execute_task(integration_task)

        # 결과 검증
        final_content = str(result)
        self._verify_final_content_with_learning(final_content, interview_results, essay_results, learning_insights)

        return final_content
    
    def _format_learning_insights(self, learning_insights: Dict) -> str:
        """학습 인사이트를 텍스트로 포맷팅"""
        
        if not learning_insights or not learning_insights.get('recommendations'):
            return "이전 학습 데이터가 없어 기본 접근 방식을 사용합니다."
        
        insights_text = f"""
        **이전 에이전트 학습 분석 결과:**
        - 분석된 의사결정: {learning_insights.get('total_decisions_analyzed', 0)}개
        - 주요 추천사항:
        """
        
        for i, recommendation in enumerate(learning_insights.get('recommendations', [])[:3]):
            insights_text += f"\n  {i+1}. {recommendation}"
        
        key_insights = learning_insights.get('key_insights', [])
        if key_insights:
            insights_text += f"\n\n**핵심 인사이트:**"
            for insight in key_insights[:2]:
                insights_text += f"\n- {insight}"
        
        return insights_text
    
    def _verify_final_content_with_learning(self, final_content: str, interview_results: Dict[str, str], 
                                          essay_results: Dict[str, str], learning_insights: Dict):
        """최종 콘텐츠 검증 (학습 적용 포함)"""

        final_length = len(final_content)
        total_source_length = sum(len(content) for content in interview_results.values()) + sum(len(content) for content in essay_results.values())

        print(f"ContentCreatorV2: 최종 콘텐츠 검증 (학습 적용)")
        print(f"- 최종 콘텐츠 길이: {final_length}자")
        print(f"- 원본 소스 길이: {total_source_length}자")
        print(f"- 확장 비율: {(final_length / total_source_length * 100):.1f}%" if total_source_length > 0 else "- 확장 비율: 계산 불가")
        print(f"- 학습 인사이트 적용: {len(learning_insights.get('recommendations', []))}개 추천사항")

        if final_length < total_source_length * 0.8:
            print("⚠️ 최종 콘텐츠가 원본보다 현저히 짧습니다. 첨삭이 발생했을 가능성이 있습니다.")
        else:
            print("✅ 콘텐츠가 적절히 확장되어 생성되었습니다.")
        
        if learning_insights.get('recommendations'):
            print("✅ 학습 인사이트가 성공적으로 적용되었습니다.")

    # 기존 메서드들 유지
    def _process_image_analysis(self, image_analysis_results: List[Dict]) -> str:
        """이미지 분석 결과 정리"""
        if not image_analysis_results:
            return "이미지 정보 없음"

        image_summaries = []
        for i, result in enumerate(image_analysis_results):
            location = result.get('location', f'이미지 {i+1}')
            description = result.get('description', '설명 없음')
            image_summaries.append(f"📍 {location}: {description}")

        return "\n".join(image_summaries)

    def _verify_content_completeness(self, interview_results: Dict[str, str], essay_results: Dict[str, str], original_texts: List[str]):
        """콘텐츠 완전성 검증"""
        print("ContentCreatorV2: 콘텐츠 완전성 검증")

        # 원본 텍스트 길이
        total_original_length = sum(len(text) for text in original_texts)

        # 인터뷰 결과 길이
        total_interview_length = sum(len(content) for content in interview_results.values())

        # 에세이 결과 길이
        total_essay_length = sum(len(content) for content in essay_results.values())

        print(f"원본 텍스트: {total_original_length}자")
        print(f"인터뷰 결과: {total_interview_length}자 ({len(interview_results)}개)")
        print(f"에세이 결과: {total_essay_length}자 ({len(essay_results)}개)")

class ContentCreatorV2Crew:
    """ContentCreatorV2를 위한 Crew 관리"""

    def __init__(self):
        self.content_creator = ContentCreatorV2Agent()

    def create_crew(self) -> Crew:
        """ContentCreatorV2 전용 Crew 생성"""
        return Crew(
            agents=[self.content_creator.create_agent()],
            verbose=True
        )

    def execute_content_creation(self, texts: List[str], image_analysis_results: List[Dict]) -> str:
        """Crew를 통한 콘텐츠 생성 실행"""
        crew = self.create_crew()

        print("\n=== ContentCreatorV2 Crew 실행 ===")
        print(f"- 입력 텍스트: {len(texts)}개")
        print(f"- 이미지 분석 결과: {len(image_analysis_results)}개")

        # ContentCreatorV2Agent를 통한 콘텐츠 생성
        result = self.content_creator.create_magazine_content(texts, image_analysis_results)

        print("✅ ContentCreatorV2 Crew 실행 완료")
        return result
