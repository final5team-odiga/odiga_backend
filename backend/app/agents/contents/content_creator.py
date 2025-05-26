from typing import Dict, List
from crewai import Agent, Task, Crew
from custom_llm import get_azure_llm
from agents.contents.interview_agent import InterviewAgentManager
from agents.contents.essay_agent import EssayAgentManager

class ContentCreatorV2Agent:
    """인터뷰와 에세이 에이전트를 통합하는 새로운 콘텐츠 생성자 - 모든 데이터 활용"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        self.interview_manager = InterviewAgentManager()
        self.essay_manager = EssayAgentManager()
        
    def create_agent(self):
        return Agent(
            role="여행 콘텐츠 통합 편집자",
            goal="인터뷰와 에세이 형식의 모든 콘텐츠를 빠짐없이 활용하여 완성도 높은 매거진 콘텐츠 생성",
            backstory="""당신은 여행 매거진의 편집장입니다. 
            인터뷰 형식의 생생한 대화와 에세이 형식의 깊이 있는 성찰을 조화롭게 엮어
            독자들에게 감동을 주는 완성된 여행 스토리를 만들어냅니다.
            특히 하위 에이전트들이 생성한 모든 콘텐츠를 빠짐없이 활용하여 
            풍부하고 완성도 높은 매거진을 만드는 전문성을 가지고 있습니다.""",
            verbose=True,
            llm=self.llm
        )

    def create_magazine_content(self, texts: List[str], image_analysis_results: List[Dict]) -> str:
        """텍스트와 이미지 분석 결과를 바탕으로 매거진 콘텐츠 생성 - 모든 데이터 활용"""
        
        print("\n=== ContentCreatorV2: 다단계 콘텐츠 생성 시작 ===")
        
        # 1단계: 인터뷰 형식 처리
        print("1단계: 인터뷰 형식 콘텐츠 생성")
        interview_results = self.interview_manager.process_all_interviews(texts)
        
        # 2단계: 에세이 형식 처리  
        print("2단계: 에세이 형식 콘텐츠 생성")
        essay_results = self.essay_manager.run_all(texts)
        
        # 3단계: 이미지 정보 정리
        image_info = self._process_image_analysis(image_analysis_results)
        
        # 4단계: 모든 콘텐츠 활용 검증
        self._verify_content_completeness(interview_results, essay_results, texts)
        
        # 5단계: 통합 매거진 콘텐츠 생성 (모든 데이터 활용)
        print("3단계: 모든 콘텐츠를 활용한 통합 매거진 생성")
        final_content = self._integrate_all_content(interview_results, essay_results, image_info, texts)
        
        return final_content

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

    def _integrate_all_content(self, interview_results: Dict[str, str], essay_results: Dict[str, str], image_info: str, original_texts: List[str]) -> str:
        """모든 콘텐츠를 활용하여 최종 매거진 콘텐츠 생성"""
        agent = self.create_agent()
        
        # 모든 인터뷰 콘텐츠 정리 (완전 활용)
        interview_content = "\n\n".join([f"=== {key} ===\n{value}" for key, value in interview_results.items()])
        
        # 모든 에세이 콘텐츠 정리 (완전 활용)
        essay_content = "\n\n".join([f"=== {key} ===\n{value}" for key, value in essay_results.items()])
        
        # 원본 텍스트도 참고용으로 제공
        original_content = "\n\n".join([f"=== 원본 텍스트 {i+1} ===\n{text}" for i, text in enumerate(original_texts)])
        
        integration_task = Task(
            description=f"""
            다음의 **모든** 인터뷰 형식 콘텐츠와 에세이 형식 콘텐츠, 그리고 이미지 정보를 바탕으로 
            **완전한** 여행 매거진 콘텐츠를 작성하세요. 
            
            **중요**: 제공된 모든 콘텐츠를 빠짐없이 활용해야 합니다. 첨삭하지 말고 모든 내용을 포함하세요.
            
            **인터뷰 형식 콘텐츠 (모두 활용):**
            {interview_content}
            
            **에세이 형식 콘텐츠 (모두 활용):**
            {essay_content}
            
            **원본 텍스트 참고:**
            {original_content}
            
            **이미지 정보:**
            {image_info}
            
            **통합 지침 (모든 데이터 활용):**
            1. **완전 활용**: 인터뷰와 에세이의 모든 내용을 빠짐없이 포함
            2. **내용 확장**: 제공된 콘텐츠를 기반으로 더 풍부한 매거진 스토리 생성
            3. **구조화**: 여행의 시간적 흐름을 고려하여 자연스럽게 구성
            4. **통합성**: 각 섹션이 독립적이면서도 전체 스토리가 연결되도록 구성
            5. **이미지 연계**: 이미지 정보를 적절한 위치에 자연스럽게 녹여냄
            6. **완성도**: 매거진 독자들이 몰입할 수 있는 완성된 스토리로 구성
            
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
            
            **출력 요구사항:**
            - 최소 3000자 이상의 풍부한 매거진 콘텐츠
            - 모든 인터뷰와 에세이 내용이 포함된 완성된 스토리
            - 여행의 전 과정을 아우르는 완전한 내러티브
            """,
            expected_output="모든 하위 에이전트 콘텐츠가 포함된 완성된 여행 매거진 콘텐츠"
        )
        
        result = agent.execute_task(integration_task)
        
        # 결과 검증
        final_content = str(result)
        self._verify_final_content(final_content, interview_results, essay_results)
        
        return final_content

    def _verify_final_content(self, final_content: str, interview_results: Dict[str, str], essay_results: Dict[str, str]):
        """최종 콘텐츠 검증"""
        final_length = len(final_content)
        total_source_length = sum(len(content) for content in interview_results.values()) + sum(len(content) for content in essay_results.values())
        
        print(f"ContentCreatorV2: 최종 콘텐츠 검증")
        print(f"- 최종 콘텐츠 길이: {final_length}자")
        print(f"- 원본 소스 길이: {total_source_length}자")
        print(f"- 확장 비율: {(final_length / total_source_length * 100):.1f}%" if total_source_length > 0 else "- 확장 비율: 계산 불가")
        
        if final_length < total_source_length * 0.8:
            print("⚠️ 최종 콘텐츠가 원본보다 현저히 짧습니다. 첨삭이 발생했을 가능성이 있습니다.")
        else:
            print("✅ 콘텐츠가 적절히 확장되어 생성되었습니다.")

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
