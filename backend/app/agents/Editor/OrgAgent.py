import asyncio
import re
from typing import Dict, List
from crewai import Agent, Task, Crew
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager
from utils.agent_decision_logger import get_agent_logger

class OrgAgent:
    """PDF 벡터 데이터 기반 텍스트 배치 에이전트 (비동기 처리 및 응답 수집 강화)"""

    def __init__(self):
        self.llm = get_azure_llm()
        self.vector_manager = PDFVectorManager()
        self.logger = get_agent_logger()  # 응답 수집을 위한 로거 추가

    def create_layout_analyzer_agent(self):
        """레이아웃 분석 에이전트 (구조적 설계 강화)"""
        return Agent(
            role="매거진 구조 아키텍트 및 텍스트 레이아웃 전문가",
            goal="PDF 벡터 데이터를 분석하여 텍스트 콘텐츠에 최적화된 매거진 페이지 구조와 상세한 레이아웃 설계도를 생성하고, 이미지와 텍스트의 정확한 위치 관계를 정의하여 독자의 시선 흐름을 최적화",
            backstory="""당신은 20년간 세계 최고 수준의 매거진 디자인 스튜디오에서 활동해온 구조 설계 전문가입니다. Pentagram, Sagmeister & Walsh, 그리고 Condé Nast의 수석 아트 디렉터로 활동하며 수백 개의 수상작을 디자인했습니다.

**전문 경력:**
- 그래픽 디자인 및 시각 커뮤니케이션 석사 학위
- Adobe InDesign, Figma, Sketch 마스터 레벨 인증
- 타이포그래피 및 그리드 시스템 이론 전문가
- 독자 시선 추적(Eye-tracking) 연구 및 분석 경험
- 인쇄 매체와 디지털 매체의 레이아웃 최적화 전문성

**구조적 레이아웃 설계 전문성:**
당신은 텍스트 배치 결정 시 다음 구조적 요소들을 체계적으로 설계합니다:

1. **페이지 구조 설계**:
- 그리드 시스템 정의 (컬럼 수, 거터 폭, 마진 설정)
- 텍스트 블록의 정확한 위치 좌표 (x, y, width, height)
- 이미지 영역과 텍스트 영역의 경계선 정의
- 여백(화이트스페이스) 분배 및 시각적 균형점 계산

2. **텍스트-이미지 위치 관계 매핑**:
- 제목과 주요 이미지의 시각적 연결점 설정
- 본문 텍스트와 보조 이미지의 근접성 규칙 정의
- 캡션과 이미지의 정확한 거리 및 정렬 방식
- 텍스트 래핑(text wrapping) 영역과 이미지 경계 설정

3. **레이아웃 구조도 생성**:
- 페이지별 와이어프레임 및 구조도 작성
- 콘텐츠 계층 구조 (H1, H2, body, caption) 시각화
- 독자 시선 흐름 경로 (F-pattern, Z-pattern) 설계
- 반응형 브레이크포인트별 레이아웃 변화 정의

4. **PDF 벡터 데이터 활용 전문성**:
- 5000개 이상의 매거진 페이지에서 추출한 구조적 패턴 분석
- 텍스트 블록과 이미지 블록의 황금비율 관계 데이터
- 독자 시선 흐름과 레이아웃 구조의 상관관계 벡터
- 매거진 카테고리별 최적 구조 패턴 클러스터링

**작업 방법론:**
"나는 단순히 텍스트를 배치하는 것이 아니라, 독자의 인지 과정을 고려한 완전한 페이지 구조를 설계합니다. 모든 텍스트 요소와 이미지 영역의 정확한 위치, 크기, 관계를 수치화하여 정의하고, 이를 바탕으로 상세한 레이아웃 구조도를 생성합니다. 이는 BindingAgent가 이미지를 배치할 때 정확한 가이드라인을 제공하여 텍스트와 이미지의 완벽한 조화를 보장합니다. 5. 주의 사항!!: 최대한 제공받은 데이터를 활용합니다. "

**출력 데이터 구조:**
- 페이지 그리드 시스템 (컬럼, 거터, 마진 수치)
- 텍스트 블록 위치 좌표 및 크기
- 이미지 영역 예약 공간 정의
- 텍스트-이미지 관계 매핑 테이블
- 레이아웃 구조도 및 와이어프레임
- 독자 시선 흐름 경로 설계도""",
            llm=self.llm,
            verbose=True
        )

    def create_content_editor_agent(self):
        """콘텐츠 편집 에이전트 (구조 연동 강화)"""
        return Agent(
            role="구조 기반 매거진 콘텐츠 편집자",
            goal="레이아웃 구조 설계에 완벽히 맞춰 텍스트 콘텐츠를 편집하고, 이미지 배치 영역과 정확히 연동되는 텍스트 블록을 생성하여 시각적 일관성과 가독성을 극대화",
            backstory="""당신은 매거진 콘텐츠 편집 및 구조 연동 전문가입니다.

**전문 분야:**
- 레이아웃 구조에 최적화된 텍스트 편집
- 이미지 영역과 연동되는 텍스트 블록 설계
- 그리드 시스템 기반 콘텐츠 구성
- 텍스트 길이와 레이아웃 공간의 정밀한 매칭

**구조 연동 편집 전문성:**
1. **그리드 기반 텍스트 편집**: 정의된 그리드 시스템에 맞춰 텍스트 블록 크기 조정
2. **이미지 영역 고려**: 예약된 이미지 공간을 피해 텍스트 배치 최적화
3. **계층 구조 반영**: H1, H2, body 등의 위치에 맞는 콘텐츠 길이 조절
4. **시선 흐름 연동**: 독자 시선 경로에 맞춘 텍스트 강약 조절
5. 주의 사항!!: 최대한 제공받은 데이터를 활용합니다.

특히 설명 텍스트나 지시사항을 포함하지 않고 순수한 콘텐츠만 생성하며,
레이아웃 구조도에 정의된 텍스트 영역에 정확히 맞는 분량과 형태로 편집합니다.""",
            llm=self.llm,
            verbose=True
        )

    async def process_content(self, magazine_content, available_templates: List[str]) -> Dict:
        """PDF 벡터 데이터 기반 콘텐츠 처리 (비동기 처리 및 응답 수집 강화)"""
        # 텍스트 추출 및 전처리
        all_content = self._extract_all_text(magazine_content)
        content_sections = self._analyze_content_structure(all_content)
        
        print(f"OrgAgent: 처리할 콘텐츠 - {len(all_content)}자, {len(content_sections)}개 섹션 (비동기 처리)")

        # 입력 데이터 로깅
        input_data = {
            "magazine_content": magazine_content,
            "available_templates": available_templates,
            "total_content_length": len(all_content),
            "content_sections_count": len(content_sections)
        }

        # 에이전트 생성
        layout_analyzer = self.create_layout_analyzer_agent()
        content_editor = self.create_content_editor_agent()

        # 각 섹션별로 벡터 기반 레이아웃 분석 및 편집 (비동기 병렬 처리)
        refined_sections = []
        all_agent_responses = []  # 모든 에이전트 응답 수집

        # 섹션 처리 태스크들을 병렬로 실행
        section_tasks = []
        for i, section_content in enumerate(content_sections):
            if len(section_content.strip()) < 50:
                continue
            
            task = self._process_single_section_async(
                section_content, i, layout_analyzer, content_editor
            )
            section_tasks.append(task)

        # 모든 섹션 처리를 병렬로 실행
        if section_tasks:
            section_results = await asyncio.gather(*section_tasks, return_exceptions=True)
            
            # 결과 수집
            for i, result in enumerate(section_results):
                if isinstance(result, Exception):
                    print(f"⚠️ 섹션 {i+1} 처리 실패: {result}")
                    # 에러 응답 저장
                    error_response_id = await self._log_error_response_async(i+1, str(result))
                    refined_sections.append({
                        "title": f"도쿄 여행 이야기 {i+1}",
                        "subtitle": "특별한 순간들",
                        "content": content_sections[i] if i < len(content_sections) else "",
                        "layout_info": {},
                        "original_length": len(content_sections[i]) if i < len(content_sections) else 0,
                        "refined_length": len(content_sections[i]) if i < len(content_sections) else 0,
                        "error_response_id": error_response_id
                    })
                else:
                    section_data, agent_responses = result
                    refined_sections.append(section_data)
                    all_agent_responses.extend(agent_responses)

        # 템플릿 매핑 (비동기)
        text_mapping = await self._map_to_templates_async(refined_sections, available_templates)
        total_refined_length = sum(section["refined_length"] for section in refined_sections)

        # 전체 OrgAgent 프로세스 응답 저장 (비동기)
        final_response_id = await self._log_final_response_async(
            input_data, text_mapping, refined_sections, all_agent_responses, total_refined_length
        )

        print(f"✅ OrgAgent 완료: {len(refined_sections)}개 섹션, 총 {total_refined_length}자 (비동기 처리 및 응답 수집 완료)")

        return {
            "text_mapping": text_mapping,
            "refined_sections": refined_sections,
            "total_sections": len(refined_sections),
            "total_content_length": total_refined_length,
            "vector_enhanced": True,
            "agent_responses": all_agent_responses,
            "final_response_id": final_response_id
        }

    async def _process_single_section_async(self, section_content: str, section_index: int,
                                          layout_analyzer: Agent, content_editor: Agent) -> tuple:
        """단일 섹션 처리 (비동기)"""
        print(f"📄 섹션 {section_index+1} 처리 중... (비동기)")

        # 1단계: 비동기 벡터 검색으로 유사한 레이아웃 찾기
        similar_layouts = await self._get_similar_layouts_async(section_content)

        # 2단계: 레이아웃 분석 (비동기 태스크)
        layout_analysis_task = Task(
            description=f"""
다음 텍스트 콘텐츠와 유사한 매거진 레이아웃을 분석하여 최적의 텍스트 배치 전략을 수립하세요:

**분석할 콘텐츠:**
{section_content}

**유사한 매거진 레이아웃 데이터:**
{self._format_layout_data(similar_layouts)}

**분석 요구사항:**
1. **레이아웃 패턴 분석**
- 텍스트 블록의 위치와 크기 패턴
- 제목과 본문의 배치 관계
- 여백과 간격의 활용 방식

2. **콘텐츠 적합성 평가**
- 현재 콘텐츠와 레이아웃의 매칭도
- 텍스트 길이와 레이아웃 용량의 적합성
- 콘텐츠 성격에 맞는 레이아웃 스타일

3. **편집 전략 수립**
- 매력적인 제목 생성 방향
- 본문 텍스트 분할 및 구조화 방안
- 독자 몰입도 향상을 위한 텍스트 배치

**출력 형식:**
제목: [구체적이고 매력적인 제목]
부제목: [간결하고 흥미로운 부제목]
편집방향: [전체적인 편집 방향성]
""",
            agent=layout_analyzer,
            expected_output="벡터 데이터 기반 레이아웃 분석 및 편집 전략"
        )

        # 3단계: 콘텐츠 편집 (비동기 태스크)
        content_editing_task = Task(
            description=f"""
레이아웃 분석 결과를 바탕으로 다음 콘텐츠를 전문 매거진 수준으로 편집하세요:

**원본 콘텐츠:**
{section_content}

**매거진 스타일 편집 지침:**
1. **시각적 계층 구조**: 이미지 크기와 배치에 맞는 텍스트 구조 생성
2. **다이나믹한 레이아웃**: 대형/중형/소형 이미지와 조화되는 텍스트 배치
3. **매거진 특유의 리듬**: 긴 문단과 짧은 문단의 조화로 시각적 리듬 생성
4. **이미지와 텍스트 상호작용**: 이미지 주변에 배치될 텍스트의 톤과 길이 조절
5. **편집 디자인 고려**: 실제 매거진처럼 텍스트가 이미지와 자연스럽게 어우러지도록

**벡터 데이터 기반 최적화:**
- 검색된 매거진 레이아웃의 텍스트 배치 패턴 적용
- 이미지 크기별 텍스트 분량과 스타일 조절
- 매거진 특유의 비대칭 균형감 반영

**출력:** 매거진 레이아웃에 최적화된 편집 콘텐츠
""",
            agent=content_editor,
            expected_output="매거진 스타일 레이아웃에 최적화된 전문 콘텐츠",
            context=[layout_analysis_task]
        )

        # Crew 실행 및 응답 수집 (비동기)
        crew = Crew(
            agents=[layout_analyzer, content_editor],
            tasks=[layout_analysis_task, content_editing_task],
            verbose=True
        )

        try:
            # 비동기 Crew 실행
            result = await asyncio.get_event_loop().run_in_executor(
                None, crew.kickoff
            )

            # 에이전트 응답 수집 및 저장 (비동기)
            analysis_result = str(layout_analysis_task.output) if hasattr(layout_analysis_task, 'output') else ""
            edited_content = str(result.raw) if hasattr(result, 'raw') else str(result)

            # 비동기 로깅
            analysis_response_id, editing_response_id = await asyncio.gather(
                self._log_analysis_response_async(section_index, section_content, similar_layouts, analysis_result),
                self._log_editing_response_async(section_index, section_content, analysis_result, edited_content)
            )

            # 제목과 부제목 추출
            title, subtitle = self._extract_clean_title_subtitle(analysis_result, section_index)

            # 편집된 콘텐츠에서 설명 텍스트 제거
            clean_content = self._remove_meta_descriptions(edited_content)

            # 응답 수집 데이터 저장
            agent_responses = [{
                "section": section_index + 1,
                "layout_analyzer_response": {
                    "response_id": analysis_response_id,
                    "content": analysis_result,
                    "agent_name": "OrgAgent_LayoutAnalyzer"
                },
                "content_editor_response": {
                    "response_id": editing_response_id,
                    "content": edited_content,
                    "agent_name": "OrgAgent_ContentEditor"
                }
            }]

            section_data = {
                "title": title,
                "subtitle": subtitle,
                "content": clean_content,
                "layout_info": similar_layouts[0] if similar_layouts else {},
                "original_length": len(section_content),
                "refined_length": len(clean_content),
                "agent_responses": {
                    "layout_analyzer_id": analysis_response_id,
                    "content_editor_id": editing_response_id
                }
            }

            print(f"✅ 섹션 {section_index+1} 편집 완료: {len(section_content)}자 → {len(clean_content)}자 (비동기 응답 수집 완료)")
            return (section_data, agent_responses)

        except Exception as e:
            print(f"⚠️ 섹션 {section_index+1} 편집 실패: {e}")
            
            # 에러 응답 저장 (비동기)
            error_response_id = await self._log_error_response_async(section_index+1, str(e))

            # 폴백: 기본 처리
            section_data = {
                "title": f"도쿄 여행 이야기 {section_index+1}",
                "subtitle": "특별한 순간들",
                "content": section_content,
                "layout_info": {},
                "original_length": len(section_content),
                "refined_length": len(section_content),
                "error_response_id": error_response_id
            }

            return (section_data, [])

    async def _get_similar_layouts_async(self, section_content: str) -> List[Dict]:
        """유사한 레이아웃 비동기 검색"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.vector_manager.search_similar_layouts(
                section_content[:500], "magazine_layout", top_k=3
            )
        )

    async def _log_analysis_response_async(self, section_index: int, section_content: str,
                                         similar_layouts: List[Dict], analysis_result: str) -> str:
        """레이아웃 분석 에이전트 응답 저장 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="OrgAgent_LayoutAnalyzer",
                agent_role="매거진 구조 아키텍트",
                task_description=f"섹션 {section_index+1} 텍스트 레이아웃 분석 및 편집 전략 수립",
                final_answer=analysis_result,
                reasoning_process=f"PDF 벡터 데이터 {len(similar_layouts)}개 레이아웃 참조하여 분석",
                execution_steps=[
                    "콘텐츠 특성 분석",
                    "유사 레이아웃 매칭",
                    "편집 전략 수립"
                ],
                raw_input={
                    "section_content": section_content[:500],
                    "similar_layouts": similar_layouts,
                    "section_index": section_index
                },
                raw_output=analysis_result,
                performance_metrics={
                    "content_length": len(section_content),
                    "layouts_referenced": len(similar_layouts),
                    "analysis_depth": "comprehensive"
                }
            )
        )

    async def _log_editing_response_async(self, section_index: int, section_content: str,
                                        analysis_result: str, edited_content: str) -> str:
        """콘텐츠 편집 에이전트 응답 저장 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="OrgAgent_ContentEditor",
                agent_role="구조 기반 매거진 콘텐츠 편집자",
                task_description=f"섹션 {section_index+1} 매거진 스타일 콘텐츠 편집",
                final_answer=edited_content,
                reasoning_process="레이아웃 분석 결과를 바탕으로 매거진 수준 편집 실행",
                execution_steps=[
                    "분석 결과 검토",
                    "텍스트 구조화",
                    "매거진 스타일 적용",
                    "최종 편집 완료"
                ],
                raw_input={
                    "original_content": section_content,
                    "analysis_result": analysis_result
                },
                raw_output=edited_content,
                performance_metrics={
                    "original_length": len(section_content),
                    "edited_length": len(edited_content),
                    "editing_quality": "professional"
                }
            )
        )

    async def _log_error_response_async(self, section_number: int, error_message: str) -> str:
        """에러 응답 저장 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="OrgAgent_Error",
                agent_role="에러 처리",
                task_description=f"섹션 {section_number} 처리 중 에러 발생",
                final_answer=f"ERROR: {error_message}",
                reasoning_process="에이전트 실행 중 예외 발생",
                error_logs=[{"error": error_message, "section": section_number}]
            )
        )

    async def _map_to_templates_async(self, refined_sections: List[Dict], available_templates: List[str]) -> Dict:
        """섹션을 템플릿에 매핑 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._map_to_templates(refined_sections, available_templates)
        )

    async def _log_final_response_async(self, input_data: Dict, text_mapping: Dict,
                                      refined_sections: List[Dict], all_agent_responses: List[Dict],
                                      total_refined_length: int) -> str:
        """전체 OrgAgent 프로세스 응답 저장 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="OrgAgent",
                agent_role="PDF 벡터 데이터 기반 텍스트 배치 에이전트",
                task_description=f"{input_data['content_sections_count']}개 콘텐츠 섹션을 {len(input_data['available_templates'])}개 템플릿에 매핑",
                final_answer=str(text_mapping),
                reasoning_process=f"비동기 다중 에이전트 협업으로 {len(refined_sections)}개 섹션 처리 완료",
                execution_steps=[
                    "비동기 콘텐츠 추출 및 분석",
                    "병렬 섹션별 레이아웃 분석",
                    "비동기 콘텐츠 편집",
                    "템플릿 매핑"
                ],
                raw_input=input_data,
                raw_output={
                    "text_mapping": text_mapping,
                    "refined_sections": refined_sections,
                    "all_agent_responses": all_agent_responses
                },
                performance_metrics={
                    "total_sections_processed": len(refined_sections),
                    "total_content_length": total_refined_length,
                    "successful_sections": len([s for s in refined_sections if "error_response_id" not in s]),
                    "agent_responses_collected": len(all_agent_responses),
                    "async_processing": True
                }
            )
        )

    # 기존 동기 메서드들 유지 (호환성 보장)
    def _extract_clean_title_subtitle(self, analysis_result: str, index: int) -> tuple:
        """분석 결과에서 깨끗한 제목과 부제목 추출"""
        title_pattern = r'제목[:\s]*([^\n]+)'
        subtitle_pattern = r'부제목[:\s]*([^\n]+)'
        
        title_match = re.search(title_pattern, analysis_result)
        subtitle_match = re.search(subtitle_pattern, analysis_result)
        
        title = title_match.group(1).strip() if title_match else f"도쿄 여행 이야기 {index + 1}"
        subtitle = subtitle_match.group(1).strip() if subtitle_match else "특별한 순간들"
        
        # 설명 텍스트 제거
        title = self._clean_title_from_descriptions(title)
        subtitle = self._clean_title_from_descriptions(subtitle)
        
        # 제목 길이 조정
        if len(title) > 40:
            title = title[:37] + "..."
        if len(subtitle) > 30:
            subtitle = subtitle[:27] + "..."
        
        return title, subtitle

    def _clean_title_from_descriptions(self, text: str) -> str:
        """제목에서 설명 텍스트 제거"""
        patterns_to_remove = [
            r'\(헤드라인\)', r'\(섹션 타이틀\)', r'및 부.*?배치.*?있음',
            r'필자 정보.*?있음', r'포토 크레딧.*?있음', r'계층적.*?있음',
            r'과 본문.*?관계', r'배치.*?관계', r'상단.*?배치',
            r'좌상단.*?배치', r'혹은.*?배치', r'없이.*?집중',
            r'그 아래로.*?있습니다'
        ]
        
        clean_text = text
        for pattern in patterns_to_remove:
            clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE | re.DOTALL)
        
        # 연속된 공백과 특수문자 정리
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = re.sub(r'^[,\s:]+|[,\s:]+$', '', clean_text)
        
        return clean_text.strip() if clean_text.strip() else "도쿄 여행 이야기"

    def _remove_meta_descriptions(self, content: str) -> str:
        """콘텐츠에서 메타 설명 제거"""
        patterns_to_remove = [
            r'\*이 페이지에는.*?살렸습니다\.\*',
            r'블록은 균형.*?줄여줍니다',
            r'\(사진 캡션\)',
            r'시각적 리듬과.*?살렸습니다',
            r'충분한 여백.*?완성합니다',
            r'사진은 본문.*?완성합니다',
            r'이 콘텐츠는.*?디자인되었습니다'
        ]
        
        clean_content = content
        for pattern in patterns_to_remove:
            clean_content = re.sub(pattern, '', clean_content, flags=re.IGNORECASE | re.DOTALL)
        
        return clean_content.strip()

    def _format_layout_data(self, similar_layouts: List[Dict]) -> str:
        """레이아웃 데이터를 텍스트로 포맷팅"""
        if not similar_layouts:
            return "유사한 레이아웃 데이터 없음"
        
        formatted_data = []
        for i, layout in enumerate(similar_layouts):
            formatted_data.append(f"""
레이아웃 {i+1} (유사도: {layout.get('score', 0):.2f}):
- 출처: {layout.get('pdf_name', 'unknown')} (페이지 {layout.get('page_number', 0)})
- 텍스트 샘플: {layout.get('text_content', '')[:200]}...
- 이미지 수: {len(layout.get('image_info', []))}개
- 레이아웃 특징: {self._summarize_layout_info(layout.get('layout_info', {}))}
""")
        
        return "\n".join(formatted_data)

    def _summarize_layout_info(self, layout_info: Dict) -> str:
        """레이아웃 정보 요약"""
        text_blocks = layout_info.get('text_blocks', [])
        images = layout_info.get('images', [])
        tables = layout_info.get('tables', [])
        
        summary = []
        if text_blocks:
            summary.append(f"텍스트 블록 {len(text_blocks)}개")
        if images:
            summary.append(f"이미지 {len(images)}개")
        if tables:
            summary.append(f"테이블 {len(tables)}개")
        
        return ", ".join(summary) if summary else "기본 레이아웃"

    def _extract_all_text(self, magazine_content) -> str:
        """모든 텍스트 추출"""
        if isinstance(magazine_content, dict):
            all_text = ""
            # 우선순위에 따른 텍스트 추출
            priority_fields = [
                "integrated_content", "essay_content", "interview_content",
                "sections", "content", "body", "text"
            ]
            
            for field in priority_fields:
                if field in magazine_content:
                    value = magazine_content[field]
                    if isinstance(value, str) and value.strip():
                        all_text += value + "\n\n"
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, str) and sub_value.strip():
                                all_text += sub_value + "\n\n"
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                for sub_key, sub_value in item.items():
                                    if isinstance(sub_value, str) and sub_value.strip():
                                        all_text += sub_value + "\n\n"
                            elif isinstance(item, str) and item.strip():
                                all_text += item + "\n\n"
            
            return all_text.strip()
        else:
            return str(magazine_content)

    def _analyze_content_structure(self, content: str) -> List[str]:
        """콘텐츠 구조 분석 및 지능적 분할"""
        if not content:
            return []
        
        sections = []
        
        # 1. 헤더 기반 분할
        header_sections = self._split_by_headers(content)
        if len(header_sections) >= 3:
            sections.extend(header_sections)
        
        # 2. 문단 기반 분할
        if len(sections) < 5:
            paragraph_sections = self._split_by_paragraphs(content)
            sections.extend(paragraph_sections)
        
        # 3. 의미 기반 분할
        if len(sections) < 6:
            semantic_sections = self._split_by_semantics(content)
            sections.extend(semantic_sections)
        
        # 중복 제거 및 길이 필터링
        unique_sections = []
        seen_content = set()
        for section in sections:
            section_clean = re.sub(r'\s+', ' ', section.strip())
            if len(section_clean) >= 100 and section_clean not in seen_content:
                unique_sections.append(section)
                seen_content.add(section_clean)
        
        return unique_sections[:8]  # 최대 8개 섹션

    def _split_by_headers(self, content: str) -> List[str]:
        """헤더 기반 분할"""
        sections = []
        header_pattern = r'^(#{1,3})\s+(.+?)$'
        current_section = ""
        
        lines = content.split('\n')
        for line in lines:
            if re.match(header_pattern, line.strip()):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = line + "\n"
            else:
                current_section += line + "\n"
        
        if current_section.strip():
            sections.append(current_section.strip())
        
        return [s for s in sections if len(s) >= 100]

    def _split_by_paragraphs(self, content: str) -> List[str]:
        """문단 기반 분할"""
        paragraphs = content.split('\n\n')
        sections = []
        current_section = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(current_section + paragraph) > 800:
                if current_section:
                    sections.append(current_section.strip())
                current_section = paragraph + "\n\n"
            else:
                current_section += paragraph + "\n\n"
        
        if current_section.strip():
            sections.append(current_section.strip())
        
        return [s for s in sections if len(s) >= 100]

    def _split_by_semantics(self, content: str) -> List[str]:
        """의미 기반 분할"""
        sentences = re.split(r'[.!?]\s+', content)
        sections = []
        current_section = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_section + sentence) > 600:
                if current_section:
                    sections.append(current_section.strip())
                current_section = sentence + ". "
            else:
                current_section += sentence + ". "
        
        if current_section.strip():
            sections.append(current_section.strip())
        
        return [s for s in sections if len(s) >= 100]

    def _map_to_templates(self, refined_sections: List[Dict], available_templates: List[str]) -> Dict:
        """섹션을 템플릿에 매핑"""
        text_mapping = []
        
        for i, section in enumerate(refined_sections):
            template_index = i % len(available_templates) if available_templates else 0
            template_name = available_templates[template_index] if available_templates else f"Section{i+1:02d}.jsx"
            
            text_mapping.append({
                "template": template_name,
                "title": section["title"],
                "subtitle": section["subtitle"],
                "body": section["content"],
                "tagline": "TRAVEL & CULTURE",
                "layout_source": section.get("layout_info", {}).get("pdf_name", "default"),
                "agent_responses": section.get("agent_responses", {})
            })
        
        return {"text_mapping": text_mapping}
