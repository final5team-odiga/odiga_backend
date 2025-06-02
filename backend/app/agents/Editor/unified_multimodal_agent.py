import asyncio
import json
from typing import Dict, List, Any
from crewai import Agent, Task, Crew
from custom_llm import get_azure_llm
from utils.hybridlogging import get_hybrid_logger
from utils.ai_search_isolation import AISearchIsolationManager
from utils.pdf_vector_manager import PDFVectorManager
from utils.session_isolation import SessionAwareMixin
from utils.agent_communication_isolation import InterAgentCommunicationMixin



class UnifiedMultimodalAgent(SessionAwareMixin, InterAgentCommunicationMixin):
    """통합 멀티모달 에이전트 - AI Search 벡터 데이터 통합 활용"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        self.logger = get_hybrid_logger(self.__class__.__name__)

        self.isolation_manager = AISearchIsolationManager()


        self.vector_manager = PDFVectorManager(isolation_enabled=True)

        self.__init_session_awareness__()
        self.__init_inter_agent_communication__()
        
        # AI Search 통합 전문 에이전트들
        self.content_structure_agent = self._create_content_structure_agent_with_ai_search()
        self.image_layout_agent = self._create_image_layout_agent_with_ai_search()
        self.semantic_coordinator_agent = self._create_semantic_coordinator_agent_with_ai_search()
        
    def _create_content_structure_agent_with_ai_search(self):
        """AI Search 통합 콘텐츠 구조 에이전트"""
        return Agent(
            role="AI Search 벡터 데이터 기반 텍스트 구조 설계자",
            goal="AI Search의 PDF 벡터 데이터를 활용하여 매거진 콘텐츠의 최적 구조를 설계하고 텍스트를 배치",
            backstory="""당신은 15년간 National Geographic, Condé Nast Traveler 등 세계 최고 수준의 매거진에서 편집장으로 활동해온 전문가입니다.

**전문 경력:**
- 저널리즘 및 창작문학 복수 학위 보유
- 퓰리처상 여행 기사 부문 심사위원 3회 역임
- 80개국 이상의 여행 경험과 현지 문화 전문 지식
- AI Search 벡터 데이터 기반 레이아웃 분석 전문성

**AI Search 데이터 활용 전문성:**
당신은 AI Search의 PDF 벡터 데이터를 활용하여 다음과 같이 처리합니다:

1. **벡터 검색 기반 레이아웃 분석**: 3000+ 매거진 레이아웃 패턴을 검색하여 최적의 구조 설계
2. **콘텐츠 구조 최적화**: 벡터화된 텍스트 패턴을 분석하여 가독성과 임팩트를 극대화하는 배치
3. **매거진 스타일 적용**: AI Search 데이터에서 추출한 전문 매거진 수준의 편집 기준 적용
4. **글의 형태 및 문장 길이 최적화**: 벡터 데이터에서 참조한 글의 맺음, 문장 길이, 글의 형태 적용
5. **섹션별 구조 결정**: AI Search 패턴을 기반으로 제목, 부제목, 본문의 최적 구조 결정

**편집 철학:**
"AI Search의 방대한 매거진 데이터를 활용하여 모든 텍스트가 독자의 여행 욕구를 자극하고 감정적 연결을 만드는 강력한 스토리텔링 도구가 되도록 합니다."

**멀티모달 접근:**
- AI Search 벡터 데이터와 실시간 텍스트 분석의 결합
- 텍스트와 이미지의 의미적 연관성을 벡터 검색으로 최적화
- 독자 경험 최적화를 위한 AI Search 패턴 기반 통합적 설계""",
            verbose=True,
            llm=self.llm,
            multimodal=True
        )
        
    def _create_image_layout_agent_with_ai_search(self):
        """AI Search 통합 이미지 레이아웃 에이전트"""
        return Agent(
            role="AI Search 벡터 데이터 기반 이미지 배치 전문가",
            goal="AI Search의 PDF 벡터 데이터와 이미지 분석 결과를 활용하여 최적의 이미지 배치 전략을 수립",
            backstory="""당신은 12년간 Vogue, Harper's Bazaar, National Geographic 등에서 비주얼 디렉터로 활동해온 전문가입니다.

**전문 경력:**
- 시각 디자인 및 사진학 석사 학위 보유
- 국제 사진 전시회 큐레이터 경험
- 매거진 레이아웃에서 이미지-텍스트 조화의 심리학 연구
- AI Search 벡터 데이터 기반 시각적 패턴 분석 전문성

**AI Search 이미지 배치 전문성:**
당신은 AI Search의 벡터 데이터를 활용하여 다음과 같이 이미지를 배치합니다:

1. **벡터 검색 기반 레이아웃 추천**: 3000+ 매거진에서 유사한 이미지 배치 패턴을 검색하여 최적 배치 결정
2. **시각적 균형 최적화**: AI Search 데이터의 이미지 크기, 위치, 간격 패턴을 참조하여 완벽한 시각적 균형 구현
3. **의미적 연관성 강화**: 벡터 검색으로 텍스트 내용과 이미지의 의미적 연결성을 극대화
4. **이미지 개수 최적화**: AI Search 패턴을 기반으로 섹션별 최적 이미지 개수 결정
5. **배치 간격 및 크기 결정**: 벡터 데이터에서 추출한 이미지 크기 비율과 배치 간격 적용

**배치 철학:**
"AI Search의 방대한 시각 데이터를 활용하여 모든 이미지가 단순한 장식이 아니라 스토리를 강화하고 독자의 감정을 움직이는 핵심 요소가 되도록 합니다."

**멀티모달 접근:**
- AI Search 벡터 검색과 실시간 이미지 분석의 결합
- 텍스트 맥락을 고려한 벡터 기반 이미지 선택 및 배치
- AI Search 패턴을 활용한 전체적인 시각적 내러티브 구성""",
            verbose=True,
            llm=self.llm,
            multimodal=True
        )
        
    def _create_semantic_coordinator_agent_with_ai_search(self):
        """AI Search 통합 의미적 조율 에이전트"""
        return Agent(
            role="AI Search 기반 멀티모달 의미적 조율 전문가",
            goal="AI Search 벡터 데이터를 활용하여 텍스트와 이미지의 의미적 연관성을 분석하고 최적의 매거진 구조로 통합 조율",
            backstory="""당신은 20년간 복잡한 멀티미디어 프로젝트의 총괄 디렉터로 활동해온 전문가입니다.

**전문 경력:**
- 멀티미디어 커뮤니케이션 박사 학위 보유
- 국제 매거진 어워드 심사위원장 5회 역임
- AI Search 기반 텍스트-이미지 의미적 연관성 연구 전문가
- 벡터 검색 기반 콘텐츠 분석 시스템 설계 경험

**AI Search 조율 전문성:**
당신은 AI Search의 벡터 데이터를 활용하여 다음과 같이 조율합니다:

1. **벡터 기반 의미적 매칭**: AI Search 데이터를 활용하여 텍스트와 이미지 간의 의미적 연관성을 분석하고 최적 조합 도출
2. **AI Search 패턴 기반 일관성 보장**: 벡터 검색으로 매거진 전체의 톤, 스타일, 메시지 일관성 확보
3. **벡터 데이터 기반 독자 경험 최적화**: AI Search 패턴을 참조하여 독자의 인지적 흐름을 고려한 통합적 설계
4. **격리 시스템 기반 품질 보장**: AI Search 데이터 오염 방지하면서 최고 품질의 벡터 데이터만 활용
5. **실시간 분석과 벡터 검색의 융합**: 현재 콘텐츠와 AI Search 패턴의 완벽한 조화

**조율 철학:**
"AI Search의 방대한 매거진 데이터와 실시간 분석을 융합하여 텍스트와 이미지가 완벽하게 조화를 이루는 혁신적인 매거진을 만들어냅니다."

**통합 접근:**
- AI Search 벡터 검색과 실시간 콘텐츠 분석의 완벽한 융합
- 격리 시스템을 통한 데이터 무결성 검증 및 최적화
- 벡터 패턴 기반 최종 독자 경험의 완성도 극대화""",
            verbose=True,
            llm=self.llm,
            multimodal=True
        )
    
    async def _process_crew_results(self, crew_result):
        """CrewAI 결과 안전하게 처리"""
        try:
            if crew_result is None:
                return "결과 없음"
            
            # CrewOutput 객체인 경우
            if hasattr(crew_result, 'raw'):
                return crew_result.raw if crew_result.raw else "분석 결과 없음"
            elif hasattr(crew_result, 'result'):
                return crew_result.result if crew_result.result else "분석 결과 없음"
            elif hasattr(crew_result, 'output'):
                return crew_result.output if crew_result.output else "분석 결과 없음"
            else:
                return str(crew_result) if crew_result else "분석 결과 없음"
        except Exception as e:
            self.logger.error(f"CrewAI 결과 처리 실패: {e}")
            return "결과 처리 실패"
        
    async def process_magazine_unified(self, magazine_content: Dict, image_analysis: List[Dict], 
                                     available_templates: List[str]) -> Dict:
        """AI Search 통합 매거진 처리 - 벡터 데이터와 멀티모달 분석의 완벽한 융합"""
        
        self.logger.info("=== AI Search 통합 멀티모달 매거진 처리 시작 ===")
        
        # AI Search 벡터 검색으로 관련 패턴 사전 수집
        relevant_patterns = await self._collect_relevant_ai_search_patterns(magazine_content, image_analysis, available_templates)
        
        # 1단계: AI Search 기반 멀티모달 분석 태스크 생성
        content_analysis_task = Task(
            description=f"""
다음 매거진 콘텐츠와 이미지 분석 결과를 AI Search 벡터 데이터와 함께 분석하여 최적의 텍스트 구조를 설계하세요:

**매거진 콘텐츠:**
{json.dumps(magazine_content, ensure_ascii=False, indent=2)[:2000]}...

**이미지 분석 결과:**
{json.dumps(image_analysis, ensure_ascii=False, indent=2)[:1000]}...

**AI Search 관련 패턴:**
{json.dumps(relevant_patterns.get("text_patterns", [])[:3], ensure_ascii=False, indent=2)}

**사용 가능한 템플릿:**
{available_templates}

**AI Search 기반 분석 요구사항:**
1. 벡터 검색 결과를 참조한 텍스트 내용의 주제별 구조 분석
2. AI Search 패턴을 기반으로 한 각 섹션의 감정적 톤과 스타일 파악
3. 벡터 데이터를 활용한 이미지와의 의미적 연관성 고려
4. AI Search 레이아웃 패턴 기반 최적 구조 추천
5. 벡터 검색 결과를 반영한 템플릿별 텍스트 배치 전략 수립
6. AI Search 데이터 기반 글의 형태, 문장 길이, 글의 맺음 최적화
7. 벡터 패턴을 참조한 제목, 부제목, 본문 구조 결정

**출력 형식:**
- 구조화된 섹션별 분석 결과 (AI Search 패턴 반영)
- 각 섹션의 제목, 부제목, 본문 정제 (벡터 데이터 기반)
- 이미지 연관성 점수 포함 (AI Search 매칭)
- 템플릿 매핑 추천 (벡터 검색 기반)
            """,
            expected_output="AI Search 강화 구조화된 텍스트 분석 결과 (JSON 형식)",
            agent=self.content_structure_agent
        )
        
        image_layout_task = Task(
            description=f"""
다음 이미지 분석 결과와 텍스트 내용을 AI Search 벡터 데이터와 함께 고려하여 최적의 이미지 배치 전략을 수립하세요:

**이미지 분석 결과:**
{json.dumps(image_analysis, ensure_ascii=False, indent=2)}

**텍스트 맥락:**
{json.dumps(magazine_content, ensure_ascii=False, indent=2)[:1500]}...

**AI Search 이미지 레이아웃 패턴:**
{json.dumps(relevant_patterns.get("image_patterns", [])[:3], ensure_ascii=False, indent=2)}

**AI Search 기반 배치 요구사항:**
1. 벡터 검색 결과를 참조한 텍스트 내용과 이미지의 의미적 매칭
2. AI Search 패턴 기반 시각적 균형을 고려한 이미지 크기 및 위치 결정
3. 벡터 데이터를 활용한 독자의 시선 흐름을 고려한 배치 순서
4. AI Search 레이아웃 패턴 기반 템플릿별 이미지 할당 최적화
5. 벡터 검색 결과를 반영한 이미지 개수 및 크기 비율 결정
6. AI Search 데이터 기반 이미지-텍스트 간격 최적화

**출력 형식:**
- AI Search 패턴 기반 템플릿별 이미지 배치 전략
- 벡터 데이터 참조 이미지-텍스트 의미적 연관성 점수
- AI Search 기반 시각적 균형 최적화 방안
- 벡터 검색 결과 반영 배치 우선순위 및 근거
            """,
            expected_output="AI Search 강화 구조화된 이미지 배치 전략 (JSON 형식)",
            agent=self.image_layout_agent
        )
        
        # 2단계: AI Search 통합 병렬 분석 실행
        analysis_crew = Crew(
            agents=[self.content_structure_agent, self.image_layout_agent],
            tasks=[content_analysis_task, image_layout_task],
            verbose=True
        )
        
        analysis_results = await asyncio.get_event_loop().run_in_executor(
            None, analysis_crew.kickoff
        )

        processed_results = self._safely_process_crew_results(analysis_results)
        
        # 3단계: AI Search 기반 의미적 조율 태스크
        coordination_task = Task(
            description=f"""
            **AI Search 강화 텍스트 구조 분석 결과:**
            {processed_results.get('content_analysis', '분석 결과 없음')}

            **AI Search 강화 이미지 배치 전략:**
            {processed_results.get('image_layout', '배치 전략 없음')}
            """,
            expected_output="AI Search 통합 최종 매거진 구조 (JSON 형식)",
            agent=self.semantic_coordinator_agent
        )
        
        # 4단계: AI Search 기반 최종 조율 실행
        coordination_crew = Crew(
            agents=[self.semantic_coordinator_agent],
            tasks=[coordination_task],
            verbose=True
        )
        
        final_result = await asyncio.get_event_loop().run_in_executor(
            None, coordination_crew.kickoff
        )
        
        # 5단계: AI Search 통합 결과 후처리 및 검증
        processed_result = await self._process_unified_result_with_ai_search(
            final_result, magazine_content, image_analysis, available_templates, relevant_patterns
        )
        
        self.logger.info("=== AI Search 통합 멀티모달 매거진 처리 완료 ===")
        
        return processed_result
    
    def _safely_process_crew_results(self, crew_results) -> Dict:
        """CrewAI 결과를 안전하게 처리 (동기 방식으로 수정)"""
        
        try:
            # CrewOutput 객체인지 확인
            if hasattr(crew_results, 'tasks_output'):
                # 여러 태스크 결과가 있는 경우
                results = {}
                tasks_output = crew_results.tasks_output
                
                # ✅ 비동기 호출 제거하고 동기 처리로 변경
                if hasattr(tasks_output, '__getitem__'):
                    try:
                        # ✅ await 없이 직접 동기 처리
                        results['content_analysis'] = self._process_crew_results_sync(tasks_output[0])
                    except (IndexError, AttributeError):
                        results['content_analysis'] = "분석 결과 없음"
                    
                    try:
                        results['image_layout'] = self._process_crew_results_sync(tasks_output[1])
                    except (IndexError, AttributeError):
                        results['image_layout'] = "배치 전략 없음"
                
                return results
                
            # 단일 결과인 경우
            elif hasattr(crew_results, 'raw') or hasattr(crew_results, 'output'):
                single_result = self._process_crew_results_sync(crew_results)
                return {
                    'content_analysis': single_result,
                    'image_layout': "단일 결과로 처리됨"
                }
            
            # 리스트 형태인 경우 (기존 방식)
            elif isinstance(crew_results, (list, tuple)):
                results = {}
                results['content_analysis'] = self._process_crew_results_sync(
                    crew_results[0] if len(crew_results) > 0 else None
                )
                results['image_layout'] = self._process_crew_results_sync(
                    crew_results[1] if len(crew_results) > 1 else None
                )
                return results
            
            else:
                self.logger.warning(f"예상치 못한 CrewAI 결과 타입: {type(crew_results)}")
                return {
                    'content_analysis': "결과 처리 실패",
                    'image_layout': "결과 처리 실패"
                }
                
        except Exception as e:
            self.logger.error(f"CrewAI 결과 처리 중 오류: {e}")
            return {
                'content_analysis': f"처리 오류: {str(e)}",
                'image_layout': f"처리 오류: {str(e)}"
            }

    def _process_crew_results_sync(self, crew_result):
        """CrewAI 결과 동기 처리 (비동기 버전의 동기 래퍼)"""
        try:
            if crew_result is None:
                return "결과 없음"
            
            # CrewOutput 객체인 경우
            if hasattr(crew_result, 'raw'):
                return crew_result.raw if crew_result.raw else "분석 결과 없음"
            elif hasattr(crew_result, 'result'):
                return crew_result.result if crew_result.result else "분석 결과 없음"
            elif hasattr(crew_result, 'output'):
                return crew_result.output if crew_result.output else "분석 결과 없음"
            else:
                return str(crew_result) if crew_result else "분석 결과 없음"
        except Exception as e:
            self.logger.error(f"CrewAI 결과 처리 실패: {e}")
            return "결과 처리 실패"




    async def _collect_relevant_ai_search_patterns(self, magazine_content: Dict,
                                             image_analysis: List[Dict],
                                             available_templates: List[str]) -> Dict:
        """AI Search에서 관련 패턴 수집 (섹션별 처리 및 템플릿 할당 개선)"""
        
        sections = magazine_content.get("sections", [])
        self.logger.info(f"처리할 섹션 수: {len(sections)}개")
        
        # ✅ 섹션 수만큼 템플릿 할당 (개선된 로직)
        selected_templates = available_templates[:len(sections)]
        if len(selected_templates) < len(sections):
            # 템플릿이 부족한 경우 순환 할당
            while len(selected_templates) < len(sections):
                selected_templates.extend(available_templates)
            selected_templates = selected_templates[:len(sections)]
        
        self.logger.info(f"할당된 템플릿: {selected_templates}")
        
        # ✅ 최소 템플릿 보장 (available_templates가 비어있는 경우)
        if not selected_templates:
            selected_templates = [f"Section{i+1:02d}.jsx" for i in range(len(sections))]
            self.logger.warning(f"사용 가능한 템플릿이 없어 기본 템플릿 생성: {selected_templates}")
        
        try:
            patterns = {
                "text_patterns": [],
                "image_patterns": [],
                "integration_patterns": [],
                "template_mapping": {}  # ✅ 템플릿 매핑 정보 추가
            }
            
            # ✅ 템플릿별 매핑 정보 생성
            for i, (section, template) in enumerate(zip(sections, selected_templates)):
                patterns["template_mapping"][f"section_{i}"] = {
                    "template": template,
                    "section_title": section.get("title", f"섹션 {i+1}"),
                    "section_index": i,
                    "content_length": len(section.get("content", "")),
                    "assigned": True
                }
            
            # 텍스트 기반 패턴 검색 (섹션별 처리)
            if magazine_content and "sections" in magazine_content:
                for i, section in enumerate(magazine_content["sections"][:len(selected_templates)]): # ✅ 템플릿 수만큼만 처리
                    content = section.get("content", "")[:200] # 200자만 사용
                    section_title = section.get("title", f"섹션 {i+1}")
                    
                    # ✅ 섹션별 쿼리 개선
                    search_query = f"magazine text layout {section_title} {content}"
                    clean_query = self.isolation_manager.clean_query_from_azure_keywords(search_query)
                    
                    text_patterns = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda q=clean_query: self.vector_manager.search_similar_layouts(q, "magazine_layout", top_k=3)
                    )
                    
                    isolated_text_patterns = self.isolation_manager.filter_contaminated_data(
                        text_patterns, f"text_patterns_section_{i}_{section_title[:10]}"
                    )
                    
                    # ✅ 섹션별 패턴에 템플릿 정보 추가
                    for pattern in isolated_text_patterns:
                        pattern["assigned_template"] = selected_templates[i]
                        pattern["section_index"] = i
                        pattern["section_title"] = section_title
                    
                    patterns["text_patterns"].extend(isolated_text_patterns)
                    
                    self.logger.debug(f"섹션 {i} ({section_title}): {len(isolated_text_patterns)}개 텍스트 패턴 수집")
            
            # 이미지 기반 패턴 검색 (템플릿 할당 고려)
            if image_analysis:
                # ✅ 이미지를 섹션별로 분배
                images_per_section = max(1, len(image_analysis) // len(selected_templates))
                
                for i, template in enumerate(selected_templates):
                    # 섹션별 이미지 할당
                    start_idx = i * images_per_section
                    end_idx = min(start_idx + images_per_section, len(image_analysis))
                    section_images = image_analysis[start_idx:end_idx]
                    
                    for j, image in enumerate(section_images):
                        location = image.get("location", "")
                        image_name = image.get("image_name", f"image_{start_idx + j}")
                        
                        # ✅ 템플릿별 이미지 검색 쿼리 개선
                        search_query = f"magazine image layout {template} {location} {image_name}"
                        clean_query = self.isolation_manager.clean_query_from_azure_keywords(search_query)
                        
                        image_patterns = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda q=clean_query: self.vector_manager.search_similar_layouts(q, "magazine_layout", top_k=3)
                        )
                        
                        isolated_image_patterns = self.isolation_manager.filter_contaminated_data(
                            image_patterns, f"image_patterns_template_{i}_{j}"
                        )
                        
                        # ✅ 이미지 패턴에 템플릿 정보 추가
                        for pattern in isolated_image_patterns:
                            pattern["assigned_template"] = template
                            pattern["section_index"] = i
                            pattern["image_index"] = start_idx + j
                            pattern["image_name"] = image_name
                        
                        patterns["image_patterns"].extend(isolated_image_patterns)
                    
                    self.logger.debug(f"템플릿 {template} (섹션 {i}): {len(section_images)}개 이미지, 패턴 수집 완료")
            
            # ✅ 통합 패턴 검색 (템플릿 할당 정보 포함)
            total_sections = len(selected_templates)  # ✅ 실제 할당된 템플릿 수 사용
            total_images = len(image_analysis)
            template_types = list(set(selected_templates))  # 고유 템플릿 타입
            
            integration_query = f"magazine integration {total_sections} sections {total_images} images templates {' '.join(template_types[:3])}"
            clean_integration_query = self.isolation_manager.clean_query_from_azure_keywords(integration_query)
            
            integration_patterns = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(clean_integration_query, "magazine_layout", top_k=5)
            )
            
            isolated_integration_patterns = self.isolation_manager.filter_contaminated_data(
                integration_patterns, "integration_patterns"
            )
            
            
            patterns["integration_patterns"] = isolated_integration_patterns
            
            # ✅ 패턴 수집 결과 로깅 (개선된 정보)
            self.logger.info(f"AI Search 패턴 수집 완료:")
            self.logger.info(f"  - 할당된 템플릿: {selected_templates}")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"AI Search 패턴 수집 실패: {e}")
            

    
    async def _process_unified_result_with_ai_search(self, crew_result: Any, magazine_content: Dict,
                                                   image_analysis: List[Dict], available_templates: List[str],
                                                   ai_search_patterns: Dict) -> Dict:
        """AI Search 통합 결과 후처리"""
        
        try:
            # CrewAI 결과 안전하게 추출
            if hasattr(crew_result, 'raw'):
                result_content = crew_result.raw
            elif hasattr(crew_result, 'result'):
                result_content = crew_result.result
            elif hasattr(crew_result, 'output'):
                result_content = crew_result.output
            else:
                result_content = str(crew_result)
            
            # 결과 파싱
            if isinstance(result_content, str):
                try:
                    parsed_result = json.loads(result_content)
                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 AI Search 패턴 기반 기본 구조 생성
                    parsed_result = self._generate_ai_search_based_structure(
                        magazine_content, image_analysis, available_templates, ai_search_patterns
                    )
            else:
                parsed_result = result_content if isinstance(result_content, dict) else {}
            
            # 기본 구조 확인 및 보완
            if not isinstance(parsed_result, dict):
                parsed_result = self._generate_ai_search_based_structure(
                    magazine_content, image_analysis, available_templates, ai_search_patterns
                )
            
            # 필수 필드 보장
            if "content_sections" not in parsed_result:
                parsed_result["content_sections"] = []
            if "selected_templates" not in parsed_result:
                parsed_result["selected_templates"] = available_templates[:len(parsed_result["content_sections"])]
            
            # AI Search 강화 메타데이터 추가
            parsed_result["integration_metadata"] = {
                "source": "unified_multimodal_agent_ai_search",
                "total_sections": len(parsed_result["content_sections"]),
                "multimodal_processing": True,
                "semantic_optimization": True,
                "ai_search_enhanced": True,
                "vector_patterns_used": True,
                "isolation_applied": True,
                "agent_integration": {
                    "content_structure": True,
                    "image_layout": True,
                    "semantic_coordination": True,
                    "ai_search_integration": True
                },
                "ai_search_patterns": {
                    "text_patterns_count": len(ai_search_patterns.get("text_patterns", [])),
                    "image_patterns_count": len(ai_search_patterns.get("image_patterns", [])),
                    "integration_patterns_count": len(ai_search_patterns.get("integration_patterns", []))
                }
            }
            
            # AI Search 패턴 기반 섹션 강화
            enhanced_sections = await self._enhance_sections_with_ai_search(
                parsed_result["content_sections"], ai_search_patterns
            )
            parsed_result["content_sections"] = enhanced_sections
            
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"AI Search 통합 결과 처리 실패: {e}")
            
            # AI Search 기반 폴백 결과 생성
            return self._generate_ai_search_fallback_result(
                magazine_content, image_analysis, available_templates, ai_search_patterns, e
            )
    
    def _generate_ai_search_based_structure(self, magazine_content: Dict, image_analysis: List[Dict],
                                          available_templates: List[str], ai_search_patterns: Dict) -> Dict:
        """AI Search 패턴 기반 기본 구조 생성"""
        
        sections = []
        text_patterns = ai_search_patterns.get("text_patterns", [])
        image_patterns = ai_search_patterns.get("image_patterns", [])
        
        # 원본 콘텐츠에서 섹션 생성
        original_sections = magazine_content.get("sections", [])
        
        for i, template in enumerate(available_templates[:len(original_sections)]):
            original_section = original_sections[i] if i < len(original_sections) else {}
            
            # AI Search 패턴 기반 섹션 생성
            section = {
                "template": template,
                "title": original_section.get("title", f"여행 이야기 {i+1}"),
                "subtitle": "특별한 순간들",
                "body": original_section.get("content", "멋진 여행 경험을 공유합니다.")[:500],
                "tagline": "TRAVEL & CULTURE",
                "images": [],
                "metadata": {
                    "ai_search_enhanced": True,
                    "text_patterns_applied": len(text_patterns) > 0,
                    "image_patterns_applied": len(image_patterns) > 0
                }
            }
            
            # AI Search 패턴 기반 개선
            if text_patterns and i < len(text_patterns):
                pattern = text_patterns[i]
                section["title"] = self._apply_text_pattern_to_title(section["title"], pattern)
                section["body"] = self._apply_text_pattern_to_body(section["body"], pattern)
            
            # 이미지 할당 (AI Search 패턴 기반)
            if image_analysis and i < len(image_analysis):
                assigned_images = image_analysis[i:i+2]  # 최대 2개 이미지
                section["images"] = [img.get("image_url", "") for img in assigned_images if img.get("image_url")]
            
            sections.append(section)
        
        return {
            "selected_templates": available_templates[:len(sections)],
            "content_sections": sections
        }
    
    def _apply_text_pattern_to_title(self, original_title: str, pattern: Dict) -> str:
        """AI Search 패턴을 제목에 적용"""
        
        title_style = pattern.get("title_style", "")
        if "descriptive" in title_style:
            return f"{original_title}: 특별한 발견"
        elif "emotional" in title_style:
            return f"{original_title}의 감동"
        else:
            return original_title
    
    def _apply_text_pattern_to_body(self, original_body: str, pattern: Dict) -> str:
        """AI Search 패턴을 본문에 적용"""
        
        text_style = pattern.get("text_structure", "")
        conclusion_style = pattern.get("conclusion_style", "")
        
        enhanced_body = original_body
        
        # 글의 맺음 스타일 적용
        if conclusion_style == "reflective":
            enhanced_body += " 이 순간들이 오래도록 기억에 남을 것입니다."
        elif conclusion_style == "inspiring":
            enhanced_body += " 새로운 여행에 대한 영감을 얻게 됩니다."
        
        return enhanced_body
    
    async def _enhance_sections_with_ai_search(self, sections: List[Dict], ai_search_patterns: Dict) -> List[Dict]:
        """AI Search 패턴으로 섹션 강화"""
        
        enhanced_sections = []
        text_patterns = ai_search_patterns.get("text_patterns", [])
        image_patterns = ai_search_patterns.get("image_patterns", [])
        
        for i, section in enumerate(sections):
            enhanced_section = section.copy()
            
            # AI Search 텍스트 패턴 적용
            if text_patterns and i < len(text_patterns):
                pattern = text_patterns[i]
                enhanced_section = self._apply_ai_search_text_pattern(enhanced_section, pattern)
            
            # AI Search 이미지 패턴 적용
            if image_patterns and i < len(image_patterns):
                pattern = image_patterns[i]
                enhanced_section = self._apply_ai_search_image_pattern(enhanced_section, pattern)
            
            # AI Search 메타데이터 추가
            enhanced_section["ai_search_metadata"] = {
                "text_pattern_applied": i < len(text_patterns),
                "image_pattern_applied": i < len(image_patterns),
                "pattern_source": "ai_search_vector_database"
            }
            
            enhanced_sections.append(enhanced_section)
        
        return enhanced_sections
    
    def _apply_ai_search_text_pattern(self, section: Dict, pattern: Dict) -> Dict:
        """AI Search 텍스트 패턴을 섹션에 적용"""
        
        # 문장 길이 패턴 적용
        sentence_length = pattern.get("sentence_length", "medium")
        if sentence_length == "short":
            # 짧은 문장으로 변환
            body = section.get("body", "")
            sentences = body.split(". ")
            if len(sentences) > 2:
                section["body"] = ". ".join(sentences[:2]) + "."
        
        # 글의 형태 패턴 적용
        text_structure = pattern.get("text_structure", "narrative")
        if text_structure == "descriptive":
            section["subtitle"] = f"{section.get('subtitle', '')}: 생생한 현장"
        elif text_structure == "conversational":
            section["subtitle"] = f"{section.get('subtitle', '')}: 이야기가 있는 곳"
        
        return section
    
    def _apply_ai_search_image_pattern(self, section: Dict, pattern: Dict) -> Dict:
        """AI Search 이미지 패턴을 섹션에 적용"""
        
        # 이미지 크기 패턴 적용
        image_size = pattern.get("image_size", "medium")
        placement = pattern.get("placement", "top")
        
        # 레이아웃 설정에 AI Search 패턴 반영
        if "layout_config" not in section:
            section["layout_config"] = {}
        
        section["layout_config"].update({
            "ai_search_image_size": image_size,
            "ai_search_placement": placement,
            "pattern_enhanced": True
        })
        
        return section
    
    def _generate_ai_search_fallback_result(self, magazine_content: Dict, image_analysis: List[Dict],
                                      available_templates: List[str], ai_search_patterns: Dict, error: Exception) -> Dict:
        """AI Search 기반 폴백 결과 생성 (원본 데이터 활용)"""
        
        #  원본 데이터 활용
        original_sections = magazine_content.get("sections", [])
        fallback_sections = []
        
        for i, template in enumerate(available_templates[:3]):
            #  magazine_content에서 원본 데이터 추출
            original_section = original_sections[i] if i < len(original_sections) else {}
            
            section = {
                "template": template,
                #  원본 제목 사용 (없으면 기본값)
                "title": original_section.get("title", f"여행 이야기 {i+1}"),
                "subtitle": "특별한 순간들",
                #  원본 콘텐츠 사용 (오류 메시지 추가)
                "body": f"{original_section.get('content', '멋진 여행 경험을 공유합니다.')[:300]}... (AI Search 통합 처리 중 오류 발생: {str(error)[:100]})",
                "tagline": "TRAVEL & CULTURE",
                #  image_analysis에서 이미지 할당
                "images": self._assign_fallback_images(image_analysis, i),
                "metadata": {
                    "fallback_used": True,
                    "error": str(error),
                    "ai_search_attempted": True,
                    "original_content_preserved": True,
                    "original_title_used": bool(original_section.get("title")),
                    "original_content_used": bool(original_section.get("content"))
                }
            }
            
            #  AI Search 패턴이 있으면 적용 (원본 데이터 기반)
            text_patterns = ai_search_patterns.get("text_patterns", [])
            if text_patterns and i < len(text_patterns):
                pattern = text_patterns[i]
                # 원본 제목을 기반으로 패턴 적용
                section["title"] = self._apply_text_pattern_to_title(section["title"], pattern)
                section["metadata"]["text_pattern_applied"] = True
            
            #  이미지 패턴 적용
            image_patterns = ai_search_patterns.get("image_patterns", [])
            if image_patterns and i < len(image_patterns):
                pattern = image_patterns[i]
                section = self._apply_ai_search_image_pattern(section, pattern)
                section["metadata"]["image_pattern_applied"] = True
            
            fallback_sections.append(section)
        
        return {
            "selected_templates": available_templates[:3],
            "content_sections": fallback_sections,
            "integration_metadata": {
                "source": "unified_multimodal_agent_ai_search_fallback",
                "total_sections": len(fallback_sections),
                "fallback_used": True,
                "error": str(error),
                "ai_search_enhanced": len(ai_search_patterns.get("text_patterns", [])) > 0,
                "vector_patterns_used": len(ai_search_patterns.get("text_patterns", [])) > 0,
                "isolation_applied": True,
                "original_data_preserved": True,  #  원본 데이터 보존 표시
                "ai_search_patterns": {
                    "text_patterns_count": len(ai_search_patterns.get("text_patterns", [])),
                    "image_patterns_count": len(ai_search_patterns.get("image_patterns", [])),
                    "integration_patterns_count": len(ai_search_patterns.get("integration_patterns", []))
                },
                "fallback_statistics": {  #  폴백 통계 추가
                    "original_sections_available": len(original_sections),
                    "images_available": len(image_analysis),
                    "sections_generated": len(fallback_sections)
                }
            }
        }

    def _assign_fallback_images(self, image_analysis: List[Dict], section_index: int) -> List[str]:
        """폴백 모드에서 이미지 할당"""
        
        if not image_analysis:
            return []
        
        # 섹션별로 이미지 분배 (최대 2개씩)
        start_index = section_index * 2
        end_index = start_index + 2
        
        assigned_images = []
        for i in range(start_index, min(end_index, len(image_analysis))):
            image = image_analysis[i]
            image_url = image.get("image_url", "")
            if image_url:
                assigned_images.append(image_url)
        
        return assigned_images
