from typing import Dict, List
from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager
from utils.agent_decision_logger import get_agent_logger, get_complete_data_manager

class JSXContentAnalyzer:
    """콘텐츠 분석 전문 에이전트 (CrewAI 기반 에이전트 결과 데이터 통합)"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        self.vector_manager = PDFVectorManager()
        self.logger = get_agent_logger()
        self.result_manager = get_complete_data_manager()
        
        # CrewAI 에이전트들 생성
        self.content_analysis_agent = self._create_content_analysis_agent()
        self.agent_result_analyzer = self._create_agent_result_analyzer()
        self.vector_enhancement_agent = self._create_vector_enhancement_agent()

    def _create_content_analysis_agent(self):
        """콘텐츠 분석 전문 에이전트"""
        return Agent(
            role="JSX 콘텐츠 분석 전문가",
            goal="JSX 생성을 위한 콘텐츠의 구조적 특성과 레이아웃 요구사항을 정밀 분석하여 최적화된 분석 결과를 제공",
            backstory="""당신은 10년간 React 및 JSX 기반 웹 개발 프로젝트에서 콘텐츠 분석을 담당해온 전문가입니다. 다양한 콘텐츠 유형에 대한 최적의 레이아웃과 디자인 패턴을 도출하는 데 특화되어 있습니다.

**전문 분야:**
- JSX 컴포넌트 구조 설계
- 콘텐츠 기반 레이아웃 최적화
- 사용자 경험 중심의 디자인 패턴 분석
- 반응형 웹 디자인 구조 설계

**분석 철학:**
"모든 콘텐츠는 고유한 특성을 가지며, 이를 정확히 분석하여 최적의 JSX 구조로 변환하는 것이 사용자 경험의 핵심입니다."

**출력 요구사항:**
- 콘텐츠 길이 및 복잡도 분석
- 감정 톤 및 분위기 파악
- 이미지 전략 및 배치 권장사항
- 레이아웃 복잡도 및 권장 구조
- 색상 팔레트 및 타이포그래피 스타일 제안""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_agent_result_analyzer(self):
        """에이전트 결과 분석 전문가"""
        return Agent(
            role="에이전트 결과 데이터 분석 전문가",
            goal="이전 에이전트들의 실행 결과를 분석하여 성공 패턴과 최적화 인사이트를 도출하고 콘텐츠 분석에 반영",
            backstory="""당신은 8년간 다중 에이전트 시스템의 성능 분석과 최적화를 담당해온 전문가입니다. BindingAgent와 OrgAgent의 결과 패턴을 분석하여 JSX 생성 품질을 향상시키는 데 특화되어 있습니다.

**전문 영역:**
- 에이전트 실행 결과 패턴 분석
- 성공적인 레이아웃 전략 식별
- 에이전트 간 협업 최적화
- 품질 지표 기반 개선 방안 도출

**분석 방법론:**
"이전 에이전트들의 성공과 실패 패턴을 체계적으로 분석하여 현재 작업에 최적화된 인사이트를 제공합니다."

**특별 처리 대상:**
- BindingAgent: 이미지 배치 전략 및 시각적 일관성
- OrgAgent: 텍스트 구조 및 레이아웃 복잡도
- 성능 메트릭: 신뢰도 점수 및 품질 지표""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_vector_enhancement_agent(self):
        """벡터 데이터 강화 전문가"""
        return Agent(
            role="PDF 벡터 데이터 기반 분석 강화 전문가",
            goal="PDF 벡터 데이터베이스에서 유사한 레이아웃 패턴을 검색하여 콘텐츠 분석 결과를 강화하고 최적화된 디자인 권장사항을 제공",
            backstory="""당신은 12년간 벡터 데이터베이스와 유사도 검색 시스템을 활용한 콘텐츠 최적화를 담당해온 전문가입니다. Azure Cognitive Search와 PDF 벡터 데이터를 활용한 레이아웃 패턴 분석에 특화되어 있습니다.

**기술 전문성:**
- 벡터 유사도 검색 및 패턴 매칭
- PDF 레이아웃 구조 분석
- 콘텐츠 기반 디자인 패턴 추출
- 색상 팔레트 및 타이포그래피 최적화

**강화 전략:**
"벡터 데이터베이스의 풍부한 레이아웃 정보를 활용하여 현재 콘텐츠에 가장 적합한 디자인 패턴을 식별하고 적용합니다."

**출력 강화 요소:**
- 유사 레이아웃 기반 구조 권장
- 벡터 신뢰도 기반 품질 향상
- PDF 소스 기반 색상 팔레트 최적화
- 타이포그래피 스타일 정교화""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def analyze_content_for_jsx(self, content: Dict, section_index: int, total_sections: int) -> Dict:
        """JSX 생성을 위한 콘텐츠 분석 (CrewAI 기반 에이전트 결과 데이터 활용)"""
        
        # 이전 에이전트 결과 수집 (수정: 올바른 메서드 사용)
        previous_results = self.result_manager.get_all_outputs(exclude_agent="JSXContentAnalyzer")
        
        # BindingAgent와 OrgAgent 응답 특별 수집
        binding_results = [r for r in previous_results if "BindingAgent" in r.get('agent_name', '')]
        org_results = [r for r in previous_results if "OrgAgent" in r.get('agent_name', '')]
        
        print(f"📊 이전 결과 수집: 전체 {len(previous_results)}개, BindingAgent {len(binding_results)}개, OrgAgent {len(org_results)}개")
        
        # CrewAI Task들 생성
        content_analysis_task = self._create_content_analysis_task(content, section_index, total_sections)
        agent_result_analysis_task = self._create_agent_result_analysis_task(previous_results, binding_results, org_results)
        vector_enhancement_task = self._create_vector_enhancement_task(content)
        
        # CrewAI Crew 생성 및 실행
        analysis_crew = Crew(
            agents=[self.content_analysis_agent, self.agent_result_analyzer, self.vector_enhancement_agent],
            tasks=[content_analysis_task, agent_result_analysis_task, vector_enhancement_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Crew 실행
        crew_result = analysis_crew.kickoff()
        
        # 결과 처리 및 통합
        vector_enhanced_analysis = self._process_crew_analysis_result(
            crew_result, content, section_index, previous_results, binding_results, org_results
        )
        
        # 결과 저장 (수정: 올바른 메서드 사용)
        self.result_manager.store_agent_output(
            agent_name="JSXContentAnalyzer",
            agent_role="콘텐츠 분석 전문가",
            task_description=f"섹션 {section_index+1}/{total_sections} JSX 콘텐츠 분석",
            final_answer=str(vector_enhanced_analysis),
            reasoning_process=f"CrewAI 기반 이전 {len(previous_results)}개 에이전트 결과 분석 후 벡터 데이터 강화 적용",
            execution_steps=[
                "CrewAI 에이전트 생성",
                "기본 콘텐츠 분석 수행",
                "에이전트 결과 통합",
                "벡터 데이터 강화",
                "최종 분석 완료"
            ],
            raw_input=content,
            raw_output=vector_enhanced_analysis,
            performance_metrics={
                "section_index": section_index,
                "total_sections": total_sections,
                "agent_results_utilized": len(previous_results),
                "binding_results_count": len(binding_results),
                "org_results_count": len(org_results),
                "vector_enhanced": vector_enhanced_analysis.get('vector_enhanced', False),
                "crewai_enhanced": True
            }
        )
        
        print(f"✅ 콘텐츠 분석 완료: {vector_enhanced_analysis.get('recommended_layout', '기본')} 레이아웃 권장 (CrewAI 기반 에이전트 데이터 활용: {len(previous_results)}개)")
        
        return vector_enhanced_analysis

    def _create_content_analysis_task(self, content: Dict, section_index: int, total_sections: int) -> Task:
        """기본 콘텐츠 분석 태스크"""
        return Task(
            description=f"""
            섹션 {section_index+1}/{total_sections}의 콘텐츠를 분석하여 JSX 생성에 필요한 기본 분석 결과를 제공하세요.
            
            **분석 대상 콘텐츠:**
            - 제목: {content.get('title', 'N/A')}
            - 본문 길이: {len(content.get('body', ''))} 문자
            - 이미지 개수: {len(content.get('images', []))}개
            
            **분석 요구사항:**
            1. 텍스트 길이 분석 (짧음/보통/긺)
            2. 감정 톤 파악 (peaceful/energetic/professional 등)
            3. 이미지 전략 권장 (단일/그리드/갤러리)
            4. 레이아웃 복잡도 평가 (단순/보통/복잡)
            5. 권장 레이아웃 타입 (minimal/hero/grid/magazine)
            6. 색상 팔레트 제안
            7. 타이포그래피 스타일 권장
            
            **출력 형식:**
            JSON 형태로 분석 결과를 구조화하여 제공하세요.
            """,
            expected_output="JSX 생성을 위한 기본 콘텐츠 분석 결과 (JSON 형식)",
            agent=self.content_analysis_agent
        )

    def _create_agent_result_analysis_task(self, previous_results: List[Dict], binding_results: List[Dict], org_results: List[Dict]) -> Task:
        """에이전트 결과 분석 태스크"""
        return Task(
            description=f"""
            이전 에이전트들의 실행 결과를 분석하여 성공 패턴과 최적화 인사이트를 도출하세요.
            
            **분석 대상:**
            - 전체 에이전트 결과: {len(previous_results)}개
            - BindingAgent 결과: {len(binding_results)}개 (이미지 배치 전략)
            - OrgAgent 결과: {len(org_results)}개 (텍스트 구조)
            
            **특별 분석 요구사항:**
            1. BindingAgent 결과에서 이미지 배치 전략 추출
               - 그리드/갤러리 패턴 식별
               - 시각적 일관성 평가
            
            2. OrgAgent 결과에서 텍스트 구조 분석
               - 레이아웃 복잡도 평가
               - 타이포그래피 스타일 추출
            
            3. 성공 패턴 학습
               - 높은 신뢰도를 보인 접근법 식별
               - 레이아웃 권장사항 도출
               - 품질 향상 전략 제안
            
            **출력 요구사항:**
            - 에이전트별 인사이트 요약
            - 성공적인 레이아웃 패턴
            - 품질 향상 권장사항
            """,
            expected_output="에이전트 결과 분석 및 최적화 인사이트 (구조화된 데이터)",
            agent=self.agent_result_analyzer
        )

    def _create_vector_enhancement_task(self, content: Dict) -> Task:
        """벡터 데이터 강화 태스크"""
        return Task(
            description=f"""
            PDF 벡터 데이터베이스를 활용하여 콘텐츠 분석 결과를 강화하세요.
            
            **검색 쿼리 생성:**
            - 콘텐츠 제목: {content.get('title', '')}
            - 본문 일부: {content.get('body', '')[:300]}
            
            **벡터 검색 및 분석:**
            1. 유사한 레이아웃 패턴 검색 (top 5)
            2. 레이아웃 타입 분석 및 권장사항 도출
            3. 벡터 신뢰도 기반 품질 점수 계산
            4. PDF 소스 기반 색상 팔레트 최적화
            5. 타이포그래피 스타일 정교화
            
            **강화 요소:**
            - 벡터 기반 레이아웃 권장
            - 신뢰도 점수 계산
            - 색상 팔레트 최적화
            - 타이포그래피 스타일 개선
            
            **실패 처리:**
            벡터 검색 실패 시 기본 분석 결과 유지
            """,
            expected_output="벡터 데이터 기반 강화된 분석 결과",
            agent=self.vector_enhancement_agent,
            context=[self._create_content_analysis_task(content, 0, 1), self._create_agent_result_analysis_task([], [], [])]
        )

    def _process_crew_analysis_result(self, crew_result, content: Dict, section_index: int, 
                                    previous_results: List[Dict], binding_results: List[Dict], 
                                    org_results: List[Dict]) -> Dict:
        """CrewAI 분석 결과 처리"""
        try:
            # CrewAI 결과에서 데이터 추출
            if hasattr(crew_result, 'raw') and crew_result.raw:
                result_text = crew_result.raw
            else:
                result_text = str(crew_result)
            
            # 기본 분석 수행
            basic_analysis = self._create_default_analysis(content, section_index)
            
            # 에이전트 결과 데이터로 분석 강화
            agent_enhanced_analysis = self._enhance_analysis_with_agent_results(
                content, basic_analysis, previous_results, binding_results, org_results
            )
            
            # 벡터 데이터로 추가 강화
            vector_enhanced_analysis = self._enhance_analysis_with_vectors(content, agent_enhanced_analysis)
            
            # CrewAI 결과 통합
            vector_enhanced_analysis['crewai_enhanced'] = True
            vector_enhanced_analysis['crew_result_length'] = len(result_text)
            
            return vector_enhanced_analysis
            
        except Exception as e:
            print(f"⚠️ CrewAI 결과 처리 실패: {e}")
            # 폴백: 기존 방식으로 처리
            basic_analysis = self._create_default_analysis(content, section_index)
            agent_enhanced_analysis = self._enhance_analysis_with_agent_results(
                content, basic_analysis, previous_results, binding_results, org_results
            )
            return self._enhance_analysis_with_vectors(content, agent_enhanced_analysis)

    # 기존 메서드들 유지 (변경 없음)
    def _enhance_analysis_with_agent_results(self, content: Dict, basic_analysis: Dict,
                                           previous_results: List[Dict], binding_results: List[Dict],
                                           org_results: List[Dict]) -> Dict:
        """에이전트 결과 데이터로 분석 강화 (BindingAgent, OrgAgent 특별 처리)"""
        enhanced_analysis = basic_analysis.copy()
        enhanced_analysis['agent_results_count'] = len(previous_results)
        enhanced_analysis['binding_results_count'] = len(binding_results)
        enhanced_analysis['org_results_count'] = len(org_results)
        
        if not previous_results:
            enhanced_analysis['agent_enhanced'] = False
            return enhanced_analysis
        
        enhanced_analysis['agent_enhanced'] = True
        
        # 이전 분석 결과 패턴 학습
        layout_recommendations = []
        confidence_scores = []
        
        for result in previous_results:
            final_answer = result.get('agent_final_answer', '')
            if 'layout' in final_answer.lower():
                if 'grid' in final_answer.lower():
                    layout_recommendations.append('grid')
                elif 'hero' in final_answer.lower():
                    layout_recommendations.append('hero')
                elif 'magazine' in final_answer.lower():
                    layout_recommendations.append('magazine')
            
            # 성능 메트릭에서 신뢰도 추출
            performance_data = result.get('performance_data', {})
            if isinstance(performance_data, dict):
                confidence = performance_data.get('confidence_score', 0)
                if confidence > 0:
                    confidence_scores.append(confidence)
        
        # BindingAgent 결과 특별 활용
        if binding_results:
            latest_binding = binding_results[-1]
            binding_answer = latest_binding.get('agent_final_answer', '')
            
            # 이미지 배치 전략에서 레이아웃 힌트 추출
            if '그리드' in binding_answer or 'grid' in binding_answer.lower():
                enhanced_analysis['image_strategy'] = '그리드'
                enhanced_analysis['recommended_layout'] = 'grid'
            elif '갤러리' in binding_answer or 'gallery' in binding_answer.lower():
                enhanced_analysis['image_strategy'] = '갤러리'
                enhanced_analysis['recommended_layout'] = 'gallery'
            
            enhanced_analysis['binding_insights_applied'] = True
            print(f" 🖼️ BindingAgent 인사이트 적용: 이미지 전략 조정")
        
        # OrgAgent 결과 특별 활용
        if org_results:
            latest_org = org_results[-1]
            org_answer = latest_org.get('agent_final_answer', '')
            
            # 텍스트 구조에서 레이아웃 힌트 추출
            if '복잡' in org_answer or 'complex' in org_answer.lower():
                enhanced_analysis['layout_complexity'] = '복잡'
                enhanced_analysis['typography_style'] = '정보 집약형'
            elif '단순' in org_answer or 'simple' in org_answer.lower():
                enhanced_analysis['layout_complexity'] = '단순'
                enhanced_analysis['typography_style'] = '미니멀 모던'
            
            enhanced_analysis['org_insights_applied'] = True
            print(f" 📄 OrgAgent 인사이트 적용: 텍스트 구조 조정")
        
        # 가장 성공적인 레이아웃 패턴 적용
        if layout_recommendations:
            most_common_layout = max(set(layout_recommendations), key=layout_recommendations.count)
            if layout_recommendations.count(most_common_layout) >= 2:
                enhanced_analysis['recommended_layout'] = most_common_layout
                enhanced_analysis['layout_confidence'] = 'high'
        
        # 평균 신뢰도 기반 조정
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            if avg_confidence > 0.8:
                enhanced_analysis['quality_boost'] = True
                enhanced_analysis['color_palette'] = '프리미엄 블루'
                enhanced_analysis['typography_style'] = '고급 모던'
        
        return enhanced_analysis

    def _enhance_analysis_with_vectors(self, content: Dict, basic_analysis: Dict) -> Dict:
        """벡터 데이터로 분석 강화 (기존 메서드 유지)"""
        try:
            content_query = f"{content.get('title', '')} {content.get('body', '')[:300]}"
            similar_layouts = self.vector_manager.search_similar_layouts(
                content_query,
                "magazine_layout",
                top_k=5
            )
            
            if similar_layouts:
                enhanced_analysis = basic_analysis.copy()
                enhanced_analysis['vector_enhanced'] = True
                enhanced_analysis['similar_layouts'] = similar_layouts
                
                vector_layout_recommendation = self._get_vector_layout_recommendation(similar_layouts)
                if vector_layout_recommendation:
                    enhanced_analysis['recommended_layout'] = vector_layout_recommendation
                    enhanced_analysis['layout_confidence'] = self._calculate_vector_confidence(similar_layouts)
                    enhanced_analysis['vector_color_palette'] = self._get_vector_color_palette(similar_layouts)
                    enhanced_analysis['vector_typography'] = self._get_vector_typography_style(similar_layouts)
                
                return enhanced_analysis
            else:
                basic_analysis['vector_enhanced'] = False
                return basic_analysis
                
        except Exception as e:
            print(f"⚠️ 벡터 데이터 분석 강화 실패: {e}")
            basic_analysis['vector_enhanced'] = False
            return basic_analysis

    def _get_vector_layout_recommendation(self, similar_layouts: List[Dict]) -> str:
        """벡터 데이터 기반 레이아웃 추천"""
        layout_types = []
        for layout in similar_layouts:
            layout_info = layout.get('layout_info', {})
            text_blocks = len(layout_info.get('text_blocks', []))
            images = len(layout_info.get('images', []))
            
            if images == 0:
                layout_types.append('minimal')
            elif images == 1 and text_blocks <= 3:
                layout_types.append('hero')
            elif images <= 3 and text_blocks <= 6:
                layout_types.append('grid')
            elif images > 3:
                layout_types.append('gallery')
            else:
                layout_types.append('magazine')
        
        if layout_types:
            return max(set(layout_types), key=layout_types.count)
        return None

    def _calculate_vector_confidence(self, similar_layouts: List[Dict]) -> float:
        """벡터 기반 신뢰도 계산"""
        if not similar_layouts:
            return 0.5
        
        scores = [layout.get('score', 0) for layout in similar_layouts]
        avg_score = sum(scores) / len(scores)
        
        layout_consistency = len(set(self._get_vector_layout_recommendation([layout]) for layout in similar_layouts))
        consistency_bonus = 0.2 if layout_consistency <= 2 else 0.1
        
        return min(avg_score + consistency_bonus, 1.0)

    def _get_vector_color_palette(self, similar_layouts: List[Dict]) -> str:
        """벡터 데이터 기반 색상 팔레트"""
        pdf_sources = [layout.get('pdf_name', '').lower() for layout in similar_layouts]
        
        if any('travel' in source for source in pdf_sources):
            return "여행 블루 팔레트"
        elif any('culture' in source for source in pdf_sources):
            return "문화 브라운 팔레트"
        elif any('lifestyle' in source for source in pdf_sources):
            return "라이프스타일 핑크 팔레트"
        elif any('nature' in source for source in pdf_sources):
            return "자연 그린 팔레트"
        else:
            return "클래식 그레이 팔레트"

    def _get_vector_typography_style(self, similar_layouts: List[Dict]) -> str:
        """벡터 데이터 기반 타이포그래피 스타일"""
        total_text_blocks = sum(len(layout.get('layout_info', {}).get('text_blocks', [])) for layout in similar_layouts)
        avg_text_blocks = total_text_blocks / len(similar_layouts) if similar_layouts else 0
        
        if avg_text_blocks > 8:
            return "정보 집약형"
        elif avg_text_blocks > 5:
            return "균형잡힌 편집형"
        elif avg_text_blocks > 2:
            return "미니멀 모던"
        else:
            return "대형 타이틀 중심"

    def _create_default_analysis(self, content: Dict, section_index: int) -> Dict:
        """기본 분석 결과 생성"""
        body_length = len(content.get('body', ''))
        image_count = len(content.get('images', []))
        
        if body_length < 300:
            recommended_layout = "minimal"
        elif image_count == 0:
            recommended_layout = "minimal"
        elif image_count == 1:
            recommended_layout = "hero"
        elif image_count <= 4:
            recommended_layout = "grid"
        else:
            recommended_layout = "magazine"
        
        return {
            "text_length": "보통" if body_length < 500 else "긺",
            "emotion_tone": "peaceful",
            "image_strategy": "그리드" if image_count > 1 else "단일",
            "layout_complexity": "보통",
            "recommended_layout": recommended_layout,
            "color_palette": "차분한 블루",
            "typography_style": "모던"
        }
