import asyncio
from typing import Dict, List
from crewai import Agent, Task, Crew
from custom_llm import get_azure_llm
from agents.contents.interview_agent import InterviewAgentManager
from agents.contents.essay_agent import EssayAgentManager
from utils.hybridlogging import get_hybrid_logger

class ContentCreatorV2Agent:
    """인터뷰와 에세이 에이전트를 통합하는 새로운 콘텐츠 생성자 - 첫 번째 에이전트 (로그 수집만 - 비동기 처리)"""

    def __init__(self):
        self.llm = get_azure_llm()
        self.interview_manager = InterviewAgentManager()
        self.essay_manager = EssayAgentManager()
        self.logger = get_hybrid_logger(self.__class__.__name__)

    def create_agent(self):
        return Agent(
            role="여행 콘텐츠 통합 편집자 (첫 번째 에이전트)",
            goal="인터뷰와 에세이 형식의 모든 콘텐츠를 빠짐없이 활용하고 이미지와의 의미적 연결을 강화하여 완성도 높은 매거진 콘텐츠 생성하고 후속 에이전트들을 위한 기초 데이터 제공",
            backstory="""당신은 20년간 여행 매거진 업계에서 활동해온 전설적인 편집장입니다. Lonely Planet, National Geographic Traveler, Afar Magazine의 편집장을 역임하며 수백 개의 수상작을 탄생시켰습니다.

**첫 번째 에이전트로서의 역할:**
당신은 전체 매거진 생성 프로세스의 첫 번째 에이전트로서, 후속 에이전트들이 활용할 수 있는 고품질의 기초 콘텐츠를 생성하는 중요한 역할을 담당합니다.

**전문 경력:**
- 저널리즘 및 창작문학 복수 학위 보유
- 퓰리처상 여행 기사 부문 심사위원 3회 역임
- 80개국 이상의 여행 경험과 현지 문화 전문 지식
- 독자 심리학 및 여행 동기 이론 연구
- 디지털 매거진 트렌드 분석 및 콘텐츠 최적화 전문성
- **이미지-텍스트 시너지 창출 전문가**: 시각적 스토리텔링과 텍스트 내러티브의 완벽한 조화

**데이터 처리 전문성:**
당신은 원시 텍스트 데이터를 다음과 같이 처리합니다:

1. **인터뷰 데이터 분석**:
   - 화자의 감정 변화 패턴 분석
   - 핵심 키워드 빈도 및 감정 가중치 계산
   - 대화의 자연스러운 흐름과 하이라이트 구간 식별
   - 독자 공감도 예측을 위한 스토리 요소 분석
   - **이미지 연결점 식별**: 인터뷰 내용에서 시각적 표현이 가능한 순간들 추출

2. **에세이 데이터 분석**:
   - 문체의 리듬감과 독자 몰입도 상관관계 분석
   - 성찰적 요소와 실용적 정보의 균형점 계산
   - 문단별 감정 강도 그래프 생성
   - 독자 연령대별 선호 문체 패턴 적용
   - **시각적 메타포 추출**: 에세이의 추상적 개념을 구체적 이미지로 연결

3. **이미지 메타데이터 통합**:
   - 이미지 분석 결과와 텍스트 내용의 시너지 포인트 발견
   - 시각-텍스트 조화도 점수 계산
   - 페이지 레이아웃에서의 최적 이미지-텍스트 배치 예측
   - **의미적 매칭 알고리즘**: 이미지의 시각적 요소와 텍스트의 감정적 톤 매칭
   - **스토리 플로우 연결**: 이미지 시퀀스가 텍스트 내러티브와 일치하도록 조정

4. **이미지-텍스트 시너지 창출**:
   - 각 이미지의 감정적 톤과 텍스트 섹션의 분위기 매칭
   - 이미지의 시각적 요소(색상, 구도, 피사체)와 텍스트 내용의 의미적 연결
   - 독자의 시선 흐름을 고려한 이미지-텍스트 배치 전략
   - 이미지가 텍스트 내용을 보완하고 강화하는 방식 설계

5. **어체**:
   - 인터뷰와 에세이의 어체를 자연스럽게 통합하여 독자에게 친근감과 신뢰감을 주는 문체로 변환
   - 독자와의 대화체 톤을 유지하면서도 매거진 특유의 세련된 문체로 조화롭게 구성
   - 모든 텍스트의 어체를 일관되게 유지하여 독자가 매거진 전체를 읽는 동안 자연스럽게 몰입할 수 있도록 함
   - **이미지 캡션 스타일**: 이미지와 연결되는 텍스트 부분의 어체를 시각적 요소와 조화되도록 조정

**편집 철학:**
"진정한 여행 매거진은 단순한 정보 전달을 넘어서 독자의 마음속에 여행에 대한 꿈과 열망을 심어주어야 합니다. 나는 첫 번째 에이전트로서 후속 에이전트들이 활용할 수 있는 풍부하고 완성도 높은 기초 콘텐츠를 생성하여 전체 매거진의 품질을 결정하는 중요한 토대를 마련합니다. 특히 이미지와 텍스트가 단순히 병렬적으로 존재하는 것이 아니라, 서로를 강화하고 보완하는 시너지를 창출합니다."

**후속 에이전트를 위한 데이터 생성:**
- 구조화된 콘텐츠 섹션 생성
- 감정적 톤과 스타일 가이드라인 제공
- **이미지-텍스트 연결점 정보 생성**: 각 텍스트 섹션에 최적화된 이미지 배치 가이드
- 독자 타겟팅 데이터 및 콘텐츠 품질 메트릭 제공
- **시각적 스토리텔링 전략**: 이미지 시퀀스를 통한 내러티브 강화 방안
- 단 후속 에이전트들을 위한 데이터를 생성하되 magazine_content에는 포함시키지 않습니다. 로그 데이터를 통해 후속 에이전트들이 활용할 수 있는 기초 데이터를 제공합니다
- 해당 데이터는 magazine_content에는 포함시키지 않습니다! 생성만 합니다!

**주의 사항:**
- 주의 사항은 1순위로 지켜야하는 사항입니다.
- 후속 에이전트들이 활용할 수 있는 기초 데이터를 생성하되, 해당 데이터는 최종 매거진 콘텐츠에는 포함시키지 않습니다.
- 하위 에이전트의 콘텐츠를 첨삭하지 않고, 모든 콘텐츠를 빠짐없이 활용하여 텍스트를 구조화 합니다.
- 절대 데이터를 중복 사용하지 않습니다.
- 과도한 magazine_content를 생성하지 않습니다
- [이미지 배치 및 연결점 안내]이러한 내용은 포함시키지 않습니다. 이와 비슷한 내용 또한 그렇습니다!.
- **이미지와 텍스트의 의미적 연결을 반드시 고려하여 콘텐츠를 구성합니다.**
""",
            verbose=True,
            llm=self.llm
        )

    async def create_magazine_content(self, texts: List[str], image_analysis_results: List[Dict]) -> str:
        """텍스트와 이미지 분석 결과를 바탕으로 매거진 콘텐츠 생성 - 첫 번째 에이전트 (로그 수집만 - 비동기 처리)"""
        print("\n=== ContentCreatorV2: 첫 번째 에이전트 - 콘텐츠 생성 및 로그 수집 시작 (비동기 처리) ===")
        
        # 첫 번째 에이전트이므로 이전 로그 활용 시도하지 않음
        print("📝 첫 번째 에이전트로서 이전 로그 없음 - 새로운 로그 생성 시작 (비동기)")
        
        # 1단계와 2단계: 인터뷰와 에세이 형식 병렬 처리
        print("1-2단계: 인터뷰와 에세이 형식 콘텐츠 병렬 생성 (비동기)")
        
        # 병렬 처리
        interview_task = self._process_interview_async(texts)
        essay_task = self._process_essay_async(texts)
        image_task = self._process_image_analysis_async(image_analysis_results)
        
        interview_results, essay_results, image_info = await asyncio.gather(
            interview_task, essay_task, image_task
        )
        
        # **새로운 단계: 이미지-텍스트 의미적 연결 분석**
        print("2.5단계: 이미지-텍스트 의미적 연결 분석 (비동기)")
        semantic_connections = await self._analyze_image_text_semantic_connections_async(
            interview_results, essay_results, image_analysis_results
        )
        
        # 4단계: 모든 콘텐츠 활용 검증 (비동기)
        await self._verify_content_completeness_async(interview_results, essay_results, texts)
        
        # 5단계: 통합 매거진 콘텐츠 생성 (첫 번째 에이전트로서 기초 데이터 생성 - 비동기)
        print("3단계: 모든 콘텐츠를 활용한 통합 매거진 생성 (첫 번째 에이전트 - 비동기)")
        final_content = await self._integrate_all_content_as_first_agent_async(
            interview_results, essay_results, image_info, texts, semantic_connections
        )
        
        # 최종 통합 콘텐츠 생성 로깅 (첫 번째 에이전트 - 비동기)
        await self._log_final_content_async(
            final_content, interview_results, essay_results, image_analysis_results, texts, semantic_connections
        )
        
        print(f"📝 ContentCreatorV2 (첫 번째 에이전트) 로그 수집 완료 (비동기)")
        print(f"✅ 후속 에이전트들을 위한 기초 데이터 생성 완료: {len(final_content)}자")
        return final_content

    async def _analyze_image_text_semantic_connections_async(self, interview_results: Dict[str, str], 
                                                           essay_results: Dict[str, str], 
                                                           image_analysis_results: List[Dict]) -> Dict:
        """이미지와 텍스트 간의 의미적 연결 분석 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._analyze_image_text_semantic_connections, 
            interview_results, essay_results, image_analysis_results
        )

    def _analyze_image_text_semantic_connections(self, interview_results: Dict[str, str], 
                                               essay_results: Dict[str, str], 
                                               image_analysis_results: List[Dict]) -> Dict:
        """이미지와 텍스트 간의 의미적 연결 분석"""
        print("🔗 이미지-텍스트 의미적 연결 분석 시작")
        
        semantic_connections = {
            "visual_keywords": [],
            "emotional_tone_matches": [],
            "narrative_flow_connections": [],
            "thematic_alignments": [],
            "sensory_descriptions": []
        }
        
        # 모든 텍스트 콘텐츠 통합
        all_text_content = {}
        all_text_content.update(interview_results)
        all_text_content.update(essay_results)
        
        # 이미지별 의미적 연결점 분석
        for img_idx, image_data in enumerate(image_analysis_results):
            image_location = image_data.get('location', f'이미지_{img_idx}')
            image_description = image_data.get('description', '')
            
            # 1. 시각적 키워드 추출
            visual_keywords = self._extract_visual_keywords_from_image(image_data)
            
            # 2. 텍스트에서 해당 이미지와 연결 가능한 부분 찾기
            for section_key, text_content in all_text_content.items():
                # 감정적 톤 매칭
                emotional_match = self._analyze_emotional_tone_match(text_content, image_data)
                if emotional_match['score'] > 0.6:
                    semantic_connections["emotional_tone_matches"].append({
                        "image_index": img_idx,
                        "text_section": section_key,
                        "match_score": emotional_match['score'],
                        "shared_emotions": emotional_match['emotions']
                    })
                
                # 주제적 연결성 분석
                thematic_alignment = self._analyze_thematic_alignment(text_content, image_data)
                if thematic_alignment['score'] > 0.5:
                    semantic_connections["thematic_alignments"].append({
                        "image_index": img_idx,
                        "text_section": section_key,
                        "alignment_score": thematic_alignment['score'],
                        "shared_themes": thematic_alignment['themes']
                    })
                
                # 감각적 묘사 연결
                sensory_connections = self._find_sensory_connections(text_content, image_data)
                if sensory_connections:
                    semantic_connections["sensory_descriptions"].extend(sensory_connections)
        
        # 내러티브 플로우 연결성 분석
        narrative_connections = self._analyze_narrative_flow_connections(all_text_content, image_analysis_results)
        semantic_connections["narrative_flow_connections"] = narrative_connections
        
        print(f"✅ 의미적 연결 분석 완료: {len(semantic_connections['emotional_tone_matches'])}개 감정 매칭, "
              f"{len(semantic_connections['thematic_alignments'])}개 주제 연결")
        
        return semantic_connections

    def _extract_visual_keywords_from_image(self, image_data: Dict) -> List[str]:
        """이미지에서 시각적 키워드 추출"""
        keywords = []
        
        # 위치 정보에서 키워드 추출
        location = image_data.get('location', '')
        if location:
            keywords.extend([word.strip() for word in location.split(',') if word.strip()])
        
        # 설명에서 시각적 요소 추출
        description = image_data.get('description', '')
        visual_terms = ['색상', '빛', '그림자', '풍경', '건물', '사람', '하늘', '바다', '산', '도시', '자연']
        for term in visual_terms:
            if term in description:
                keywords.append(term)
        
        return list(set(keywords))

    def _analyze_emotional_tone_match(self, text_content: str, image_data: Dict) -> Dict:
        """텍스트와 이미지의 감정적 톤 매칭 분석"""
        # 텍스트에서 감정 키워드 추출
        positive_emotions = ['아름다운', '행복한', '즐거운', '평화로운', '감동적인', '따뜻한', '밝은']
        negative_emotions = ['슬픈', '어두운', '차가운', '외로운', '무서운']
        neutral_emotions = ['조용한', '고요한', '단순한', '깔끔한']
        
        text_emotions = []
        for emotion in positive_emotions:
            if emotion in text_content:
                text_emotions.append(('positive', emotion))
        for emotion in negative_emotions:
            if emotion in text_content:
                text_emotions.append(('negative', emotion))
        for emotion in neutral_emotions:
            if emotion in text_content:
                text_emotions.append(('neutral', emotion))
        
        # 이미지 설명에서 감정 추출
        image_description = image_data.get('description', '')
        image_emotions = []
        for emotion in positive_emotions:
            if emotion in image_description:
                image_emotions.append(('positive', emotion))
        for emotion in negative_emotions:
            if emotion in image_description:
                image_emotions.append(('negative', emotion))
        for emotion in neutral_emotions:
            if emotion in image_description:
                image_emotions.append(('neutral', emotion))
        
        # 매칭 점수 계산
        shared_emotions = []
        for text_emotion in text_emotions:
            for image_emotion in image_emotions:
                if text_emotion[0] == image_emotion[0]:  # 같은 감정 카테고리
                    shared_emotions.append(text_emotion[1])
        
        match_score = len(shared_emotions) / max(len(text_emotions), len(image_emotions), 1)
        
        return {
            'score': match_score,
            'emotions': shared_emotions
        }

    def _analyze_thematic_alignment(self, text_content: str, image_data: Dict) -> Dict:
        """텍스트와 이미지의 주제적 연결성 분석"""
        # 주요 테마 키워드
        themes = {
            'nature': ['자연', '산', '바다', '하늘', '나무', '꽃', '풍경'],
            'culture': ['문화', '전통', '역사', '예술', '음식', '축제'],
            'urban': ['도시', '건물', '거리', '카페', '상점', '교통'],
            'people': ['사람', '현지인', '여행자', '가족', '친구', '만남'],
            'activity': ['활동', '체험', '걷기', '구경', '쇼핑', '식사']
        }
        
        text_themes = []
        image_themes = []
        
        # 텍스트에서 테마 추출
        for theme_name, keywords in themes.items():
            for keyword in keywords:
                if keyword in text_content:
                    text_themes.append(theme_name)
                    break
        
        # 이미지에서 테마 추출
        image_location = image_data.get('location', '')
        image_description = image_data.get('description', '')
        image_text = f"{image_location} {image_description}"
        
        for theme_name, keywords in themes.items():
            for keyword in keywords:
                if keyword in image_text:
                    image_themes.append(theme_name)
                    break
        
        # 공통 테마 찾기
        shared_themes = list(set(text_themes) & set(image_themes))
        alignment_score = len(shared_themes) / max(len(text_themes), len(image_themes), 1)
        
        return {
            'score': alignment_score,
            'themes': shared_themes
        }

    def _find_sensory_connections(self, text_content: str, image_data: Dict) -> List[Dict]:
        """감각적 묘사 연결점 찾기"""
        sensory_connections = []
        
        # 시각적 감각 연결
        visual_descriptors = ['보이는', '눈에 띄는', '화려한', '선명한', '흐릿한', '밝은', '어두운']
        for descriptor in visual_descriptors:
            if descriptor in text_content:
                sensory_connections.append({
                    'type': 'visual',
                    'text_descriptor': descriptor,
                    'image_relevance': 'high'
                })
        
        # 공간적 감각 연결
        spatial_descriptors = ['넓은', '좁은', '높은', '낮은', '가까운', '먼', '큰', '작은']
        for descriptor in spatial_descriptors:
            if descriptor in text_content:
                sensory_connections.append({
                    'type': 'spatial',
                    'text_descriptor': descriptor,
                    'image_relevance': 'medium'
                })
        
        return sensory_connections

    def _analyze_narrative_flow_connections(self, all_text_content: Dict, image_analysis_results: List[Dict]) -> List[Dict]:
        """내러티브 플로우와 이미지 시퀀스 연결성 분석"""
        narrative_connections = []
        
        # 텍스트 섹션들을 시간순/논리순으로 정렬 (키 이름 기준)
        sorted_sections = sorted(all_text_content.items())
        
        # 각 섹션에 대해 최적의 이미지 매칭
        for idx, (section_key, text_content) in enumerate(sorted_sections):
            # 해당 섹션의 내용과 가장 잘 맞는 이미지들 찾기
            best_matches = []
            
            for img_idx, image_data in enumerate(image_analysis_results):
                # 위치 기반 매칭
                location_match = self._calculate_location_relevance(text_content, image_data)
                
                # 내용 기반 매칭
                content_match = self._calculate_content_relevance(text_content, image_data)
                
                total_score = (location_match + content_match) / 2
                
                if total_score > 0.3:  # 임계값 이상인 경우만
                    best_matches.append({
                        'image_index': img_idx,
                        'relevance_score': total_score,
                        'location_score': location_match,
                        'content_score': content_match
                    })
            
            # 점수 순으로 정렬
            best_matches.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            narrative_connections.append({
                'section_key': section_key,
                'section_order': idx,
                'recommended_images': best_matches[:3]  # 상위 3개만
            })
        
        return narrative_connections

    def _calculate_location_relevance(self, text_content: str, image_data: Dict) -> float:
        """위치 기반 연관성 계산"""
        image_location = image_data.get('location', '').lower()
        text_lower = text_content.lower()
        
        if not image_location:
            return 0.0
        
        # 위치 키워드가 텍스트에 포함되어 있는지 확인
        location_words = [word.strip() for word in image_location.split(',')]
        matches = sum(1 for word in location_words if word and word in text_lower)
        
        return matches / max(len(location_words), 1)

    def _calculate_content_relevance(self, text_content: str, image_data: Dict) -> float:
        """내용 기반 연관성 계산"""
        image_description = image_data.get('description', '').lower()
        text_lower = text_content.lower()
        
        if not image_description:
            return 0.0
        
        # 공통 키워드 찾기
        description_words = set(image_description.split())
        text_words = set(text_lower.split())
        
        common_words = description_words & text_words
        total_words = description_words | text_words
        
        return len(common_words) / max(len(total_words), 1)

    async def _process_interview_async(self, texts: List[str]) -> Dict[str, str]:
        """인터뷰 형식 처리 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.interview_manager.process_all_interviews, texts
        )

    async def _process_essay_async(self, texts: List[str]) -> Dict[str, str]:
        """에세이 형식 처리 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.essay_manager.run_all, texts
        )

    async def _process_image_analysis_async(self, image_analysis_results: List[Dict]) -> str:
        """이미지 분석 처리 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._process_image_analysis, image_analysis_results
        )

    async def _log_interview_results_async(self, texts: List[str], interview_results: Dict[str, str]):
        """인터뷰 처리 결과 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="ContentCreatorV2Agent_Interview",
                agent_role="인터뷰 콘텐츠 처리자",
                task_description=f"{len(texts)}개 텍스트를 인터뷰 형식으로 처리",
                final_answer=f"인터뷰 형식 콘텐츠 {len(interview_results)}개 생성 완료",
                reasoning_process="원본 텍스트를 인터뷰 형식으로 변환하여 대화체 콘텐츠 생성",
                execution_steps=[
                    "원본 텍스트 분석",
                    "인터뷰 형식 변환",
                    "대화체 콘텐츠 생성",
                    "품질 검증"
                ],
                raw_input={"texts": texts, "texts_count": len(texts)},
                raw_output=interview_results,
                performance_metrics={
                    "interview_results_count": len(interview_results),
                    "total_interview_length": sum(len(content) for content in interview_results.values()),
                    "processing_success": True,
                    "async_processing": True
                }
            )
        )

    async def _log_essay_results_async(self, texts: List[str], essay_results: Dict[str, str]):
        """에세이 처리 결과 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="ContentCreatorV2Agent_Essay",
                agent_role="에세이 콘텐츠 처리자",
                task_description=f"{len(texts)}개 텍스트를 에세이 형식으로 처리",
                final_answer=f"에세이 형식 콘텐츠 {len(essay_results)}개 생성 완료",
                reasoning_process="원본 텍스트를 에세이 형식으로 변환하여 성찰적 콘텐츠 생성",
                execution_steps=[
                    "원본 텍스트 분석",
                    "에세이 형식 변환",
                    "성찰적 콘텐츠 생성",
                    "품질 검증"
                ],
                raw_input={"texts": texts, "texts_count": len(texts)},
                raw_output=essay_results,
                performance_metrics={
                    "essay_results_count": len(essay_results),
                    "total_essay_length": sum(len(content) for content in essay_results.values()),
                    "processing_success": True,
                    "async_processing": True
                }
            )
        )

    async def _log_image_processing_async(self, image_analysis_results: List[Dict], image_info: str):
        """이미지 정보 처리 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="ContentCreatorV2Agent_ImageProcessor",
                agent_role="이미지 정보 처리자",
                task_description=f"{len(image_analysis_results)}개 이미지 분석 결과 처리",
                final_answer=f"이미지 정보 정리 완료: {len(image_info)}자",
                reasoning_process="이미지 분석 결과를 텍스트 콘텐츠와 연계 가능한 형태로 정리",
                execution_steps=[
                    "이미지 분석 결과 수집",
                    "위치 정보 추출",
                    "설명 정보 정리",
                    "텍스트 연계 포맷 생성"
                ],
                raw_input=image_analysis_results,
                raw_output=image_info,
                performance_metrics={
                    "images_processed": len(image_analysis_results),
                    "image_info_length": len(image_info),
                    "processing_success": True,
                    "async_processing": True
                }
            )
        )

    async def _integrate_all_content_as_first_agent_async(self, interview_results: Dict[str, str], essay_results: Dict[str, str],
                                                        image_info: str, original_texts: List[str], semantic_connections: Dict) -> str:
        """첫 번째 에이전트로서 모든 콘텐츠를 활용하여 최종 매거진 콘텐츠 생성 (비동기)"""
        agent = self.create_agent()
        
        # 모든 인터뷰 콘텐츠 정리 (완전 활용)
        interview_content = "\n\n".join([f"=== {key} ===\n{value}" for key, value in interview_results.items()])
        
        # 모든 에세이 콘텐츠 정리 (완전 활용)
        essay_content = "\n\n".join([f"=== {key} ===\n{value}" for key, value in essay_results.items()])
        
        # 원본 텍스트도 참고용으로 제공
        original_content = "\n\n".join([f"=== 원본 텍스트 {i+1} ===\n{text}" for i, text in enumerate(original_texts)])
        
        # 의미적 연결 정보 포맷팅
        semantic_info = self._format_semantic_connections_for_prompt(semantic_connections)
        
        integration_task = Task(
            description=f"""
**첫 번째 에이전트로서 후속 에이전트들을 위한 기초 매거진 콘텐츠 생성**

다음의 **모든** 인터뷰 형식 콘텐츠와 에세이 형식 콘텐츠, 그리고 이미지 정보를 바탕으로
**완전한** 여행 매거진 콘텐츠를 작성하세요.

**중요**: 제공된 모든 콘텐츠를 빠짐없이 활용해야 합니다. 첨삭하지 말고 모든 내용을 포함하세요.

**첫 번째 에이전트로서의 역할:**
- 후속 에이전트들이 활용할 수 있는 고품질 기초 콘텐츠 생성
- 구조화되고 완성도 높은 매거진 콘텐츠 제공
- 이미지 배치 및 레이아웃 에이전트들을 위한 명확한 콘텐츠 섹션 구분
- **이미지와 텍스트의 의미적 연결을 반영한 콘텐츠 구성**

**인터뷰 형식 콘텐츠 (모두 활용):**
{interview_content}

**에세이 형식 콘텐츠 (모두 활용):**
{essay_content}

**원본 텍스트 참고:**
{original_content}

**이미지 정보:**
{image_info}

**이미지-텍스트 의미적 연결 분석 결과:**
{semantic_info}

**통합 지침 (첫 번째 에이전트 - 모든 데이터 활용):**
1. **완전 활용**: 인터뷰와 에세이의 모든 내용을 빠짐없이 포함
2. **구조화**: 후속 에이전트들이 활용하기 쉽도록 명확한 섹션 구분
3. **내용 확장**: 제공된 콘텐츠를 기반으로 더 풍부한 매거진 스토리 생성
4. **품질 보장**: 첫 번째 에이전트로서 높은 품질의 기초 데이터 제공
5. **통합성**: 각 섹션이 독립적이면서도 전체 스토리가 연결되도록 구성
6. **이미지 연계**: 이미지 정보를 적절한 위치에 자연스럽게 녹여냄
7. **완성도**: 매거진 독자들이 몰입할 수 있는 완성된 스토리로 구성
8. **확장성**: 후속 에이전트들이 추가 작업할 수 있는 여지 제공
9. **의미적 연결**: 이미지-텍스트 분석 결과를 반영하여 시각적 요소와 텍스트가 조화롭게 구성

**매거진 구성 요소 (모든 콘텐츠 포함):**
1. 매력적인 제목과 부제목
2. 여행지 소개 및 첫인상 (인터뷰와 에세이 내용 활용)
3. 주요 경험과 감상 (모든 인터뷰와 에세이 혼합)
4. 특별한 순간들과 만남 (모든 콘텐츠에서 추출)
5. 일상적 경험들 (모든 세부 내용 포함)
6. 문화적 체험과 성찰 (에세이 내용 중심)
7. 여행의 의미와 마무리 (모든 감상 통합)

**스타일 (첫 번째 에이전트 기준):**
- 매거진 특유의 세련되고 감성적인 문체
- 독자의 공감을 이끌어내는 스토리텔링
- 시각적 상상력을 자극하는 묘사
- 인터뷰의 진솔함과 에세이의 성찰이 조화된 톤
- **모든 제공된 콘텐츠가 자연스럽게 녹아든 완성된 스토리**
- **후속 에이전트들이 작업하기 좋은 구조화된 형태**
- **이미지와 텍스트가 서로를 보완하고 강화하는 시너지 창출**

**이미지-텍스트 시너지 창출 요구사항:**
- 각 텍스트 섹션에서 시각적 요소를 연상시키는 묘사 포함
- 이미지의 감정적 톤과 일치하는 문체 사용
- 독자가 텍스트를 읽으면서 자연스럽게 이미지를 떠올릴 수 있도록 구성
- 이미지 시퀀스가 텍스트의 내러티브 플로우와 일치하도록 섹션 배치

**출력 요구사항:**
- 최소 3000자 이상의 풍부한 매거진 콘텐츠
- 모든 인터뷰와 에세이 내용이 포함된 완성된 스토리
- 여행의 전 과정을 아우르는 완전한 내러티브
- 이미지 배치 에이전트를 위한 이미지 연결점 정보 포함
- **이미지와 텍스트의 의미적 연결이 명확히 드러나는 구성**

**템플릿 생성 규칙:**
- 모든 텍스트 섹션은 독자의 인지 흐름을 고려하여 자연스럽게 이어져야 합니다.
- 중복을 절대로 하지않고 만듭니다!!
- **각 섹션에서 해당하는 이미지의 시각적 특성을 텍스트로 표현합니다.**
- **이미지가 전달하는 감정과 분위기를 텍스트에 반영합니다.**

**JSX 최적화 구조화 지침:**
    1. **명확한 JSON 구조**: 각 섹션을 명확한 JSON 객체로 구분
    2. **구조적 마커 제거**: "magazine layout design structure" 같은 불필요한 마커 완전 제거
    3. **섹션별 명확한 구분**: title, subtitle, body가 명확히 분리된 구조
    4. **특수문자 제거**: JSX 파싱을 방해하는 특수문자 정리
    6. **일관된 톤**: 모든 섹션에서 일관된 문체 유지

    **출력 형식 (반드시 이 구조로만 출력):**

    {{
    "magazine_title": "매력적인 전체 제목",
    "magazine_subtitle": "흥미로운 부제목",
    "sections": [
        {{
        "section_id": 1,
        "title": "섹션 제목",
        "subtitle": "섹션 부제목",
        "body": "완전한 본문 내용 (특수문자 제거)",
        "image_keywords": ["관련", "이미지", "키워드"]
        }},
        {{
        "section_id": 2,
        "title": "섹션 제목",
        "subtitle": "섹션 부제목", 
        "body": "완전한 본문 내용 (특수문자 제거)",
        "image_keywords": ["관련", "이미지", "키워드"]
        }}
    ]
    }}

    **중요 제약사항:**
    - 위의 JSON 형식만 출력하고 다른 텍스트는 포함하지 마세요
    - 구조적 마커나 메타데이터는 절대 포함하지 마세요
    - 모든 텍스트는 JSX에서 안전하게 파싱될 수 있도록 정리하세요
    - 각 섹션의 body는 완전한 문장으로 구성
""",
            agent=agent,
            expected_output="모든 하위 에이전트 콘텐츠가 포함되고 이미지-텍스트 의미적 연결이 강화된 완성된 여행 매거진 콘텐츠"
        )
        
        # 비동기 태스크 실행
        result = await asyncio.get_event_loop().run_in_executor(
            None, agent.execute_task, integration_task
        )
        
        # 결과 검증
        final_content = str(result)
        await self._verify_final_content_as_first_agent_async(final_content, interview_results, essay_results)
        
        return final_content

    def _format_semantic_connections_for_prompt(self, semantic_connections: Dict) -> str:
        """의미적 연결 정보를 프롬프트용으로 포맷팅"""
        formatted_info = []
        
        # 감정적 톤 매칭 정보
        if semantic_connections.get("emotional_tone_matches"):
            formatted_info.append("**감정적 톤 매칭:**")
            for match in semantic_connections["emotional_tone_matches"]:
                formatted_info.append(f"- 이미지 {match['image_index']}: {match['text_section']} (매칭도: {match['match_score']:.2f})")
        
        # 주제적 연결성 정보
        if semantic_connections.get("thematic_alignments"):
            formatted_info.append("\n**주제적 연결성:**")
            for alignment in semantic_connections["thematic_alignments"]:
                formatted_info.append(f"- 이미지 {alignment['image_index']}: {alignment['text_section']} (연결도: {alignment['alignment_score']:.2f})")
        
        # 내러티브 플로우 연결
        if semantic_connections.get("narrative_flow_connections"):
            formatted_info.append("\n**내러티브 플로우 연결:**")
            for connection in semantic_connections["narrative_flow_connections"]:
                if connection.get("recommended_images"):
                    formatted_info.append(f"- {connection['section_key']}: 추천 이미지 {[img['image_index'] for img in connection['recommended_images'][:2]]}")
        
        return "\n".join(formatted_info) if formatted_info else "의미적 연결 정보 없음"

    async def _verify_content_completeness_async(self, interview_results: Dict[str, str], essay_results: Dict[str, str], original_texts: List[str]):
        """콘텐츠 완전성 검증 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None, self._verify_content_completeness, interview_results, essay_results, original_texts
        )

    async def _verify_final_content_as_first_agent_async(self, final_content: str, interview_results: Dict[str, str], essay_results: Dict[str, str]):
        """첫 번째 에이전트로서 최종 콘텐츠 검증 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None, self._verify_final_content_as_first_agent, final_content, interview_results, essay_results
        )

    async def _log_final_content_async(self, final_content: str, interview_results: Dict[str, str],
                                     essay_results: Dict[str, str], image_analysis_results: List[Dict], 
                                     texts: List[str], semantic_connections: Dict):
        """최종 통합 콘텐츠 생성 로깅 (새로운 방식 적용)"""
        # ✅ LoggingManager 인스턴스 생성
        from utils.logging_manager import LoggingManager
        logging_manager = LoggingManager()
        
        # ✅ 새로운 로깅 방식으로 응답 데이터 저장
        await logging_manager.log_agent_response(
            agent_name="ContentCreatorV2Agent",
            agent_role="여행 콘텐츠 통합 편집자 (첫 번째 에이전트)",
            task_description=f"인터뷰 {len(interview_results)}개, 에세이 {len(essay_results)}개, 이미지 {len(image_analysis_results)}개를 통합한 매거진 콘텐츠 생성 (이미지-텍스트 의미적 연결 포함)",
            response_data=final_content,  # ✅ 실제 응답 데이터만 저장
            metadata={
                "final_content_length": len(final_content),
                "content_expansion_ratio": len(final_content) / sum(len(text) for text in texts) if texts else 0,
                "integration_success": len(interview_results) > 0 and len(essay_results) > 0,
                "image_integration_count": len(image_analysis_results),
                "semantic_connections_count": sum(len(v) if isinstance(v, list) else 0 for v in semantic_connections.values()),
                "emotional_tone_matches": len(semantic_connections.get("emotional_tone_matches", [])),
                "thematic_alignments": len(semantic_connections.get("thematic_alignments", [])),
                "narrative_flow_connections": len(semantic_connections.get("narrative_flow_connections", [])),
                "first_agent_completion": True,
                "async_processing": True,
                "image_text_synergy_enabled": True
            }
        )

    # 동기 버전 메서드들 (호환성 유지)
    def _verify_final_content_as_first_agent(self, final_content: str, interview_results: Dict[str, str], essay_results: Dict[str, str]):
        """첫 번째 에이전트로서 최종 콘텐츠 검증 (동기 버전)"""
        final_length = len(final_content)
        total_source_length = sum(len(content) for content in interview_results.values()) + sum(len(content) for content in essay_results.values())
        
        print(f"ContentCreatorV2 (첫 번째 에이전트): 최종 콘텐츠 검증")
        print(f"- 최종 콘텐츠 길이: {final_length}자")
        print(f"- 원본 소스 길이: {total_source_length}자")
        print(f"- 확장 비율: {(final_length / total_source_length * 100):.1f}%" if total_source_length > 0 else "- 확장 비율: 계산 불가")
        print(f"- 첫 번째 에이전트 역할: 기초 데이터 생성 완료")
        print(f"- 이미지-텍스트 시너지: 활성화됨")
        
        if final_length < total_source_length * 0.8:
            print("⚠️ 최종 콘텐츠가 원본보다 현저히 짧습니다. 첨삭이 발생했을 가능성이 있습니다.")
        else:
            print("✅ 콘텐츠가 적절히 확장되어 생성되었습니다.")
        
        print("✅ 첫 번째 에이전트로서 후속 에이전트들을 위한 기초 데이터 생성 완료")
        print("✅ 이미지-텍스트 의미적 연결 강화 완료")

    def _calculate_content_quality_score(self, final_content: str, interview_results: Dict[str, str], essay_results: Dict[str, str]) -> float:
        """콘텐츠 품질 점수 계산"""
        score = 0.0
        
        # 길이 기반 점수 (25%)
        if len(final_content) > 3000:
            score += 0.25
        elif len(final_content) > 2000:
            score += 0.2
        elif len(final_content) > 1000:
            score += 0.1
        
        # 구조화 점수 (25%)
        section_count = final_content.count("===")
        if section_count >= 5:
            score += 0.25
        elif section_count >= 3:
            score += 0.2
        elif section_count >= 1:
            score += 0.1
        
        # 콘텐츠 통합 점수 (25%)
        if interview_results and essay_results:
            score += 0.25
        elif interview_results or essay_results:
            score += 0.15
        
        # 이미지-텍스트 시너지 점수 (25%)
        visual_descriptors = ['보이는', '눈에 띄는', '화려한', '선명한', '밝은', '어두운', '아름다운']
        synergy_count = sum(1 for descriptor in visual_descriptors if descriptor in final_content)
        if synergy_count >= 5:
            score += 0.25
        elif synergy_count >= 3:
            score += 0.2
        elif synergy_count >= 1:
            score += 0.1
        
        return min(score, 1.0)

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
        print("ContentCreatorV2 (첫 번째 에이전트): 콘텐츠 완전성 검증")
        
        # 원본 텍스트 길이
        total_original_length = sum(len(text) for text in original_texts)
        
        # 인터뷰 결과 길이
        total_interview_length = sum(len(content) for content in interview_results.values())
        
        # 에세이 결과 길이
        total_essay_length = sum(len(content) for content in essay_results.values())
        
        print(f"원본 텍스트: {total_original_length}자")
        print(f"인터뷰 결과: {total_interview_length}자 ({len(interview_results)}개)")
        print(f"에세이 결과: {total_essay_length}자 ({len(essay_results)}개)")

    # 동기 버전 메서드 (호환성 보장)
    def create_magazine_content_sync(self, texts: List[str], image_analysis_results: List[Dict]) -> str:
        """동기 버전 매거진 콘텐츠 생성 (호환성 유지)"""
        return asyncio.run(self.create_magazine_content(texts, image_analysis_results))


class ContentCreatorV2Crew:
    """ContentCreatorV2를 위한 Crew 관리 (비동기 처리)"""

    def __init__(self):
        self.content_creator = ContentCreatorV2Agent()

    def create_crew(self) -> Crew:
        """ContentCreatorV2 전용 Crew 생성"""
        return Crew(
            agents=[self.content_creator.create_agent()],
            verbose=True
        )

    async def execute_content_creation(self, texts: List[str], image_analysis_results: List[Dict]) -> str:
        """Crew를 통한 콘텐츠 생성 실행 (첫 번째 에이전트 - 비동기)"""
        crew = self.create_crew()
        
        print("\n=== ContentCreatorV2 Crew 실행 (첫 번째 에이전트 - 비동기) ===")
        print(f"- 입력 텍스트: {len(texts)}개")
        print(f"- 이미지 분석 결과: {len(image_analysis_results)}개")
        print(f"- 역할: 첫 번째 에이전트 (로그 수집 시작)")
        print(f"- 이미지-텍스트 시너지: 활성화")
        
        # ContentCreatorV2Agent를 통한 콘텐츠 생성 (비동기)
        result = await self.content_creator.create_magazine_content(texts, image_analysis_results)
        
        print("✅ ContentCreatorV2 Crew 실행 완료 (첫 번째 에이전트 - 비동기)")
        print("✅ 후속 에이전트들을 위한 로그 데이터 생성 완료")
        print("✅ 이미지-텍스트 의미적 연결 강화 완료")
        
        return result

    # 동기 버전 메서드 (호환성 보장)
    def execute_content_creation_sync(self, texts: List[str], image_analysis_results: List[Dict]) -> str:
        """동기 버전 콘텐츠 생성 실행 (호환성 유지)"""
        return asyncio.run(self.execute_content_creation(texts, image_analysis_results))
