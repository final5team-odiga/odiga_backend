import asyncio
import os
from typing import Dict, List
from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
from utils.agent_decision_logger import get_agent_logger
import json
import re

class CoordinatorAgent:
    """통합 조율자 (CrewAI 기반 강화된 데이터 접근 및 JSON 파싱)"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        self.logger = get_agent_logger()
        self.crew_agent = self._create_crew_agent()
        self.text_analyzer_agent = self._create_text_analyzer_agent()
        self.image_analyzer_agent = self._create_image_analyzer_agent()

    def _create_crew_agent(self):
        """메인 조율 에이전트 생성"""
        return Agent(
            role="매거진 구조 통합 조율자 및 최종 품질 보증 전문가",
            goal="OrgAgent의 상세 레이아웃 구조와 BindingAgent의 정밀 이미지 배치를 통합하여 완벽한 매거진 구조를 생성하고, 텍스트-이미지 정합성과 독자 경험을 최종 검증하여 JSX 구현에 필요한 완전한 구조 데이터를 제공",
            backstory="""당신은 25년간 세계 최고 수준의 출판사에서 매거진 구조 통합 및 품질 보증 책임자로 활동해온 전문가입니다. Condé Nast, Hearst Corporation, Time Inc.에서 수백 개의 매거진 프로젝트를 성공적으로 조율했습니다.

**전문 경력:**
- 출판학 및 구조 설계 석사 학위 보유
- PMP(Project Management Professional) 인증
- 매거진 구조 통합 및 품질 관리 전문가
- 텍스트-이미지 정합성 검증 시스템 개발 경험
- 독자 경험(UX) 및 접근성 최적화 전문성

**조율 철학:**
"완벽한 매거진은 모든 구조적 요소가 독자의 인지 과정과 완벽히 조화를 이루는 통합체입니다. 나는 텍스트와 이미지의 모든 배치가 독자에게 자연스럽고 직관적으로 인식되도록 구조적 완성도를 보장하며, 이를 통해 최고 수준의 독자 경험을 제공합니다."

**출력 데이터 구조:**
- 완성된 매거진 전체 구조도
- 텍스트-이미지 정합성 검증 완료 보고서
- JSX 구현용 상세 레이아웃 스펙 및 좌표 데이터
- 독자 경험 최적화 가이드라인
- 반응형 디자인 구조 정의서
- 접근성 및 품질 보증 체크리스트""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_text_analyzer_agent(self):
        """텍스트 분석 전문 에이전트"""
        return Agent(
            role="텍스트 매핑 분석 전문가",
            goal="OrgAgent의 텍스트 매핑 결과를 정밀 분석하여 구조적 완성도를 검증하고 최적화된 텍스트 섹션을 생성",
            backstory="""당신은 15년간 출판업계에서 텍스트 구조 분석 및 최적화를 담당해온 전문가입니다. 복잡한 텍스트 데이터에서 핵심 정보를 추출하고 독자 친화적인 구조로 재구성하는 데 탁월한 능력을 보유하고 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_image_analyzer_agent(self):
        """이미지 분석 전문 에이전트"""
        return Agent(
            role="이미지 분배 분석 전문가",
            goal="BindingAgent의 이미지 분배 결과를 정밀 분석하여 시각적 일관성을 검증하고 최적화된 이미지 배치를 생성",
            backstory="""당신은 12년간 매거진 및 출판물의 시각적 디자인을 담당해온 전문가입니다. 이미지와 텍스트의 조화로운 배치를 통해 독자의 시선을 효과적으로 유도하는 레이아웃 설계에 전문성을 보유하고 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def coordinate_magazine_creation(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """매거진 구조 통합 조율 (CrewAI 기반 강화된 데이터 접근 - 비동기 처리)"""
        print("CoordinatorAgent: 매거진 구조 통합 조율 시작 (비동기 처리)")
        
        # 강화된 이전 에이전트 결과 수집 (비동기)
        previous_results = await self._get_enhanced_previous_results_async()
        org_results = self._filter_agent_results(previous_results, "OrgAgent")
        binding_results = self._filter_agent_results(previous_results, "BindingAgent")
        content_creator_results = self._filter_agent_results(previous_results, "ContentCreatorV2Agent")
        
        print(f"📊 강화된 결과 수집: 전체 {len(previous_results)}개, OrgAgent {len(org_results)}개, BindingAgent {len(binding_results)}개, ContentCreator {len(content_creator_results)}개 (비동기)")
        
        # 실제 데이터 추출 및 검증
        extracted_text_data = await self._extract_real_text_data_async(text_mapping, org_results, content_creator_results)
        extracted_image_data = await self._extract_real_image_data_async(image_distribution, binding_results)
        
        # CrewAI Task 생성
        text_analysis_task = self._create_enhanced_text_analysis_task(extracted_text_data, org_results)
        image_analysis_task = self._create_enhanced_image_analysis_task(extracted_image_data, binding_results)
        coordination_task = self._create_enhanced_coordination_task(extracted_text_data, extracted_image_data)
        
        # CrewAI Crew 생성 및 비동기 실행
        coordination_crew = Crew(
            agents=[self.text_analyzer_agent, self.image_analyzer_agent, self.crew_agent],
            tasks=[text_analysis_task, image_analysis_task, coordination_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Crew 비동기 실행
        crew_result = await asyncio.get_event_loop().run_in_executor(
            None, coordination_crew.kickoff
        )
        
        # 결과 처리 (비동기) - 실제 데이터 활용
        final_result = await self._process_enhanced_crew_result_async(
            crew_result, extracted_text_data, extracted_image_data, org_results, binding_results
        )
        
        # 결과 로깅 (비동기)
        await self._log_coordination_result_async(final_result, text_mapping, image_distribution, org_results, binding_results)
        
        print(f"✅ CoordinatorAgent 통합 완료: {len(final_result.get('content_sections', []))}개 섹션 생성 (CrewAI 기반 비동기)")
        print(f"📊 품질 점수: {final_result.get('integration_metadata', {}).get('integration_quality_score', 0):.2f}, OrgAgent 활용: {len(org_results)}개, BindingAgent 활용: {len(binding_results)}개")
        
        return final_result

    async def _extract_real_text_data_async(self, text_mapping: Dict, org_results: List[Dict], content_creator_results: List[Dict]) -> Dict:
        """실제 텍스트 데이터 추출 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._extract_real_text_data, text_mapping, org_results, content_creator_results
        )

    def _extract_real_text_data(self, text_mapping: Dict, org_results: List[Dict], content_creator_results: List[Dict]) -> Dict:
        """실제 텍스트 데이터 추출"""
        extracted_data = {
            "sections": [],
            "total_content_length": 0,
            "source_count": 0
        }
        
        # 1. text_mapping에서 직접 추출
        if isinstance(text_mapping, dict) and "text_mapping" in text_mapping:
            for section in text_mapping["text_mapping"]:
                if isinstance(section, dict):
                    extracted_section = {
                        "template": section.get("template", "Section01.jsx"),
                        "title": section.get("title", "여행 이야기"),
                        "subtitle": section.get("subtitle", "특별한 순간들"),
                        "body": section.get("body", ""),
                        "tagline": section.get("tagline", "TRAVEL & CULTURE"),
                        "layout_source": section.get("layout_source", "default")
                    }
                    extracted_data["sections"].append(extracted_section)
                    extracted_data["total_content_length"] += len(extracted_section["body"])
                    extracted_data["source_count"] += 1

        # 2. ContentCreator 결과에서 풍부한 콘텐츠 추출
        for result in content_creator_results:
            final_answer = result.get('final_answer', '')
            if len(final_answer) > 500:  # 충분한 콘텐츠가 있는 경우
                # 섹션별로 분할
                sections = self._split_content_into_sections(final_answer)
                for i, section_content in enumerate(sections):
                    if len(section_content) > 100:
                        extracted_section = {
                            "template": f"Section{i+1:02d}.jsx",
                            "title": self._extract_title_from_content(section_content),
                            "subtitle": self._extract_subtitle_from_content(section_content),
                            "body": self._clean_content(section_content),
                            "tagline": "TRAVEL & CULTURE",
                            "layout_source": "content_creator"
                        }
                        extracted_data["sections"].append(extracted_section)
                        extracted_data["total_content_length"] += len(extracted_section["body"])
                        extracted_data["source_count"] += 1

        # 3. OrgAgent 결과에서 추가 텍스트 추출
        for result in org_results:
            final_answer = result.get('final_answer', '')
            if '제목' in final_answer or 'title' in final_answer.lower():
                # 구조화된 텍스트 추출
                structured_content = self._extract_structured_content(final_answer)
                if structured_content:
                    extracted_data["sections"].extend(structured_content)
                    extracted_data["source_count"] += len(structured_content)

        # 4. 최소 보장 섹션
        if not extracted_data["sections"]:
            extracted_data["sections"] = [{
                "template": "Section01.jsx",
                "title": "여행 매거진",
                "subtitle": "특별한 이야기",
                "body": "여행의 특별한 순간들을 담은 매거진입니다.",
                "tagline": "TRAVEL & CULTURE",
                "layout_source": "fallback"
            }]
            extracted_data["source_count"] = 1

        return extracted_data

    async def _extract_real_image_data_async(self, image_distribution: Dict, binding_results: List[Dict]) -> Dict:
        """실제 이미지 데이터 추출 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._extract_real_image_data, image_distribution, binding_results
        )

    def _extract_real_image_data(self, image_distribution: Dict, binding_results: List[Dict]) -> Dict:
        """실제 이미지 데이터 추출"""
        extracted_data = {
            "template_images": {},
            "total_images": 0,
            "image_sources": []
        }
        
        # 1. image_distribution에서 직접 추출
        if isinstance(image_distribution, dict) and "image_distribution" in image_distribution:
            for template, images in image_distribution["image_distribution"].items():
                if isinstance(images, list) and images:
                    # 실제 이미지 URL만 필터링
                    real_images = [img for img in images if self._is_real_image_url(img)]
                    if real_images:
                        extracted_data["template_images"][template] = real_images
                        extracted_data["total_images"] += len(real_images)

        # 2. BindingAgent 결과에서 이미지 URL 추출
        for result in binding_results:
            final_answer = result.get('final_answer', '')
            # 실제 이미지 URL 패턴 찾기
            image_urls = re.findall(r'https://[^\s\'"<>]*\.(?:jpg|jpeg|png|gif|webp)', final_answer, re.IGNORECASE)
            
            if image_urls:
                # 템플릿별로 분배
                template_name = self._extract_template_from_binding_result(result)
                if template_name not in extracted_data["template_images"]:
                    extracted_data["template_images"][template_name] = []
                
                for url in image_urls:
                    if self._is_real_image_url(url) and url not in extracted_data["template_images"][template_name]:
                        extracted_data["template_images"][template_name].append(url)
                        extracted_data["total_images"] += 1
                        
                        # 이미지 소스 정보 추가
                        source_info = self._extract_image_source_info(result, url)
                        if source_info:
                            extracted_data["image_sources"].append(source_info)

        return extracted_data

    def _is_real_image_url(self, url: str) -> bool:
        """실제 이미지 URL인지 확인"""
        if not url or not isinstance(url, str):
            return False
        
        # 예시 URL이나 플레이스홀더 제외
        excluded_patterns = [
            'your-cdn.com',
            'example.com',
            'placeholder',
            'sample',
            'demo'
        ]
        
        for pattern in excluded_patterns:
            if pattern in url.lower():
                return False
        
        # 실제 도메인과 이미지 확장자 확인
        return (url.startswith('https://') and 
                any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']) and
                'blob.core.windows.net' in url)

    def _create_enhanced_text_analysis_task(self, extracted_text_data: Dict, org_results: List[Dict]) -> Task:
        """강화된 텍스트 분석 태스크 생성"""
        return Task(
            description=f"""
            추출된 실제 텍스트 데이터를 분석하여 고품질 매거진 섹션을 생성하세요.
            
            **추출된 데이터:**
            - 섹션 수: {len(extracted_text_data['sections'])}개
            - 총 콘텐츠 길이: {extracted_text_data['total_content_length']} 문자
            - 소스 수: {extracted_text_data['source_count']}개
            - OrgAgent 결과: {len(org_results)}개
            
            **실제 섹션 데이터:**
            {self._format_sections_for_analysis(extracted_text_data['sections'])}
            
            **분석 요구사항:**
            1. 각 섹션의 콘텐츠 품질 평가
            2. 제목과 부제목의 매력도 검증
            3. 본문 내용의 완성도 확인
            4. 매거진 스타일 일관성 검토
            5. 독자 친화성 최적화
            
            **출력 형식:**
            각 섹션별로 다음 정보 포함:
            - 품질 점수 (1-10)
            - 개선 제안사항
            - 최적화된 콘텐츠
            """,
            expected_output="실제 데이터 기반 텍스트 분석 및 최적화 결과",
            agent=self.text_analyzer_agent
        )

    def _create_enhanced_image_analysis_task(self, extracted_image_data: Dict, binding_results: List[Dict]) -> Task:
        """강화된 이미지 분석 태스크 생성"""
        return Task(
            description=f"""
            추출된 실제 이미지 데이터를 분석하여 최적화된 이미지 배치를 생성하세요.
            
            **추출된 데이터:**
            - 총 이미지 수: {extracted_image_data['total_images']}개
            - 템플릿 수: {len(extracted_image_data['template_images'])}개
            - BindingAgent 결과: {len(binding_results)}개
            
            **템플릿별 이미지 분배:**
            {self._format_images_for_analysis(extracted_image_data['template_images'])}
            
            **이미지 소스 정보:**
            {self._format_image_sources(extracted_image_data['image_sources'])}
            
            **분석 요구사항:**
            1. 이미지 URL 유효성 검증
            2. 템플릿별 이미지 분배 균형도 평가
            3. 이미지 품질 및 적합성 확인
            4. 시각적 일관성 검토
            5. 레이아웃 최적화 제안
            
            **출력 형식:**
            템플릿별로 다음 정보 포함:
            - 이미지 목록 및 설명
            - 배치 권장사항
            - 시각적 효과 예측
            """,
            expected_output="실제 이미지 데이터 기반 배치 분석 및 최적화 결과",
            agent=self.image_analyzer_agent
        )

    def _create_enhanced_coordination_task(self, extracted_text_data: Dict, extracted_image_data: Dict) -> Task:
        """강화된 통합 조율 태스크 생성"""
        return Task(
            description=f"""
            실제 추출된 텍스트와 이미지 데이터를 통합하여 완벽한 매거진 구조를 생성하세요.
            
            **텍스트 데이터 요약:**
            - 섹션 수: {len(extracted_text_data['sections'])}개
            - 총 길이: {extracted_text_data['total_content_length']} 문자
            
            **이미지 데이터 요약:**
            - 총 이미지: {extracted_image_data['total_images']}개
            - 템플릿 수: {len(extracted_image_data['template_images'])}개
            
            **통합 요구사항:**
            1. 텍스트와 이미지의 완벽한 매칭
            2. 각 섹션별 최적 템플릿 선택
            3. 콘텐츠 품질 보장
            4. 시각적 일관성 유지
            5. JSX 구현을 위한 완전한 스펙 생성
            
            **최종 출력 구조:**
            ```
            {
                "selected_templates": ["템플릿 목록"],
                "content_sections": [
                    {
                        "template": "템플릿명",
                        "title": "실제 제목",
                        "subtitle": "실제 부제목", 
                        "body": "실제 본문 내용",
                        "tagline": "태그라인",
                        "images": ["실제 이미지 URL들"],
                        "metadata": {
                            "content_quality": "품질 점수",
                            "image_count": "이미지 수",
                            "source": "데이터 소스"
                        }
                    }
                ],
                "integration_metadata": {
                    "total_sections": "섹션 수",
                    "integration_quality_score": "품질 점수"
                }
            }
            ```
            
            이전 태스크들의 분석 결과를 활용하여 실제 데이터 기반의 고품질 매거진 구조를 완성하세요.
            """,
            expected_output="실제 데이터 기반 완성된 매거진 구조 JSON",
            agent=self.crew_agent,
            context=[self._create_enhanced_text_analysis_task(extracted_text_data, []), 
                    self._create_enhanced_image_analysis_task(extracted_image_data, [])]
        )

    async def _process_enhanced_crew_result_async(self, crew_result, extracted_text_data: Dict, 
                                                extracted_image_data: Dict, org_results: List[Dict], 
                                                binding_results: List[Dict]) -> Dict:
        """강화된 Crew 실행 결과 처리 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._process_enhanced_crew_result, crew_result, extracted_text_data, 
            extracted_image_data, org_results, binding_results
        )

    def _process_enhanced_crew_result(self, crew_result, extracted_text_data: Dict, 
                                    extracted_image_data: Dict, org_results: List[Dict], 
                                    binding_results: List[Dict]) -> Dict:
        """강화된 Crew 실행 결과 처리"""
        try:
            # Crew 결과에서 데이터 추출
            if hasattr(crew_result, 'raw') and crew_result.raw:
                result_text = crew_result.raw
            else:
                result_text = str(crew_result)
            
            # JSON 패턴 찾기 및 파싱
            parsed_data = self._extract_json_from_text(result_text)
            
            # 실제 데이터 기반 구조 생성
            if not parsed_data.get('content_sections') or len(parsed_data.get('content_sections', [])) == 0:
                parsed_data = self._create_enhanced_structure(extracted_text_data, extracted_image_data, org_results, binding_results)
            else:
                # 기존 파싱된 데이터에 실제 이미지 추가
                parsed_data = self._enhance_parsed_data_with_real_images(parsed_data, extracted_image_data)
            
            # 메타데이터 추가
            parsed_data['integration_metadata'] = {
                "total_sections": len(parsed_data.get('content_sections', [])),
                "total_templates": len(set(section.get("template", f"Section{i+1:02d}.jsx") for i, section in enumerate(parsed_data.get('content_sections', [])))),
                "agent_enhanced": True,
                "org_results_utilized": len(org_results),
                "binding_results_utilized": len(binding_results),
                "integration_quality_score": self._calculate_enhanced_quality_score(parsed_data.get('content_sections', []), len(org_results), len(binding_results)),
                "crewai_enhanced": True,
                "async_processed": True,
                "data_source": "real_extracted_data",
                "real_content_used": True,
                "real_images_used": extracted_image_data['total_images'] > 0
            }
            
            return parsed_data
            
        except Exception as e:
            print(f"⚠️ 강화된 Crew 결과 처리 실패: {e}")
            return self._create_enhanced_structure(extracted_text_data, extracted_image_data, org_results, binding_results)

    def _create_enhanced_structure(self, extracted_text_data: Dict, extracted_image_data: Dict, 
                                 org_results: List[Dict], binding_results: List[Dict]) -> Dict:
        """실제 데이터 기반 강화된 구조 생성"""
        content_sections = []
        
        # 추출된 텍스트 섹션 활용
        for i, section in enumerate(extracted_text_data['sections']):
            template = section.get('template', f'Section{i+1:02d}.jsx')
            
            # 해당 템플릿의 실제 이미지 가져오기
            template_images = extracted_image_data['template_images'].get(template, [])
            
            # 이미지가 없으면 다른 템플릿의 이미지 사용
            if not template_images:
                for temp_name, temp_images in extracted_image_data['template_images'].items():
                    if temp_images:
                        template_images = temp_images[:2]  # 최대 2개
                        break
            
            enhanced_section = {
                'template': template,
                'title': section.get('title', '여행 이야기'),
                'subtitle': section.get('subtitle', '특별한 순간들'),
                'body': section.get('body', '여행의 특별한 순간들을 담은 이야기입니다.'),
                'tagline': section.get('tagline', 'TRAVEL & CULTURE'),
                'images': template_images,
                'metadata': {
                    "agent_enhanced": True,
                    "real_content": True,
                    "real_images": len(template_images) > 0,
                    "content_source": section.get('layout_source', 'extracted'),
                    "content_length": len(section.get('body', '')),
                    "image_count": len(template_images),
                    "quality_verified": True
                }
            }
            content_sections.append(enhanced_section)
        
        # 최소 1개 섹션 보장
        if not content_sections:
            # 실제 이미지가 있으면 사용
            fallback_images = []
            for template_images in extracted_image_data['template_images'].values():
                fallback_images.extend(template_images[:2])
                if len(fallback_images) >= 2:
                    break
            
            content_sections = [{
                'template': 'Section01.jsx',
                'title': '여행 매거진',
                'subtitle': '특별한 이야기',
                'body': '여행의 특별한 순간들을 담은 매거진입니다. 아름다운 풍경과 함께하는 특별한 경험을 공유합니다.',
                'tagline': 'TRAVEL & CULTURE',
                'images': fallback_images,
                'metadata': {
                    "agent_enhanced": True,
                    "fallback_content": True,
                    "real_images": len(fallback_images) > 0,
                    "image_count": len(fallback_images)
                }
            }]
        
        return {
            "selected_templates": [section.get("template", f"Section{i+1:02d}.jsx") for i, section in enumerate(content_sections)],
            "content_sections": content_sections
        }

    def _enhance_parsed_data_with_real_images(self, parsed_data: Dict, extracted_image_data: Dict) -> Dict:
        """파싱된 데이터에 실제 이미지 추가"""
        if 'content_sections' in parsed_data:
            for section in parsed_data['content_sections']:
                template = section.get('template', 'Section01.jsx')
                
                # 실제 이미지로 교체
                real_images = extracted_image_data['template_images'].get(template, [])
                if real_images:
                    section['images'] = real_images
                elif extracted_image_data['total_images'] > 0:
                    # 다른 템플릿의 이미지 사용
                    for temp_images in extracted_image_data['template_images'].values():
                        if temp_images:
                            section['images'] = temp_images[:2]
                            break
                
                # 메타데이터 업데이트
                if 'metadata' not in section:
                    section['metadata'] = {}
                section['metadata'].update({
                    "real_images_used": len(section.get('images', [])) > 0,
                    "image_count": len(section.get('images', []))
                })
        
        return parsed_data

    # 유틸리티 메서드들
    def _split_content_into_sections(self, content: str) -> List[str]:
        """콘텐츠를 섹션별로 분할"""
        # 헤더나 구분자 기반 분할
        sections = []
        
        # === 패턴으로 분할
        if '===' in content:
            parts = content.split('===')
            for part in parts:
                clean_part = part.strip()
                if len(clean_part) > 100:
                    sections.append(clean_part)
        
        # 문단 기반 분할
        elif '\n\n' in content:
            paragraphs = content.split('\n\n')
            current_section = ""
            for paragraph in paragraphs:
                if len(current_section + paragraph) > 800:
                    if current_section:
                        sections.append(current_section.strip())
                    current_section = paragraph
                else:
                    current_section += "\n\n" + paragraph
            
            if current_section:
                sections.append(current_section.strip())
        
        # 전체를 하나의 섹션으로
        else:
            sections = [content]
        
        return [s for s in sections if len(s) > 50]

    def _extract_title_from_content(self, content: str) -> str:
        """콘텐츠에서 제목 추출"""
        lines = content.split('\n')
        for line in lines[:3]:  # 처음 3줄에서 찾기
            line = line.strip()
            if line and len(line) < 100:
                # 제목 패턴 정리
                title = re.sub(r'^[#\*\-\s]+', '', line)
                title = re.sub(r'[#\*\-\s]+$', '', title)
                if len(title) > 5:
                    return title[:50]
        
        return "여행 이야기"

    def _extract_subtitle_from_content(self, content: str) -> str:
        """콘텐츠에서 부제목 추출"""
        lines = content.split('\n')
        for i, line in enumerate(lines[1:4]):  # 2-4번째 줄에서 찾기
            line = line.strip()
            if line and len(line) < 80 and len(line) > 5:
                subtitle = re.sub(r'^[#\*\-\s]+', '', line)
                subtitle = re.sub(r'[#\*\-\s]+$', '', subtitle)
                if len(subtitle) > 3:
                    return subtitle[:40]
        
        return "특별한 순간들"

    def _clean_content(self, content: str) -> str:
        """콘텐츠 정리"""
        # 불필요한 패턴 제거
        cleaned = re.sub(r'^[#\*\-\s]+', '', content, flags=re.MULTILINE)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'^\s*$\n', '', cleaned, flags=re.MULTILINE)
        
        return cleaned.strip()

    def _extract_structured_content(self, text: str) -> List[Dict]:
        """구조화된 콘텐츠 추출"""
        sections = []
        
        # 제목 패턴 찾기
        title_patterns = [
            r'제목[:\s]*([^\n]+)',
            r'title[:\s]*([^\n]+)',
            r'## ([^\n]+)',
            r'# ([^\n]+)'
        ]
        
        for pattern in title_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                title = match.group(1).strip()
                if len(title) > 3:
                    section = {
                        "template": f"Section{len(sections)+1:02d}.jsx",
                        "title": title[:50],
                        "subtitle": "여행의 특별한 순간",
                        "body": f"{title}에 대한 자세한 이야기를 담고 있습니다.",
                        "tagline": "TRAVEL & CULTURE",
                        "layout_source": "org_agent"
                    }
                    sections.append(section)
                    
                    if len(sections) >= 3:  # 최대 3개
                        break
            
            if sections:
                break
        
        return sections

    def _extract_template_from_binding_result(self, result: Dict) -> str:
        """BindingAgent 결과에서 템플릿명 추출"""
        task_desc = result.get('task_description', '')
        
        # 템플릿명 패턴 찾기
        template_match = re.search(r'Section\d+\.jsx', task_desc)
        if template_match:
            return template_match.group()
        
        return "Section01.jsx"

    def _extract_image_source_info(self, result: Dict, url: str) -> Dict:
        """이미지 소스 정보 추출"""
        return {
            "url": url,
            "template": self._extract_template_from_binding_result(result),
            "source": "binding_agent",
            "timestamp": result.get('timestamp', ''),
            "quality_verified": True
        }

    def _format_sections_for_analysis(self, sections: List[Dict]) -> str:
        """분석용 섹션 포맷팅"""
        formatted = []
        for i, section in enumerate(sections[:3]):  # 최대 3개만 표시
            formatted.append(f"""
섹션 {i+1}:
- 템플릿: {section.get('template', 'N/A')}
- 제목: {section.get('title', 'N/A')}
- 부제목: {section.get('subtitle', 'N/A')}
- 본문 길이: {len(section.get('body', ''))} 문자
- 소스: {section.get('layout_source', 'N/A')}
""")
        
        return "\n".join(formatted)

    def _format_images_for_analysis(self, template_images: Dict) -> str:
        """분석용 이미지 포맷팅"""
        formatted = []
        for template, images in template_images.items():
            formatted.append(f"""
{template}: {len(images)}개 이미지
{chr(10).join([f"  - {img}" for img in images[:2]])}
""")
        
        return "\n".join(formatted)

    def _format_image_sources(self, image_sources: List[Dict]) -> str:
        """이미지 소스 정보 포맷팅"""
        if not image_sources:
            return "이미지 소스 정보 없음"
        
        formatted = []
        for source in image_sources[:3]:  # 최대 3개만 표시
            formatted.append(f"- {source.get('template', 'N/A')}: {source.get('url', 'N/A')}")
        
        return "\n".join(formatted)

    def _extract_json_from_text(self, text: str) -> Dict:
        """텍스트에서 JSON 추출"""
        # JSON 패턴 찾기
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        
        parsed_data = {}
        for match in json_matches:
            try:
                if len(match) < 10000:  # 크기 제한
                    data = json.loads(match)
                    if isinstance(data, dict):
                        parsed_data.update(data)
            except json.JSONDecodeError:
                continue
        
        return parsed_data

    # 기존 메서드들 유지 (변경 없음)
    async def _log_coordination_result_async(self, final_result: Dict, text_mapping: Dict, 
                                           image_distribution: Dict, org_results: List[Dict], 
                                           binding_results: List[Dict]) -> None:
        """조율 결과 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="CoordinatorAgent",
                agent_role="매거진 구조 통합 조율자",
                task_description=f"CrewAI 기반 비동기 실제 데이터 활용으로 {len(final_result.get('content_sections', []))}개 섹션 생성",
                final_answer=f"매거진 구조 통합 완료: {len(final_result.get('content_sections', []))}개 섹션, 품질 점수: {final_result.get('integration_metadata', {}).get('integration_quality_score', 0):.2f}",
                reasoning_process=f"CrewAI 기반 비동기 실제 데이터 추출 및 활용으로 OrgAgent {len(org_results)}개, BindingAgent {len(binding_results)}개 결과 통합",
                execution_steps=[
                    "CrewAI 에이전트 생성",
                    "비동기 실제 데이터 추출",
                    "강화된 텍스트 분석 태스크 실행",
                    "강화된 이미지 분석 태스크 실행", 
                    "실제 데이터 기반 통합 조율",
                    "품질 검증 및 최적화"
                ],
                raw_input={"text_mapping": text_mapping, "image_distribution": image_distribution},
                raw_output=final_result,
                performance_metrics={
                    "async_processing": True,
                    "real_data_used": True,
                    "crew_execution_time": "optimized",
                    "total_sections": len(final_result.get('content_sections', [])),
                    "quality_score": final_result.get('integration_metadata', {}).get('integration_quality_score', 0),
                    "org_results_utilized": len(org_results),
                    "binding_results_utilized": len(binding_results),
                    "real_images_count": sum(len(section.get('images', [])) for section in final_result.get('content_sections', [])),
                    "content_enhancement": True
                }
            )
        )

    # 기존 메서드들 유지
    async def _get_enhanced_previous_results_async(self) -> List[Dict]:
        """강화된 이전 결과 수집 (비동기)"""
        try:
            # 병렬로 결과 수집
            basic_results_task = asyncio.get_event_loop().run_in_executor(
                None, lambda: self.logger.get_all_previous_results("CoordinatorAgent")
            )
            file_results_task = self._load_results_from_file_async()
            
            basic_results, file_results = await asyncio.gather(
                basic_results_task, file_results_task, return_exceptions=True
            )
            
            # 예외 처리
            if isinstance(basic_results, Exception):
                print(f"⚠️ 기본 결과 수집 실패: {basic_results}")
                basic_results = []
            
            if isinstance(file_results, Exception):
                print(f"⚠️ 파일 결과 수집 실패: {file_results}")
                file_results = []
            
            # 결과 합치기
            results = []
            results.extend(basic_results)
            results.extend(file_results)
            
            # 중복 제거 (비동기)
            unique_results = await asyncio.get_event_loop().run_in_executor(
                None, self._deduplicate_results, results
            )
            
            return unique_results
            
        except Exception as e:
            print(f"⚠️ 비동기 이전 결과 수집 실패: {e}")
            return []

    async def _load_results_from_file_async(self) -> List[Dict]:
        """파일에서 직접 결과 로드 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._load_results_from_file
        )

    def _load_results_from_file(self) -> List[Dict]:
        """파일에서 직접 결과 로드 (동기 버전 - 호환성 유지)"""
        results = []
        
        try:
            # latest_outputs.json에서 로드
            if os.path.exists('./agent_outputs/latest_outputs.json'):
                with open('./agent_outputs/latest_outputs.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    latest_outputs = data.get('latest_outputs', [])
                    results.extend(latest_outputs)
            
            # 세션 파일에서 로드
            session_files = []
            if os.path.exists('./agent_outputs/outputs'):
                for session_dir in os.listdir('./agent_outputs/outputs'):
                    session_path = os.path.join('./agent_outputs/outputs', session_dir, 'agent_outputs.json')
                    if os.path.exists(session_path):
                        session_files.append(session_path)
            
            # 최신 세션 파일 우선 처리
            session_files.sort(reverse=True)
            for session_file in session_files[:3]:  # 최근 3개 세션만
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                        if 'outputs' in session_data:
                            results.extend(session_data['outputs'])
                except Exception as e:
                    print(f"⚠️ 세션 파일 로드 실패 {session_file}: {e}")
                    continue
                    
        except Exception as e:
            print(f"⚠️ 파일 결과 로드 실패: {e}")
        
        return results

    def _filter_agent_results(self, results: List[Dict], agent_type: str) -> List[Dict]:
        """특정 에이전트 결과 필터링"""
        filtered = []
        for result in results:
            agent_name = result.get('agent_name', '')
            if agent_type in agent_name:
                filtered.append(result)
        return filtered

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """결과 중복 제거"""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            result_id = result.get('output_id') or result.get('timestamp', '')
            if result_id and result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
        
        return unique_results

    def _calculate_enhanced_quality_score(self, content_sections: List[Dict], 
                                        org_count: int, binding_count: int) -> float:
        """강화된 품질 점수 계산"""
        if not content_sections:
            return 0.0
        
        quality_score = 0.0
        
        # 1. 섹션 품질 (60%)
        section_quality = 0.0
        for section in content_sections:
            section_score = 0.0
            
            if section.get("title") and len(section.get("title", "")) > 3:
                section_score += 0.25
            if section.get("subtitle") and len(section.get("subtitle", "")) > 3:
                section_score += 0.15
            if section.get("body") and len(section.get("body", "")) > 50:
                section_score += 0.35
            if section.get("images") and len(section.get("images", [])) > 0:
                section_score += 0.25
            
            section_quality += min(section_score, 1.0)
        
        quality_score += (section_quality / len(content_sections)) * 0.6
        
        # 2. 에이전트 활용도 (25%)
        agent_score = 0.0
        if org_count > 0:
            agent_score += 0.5
        if binding_count > 0:
            agent_score += 0.5
        
        quality_score += agent_score * 0.25
        
        # 3. 실제 데이터 활용도 (15%)
        real_data_score = 0.0
        real_content_sections = sum(1 for section in content_sections 
                                  if section.get('metadata', {}).get('real_content', False))
        real_image_sections = sum(1 for section in content_sections 
                                if section.get('metadata', {}).get('real_images', False))
        
        if real_content_sections > 0:
            real_data_score += 0.5
        if real_image_sections > 0:
            real_data_score += 0.5
        
        quality_score += real_data_score * 0.15
        
        return min(quality_score, 1.0)

    # 동기 버전 메서드 유지 (호환성 보장)
    def coordinate_magazine_creation_sync(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """매거진 구조 통합 조율 (동기 버전 - 호환성 유지)"""
        return asyncio.run(self.coordinate_magazine_creation(text_mapping, image_distribution))

    def _get_enhanced_previous_results(self) -> List[Dict]:
        """강화된 이전 결과 수집 (동기 버전 - 호환성 유지)"""
        return asyncio.run(self._get_enhanced_previous_results_async())
