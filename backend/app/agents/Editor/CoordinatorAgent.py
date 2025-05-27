import os
from typing import Dict, List
from crewai import Agent
from custom_llm import get_azure_llm

class CoordinatorAgent:
    """단순화된 통합 조율자"""
    
    def __init__(self):
        self.llm = get_azure_llm()
    
    def create_agent(self):
        return Agent(
            role="매거진 통합 조율자",
            goal="텍스트와 이미지를 통합하여 최종 데이터 생성",
            backstory="""당신은 18년간 세계 최고 수준의 출판사에서 편집 디렉터 및 품질 관리 책임자로 활동해온 전문가입니다. Condé Nast, Hearst Corporation, Time Inc.에서 수백 개의 매거진 프로젝트를 성공적으로 조율했습니다.

            **전문 경력:**
            - 출판학 및 프로젝트 관리 석사 학위 보유
            - PMP(Project Management Professional) 인증
            - 품질 관리 및 프로세스 최적화 전문가
            - 크로스 플랫폼 출판 워크플로우 설계 경험
            - 독자 경험(UX) 및 접근성 최적화 전문성

            **다중 에이전트 데이터 통합 전문성:**
            당신은 최종 매거진 생성 시 다음 데이터들을 종합적으로 조율합니다:

            1. **에이전트 간 작업 결과 통합**:
            - OrgAgent의 텍스트 배치 데이터와 BindingAgent의 이미지 배치 데이터 조화
            - 각 에이전트의 신뢰도 점수와 작업 품질 메트릭 분석
            - 에이전트 간 의견 충돌 시 최적 해결책 도출
            - 전체 페이지 레이아웃의 시각적 균형과 일관성 검증

            2. **품질 보증 데이터 분석**:
            - 매거진 전체의 시각적 일관성 점수 계산
            - 독자 경험 흐름의 연속성 및 자연스러움 평가
            - 접근성 가이드라인 준수도 검증
            - 다양한 디바이스에서의 렌더링 품질 보장

            3. **독자 경험 최적화 데이터**:
            - 페이지 간 전환의 자연스러움 측정
            - 정보 계층 구조의 명확성 및 논리성 평가
            - 독자 피로도 최소화를 위한 콘텐츠 밀도 조절
            - 감정적 몰입도 유지를 위한 페이스 조절

            4. **학습 데이터 통합 및 피드백**:
            - 이전 에이전트들의 의사결정 로그 분석
            - 성공적인 협업 패턴과 개선이 필요한 영역 식별
            - 에이전트 간 커뮤니케이션 효율성 측정
            - 전체 시스템의 성능 최적화 포인트 발견

            **조율 철학:**
            "훌륭한 매거진은 개별 요소들의 단순한 합이 아니라, 모든 요소가 하나의 완성된 경험을 만들어내는 유기적 통합체입니다. 나는 각 전문 에이전트의 뛰어난 작업 결과를 존중하면서도, 전체적인 관점에서 독자에게 최고의 경험을 제공할 수 있도록 모든 요소를 조화롭게 통합합니다."

            **학습 데이터 활용 전략:**
            - 이전 조율 작업의 성공/실패 패턴을 분석하여 의사결정 기준 개선
            - 각 에이전트의 작업 품질 변화 추이를 모니터링하여 협업 최적화
            - 독자 피드백과 매거진 구성 요소의 상관관계 분석
            - 전체 시스템의 효율성 지표를 통한 워크플로우 지속 개선""",
            verbose=True,
            llm=self.llm
        )
    
    def coordinate_magazine_creation(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """단순화된 데이터 통합 - 설명 텍스트 제거"""
        
        # 텍스트 데이터 추출
        text_sections = text_mapping.get("text_mapping", [])
        
        # 이미지 데이터 추출
        image_data = image_distribution.get("image_distribution", {})
        
        # 통합 데이터 생성
        content_sections = []
        selected_templates = []
        
        for section in text_sections:
            template_name = section.get("template", "")
            selected_templates.append(template_name)
            
            # 해당 템플릿의 이미지 가져오기
            section_images = image_data.get(template_name, [])
            
            # 제목과 부제목에서 설명 텍스트 제거
            clean_title = self._clean_title_text(section.get("title", "여행 이야기"))
            clean_subtitle = self._clean_subtitle_text(section.get("subtitle", "특별한 순간들"))
            
            content_sections.append({
                "template": template_name,
                "title": clean_title,
                "subtitle": clean_subtitle,
                "body": section.get("body", "여행의 아름다운 기억들"),
                "tagline": section.get("tagline", "TRAVEL & CULTURE"),
                "images": section_images
            })
        
        final_result = {
            "selected_templates": selected_templates,
            "content_sections": content_sections
        }
        
        print(f"✅ 통합 완료: {len(content_sections)}개 섹션 생성")
        return final_result
    
    def _clean_title_text(self, title: str) -> str:
        """제목에서 설명 텍스트 제거"""
        import re
        
        # 제거할 패턴들
        patterns_to_remove = [
            r'\(헤드라인\)',
            r'\(섹션 타이틀\)',
            r'및 부제목.*?배치되어 있음',
            r'필자 정보.*?배치되어 있음',
            r'포토 크레딧.*?배치되어 있음',
            r'계층적으로.*?배치되어 있음',
            r'과 본문의 배치 관계:',
            r'과 본문 배치:',
            r'배치:.*?배치되며',
            r'은 상단에.*?배치되며',
            r'혹은 좌상단에.*?줍니다',
            r'상단 혹은.*?강조합니다',
            r'없이 단일.*?집중시킵니다',
            r'과 소제목.*?있습니다',
            r'그 아래로.*?줄여줍니다',
            r'본문.*?구분할 수 있는.*?있습니다',
            r'콘텐츠의 각 요소.*?있습니다',
            r', 그 아래로.*?있습니다'
        ]
        
        clean_title = title
        for pattern in patterns_to_remove:
            clean_title = re.sub(pattern, '', clean_title, flags=re.IGNORECASE | re.DOTALL)
        
        # 연속된 공백과 특수문자 정리
        clean_title = re.sub(r'\s+', ' ', clean_title)
        clean_title = re.sub(r'^[,\s]+|[,\s]+$', '', clean_title)
        
        # 빈 문자열이면 기본값 반환
        if not clean_title.strip():
            return "도쿄 여행 이야기"
        
        return clean_title.strip()
    
    def _clean_subtitle_text(self, subtitle: str) -> str:
        """부제목에서 설명 텍스트 제거"""
        import re
        
        patterns_to_remove = [
            r'필자 정보.*?배치되어 있음',
            r'포토 크레딧.*?배치되어 있음',
            r'계층적으로.*?배치되어 있음'
        ]
        
        clean_subtitle = subtitle
        for pattern in patterns_to_remove:
            clean_subtitle = re.sub(pattern, '', clean_subtitle, flags=re.IGNORECASE | re.DOTALL)
        
        clean_subtitle = re.sub(r'\s+', ' ', clean_subtitle)
        clean_subtitle = re.sub(r'^[,\s]+|[,\s]+$', '', clean_subtitle)
        
        if not clean_subtitle.strip():
            return "특별한 순간들"
        
        return clean_subtitle.strip()
