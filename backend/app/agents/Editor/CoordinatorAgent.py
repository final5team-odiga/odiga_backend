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
            backstory="매거진 데이터를 통합하는 전문가입니다.",
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
