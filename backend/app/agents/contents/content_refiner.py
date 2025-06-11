import asyncio
import re
import json
from typing import Dict, List, Any, Tuple
from ...custom_llm import get_azure_llm
from ...utils.log.hybridlogging import get_hybrid_logger

class ContentRefiner:
    """콘텐츠 분량 검토 및 지능적 분할을 담당하는 클래스"""
    
    def __init__(self, max_section_length: int = 1000):
        self.llm = get_azure_llm()
        self.logger = get_hybrid_logger(self.__class__.__name__)
        self.max_section_length = max_section_length
    
    async def refine_content(self, sections: List[Dict]) -> List[Dict]:
        """모든 섹션의 콘텐츠를 검토하고 필요시 분할"""
        
        self.logger.info(f"콘텐츠 분량 검토 및 분할 시작: {len(sections)}개 섹션")
        refined_sections = []
        
        for section in sections:
            if len(section.get('body', '')) > self.max_section_length:
                self.logger.info(f"섹션 '{section['title']}' 분할 필요 (길이: {len(section['body'])}자)")
                split_sections = await self._split_long_section(section)
                refined_sections.extend(split_sections)
                self.logger.info(f"섹션 '{section['title']}' {len(split_sections)}개 하위 섹션으로 분할 완료")
            else:
                refined_sections.append(section)
        
        self.logger.info(f"콘텐츠 분량 검토 및 분할 완료: {len(refined_sections)}개 섹션")
        return refined_sections
    
    async def _split_long_section(self, section: Dict) -> List[Dict]:
        """긴 섹션을 자연스러운 하위 섹션으로 분할"""
        
        section_id = section.get('section_id', '0')
        title = section.get('title', '')
        body = section.get('body', '')
        
        prompt = f"""
당신은 전문 편집자입니다. 아래 제공된 섹션은 너무 길어서 여러 개의 하위 섹션으로 나눠야 합니다.

**섹션 제목:** {title}

**섹션 내용:**
{body}

**작업 지시:**
1. 위 내용을 2-3개의 자연스러운 하위 섹션으로 나누세요.
2. 각 하위 섹션에는 원본 섹션 번호에 하위 번호를 붙인 ID(예: {section_id}-1, {section_id}-2)와 적절한 소제목, 그리고 내용에 어울리는 **부제목**을 부여하세요.
3. 내용을 나눌 때는 반드시 문장 단위로 분할하세요. 절대로 문장 중간에서 끊지 마세요.
4. 원본의 모든 내용이 빠짐없이 포함되어야 합니다.
5. 각 하위 섹션은 최대한 비슷한 길이를 가지도록 균형있게 나누세요.

**출력 형식:**
아래의 JSON 형식으로 출력하세요. 다른 설명이나 주석은 포함하지 마세요.

```json
[
  {{
    "sub_section_id": "{section_id}-1",
    "title": "첫 번째 하위 섹션 제목",
    "subtitle": "첫 번째 하위 섹션 부제목",
    "body": "첫 번째 하위 섹션 내용..."
  }},
  {{
    "sub_section_id": "{section_id}-2",
    "title": "두 번째 하위 섹션 제목",
    "subtitle": "두 번째 하위 섹션 부제목",
    "body": "두 번째 하위 섹션 내용..."
  }},
  {{
    "sub_section_id": "{section_id}-3",
    "title": "세 번째 하위 섹션 제목",
    "subtitle": "세 번째 하위 섹션 부제목",
    "body": "세 번째 하위 섹션 내용..."
  }}
]
```

**중요 지침:**
- 각 하위 섹션의 내용은 완전한 문장으로 시작하고 완전한 문장으로 끝나야 합니다.
- 하위 섹션 간에 내용이 중복되지 않도록 하세요.
- 원본 내용의 흐름과 맥락을 유지하세요.
"""
        
        try:
            # LLM을 통한 섹션 분할 (ainvoke 사용)
            response = await self.llm.ainvoke(prompt)
            
            # JSON 응답 추출 및 파싱
            # JSON 부분만 추출
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
            
            # 불필요한 마크다운이나 설명 제거
            json_str = re.sub(r'```(json)?|```', '', json_str).strip()
            
            # JSON 파싱
            sub_sections = json.loads(json_str)
            
            # 각 하위 섹션에 원본 섹션 정보 추가
            for sub_section in sub_sections:
                sub_section['parent_section_id'] = section_id
                sub_section['parent_section_title'] = title
            
            # 문장 경계 검증
            sub_sections = self._verify_sentence_boundaries(sub_sections)
            
            return sub_sections
            
        except Exception as e:
            self.logger.error(f"섹션 분할 실패: {e}")
            # 실패 시 원본 섹션 그대로 반환
            return [section]
    
    def _verify_sentence_boundaries(self, sub_sections: List[Dict]) -> List[Dict]:
        """각 하위 섹션의 시작과 끝이 완전한 문장인지 확인"""
        
        # 간단한 문장 종결 패턴 (마침표, 물음표, 느낌표로 끝나는지)
        end_pattern = r'[.!?][\s"\']*$'
        
        for i, section in enumerate(sub_sections):
            body = section.get('body', '')
            
            # 첫 번째 섹션이 아닌 경우, 시작 부분이 완전한 문장인지 확인
            if i > 0:
                # 첫 글자가 대문자로 시작하는지 확인 (영어 기준)
                if body and not body[0].isupper() and body[0].isalpha():
                    self.logger.warning(f"하위 섹션 {section.get('sub_section_id')}가 완전한 문장으로 시작하지 않을 수 있습니다.")
            
            # 마지막 부분이 완전한 문장인지 확인
            if body and not re.search(end_pattern, body):
                self.logger.warning(f"하위 섹션 {section.get('sub_section_id')}가 완전한 문장으로 끝나지 않을 수 있습니다.")
                # 필요시 여기에 문장 경계 수정 로직 추가 가능
        
        return sub_sections 