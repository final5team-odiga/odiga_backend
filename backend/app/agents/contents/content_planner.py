import asyncio
from typing import Dict, List, Any
from custom_llm import get_azure_llm
from utils.log.hybridlogging import get_hybrid_logger

class ContentPlannerAgent:
    """콘텐츠 분석 및 구조 설계를 담당하는 에이전트"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        self.logger = get_hybrid_logger(self.__class__.__name__)
    
    async def analyze_and_plan_structure(self, interview_results: Dict[str, str], 
                                        essay_results: Dict[str, str],
                                        image_analysis_results: List[Dict]) -> Dict:
        """콘텐츠 분석 및 최적의 섹션 구조 설계"""
        
        self.logger.info("콘텐츠 분석 및 섹션 구조 설계 시작")
        
        # 모든 인터뷰 콘텐츠 정리
        interview_content = "\n\n".join([f"=== {key} ===\n{value}" for key, value in interview_results.items()])
        
        # 모든 에세이 콘텐츠 정리
        essay_content = "\n\n".join([f"=== {key} ===\n{value}" for key, value in essay_results.items()])
        
        # 이미지 분석 정보 정리
        image_info = self._process_image_analysis(image_analysis_results)
        
        # 프롬프트 구성
        prompt = f"""
당신은 전문 편집장입니다. 주어진 텍스트 묶음을 분석하여 최적의 매거진 구조를 설계해야 합니다.

**인터뷰 형식 콘텐츠:**
{interview_content[:3000]}... (생략)

**에세이 형식 콘텐츠:**
{essay_content[:3000]}... (생략)

**이미지 정보:**
{image_info[:1000]}... (생략)

**작업 지시:**
1. 제공된 콘텐츠의 내용과 분량을 분석하세요.
2. 자연스러운 내러티브 흐름과 주제 전환점을 찾아내세요.
3. 내용의 양과 복잡성을 고려하여 최적의 섹션 개수를 결정하세요. (섹션 개수는 고정되지 않고, 콘텐츠에 따라 달라질 수 있습니다)
4. 각 섹션의 핵심 내용을 요약하고, 매력적인 섹션 제목을 정해주세요.

**출력 형식:**
아래의 JSON 형식으로 출력하세요. 다른 설명이나 주석은 포함하지 마세요.
estimated_length 필드에는 "짧음", "중간", "김" 중 하나의 값만 입력하세요.

```json
{{
  "proposed_title": "매거진의 전체 제목",
  "proposed_subtitle": "매거진의 부제목",
  "sections": [
    {{
      "section_id": "1",
      "title": "첫 번째 섹션 제목",
      "subtitle": "첫 번째 섹션 부제목",
      "summary": "이 섹션에서 다룰 내용에 대한 요약과 주요 키워드",
      "estimated_length": "중간"
    }}
  ]
}}
```

**중요 지침:**
- 섹션 개수는 콘텐츠의 양과 복잡성에 따라 자유롭게 결정하세요. (최소 3개, 최대 10개 권장)
- 각 섹션은 하나의 일관된 주제나 경험을 다루어야 합니다.
- 제목과 부제목은 매력적이고 독자의 관심을 끌 수 있어야 합니다.
- 모든 원본 콘텐츠가 어느 섹션에든 포함되도록 계획하세요.
"""
        
        try:
            # LLM을 통한 구조 설계 (ainvoke 사용)
            response = await self.llm.ainvoke(prompt)
            
            # JSON 응답 추출 및 파싱
            import json
            import re
            
            # JSON 부분만 추출
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
            
            # 불필요한 마크다운이나 설명 제거
            json_str = re.sub(r'```(json)?|```', '', json_str).strip()
            
            # JSON 파싱
            try:
                structure_plan = json.loads(json_str)
                self.logger.info(f"구조 설계 완료: {len(structure_plan.get('sections', []))}개 섹션")
                return structure_plan
            except Exception as e:
                self.logger.error(f"JSON 파싱 실패: {str(e)}")
                return self._create_default_structure_plan()
            
        except Exception as e:
            self.logger.error(f"구조 설계 실패: {str(e)}")
            # 실패 시 기본 구조 반환
            return self._create_default_structure_plan()
    
    def _process_image_analysis(self, image_analysis_results: List[Dict]) -> str:
        """이미지 분석 결과를 텍스트로 변환"""
        if not image_analysis_results:
            return "이미지 정보 없음"
        
        image_descriptions = []
        for i, img in enumerate(image_analysis_results):
            description = f"이미지 {i+1}: "
            if "description" in img:
                description += img["description"]
            elif "caption" in img:
                description += img["caption"]
            else:
                description += "설명 없음"
                
            if "location" in img:
                description += f" (위치: {img['location']})"
            
            image_descriptions.append(description)
        
        return "\n".join(image_descriptions)
    
    def _create_default_structure_plan(self) -> Dict:
        """기본 구조 계획 생성 (오류 발생 시 폴백)"""
        return {
            "proposed_title": "여행 경험",
            "proposed_subtitle": "특별한 순간들",
            "sections": [
                {
                    "section_id": "1",
                    "title": "여행의 시작",
                    "subtitle": "새로운 경험",
                    "summary": "여행의 시작과 첫인상에 대한 내용",
                    "estimated_length": "중간"
                },
                {
                    "section_id": "2",
                    "title": "주요 경험",
                    "subtitle": "특별한 순간들",
                    "summary": "여행 중 겪은 주요 경험과 감상",
                    "estimated_length": "김"
                },
                {
                    "section_id": "3",
                    "title": "마무리",
                    "subtitle": "여행을 마치며",
                    "summary": "여행의 마무리와 전체 소감",
                    "estimated_length": "짧음"
                }
            ]
        } 