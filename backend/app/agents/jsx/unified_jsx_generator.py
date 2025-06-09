import asyncio
import json
import time
import re
from typing import Dict, List, Any
from custom_llm import get_azure_llm
from utils.isolation.ai_search_isolation import AISearchIsolationManager
from utils.isolation.session_isolation import SessionAwareMixin
from utils.isolation.agent_communication_isolation import InterAgentCommunicationMixin
from utils.log.logging_manager import LoggingManager

class UnifiedJSXGenerator(SessionAwareMixin, InterAgentCommunicationMixin):
    """통합 JSX 생성기 - 템플릿 코드와 콘텐츠를 결합하여 최종 JSX 컴포넌트 생성"""
    
    def __init__(self, logger: Any = None):
        self.llm = get_azure_llm()
        self.logger = logger
        self._setup_logging_system()
        
        self.isolation_manager = AISearchIsolationManager()
        self.__init_session_awareness__()
        self.__init_inter_agent_communication__()

    def _setup_logging_system(self):
        """로그 저장 시스템 설정"""
        self.log_enabled = True
        self.response_counter = 0

    def set_logger(self, logger):
        """로거 설정 (외부 주입)"""
        self.logger = logger
        self.logging_manager = LoggingManager(self.logger)
        
    async def process_data(self, input_data):
        """에이전트 인터페이스 구현"""
        result = await self._do_work(input_data)
        
        if self.logger and hasattr(self, 'logging_manager'):
            await self.logging_manager.log_agent_response(
                agent_name=self.__class__.__name__,
                agent_role="JSX 컴포넌트 생성기",
                task_description="템플릿과 콘텐츠를 결합한 JSX 컴포넌트 생성",
                response_data=result,
                metadata={"integrated_processing": True}
            )
        
        return result

    async def generate_jsx_from_template(self, content_data: Dict, template_code: str) -> Dict:
        """템플릿 코드를 기반으로 JSX 생성"""
        try:
            self.logger.info(f"JSX 생성 시작: {content_data.get('title', '제목 없음')}")
            
            # 하위 섹션 여부 확인
            is_subsection = content_data.get("metadata", {}).get("is_subsection", False)
            
            # 프롬프트에 하위 섹션 정보 추가
            subsection_info = ""
            if is_subsection:
                parent_section_id = content_data.get("metadata", {}).get("parent_section_id", "")
                parent_section_title = content_data.get("metadata", {}).get("parent_section_title", "")
                subsection_info = f"""
이 JSX 컴포넌트는 하위 섹션입니다:
- 상위 섹션 ID: {parent_section_id}
- 상위 섹션 제목: {parent_section_title}
- 제목에는 이미 상위 섹션 정보가 포함되어 있습니다
"""
            
            # 선택된 템플릿 적용 (지능형 변환)
            try:
                if template_code and len(template_code) > 0:
                    # 이 메서드는 이미 딕셔너리를 반환하므로 수정 불필요
                    jsx_result = await self._generate_intelligent_jsx(content_data, template_code, subsection_info)
                else:
                    # ⚠️ 폴백 함수의 반환값을 딕셔너리로 감싸서 일관성 유지
                    substituted_code = self._simple_template_substitution(content_data, self._get_default_template())
                    jsx_result = {
                        "title": content_data.get("title", "제목 없음"),
                        "jsx_code": substituted_code
                    }
            except Exception as jsx_error:
                self.logger.error(f"지능형 JSX 생성 실패 (폴백 사용): {jsx_error}")
                jsx_result = self._simple_template_substitution(content_data, self._get_default_template())
            
            # 메타데이터 추가
            jsx_result["metadata"] = {
                "template_applied": True,
                "generation_method": "intelligent_jsx_generation",
                "template_name": template_code.split("/")[-1] if "/" in template_code else template_code,
                "generation_timestamp": time.time(),
                "is_subsection": is_subsection
            }
            
            # 하위 섹션인 경우 추가 메타데이터
            if is_subsection:
                jsx_result["metadata"]["parent_section_id"] = content_data.get("metadata", {}).get("parent_section_id", "")
                jsx_result["metadata"]["parent_section_title"] = content_data.get("metadata", {}).get("parent_section_title", "")
            
            self.logger.info(f"JSX 생성 완료: {content_data.get('title', '제목 없음')}")
            return jsx_result
            
        except Exception as e:
            self.logger.error(f"JSX 생성 과정 실패: {e}")
            return self._create_fallback_jsx(content_data, str(e))

    async def _generate_intelligent_jsx(self, content_data: Dict, template_code: str, subsection_info: str = "") -> Dict:
        """
        AI를 활용한 지능형 JSX 생성 (ainvoke 사용 및 안정성 강화 버전)
        """
        title = content_data.get("title", "제목 없음")
        
        try:
            # ✅ 1. 프롬프트 생성을 별도 메서드로 분리하여 가독성 향상
            prompt = self._create_jsx_generation_prompt(content_data, template_code, subsection_info)

            # ✅ 2. 올바른 AI 모델 메서드(ainvoke) 호출
            # custom_llm.py (paste.txt)에 정의된 ainvoke를 사용합니다.
            if not hasattr(self.llm, 'ainvoke'):
                raise AttributeError("LLM 객체에 ainvoke 메서드가 없습니다. custom_llm.py를 확인하세요.")
            
            self.logger.info(f"'{title}' 섹션에 대한 지능형 JSX 생성을 시작합니다...")
            generated_code = await self.llm.ainvoke(prompt)
            
            if not generated_code or not isinstance(generated_code, str):
                raise ValueError("LLM으로부터 유효한 JSX 코드를 받지 못했습니다.")

            # ✅ 3. LLM 응답에서 순수 JSX 코드만 추출하는 후처리 단계 추가
            extracted_code = self._extract_jsx_code(generated_code)

            self.logger.info(f"'{title}' 섹션 JSX 생성 완료.")
            return {
                "title": title,
                "jsx_code": extracted_code
            }
        except Exception as e:
            # ✅ 4. 강화된 예외 처리 및 명확한 폴백
            self.logger.error(f"'{title}' 섹션의 지능형 JSX 생성 실패 (폴백 사용): {e}")
            # 오류 발생 시, 간단한 치환 기반의 폴백 로직으로 전환
            fallback_result = self._simple_template_substitution(content_data, self._get_default_template())
            return fallback_result


    def _create_jsx_generation_prompt(self, content_data: Dict, template_code: str, subsection_info: str = "") -> str:
        """JSX 생성용 LLM 프롬프트를 구성합니다."""
        
        title = content_data.get("title", "제목 없음")
        pascal_case_title = ''.join(word.capitalize() for word in title.replace('-', ' ').replace('_', ' ').split())
        
        # 이미지 데이터 요약
        image_data_json = self._format_image_data_for_prompt(content_data.get("images", []))
        
        return f"""# JSX 컴포넌트 생성 작업

    ## 기본 정보
    - 컴포넌트 제목: {title}
    - 컴포넌트 이름 (PascalCase): {pascal_case_title}
    - 이미지 수: {len(content_data.get("images", []))}
    {subsection_info}

    ## 콘텐츠 데이터 요약
    {{
    "title": "{title}",
    "subtitle": "{content_data.get("subtitle", "")}",
    "content": "{content_data.get("content", "")[:300]}...(생략)",
    "images": {image_data_json[:300]}...(생략)
    }}

    text

    ## 기준 템플릿 코드
    {template_code}

    text

    ## 작업 지시사항
    1. 위 '기준 템플릿 코드'를 기반으로, '콘텐츠 데이터'를 표시하는 완성도 높은 React 컴포넌트를 생성하세요.
    2. 템플릿의 기본 구조와 스타일은 최대한 유지하되, 데이터에 맞게 내용을 채워 넣으세요.
    3. 생성할 컴포넌트의 이름은 제시된 '컴포넌트 이름 (PascalCase)'을 사용하세요.
    4. 이미지가 있다면, 템플릿 내의 적절한 위치에 `<img>` 태그를 사용하여 배치하세요. 이미지의 `src`는 `props.images[n].url`, `alt`는 `props.images[n].alt`를 사용하세요.
    5. 모든 스타일은 인라인 스타일 또는 Tailwind CSS 클래스를 사용하여 적용하세요.
    6. 최종 결과물은 `import React from "react";`를 포함한 완전한 JSX 파일 형식이어야 합니다.

    ## 출력 형식
    오직 유효한 JSX 코드만 제공하세요. 다른 설명, 주석, 마크다운 코드 블록(` `````` `)을 포함하지 마세요.
    """

    def _extract_jsx_code(self, response: str) -> str:
        """LLM의 응답에서 순수 JSX 코드만 추출합니다."""
        if '```' in response:
            match = re.search(r'```(?:jsx)?\s*([\s\S]+?)\s*```', response)
            if match:
                return match.group(1).strip()
        
        # 코드 블록이 없는 경우, 불필요한 서론/결론 제거 시도
        lines = response.strip().split('\n')
        if lines and 'export default' in response:
            # 'export default'가 포함된 첫 줄부터 시작
            for i, line in enumerate(lines):
                if 'export default' in line:
                    return '\n'.join(lines[i:])
        
        return response.strip()
    
    def _process_content_data(self, content_data: Dict) -> Dict:
        """콘텐츠 데이터 정제 및 구조화"""
        processed = {
            "title": content_data.get("title", "제목 없음"),
            "subtitle": content_data.get("subtitle", ""),
            "content": content_data.get("content", ""),
            "images": content_data.get("images", []),
            "metadata": content_data.get("metadata", {}),
            "layout_type": content_data.get("layout_type", "default")
        }
        
        # 이미지 URL 정제
        if processed["images"]:
            processed_images = []
            for img in processed["images"]:
                if isinstance(img, dict):
                    processed_images.append({
                        "url": img.get("image_url", img.get("url", "")),
                        "alt": img.get("description", img.get("alt", processed["title"])),
                        "caption": img.get("caption", "")
                    })
                elif isinstance(img, str):
                    processed_images.append({
                        "url": img,
                        "alt": processed["title"],
                        "caption": ""
                    })
            processed["images"] = processed_images
        
        return processed

    def _validate_template_code(self, template_code: str) -> str:
        """템플릿 코드 검증 및 정제"""
        if not template_code or not isinstance(template_code, str):
            return self._get_default_template()
        
        # 기본적인 React 컴포넌트 구조 확인
        if "export default" not in template_code and "function" not in template_code:
            return self._get_default_template()
        
        # 위험한 코드 패턴 제거
        dangerous_patterns = ["eval(", "Function(", "setTimeout(", "setInterval("]
        for pattern in dangerous_patterns:
            if pattern in template_code:
                if self.logger:
                    self.logger.warning(f"위험한 패턴 감지: {pattern}")
                return self._get_default_template()
        
        return template_code

    def _simple_template_substitution(self, content_data: Dict, template_code: str) -> Dict:
        jsx_code = template_code
        
        # 기본 치환
        substitutions = {
            "{props.title}": content_data.get('title', ''),
            "{props.subtitle}": content_data.get('subtitle', ''),
            "{props.content}": content_data.get('content', ''),
            "{props.body}": content_data.get('content', '')
        }
        
        for placeholder, value in substitutions.items():
            jsx_code = jsx_code.replace(placeholder, str(value)) # str()로 안정성 확보
        
        # 이미지 처리
        if content_data.get('images') and "{props.images}" in jsx_code:
            images_jsx = self._generate_images_jsx(content_data['images'])
            jsx_code = jsx_code.replace("{props.images}", images_jsx)
        
        # ✅ 딕셔너리 형태로 반환
        return {
            "title": content_data.get("title", "제목 없음"),
            "jsx_code": jsx_code
        }

    def _generate_images_jsx(self, images: List[Dict]) -> str:
        """이미지 JSX 생성"""
        if not images:
            return ""
        
        images_jsx = []
        for img in images:
            img_jsx = f'<img src="{img["url"]}" alt="{img["alt"]}" className="w-full h-auto rounded-lg mb-4" />'
            if img.get("caption"):
                img_jsx += f'<p className="text-sm text-gray-600 mb-4">{img["caption"]}</p>'
            images_jsx.append(img_jsx)
        
        return "\n".join(images_jsx)

    def _validate_generated_jsx(self, jsx_code: str) -> str:
        """생성된 JSX 검증"""
        if not jsx_code:
            return self._get_default_jsx_with_content({})
        
        # 기본적인 JSX 구조 확인
        if "export default" not in jsx_code:
            jsx_code = f"export default function GeneratedComponent() {{\n  return (\n    {jsx_code}\n  );\n}}"
        
        return jsx_code

    def _add_component_metadata(self, jsx_code: str, content_data: Dict, template_code: str) -> Dict:
        """컴포넌트 메타데이터 추가"""
        return {
            "title": content_data.get("title", "제목 없음"),
            "jsx_code": jsx_code,
            "metadata": {
                "template_applied": True,
                "generation_method": "ai_enhanced",
                "content_type": content_data.get("layout_type", "default"),
                "image_count": len(content_data.get("images", [])),
                "has_subtitle": bool(content_data.get("subtitle")),
                "content_length": len(content_data.get("content", "")),
                "generation_timestamp": time.time()
            }
        }

    def _create_fallback_jsx(self, content_data: Dict, error_message: str) -> Dict:
        """폴백 JSX 생성"""
        fallback_jsx = self._get_default_jsx_with_content(content_data)
        
        return {
            "title": content_data.get("title", "제목 없음"),
            "jsx_code": fallback_jsx,
            "metadata": {
                "template_applied": False,
                "generation_method": "fallback",
                "error": error_message,
                "generation_timestamp": time.time()
            }
        }

    def _get_default_template(self) -> str:
        """기본 템플릿 반환"""
        return """
        export default function DefaultTemplate(props) {
          return (
            <div className="section-container p-4 my-8">
              <h2 className="text-2xl font-bold mb-2">{props.title}</h2>
              {props.subtitle && <h3 className="text-xl mb-4">{props.subtitle}</h3>}
              <div className="content" dangerouslySetInnerHTML={{ __html: props.content }} />
            </div>
          );
        }
        """

    def _get_default_jsx_with_content(self, content_data: Dict) -> str:
        """콘텐츠가 포함된 기본 JSX"""
        title = content_data.get("title", "제목 없음")
        subtitle = content_data.get("subtitle", "")
        content = content_data.get("content", "")
        
        jsx = f"""
        export default function DefaultSection(props) {{
          return (
            <div className="section-container p-4 my-8">
              <h2 className="text-2xl font-bold mb-2">{title}</h2>
              {f'<h3 className="text-xl mb-4">{subtitle}</h3>' if subtitle else ''}
              <div className="content" dangerouslySetInnerHTML={{ __html: "{content}" }} />
            </div>
          );
        }}
        """
        
        return jsx

    def _extract_jsx_code(self, response: str) -> str:
        """AI 응답에서 JSX 코드 추출"""
        # 코드 블록 마커 제거
        if "```jsx" in response:
            start = response.find("```jsx") + 6
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # 마커가 없으면 전체 응답 사용
        return response.strip()
    
    async def _get_ai_generated_jsx(self, prompt: str) -> str:
        """AI 모델을 사용하여 JSX 코드 생성"""
        try:
            if hasattr(self.llm, 'agenerate_text'):
                response = await self.llm.agenerate_text(prompt)
            else:
                # 동기 메서드 사용
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.llm.generate_text, prompt
                )
            return response
        except Exception as e:
            self.logger.error(f"AI 생성 실패: {e}")
            raise e
            
    def _format_image_data_for_prompt(self, images: List[Dict]) -> str:
        """이미지 데이터를 프롬프트용으로 포맷팅"""
        if not images:
            return "[]"
        
        # 중요 필드만 포함하여 간략화
        simplified_images = []
        for img in images[:3]:  # 최대 3개만 포함
            simplified_img = {
                "url": img.get("url", ""),
                "alt_text": img.get("alt_text", img.get("caption", "")),
                "width": img.get("width", 800),
                "height": img.get("height", 600),
                "caption": img.get("caption", "")
            }
            simplified_images.append(simplified_img)
        
        # JSON 문자열로 변환
        try:
            return json.dumps(simplified_images, ensure_ascii=False)
        except:
            return "[]"
    
    async def _load_template_file(self, template_path: str) -> str:
        """템플릿 파일 로드"""
        try:
            # 경로 정규화
            if not template_path.startswith("/"):
                template_path = f"jsx_templates/{template_path}"
            
            # 파일 읽기
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"템플릿 파일 로드 실패: {e}")
            return self._get_default_template()