import asyncio
import json
import time
import traceback
import re
from typing import Dict, List, Any
from custom_llm import get_azure_llm
from utils.log.hybridlogging import get_hybrid_logger
from utils.isolation.ai_search_isolation import AISearchIsolationManager
from utils.data.pdf_vector_manager import PDFVectorManager
from utils.isolation.session_isolation import SessionAwareMixin
from utils.isolation.agent_communication_isolation import InterAgentCommunicationMixin

class UnifiedJSXGenerator(SessionAwareMixin, InterAgentCommunicationMixin):
    """통합 JSX 생성기"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_hybrid_logger("UnifiedJSXGenerator")
        self.llm = get_azure_llm()
        self.pdf_vector_manager = PDFVectorManager()
        self.isolation_manager = AISearchIsolationManager()
        
    async def generate_jsx_component(self, component_data: Dict) -> str:
        """JSX 컴포넌트 생성"""
        try:
            # 컴포넌트 데이터 검증
            if not self._validate_component_data(component_data):
                raise ValueError("유효하지 않은 컴포넌트 데이터")
            
            # JSX 생성
            jsx_content = await self._generate_jsx_content(component_data)
            
            # 결과 검증
            if not self._validate_jsx_content(jsx_content):
                raise ValueError("유효하지 않은 JSX 생성 결과")
            
            return jsx_content
            
        except Exception as e:
            self.logger.error(f"JSX 컴포넌트 생성 실패: {e}")
            raise
    
    async def generate_jsx_from_template(self, content_data: Dict, template_code: str) -> Dict:
        """
        콘텐츠 데이터와 템플릿 코드를 결합하여 최종 JSX 컴포넌트 생성
        
        Args:
            content_data (Dict): 섹션 콘텐츠 데이터 (title, final_content, metadata 등)
            template_code (str): 템플릿 JSX 코드
            
        Returns:
            Dict: 최종 JSX 컴포넌트 정보
        """
        try:
            self.logger.info(f"템플릿 기반 JSX 생성 시작: {content_data.get('title', '제목 없음')}")
            
            # 콘텐츠 데이터 추출
            title = content_data.get('title', '')
            subtitle = content_data.get('subtitle', '')
            content = content_data.get('final_content', '')
            metadata = content_data.get('metadata', {})
            
            # HTML 콘텐츠 생성 (마크다운 -> HTML)
            html_content = await self._convert_content_to_html(content)
            
            # 템플릿에 콘텐츠 삽입을 위한 프롬프트 생성
            prompt = self._create_template_injection_prompt(
                template_code=template_code,
                title=title,
                subtitle=subtitle,
                content=html_content,
                metadata=metadata
            )
            
            # LLM을 통해 템플릿에 콘텐츠 삽입
            response = await self.llm.agenerate_text(prompt)
            jsx_result = self._extract_jsx_from_response(response)
            
            # 결과 검증
            if not self._validate_jsx_content(jsx_result):
                self.logger.warning("생성된 JSX가 유효하지 않습니다. 기본 템플릿으로 대체합니다.")
                jsx_result = self._create_fallback_jsx(title, subtitle, html_content)
            
            # 결과 반환
            return {
                "title": title,
                "jsx_code": jsx_result,
                "metadata": {
                    "template_applied": True,
                    "style_attributes": metadata
                }
            }
            
        except Exception as e:
            self.logger.error(f"템플릿 기반 JSX 생성 실패: {str(e)}")
            # 오류 발생 시 기본 템플릿 반환
            return {
                "title": content_data.get('title', '오류 발생'),
                "jsx_code": self._create_fallback_jsx(
                    content_data.get('title', ''),
                    content_data.get('subtitle', ''),
                    content_data.get('final_content', '')
                ),
                "metadata": {
                    "template_applied": False,
                    "error": str(e)
                }
            }
    
    async def _convert_content_to_html(self, content: str) -> str:
        """마크다운 또는 일반 텍스트를 HTML로 변환"""
        try:
            # 간단한 마크다운 변환 (실제로는 마크다운 라이브러리 사용 권장)
            prompt = f"""
            다음 텍스트를 HTML로 변환해주세요. 마크다운 문법이 있다면 적절히 처리하세요:
            
            {content}
            
            HTML만 반환하세요. 설명이나 다른 텍스트는 포함하지 마세요.
            """
            
            response = await self.llm.agenerate_text(prompt)
            html = self._extract_html_from_response(response)
            return html
        except Exception as e:
            self.logger.error(f"HTML 변환 실패: {e}")
            # 기본 HTML 래핑
            return f"<div>{content}</div>"
    
    def _extract_html_from_response(self, response: str) -> str:
        """LLM 응답에서 HTML 코드 추출"""
        html_pattern = r'```html\s*([\s\S]*?)\s*```|<html[\s\S]*?</html>|<div[\s\S]*?</div>|<p[\s\S]*?</p>'
        matches = re.findall(html_pattern, response)
        
        if matches:
            return matches[0].strip()
        
        # HTML 태그가 없는 경우 전체 응답을 반환
        return response.strip()
    
    def _extract_jsx_from_response(self, response: str) -> str:
        """LLM 응답에서 JSX 코드 추출"""
        jsx_pattern = r'```jsx\s*([\s\S]*?)\s*```|```js\s*([\s\S]*?)\s*```|```\s*([\s\S]*?)\s*```'
        matches = re.findall(jsx_pattern, response)
        
        if matches:
            # 첫 번째 매치에서 비어있지 않은 그룹 찾기
            for match in matches:
                for group in match:
                    if group.strip():
                        return group.strip()
        
        # 코드 블록이 없는 경우 전체 응답을 반환
        return response.strip()
    
    def _create_template_injection_prompt(self, template_code: str, title: str, subtitle: str, 
                                         content: str, metadata: Dict) -> str:
        """템플릿에 콘텐츠를 삽입하기 위한 프롬프트 생성"""
        # 메타데이터에서 스타일 정보 추출
        style = metadata.get('style', '')
        emotion = metadata.get('emotion', '')
        keywords = metadata.get('keywords', [])
        
        # 키워드를 문자열로 변환
        keywords_str = ', '.join(keywords) if isinstance(keywords, list) else str(keywords)
        
        return f"""
        다음 JSX 템플릿 코드에 제공된 콘텐츠를 삽입해주세요:
        
        # 템플릿 코드:
        ```jsx
        {template_code}
        ```
        
        # 삽입할 콘텐츠:
        제목: {title}
        부제목: {subtitle}
        
        콘텐츠(HTML):
        ```html
        {content}
        ```
        
        # 스타일 정보:
        - 스타일: {style}
        - 감정 톤: {emotion}
        - 키워드: {keywords_str}
        
        # 지침:
        1. 템플릿의 구조와 스타일을 유지하세요.
        2. 제목, 부제목, 콘텐츠를 적절한 위치에 삽입하세요.
        3. props 구조에 맞게 데이터를 전달하세요.
        4. HTML 콘텐츠는 dangerouslySetInnerHTML 또는 적절한 방식으로 렌더링하세요.
        5. 템플릿에 없는 요소는 추가하지 마세요.
        
        JSX 코드만 반환하세요. 설명이나 다른 텍스트는 포함하지 마세요.
        """
    
    def _create_fallback_jsx(self, title: str, subtitle: str, content: str) -> str:
        """오류 발생 시 사용할 기본 JSX 템플릿"""
        subtitle_element = '<h3 className="text-xl mb-4">' + subtitle + '</h3>' if subtitle else ''
        return f"""
        export default function DefaultSection(props) {{
          return (
            <div className="section-container p-4 my-8">
              <h2 className="text-2xl font-bold mb-2">{title}</h2>
              {subtitle_element}
              <div className="content" dangerouslySetInnerHTML={{{{ __html: "{content}" }}}} />
            </div>
          );
        }}
        """
    
    def _validate_component_data(self, data: Dict) -> bool:
        """컴포넌트 데이터 유효성 검사"""
        required_fields = ['type', 'content', 'style']
        return all(field in data for field in required_fields)
    
    async def _generate_jsx_content(self, data: Dict) -> str:
        """JSX 내용 생성"""
        component_type = data['type']
        content = data['content']
        style = data['style']
        
        # 컴포넌트 타입별 템플릿 적용
        if component_type == 'text':
            return self._generate_text_component(content, style)
        elif component_type == 'image':
            return self._generate_image_component(content, style)
        elif component_type == 'container':
            return await self._generate_container_component(content, style)
        else:
            raise ValueError(f"지원하지 않는 컴포넌트 타입: {component_type}")
    
    def _validate_jsx_content(self, content: str) -> bool:
        """JSX 내용 유효성 검사"""
        # 기본 구문 검사
        if not content or not isinstance(content, str):
            return False
        
        # JSX 태그 균형 검사
        tag_pattern = r'<(\w+)[^>]*>'
        closing_pattern = r'</(\w+)>'
        
        opening_tags = re.findall(tag_pattern, content)
        closing_tags = re.findall(closing_pattern, content)
        
        return len(opening_tags) == len(closing_tags)
    
    def _generate_text_component(self, content: str, style: Dict) -> str:
        """텍스트 컴포넌트 생성"""
        style_str = self._convert_style_to_string(style)
        return f'<div style={{{style_str}}}>{content}</div>'
    
    def _generate_image_component(self, content: Dict, style: Dict) -> str:
        """이미지 컴포넌트 생성"""
        style_str = self._convert_style_to_string(style)
        return f'<img src="{content["src"]}" alt="{content.get("alt", "")}" style={{{style_str}}} />'
    
    async def _generate_container_component(self, content: List[Dict], style: Dict) -> str:
        """컨테이너 컴포넌트 생성"""
        style_str = self._convert_style_to_string(style)
        
        # 내부 컴포넌트 생성
        inner_components = []
        for item in content:
            component = await self.generate_jsx_component(item)
            inner_components.append(component)
        
        return f'<div style={{{style_str}}}>{" ".join(inner_components)}</div>'
    
    def _convert_style_to_string(self, style: Dict) -> str:
        """스타일 객체를 문자열로 변환"""
        style_items = []
        for key, value in style.items():
            # camelCase로 변환
            key = self._to_camel_case(key)
            style_items.append(f'"{key}": "{value}"')
        return "{" + ", ".join(style_items) + "}"
    
    def _to_camel_case(self, snake_str: str) -> str:
        """snake_case를 camelCase로 변환"""
        components = snake_str.split('_')
        return components[0] + ''.join(x.title() for x in components[1:]) 