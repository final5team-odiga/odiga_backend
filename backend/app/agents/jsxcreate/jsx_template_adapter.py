import re
import asyncio
from typing import Dict, List
from crewai import Agent, Task
from custom_llm import get_azure_llm


class JSXTemplateAdapter:
    """실제 JSX 템플릿을 콘텐츠에 맞게 적용하는 에이전트 (이미지 URL 완전 통합)"""

    def __init__(self):
        self.llm = get_azure_llm()

    async def adapt_template_to_content(self, template_info: Dict, content: Dict, component_name: str) -> str:
        """템플릿을 콘텐츠에 맞게 적용 (이미지 URL 완전 통합)"""

        original_jsx = template_info.get('original_jsx', '')

        if not original_jsx:
            print(f"⚠️ 원본 JSX 없음 - 폴백 생성")
            return await self._create_fallback_adaptation(template_info, content, component_name)

        print(f"  🔧 실제 템플릿 구조 적용 시작 (이미지 URL 통합)")

        # 실제 템플릿 구조를 완전히 보존하면서 콘텐츠만 교체
        adapted_jsx = self._preserve_structure_adapt_content(
            original_jsx, template_info, content, component_name)

        # 이미지 URL 강제 통합
        adapted_jsx = self._force_integrate_image_urls(adapted_jsx, content)

        # 벡터 데이터 기반 스타일 조정
        if template_info.get('vector_matched', False):
            adapted_jsx = self._apply_vector_style_enhancements(
                adapted_jsx, template_info)

        # 마크다운 블록 제거 및 최종 검증
        adapted_jsx = self._remove_markdown_blocks_and_validate(
            adapted_jsx, content, component_name)

        print(f"  ✅ 실제 구조 보존 및 이미지 통합 완료")
        return adapted_jsx

    def _force_integrate_image_urls(self, jsx_code: str, content: Dict) -> str:
        """이미지 URL 강제 통합"""

        images = content.get('images', [])
        if not images:
            print(f"    📷 이미지 없음 - 플레이스홀더 유지")
            return jsx_code

        print(f"    📷 {len(images)}개 이미지 URL 통합 시작")

        # 1. 기존 이미지 태그에 실제 URL 적용
        jsx_code = self._replace_existing_image_tags(jsx_code, images)

        # 2. 이미지 props 교체
        jsx_code = self._replace_image_props(jsx_code, images)

        # 3. 이미지가 없는 경우 새로 추가
        jsx_code = self._add_missing_images(jsx_code, images)

        print(f"    ✅ 이미지 URL 통합 완료")
        return jsx_code

    def _replace_existing_image_tags(self, jsx_code: str, images: List[str]) -> str:
        """기존 이미지 태그에 실제 URL 적용"""

        # img 태그의 src 속성 찾기 및 교체
        img_pattern = r'<img\s+([^>]*?)src="([^"]*)"([^>]*?)/?>'

        def replace_img_src(match):
            before_src = match.group(1)
            old_src = match.group(2)
            after_src = match.group(3)

            # 첫 번째 이미지로 교체
            if images and images:
                new_src = images
                return f'<img {before_src}src="{new_src}"{after_src} />'

            return match.group(0)

        jsx_code = re.sub(img_pattern, replace_img_src, jsx_code)

        # styled img 컴포넌트의 src 속성 교체
        styled_img_pattern = r'<(\w*[Ii]mage?\w*)\s+([^>]*?)src="([^"]*)"([^>]*?)/?>'

        def replace_styled_img_src(match):
            component_name = match.group(1)
            before_src = match.group(2)
            old_src = match.group(3)
            after_src = match.group(4)

            # 이미지 인덱스 추출 시도
            img_index = self._extract_image_index_from_component(
                component_name)

            if img_index < len(images) and images[img_index]:
                new_src = images[img_index]
                return f'<{component_name} {before_src}src="{new_src}"{after_src} />'
            elif images and images:
                new_src = images
                return f'<{component_name} {before_src}src="{new_src}"{after_src} />'

            return match.group(0)

        jsx_code = re.sub(styled_img_pattern, replace_styled_img_src, jsx_code)

        return jsx_code

    def _replace_image_props(self, jsx_code: str, images: List[str]) -> str:
        """이미지 props 교체"""

        # 다양한 이미지 prop 패턴 교체
        image_prop_patterns = [
            (r'\{imageUrl\}', 0),
            (r'\{imageUrl1\}', 0),
            (r'\{imageUrl2\}', 1),
            (r'\{imageUrl3\}', 2),
            (r'\{imageUrl4\}', 3),
            (r'\{imageUrl5\}', 4),
            (r'\{imageUrl6\}', 5),
            (r'\{image\}', 0),
            (r'\{heroImage\}', 0),
            (r'\{featuredImage\}', 0),
            (r'\{mainImage\}', 0)
        ]

        for pattern, index in image_prop_patterns:
            if index < len(images) and images[index]:
                jsx_code = re.sub(pattern, images[index], jsx_code)

        return jsx_code

    def _add_missing_images(self, jsx_code: str, images: List[str]) -> str:
        """이미지가 없는 경우 새로 추가"""

        # 이미지 태그가 전혀 없는 경우 추가
        if '<img' not in jsx_code and 'Image' not in jsx_code and images:

            # Container 내부에 이미지 갤러리 추가
            container_pattern = r'(<Container[^>]*>)(.*?)(</Container>)'

            def add_image_gallery(match):
                container_open = match.group(1)
                container_content = match.group(2)
                container_close = match.group(3)

                # 이미지 갤러리 생성
                image_gallery = self._create_image_gallery_jsx(images)

                # 기존 콘텐츠 뒤에 이미지 갤러리 추가
                new_content = container_content + '\n      ' + image_gallery

                return container_open + new_content + '\n    ' + container_close

            jsx_code = re.sub(container_pattern,
                              add_image_gallery, jsx_code, flags=re.DOTALL)

        return jsx_code

    def _create_image_gallery_jsx(self, images: List[str]) -> str:
        """이미지 갤러리 JSX 생성"""

        # 이미지 태그들 생성
        image_tags = []
        for i, img_url in enumerate(images[:6]):
            if img_url and img_url.strip():
                image_tags.append(
                    f'        <TravelImage src="{img_url}" alt="Travel {i+1}" />')

        if not image_tags:
            return ''

        # Styled Component가 없으면 추가
        styled_component = '''
const TravelImage = styled.img`
  width: 100%;
  height: 200px;
  object-fit: cover;
  border-radius: 8px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  margin-bottom: 20px;
`;

const ImageGallery = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin: 40px 0;
`;'''

        gallery_jsx = f'''
      <ImageGallery>
{chr(10).join(image_tags)}
      </ImageGallery>'''

        return gallery_jsx

    def _extract_image_index_from_component(self, component_name: str) -> int:
        """컴포넌트 이름에서 이미지 인덱스 추출"""

        # 숫자 패턴 찾기
        match = re.search(r'(\d+)', component_name)
        if match:
            return int(match.group(1)) - 1

        # 특정 이름 패턴 매핑
        name_mapping = {
            'heroimage': 0,
            'featuredimage': 0,
            'mainimage': 0,
            'secondimage': 1,
            'thirdimage': 2
        }

        component_lower = component_name.lower()
        for name, index in name_mapping.items():
            if name in component_lower:
                return index

        return 0

    def _remove_markdown_blocks_and_validate(self, jsx_code: str, content: Dict, component_name: str) -> str:
        """마크다운 블록 제거 및 최종 검증"""

        print(f"    🧹 마크다운 블록 제거 시작")

        # 1. 마크다운 코드 블록 완전 제거
        jsx_code = self._remove_all_markdown_blocks(jsx_code)

        # 2. 기본 구조 검증
        jsx_code = self._validate_basic_structure(jsx_code, component_name)

        # 3. 실제 콘텐츠 포함 확인
        jsx_code = self._ensure_content_inclusion(jsx_code, content)

        # 4. 문법 오류 수정
        jsx_code = self._fix_syntax_errors(jsx_code)

        print(f"    ✅ 마크다운 제거 및 검증 완료")
        return jsx_code

    def _remove_all_markdown_blocks(self, jsx_code: str) -> str:
        """모든 마크다운 블록 제거"""

        # ```jsx, `````` 등 모든 마크다운 블록 제거
        jsx_code = re.sub(r'```[\s\S]*?```', '', jsx_code)
        jsx_code = re.sub(r'```\n?', '', jsx_code)

        # 연속된 백틱 제거
        jsx_code = re.sub(r'`{3,}', '', jsx_code)

        # 마크다운 주석 제거
        jsx_code = re.sub(r'<!--.*?-->', '', jsx_code, flags=re.DOTALL)

        # 불필요한 설명 텍스트 제거
        jsx_code = re.sub(r'^(이 코드는|다음은|아래는).*?\n', '',
                          jsx_code, flags=re.MULTILINE)
        jsx_code = re.sub(r'위의? 코드.*?\n', '', jsx_code, flags=re.MULTILINE)

        return jsx_code.strip()

    def _validate_basic_structure(self, jsx_code: str, component_name: str) -> str:
        """기본 구조 검증"""

        # import 문 확인
        if 'import React' not in jsx_code:
            jsx_code = 'import React from "react";\n' + jsx_code

        if 'import styled' not in jsx_code:
            jsx_code = jsx_code.replace(
                'import React from "react";',
                'import React from "react";\nimport styled from "styled-components";'
            )

        # export 문 확인
        if f'export const {component_name}' not in jsx_code:
            jsx_code = re.sub(
                r'export const \w+',
                f'export const {component_name}',
                jsx_code
            )

        return jsx_code

    def _ensure_content_inclusion(self, jsx_code: str, content: Dict) -> str:
        """실제 콘텐츠 포함 확인"""

        title = content.get('title', '')
        subtitle = content.get('subtitle', '')
        body = content.get('body', '')

        # 콘텐츠가 포함되지 않은 경우 강제 추가
        if title and title not in jsx_code:
            jsx_code = jsx_code.replace('{title}', title)
            jsx_code = jsx_code.replace(
                '<Title></Title>', f'<Title>{title}</Title>')
            jsx_code = jsx_code.replace('<Title/>', f'<Title>{title}</Title>')

        if subtitle and subtitle not in jsx_code:
            jsx_code = jsx_code.replace('{subtitle}', subtitle)
            jsx_code = jsx_code.replace(
                '<Subtitle></Subtitle>', f'<Subtitle>{subtitle}</Subtitle>')

        if body and body not in jsx_code:
            jsx_code = jsx_code.replace('{body}', body)
            jsx_code = jsx_code.replace(
                '<Content></Content>', f'<Content>{body}</Content>')

        return jsx_code

    def _fix_syntax_errors(self, jsx_code: str) -> str:
        """문법 오류 수정"""

        # 이중 중괄호 수정
        jsx_code = re.sub(r'\{\{([^}]+)\}\}', r'{\1}', jsx_code)

        # className 수정
        jsx_code = jsx_code.replace('class=', 'className=')

        # 빈 JSX 표현식 제거
        jsx_code = re.sub(r'\{\s*\}', '', jsx_code)

        # 연속된 빈 줄 정리
        jsx_code = re.sub(r'\n\s*\n\s*\n', '\n\n', jsx_code)

        # 중괄호 매칭 확인
        open_braces = jsx_code.count('{')
        close_braces = jsx_code.count('}')

        if open_braces != close_braces:
            if open_braces > close_braces:
                jsx_code += '}' * (open_braces - close_braces)
            else:
                jsx_code = jsx_code.rstrip('}') + '}' * open_braces

        # 마지막 }; 확인
        if not jsx_code.rstrip().endswith('};'):
            jsx_code = jsx_code.rstrip() + '\n};'

        return jsx_code

    # 기존 메서드들 유지...
    def _preserve_structure_adapt_content(self, original_jsx: str, template_info: Dict, content: Dict, component_name: str) -> str:
        """원본 구조를 완전히 보존하면서 콘텐츠만 교체 (기존 메서드 유지)"""

        adapted_jsx = original_jsx

        # 1. 컴포넌트 이름 변경 (구조 유지)
        adapted_jsx = re.sub(
            r'export const \w+',
            f'export const {component_name}',
            adapted_jsx
        )

        # 2. Props 구조 분석 및 실제 값으로 교체
        props = template_info.get('props', [])

        if props:
            # Props 함수 시그니처를 제거하고 직접 값 사용
            props_pattern = r'$$\s*\{\s*([^}]+)\s*\}\s*$$\s*=>'
            adapted_jsx = re.sub(props_pattern, '() =>', adapted_jsx)

            # 각 prop을 실제 값으로 교체
            for prop in props:
                prop = prop.strip()
                if prop == 'title':
                    adapted_jsx = adapted_jsx.replace(
                        f'{{{prop}}}', content.get('title', '도쿄 여행 이야기'))
                elif prop == 'subtitle':
                    adapted_jsx = adapted_jsx.replace(
                        f'{{{prop}}}', content.get('subtitle', '특별한 순간들'))
                elif prop == 'body':
                    adapted_jsx = adapted_jsx.replace(
                        f'{{{prop}}}', content.get('body', '여행의 아름다운 기억들'))
                elif prop == 'tagline':
                    adapted_jsx = adapted_jsx.replace(
                        f'{{{prop}}}', content.get('tagline', 'TRAVEL & CULTURE'))

        return adapted_jsx

    def _apply_vector_style_enhancements(self, jsx_code: str, template_info: Dict) -> str:
        """벡터 데이터 기반 스타일 향상 (기존 메서드 유지)"""

        similar_layouts = template_info.get('similar_pdf_layouts', [])
        if not similar_layouts:
            return jsx_code

        # PDF 매거진에서 추출한 색상 팔레트 적용
        color_enhancements = self._extract_colors_from_vector_data(
            similar_layouts)

        if color_enhancements:
            # 기존 색상을 벡터 기반 색상으로 교체
            for old_color, new_color in color_enhancements.items():
                jsx_code = jsx_code.replace(old_color, new_color)

        return jsx_code

    def _extract_colors_from_vector_data(self, similar_layouts: List[Dict]) -> Dict[str, str]:
        """벡터 데이터에서 색상 팔레트 추출 (기존 메서드 유지)"""

        # PDF 소스별 추천 색상 매핑
        color_mappings = {
            'travel': {
                '#2c3e50': '#1e3a8a',  # 더 깊은 블루
                '#7f8c8d': '#64748b',  # 슬레이트 그레이
                '#f8f9fa': '#f1f5f9'   # 라이트 블루 그레이
            },
            'culture': {
                '#2c3e50': '#7c2d12',  # 따뜻한 브라운
                '#7f8c8d': '#a3a3a3',  # 뉴트럴 그레이
                '#f8f9fa': '#fef7ed'   # 따뜻한 베이지
            },
            'lifestyle': {
                '#2c3e50': '#be185d',  # 핑크
                '#7f8c8d': '#9ca3af',  # 쿨 그레이
                '#f8f9fa': '#fdf2f8'   # 라이트 핑크
            }
        }

        # 가장 많이 매칭된 카테고리 찾기
        pdf_sources = [layout.get('pdf_name', '').lower()
                       for layout in similar_layouts]

        if any('travel' in source for source in pdf_sources):
            return color_mappings['travel']
        elif any('culture' in source for source in pdf_sources):
            return color_mappings['culture']
        elif any('lifestyle' in source for source in pdf_sources):
            return color_mappings['lifestyle']

        return {}

    async def _create_fallback_adaptation(self, template_info: Dict, content: Dict, component_name: str) -> str:
        """폴백 적용 (이미지 포함)"""

        title = content.get('title', '여행 이야기')
        subtitle = content.get('subtitle', '특별한 순간들')
        body = content.get('body', '여행의 기억들')
        images = content.get('images', [])
        tagline = content.get('tagline', 'TRAVEL & CULTURE')

        # 이미지 태그 생성
        image_tags = []
        for i, img_url in enumerate(images[:4]):
            if img_url and img_url.strip():
                image_tags.append(
                    f'        <TravelImage src="{img_url}" alt="Travel {i+1}" />')

        image_jsx = '\n'.join(image_tags) if image_tags else ''
        image_gallery = f'''
      <ImageGallery>
{image_jsx}
      </ImageGallery>''' if image_jsx else ''

        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 1000px;
  margin: 0 auto;
  padding: 40px 20px;
`;

const Title = styled.h1`
  font-size: 2em;
  margin-bottom: 20px;
`;

const Subtitle = styled.h2`
  font-size: 1.2em;
  margin-bottom: 30px;
`;

const Content = styled.div`
  line-height: 1.6;
  margin-bottom: 30px;
`;

const ImageGallery = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin: 40px 0;
`;

const TravelImage = styled.img`
  width: 100%;
  height: 200px;
  object-fit: cover;
  border-radius: 8px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
`;

const Tagline = styled.div`
  text-align: center;
  font-size: 0.9em;
  color: #666;
  margin-top: 30px;
`;

export const {component_name} = () => {{
  return (
    <Container>
      <Title>{title}</Title>
      <Subtitle>{subtitle}</Subtitle>
      <Content>{body}</Content>{image_gallery}
      <Tagline>{tagline}</Tagline>
    </Container>
  );
}};'''
