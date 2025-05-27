import os
import re
import json
from typing import Dict, List, Tuple
from crewai import Agent, Task
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager

class JSXTemplateAnalyzer:
    """jsx_templates 폴더의 실제 템플릿을 분석하는 에이전트 (벡터 데이터 통합)"""

    def __init__(self):
        self.llm = get_azure_llm()
        self.templates_cache = {}
        self.vector_manager = PDFVectorManager()

    def analyze_jsx_templates(self, templates_dir: str = "jsx_templates") -> Dict[str, Dict]:
        """jsx_templates 폴더의 모든 템플릿 분석 (벡터 데이터 활용)"""

        if not os.path.exists(templates_dir):
            print(f"❌ 템플릿 폴더 없음: {templates_dir}")
            return {}

        jsx_files = [f for f in os.listdir(templates_dir) if f.endswith('.jsx')]

        if not jsx_files:
            print(f"❌ JSX 템플릿 파일 없음: {templates_dir}")
            return {}

        print(f"📁 {len(jsx_files)}개 JSX 템플릿 분석 시작 (벡터 데이터 통합)")

        analyzed_templates = {}

        for jsx_file in jsx_files:
            file_path = os.path.join(templates_dir, jsx_file)
            template_analysis = self._analyze_single_template(file_path, jsx_file)
            
            # 벡터 데이터와 연결
            template_analysis = self._enhance_with_vector_data(template_analysis, jsx_file)
            
            analyzed_templates[jsx_file] = template_analysis
            print(f"✅ {jsx_file} 분석 완료: {template_analysis['layout_type']} (벡터 매칭: {template_analysis['vector_matched']})")

        self.templates_cache = analyzed_templates
        return analyzed_templates

    def _enhance_with_vector_data(self, template_analysis: Dict, jsx_file: str) -> Dict:
        """벡터 데이터로 템플릿 분석 강화"""
        
        try:
            # 템플릿의 레이아웃 특성을 쿼리로 변환
            layout_query = self._create_layout_query_from_template(template_analysis)
            
            # 벡터 검색으로 유사한 매거진 레이아웃 찾기
            similar_layouts = self.vector_manager.search_similar_layouts(
                layout_query, 
                "magazine_layout", 
                top_k=3
            )
            
            # 벡터 데이터로 템플릿 특성 보강
            if similar_layouts:
                template_analysis['vector_matched'] = True
                template_analysis['similar_pdf_layouts'] = similar_layouts
                template_analysis['layout_confidence'] = self._calculate_layout_confidence(template_analysis, similar_layouts)
                template_analysis['recommended_usage'] = self._determine_usage_from_vectors(similar_layouts)
            else:
                template_analysis['vector_matched'] = False
                template_analysis['similar_pdf_layouts'] = []
                template_analysis['layout_confidence'] = 0.5
                template_analysis['recommended_usage'] = 'general'
                
        except Exception as e:
            print(f"⚠️ 벡터 데이터 통합 실패 ({jsx_file}): {e}")
            template_analysis['vector_matched'] = False
            template_analysis['similar_pdf_layouts'] = []
            template_analysis['layout_confidence'] = 0.3
            
        return template_analysis

    def _create_layout_query_from_template(self, template_analysis: Dict) -> str:
        """템플릿 분석 결과를 벡터 검색 쿼리로 변환"""
        
        layout_type = template_analysis['layout_type']
        image_count = template_analysis['image_strategy']
        complexity = template_analysis['complexity_level']
        features = template_analysis['layout_features']
        
        # 템플릿 특성을 자연어 쿼리로 변환
        query_parts = [
            f"{layout_type} magazine layout",
            f"{image_count} images" if image_count > 0 else "text focused",
            f"{complexity} complexity design",
            "grid system" if template_analysis['grid_structure'] else "flexible layout"
        ]
        
        # 특징 추가
        if 'fixed_height' in features:
            query_parts.append("fixed height sections")
        if 'vertical_layout' in features:
            query_parts.append("vertical column layout")
        if 'gap_spacing' in features:
            query_parts.append("spaced elements design")
            
        return " ".join(query_parts)

    def _calculate_layout_confidence(self, template_analysis: Dict, similar_layouts: List[Dict]) -> float:
        """벡터 매칭 기반 레이아웃 신뢰도 계산"""
        
        if not similar_layouts:
            return 0.3
            
        # 유사도 점수 평균
        avg_similarity = sum(layout.get('score', 0) for layout in similar_layouts) / len(similar_layouts)
        
        # 템플릿 복잡도와 매칭 정도
        complexity_bonus = 0.2 if template_analysis['complexity_level'] == 'moderate' else 0.1
        
        # 이미지 전략 매칭 보너스
        image_bonus = 0.1 if template_analysis['image_strategy'] > 0 else 0.05
        
        confidence = min(avg_similarity + complexity_bonus + image_bonus, 1.0)
        return round(confidence, 2)

    def _determine_usage_from_vectors(self, similar_layouts: List[Dict]) -> str:
        """벡터 데이터 기반 사용 용도 결정"""
        
        if not similar_layouts:
            return 'general'
            
        # PDF 소스 분석
        pdf_sources = [layout.get('pdf_name', '') for layout in similar_layouts]
        
        # 매거진 타입 추론
        if any('travel' in source.lower() for source in pdf_sources):
            return 'travel_focused'
        elif any('culture' in source.lower() for source in pdf_sources):
            return 'culture_focused'
        elif any('lifestyle' in source.lower() for source in pdf_sources):
            return 'lifestyle_focused'
        else:
            return 'editorial'

    def get_best_template_for_content(self, content: Dict, analysis: Dict) -> str:
        """콘텐츠에 가장 적합한 템플릿 선택 (벡터 데이터 활용)"""

        if not self.templates_cache:
            return "Section01.jsx"

        image_count = len(content.get('images', []))
        text_length = len(content.get('body', ''))
        content_emotion = analysis.get('emotion_tone', 'neutral')

        # 콘텐츠 기반 벡터 검색
        content_query = f"{content.get('title', '')} {content.get('body', '')[:200]}"
        content_vectors = self.vector_manager.search_similar_layouts(
            content_query, 
            "magazine_layout", 
            top_k=5
        )

        best_template = None
        best_score = 0

        for template_name, template_info in self.templates_cache.items():
            score = 0

            # 기본 매칭 점수
            template_images = template_info['image_strategy']
            if image_count == 0 and template_images == 0:
                score += 30
            elif image_count == 1 and template_images == 1:
                score += 30
            elif image_count > 1 and template_images > 1:
                score += 20

            # 텍스트 길이 매칭
            if text_length < 300 and template_info['layout_type'] in ['simple', 'hero']:
                score += 20
            elif text_length > 500 and template_info['layout_type'] in ['grid', 'gallery']:
                score += 20

            # 벡터 데이터 기반 보너스 점수
            if template_info.get('vector_matched', False):
                score += template_info.get('layout_confidence', 0) * 30
                
                # 콘텐츠 벡터와 템플릿 벡터 매칭
                template_vectors = template_info.get('similar_pdf_layouts', [])
                vector_match_bonus = self._calculate_vector_content_match(content_vectors, template_vectors)
                score += vector_match_bonus * 20

            # 감정 톤 매칭
            recommended_usage = template_info.get('recommended_usage', 'general')
            if content_emotion == 'peaceful' and 'culture' in recommended_usage:
                score += 15
            elif content_emotion == 'exciting' and 'travel' in recommended_usage:
                score += 15

            if score > best_score:
                best_score = score
                best_template = template_name

        selected_template = best_template or "Section01.jsx"
        
        # 선택 이유 로깅
        selected_info = self.templates_cache.get(selected_template, {})
        print(f"  🎯 템플릿 선택: {selected_template}")
        print(f"     - 점수: {best_score}")
        print(f"     - 벡터 매칭: {selected_info.get('vector_matched', False)}")
        print(f"     - 신뢰도: {selected_info.get('layout_confidence', 0)}")
        print(f"     - 용도: {selected_info.get('recommended_usage', 'general')}")

        return selected_template

    def _calculate_vector_content_match(self, content_vectors: List[Dict], template_vectors: List[Dict]) -> float:
        """콘텐츠 벡터와 템플릿 벡터 간 매칭 점수"""
        
        if not content_vectors or not template_vectors:
            return 0.0
            
        # PDF 소스 기반 매칭
        content_sources = set(v.get('pdf_name', '') for v in content_vectors)
        template_sources = set(v.get('pdf_name', '') for v in template_vectors)
        
        # 공통 소스 비율
        common_sources = content_sources.intersection(template_sources)
        if content_sources:
            match_ratio = len(common_sources) / len(content_sources)
            return min(match_ratio, 1.0)
            
        return 0.0

    def _analyze_single_template(self, file_path: str, file_name: str) -> Dict:
        """개별 JSX 템플릿 분석 (기존 메서드 유지)"""

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                jsx_content = f.read()

            # 기본 정보 추출
            component_name = self._extract_component_name(jsx_content)
            props = self._extract_props(jsx_content)
            styled_components = self._extract_styled_components(jsx_content)
            layout_structure = self._analyze_layout_structure(jsx_content)

            return {
                'file_name': file_name,
                'component_name': component_name,
                'props': props,
                'styled_components': styled_components,
                'layout_type': layout_structure['type'],
                'layout_features': layout_structure['features'],
                'grid_structure': layout_structure['grid'],
                'image_strategy': layout_structure['images'],
                'text_strategy': layout_structure['text'],
                'complexity_level': layout_structure['complexity'],
                'original_jsx': jsx_content
            }

        except Exception as e:
            print(f"⚠️ {file_name} 분석 실패: {e}")
            return self._create_default_template_analysis(file_name)

    # 기존 메서드들 유지 (변경 없음)
    def _extract_component_name(self, jsx_content: str) -> str:
        """컴포넌트 이름 추출"""
        match = re.search(r'export const (\w+)', jsx_content)
        return match.group(1) if match else "UnknownComponent"

    def _extract_props(self, jsx_content: str) -> List[str]:
        """Props 추출"""
        props_match = re.search(r'\(\s*\{\s*([^}]+)\s*\}\s*\)', jsx_content)
        if props_match:
            props_str = props_match.group(1)
            props = [prop.strip() for prop in props_str.split(',')]
            return [prop for prop in props if prop]
        return []

    def _extract_styled_components(self, jsx_content: str) -> List[Dict]:
        """Styled Components 추출"""
        styled_components = []
        pattern = r'const\s+(\w+)\s*=\s*styled\.(\w+)`([^`]*)`'
        matches = re.findall(pattern, jsx_content, re.DOTALL)

        for comp_name, element_type, css_content in matches:
            styled_components.append({
                'name': comp_name,
                'element': element_type,
                'css': css_content.strip(),
                'properties': self._extract_css_properties(css_content)
            })

        return styled_components

    def _extract_css_properties(self, css_content: str) -> Dict:
        """CSS 속성 분석"""
        properties = {
            'display': 'block',
            'position': 'static',
            'grid': False,
            'flex': False,
            'absolute': False
        }

        if 'display: flex' in css_content or 'display: inline-flex' in css_content:
            properties['display'] = 'flex'
            properties['flex'] = True

        if 'display: grid' in css_content:
            properties['display'] = 'grid'
            properties['grid'] = True

        if 'position: absolute' in css_content:
            properties['position'] = 'absolute'
            properties['absolute'] = True

        return properties

    def _analyze_layout_structure(self, jsx_content: str) -> Dict:
        """레이아웃 구조 분석"""
        image_count = jsx_content.count('styled.img')

        if 'position: absolute' in jsx_content:
            layout_type = 'overlay'
        elif 'display: grid' in jsx_content or 'display: inline-flex' in jsx_content:
            if image_count == 0:
                layout_type = 'text_only'
            elif image_count == 1:
                layout_type = 'hero'
            elif image_count <= 3:
                layout_type = 'grid'
            else:
                layout_type = 'gallery'
        else:
            layout_type = 'simple'

        features = []
        if 'height: 800px' in jsx_content:
            features.append('fixed_height')
        if 'max-width: 1000px' in jsx_content:
            features.append('max_width_constrained')
        if 'gap:' in jsx_content:
            features.append('gap_spacing')
        if 'flex-direction: column' in jsx_content:
            features.append('vertical_layout')

        styled_comp_count = jsx_content.count('const Styled')
        if styled_comp_count <= 3:
            complexity = 'simple'
        elif styled_comp_count <= 6:
            complexity = 'moderate'
        else:
            complexity = 'complex'

        return {
            'type': layout_type,
            'features': features,
            'grid': 'display: grid' in jsx_content,
            'images': image_count,
            'text': jsx_content.count('font-size:'),
            'complexity': complexity
        }

    def _create_default_template_analysis(self, file_name: str) -> Dict:
        """기본 템플릿 분석 결과"""
        return {
            'file_name': file_name,
            'component_name': 'DefaultComponent',
            'props': ['title', 'subtitle', 'body', 'imageUrl'],
            'styled_components': [],
            'layout_type': 'simple',
            'layout_features': [],
            'grid_structure': False,
            'image_strategy': 1,
            'text_strategy': 3,
            'complexity_level': 'simple',
            'original_jsx': '',
            'vector_matched': False,
            'similar_pdf_layouts': [],
            'layout_confidence': 0.3,
            'recommended_usage': 'general'
        }
