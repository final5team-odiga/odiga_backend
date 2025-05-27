import os
import re
import json
import asyncio
from agents.jsxcreate.jsx_content_analyzer import JSXContentAnalyzer
from agents.jsxcreate.jsx_layout_designer import JSXLayoutDesigner
from agents.jsxcreate.jsx_code_generator import JSXCodeGenerator
from agents.jsxcreate.jsx_template_analyzer import JSXTemplateAnalyzer
from agents.jsxcreate.jsx_template_adapter import JSXTemplateAdapter
from typing import Dict, List
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager
from utils.agent_decision_logger import get_agent_logger

class JSXCreatorAgent:
    """다중 에이전트 조율자 - JSX 생성 총괄 (의사결정 로깅 포함)"""

    def __init__(self):
        self.llm = get_azure_llm()
        self.vector_manager = PDFVectorManager()
        self.logger = get_agent_logger()

        # 전문 에이전트들 초기화
        self.content_analyzer = JSXContentAnalyzer()
        self.layout_designer = JSXLayoutDesigner()
        self.code_generator = JSXCodeGenerator()

        # 템플릿 관련 에이전트 추가
        self.template_analyzer = JSXTemplateAnalyzer()
        self.template_adapter = JSXTemplateAdapter()

    async def generate_jsx_components_async(self, template_data_path: str, templates_dir: str = "jsx_templates") -> List[Dict]:
        """jsx_templates 우선 사용하는 비동기 생성 (학습 기반)"""
        
        # 이전 의사결정 로그에서 학습 인사이트 획득
        learning_insights = self.logger.get_learning_insights("JSXCreatorAgent")
        print(f"📚 JSXCreatorAgent 학습 인사이트: {len(learning_insights.get('recommendations', []))}개 추천사항")

        # jsx_templates가 있으면 템플릿 기반으로, 없으면 기존 방식으로
        if os.path.exists(templates_dir) and any(f.endswith('.jsx') for f in os.listdir(templates_dir)):
            print(f"📁 jsx_templates 폴더 발견 - 템플릿 기반 생성 모드")
            generated_components = self.generate_jsx_components_with_templates_and_learning(
                template_data_path, templates_dir, learning_insights
            )
        else:
            print(f"📁 jsx_templates 없음 - 다중 에이전트 생성 모드")
            generated_components = self.generate_jsx_components_with_multi_agents_and_learning(
                template_data_path, templates_dir, learning_insights
            )

        if not generated_components:
            return []

        # 전체 JSX 생성 과정 로깅
        total_components = len(generated_components)
        successful_components = len([c for c in generated_components if c.get('jsx_code')])
        
        self.logger.log_agent_decision(
            agent_name="JSXCreatorAgent",
            agent_role="JSX 생성 총괄 조율자",
            input_data={
                "template_data_path": template_data_path,
                "templates_dir": templates_dir,
                "learning_insights_applied": len(learning_insights.get('recommendations', [])) > 0
            },
            decision_process={
                "generation_mode": "template_based" if os.path.exists(templates_dir) else "multi_agent",
                "learning_insights_count": len(learning_insights.get('recommendations', []))
            },
            output_result={
                "total_components": total_components,
                "successful_components": successful_components,
                "success_rate": successful_components / max(total_components, 1)
            },
            reasoning=f"JSX 생성 완료: {successful_components}/{total_components} 성공, 학습 인사이트 적용",
            confidence_score=0.9,
            context={"learning_insights": learning_insights},
            performance_metrics={
                "generation_efficiency": successful_components / max(total_components, 1),
                "learning_application_rate": 1.0 if learning_insights.get('recommendations') else 0.0
            }
        )

        print(f"✅ JSX 생성 완료: {len(generated_components)}개 컴포넌트 (학습 기반)")
        return generated_components

    def generate_jsx_components_with_templates_and_learning(self, template_data_path: str, templates_dir: str, learning_insights: Dict) -> List[Dict]:
        """jsx_templates를 활용한 JSX 생성 (학습 인사이트 적용)"""

        # 1. jsx_templates 폴더 분석
        print(f"\n📁 jsx_templates 폴더 분석 시작 (학습 기반)")
        template_analysis = self.template_analyzer.analyze_jsx_templates(templates_dir)

        if not template_analysis:
            print(f"⚠️ jsx_templates 분석 실패 - 다중 에이전트 모드로 전환")
            return self.generate_jsx_components_with_multi_agents_and_learning(template_data_path, templates_dir, learning_insights)

        # 2. template_data.json 읽기
        try:
            with open(template_data_path, 'r', encoding='utf-8') as f:
                file_content = f.read()

            template_data = self._safe_parse_json(file_content)

            if template_data is None:
                print(f"❌ template_data.json 파싱 실패")
                return []

        except Exception as e:
            print(f"template_data.json 읽기 오류: {str(e)}")
            return []

        # 데이터 검증
        if not isinstance(template_data, dict) or "content_sections" not in template_data:
            print(f"❌ 잘못된 template_data 구조")
            return []

        # 3. 콘텐츠 섹션별 템플릿 매핑 및 적용 (학습 기반)
        generated_components = []
        content_sections = template_data.get("content_sections", [])

        for i, content_section in enumerate(content_sections):
            if not isinstance(content_section, dict):
                continue

            component_name = f"Template{i+1:02d}Adapted"

            print(f"\n=== {component_name} 템플릿 기반 생성 시작 (학습 적용) ===")

            # 콘텐츠 정제 (학습 인사이트 적용)
            clean_content = self._clean_content_section_with_learning(content_section, learning_insights)

            # 콘텐츠 분석
            content_analysis = self.content_analyzer.analyze_content_for_jsx(
                clean_content, i, len(content_sections)
            )

            # 최적 템플릿 선택 (학습 인사이트 고려)
            best_template_name = self._select_template_with_learning(
                clean_content, content_analysis, template_analysis, learning_insights
            )

            best_template_info = template_analysis[best_template_name]

            print(f"  📋 선택된 템플릿: {best_template_name} (학습 기반)")
            print(f"  🎨 레이아웃 타입: {best_template_info['layout_type']}")

            # 템플릿을 콘텐츠에 적용
            jsx_code = self.template_adapter.adapt_template_to_content(
                best_template_info, clean_content, component_name
            )

            # 기본 검증 및 수정
            jsx_code = self._validate_template_adapted_jsx(jsx_code, clean_content, component_name)

            # 개별 컴포넌트 생성 로깅
            self.logger.log_agent_interaction(
                source_agent="JSXTemplateAdapter",
                target_agent="JSXCreatorAgent",
                interaction_type="component_generation",
                data_transferred={
                    "component_name": component_name,
                    "template_used": best_template_name,
                    "jsx_code_length": len(jsx_code),
                    "learning_applied": True
                }
            )

            generated_components.append({
                'name': component_name,
                'file': f"{component_name}.jsx",
                'jsx_code': jsx_code,
                'template_name': best_template_name,
                'approach': 'template_based_learning',
                'source_template': best_template_info,
                'content_analysis': content_analysis,
                'learning_insights_applied': True
            })

            print(f"✅ 템플릿 기반 JSX 생성 완료: {component_name} (학습 적용)")

        return generated_components

    def generate_jsx_components_with_multi_agents_and_learning(self, template_data_path: str, templates_dir: str, learning_insights: Dict) -> List[Dict]:
        """다중 에이전트 협업 JSX 생성 (학습 인사이트 적용)"""

        # template_data.json 읽기
        try:
            with open(template_data_path, 'r', encoding='utf-8') as f:
                file_content = f.read()

            template_data = self._safe_parse_json(file_content)

            if template_data is None:
                print(f"❌ template_data.json 파싱 실패: {template_data_path}")
                return []

        except Exception as e:
            print(f"template_data.json 읽기 오류: {str(e)}")
            return []

        # 데이터 검증
        if not isinstance(template_data, dict) or "content_sections" not in template_data:
            print(f"❌ 잘못된 template_data 구조")
            return []

        generated_components = []
        content_sections = template_data.get("content_sections", [])

        for i, content_section in enumerate(content_sections):
            if not isinstance(content_section, dict):
                continue

            template_name = content_section.get("template", f"Section{i+1:02d}.jsx")
            component_name = f"{template_name.replace('.jsx', '')}MultiAgent{i+1}"

            print(f"\n=== {component_name} 다중 에이전트 협업 시작 (학습 적용) ===")

            # 콘텐츠 정제 (학습 인사이트 적용)
            clean_content = self._clean_content_section_with_learning(content_section, learning_insights)

            # 4단계 다중 에이전트 협업 프로세스 (학습 기반)
            jsx_code = self._multi_agent_jsx_pipeline_with_learning(
                clean_content,
                component_name,
                i,
                len(content_sections),
                learning_insights
            )

            generated_components.append({
                'name': component_name,
                'file': f"{component_name}.jsx",
                'jsx_code': jsx_code,
                'template_name': template_name,
                'approach': 'multi_agent_learning',
                'pipeline_completed': True,
                'content_analysis': {'multi_agent_collaboration': True, 'learning_applied': True},
                'learning_insights_applied': True
            })

            print(f"✅ 다중 에이전트 JSX 생성 완료: {component_name} (학습 적용)")

        return generated_components

    def _select_template_with_learning(self, content: Dict, analysis: Dict, template_analysis: Dict, learning_insights: Dict) -> str:
        """학습 인사이트를 적용한 템플릿 선택"""
        
        # 기본 템플릿 선택
        base_selection = self.template_analyzer.get_best_template_for_content(content, analysis)
        
        # 학습 인사이트 적용
        recommendations = learning_insights.get('recommendations', [])
        for recommendation in recommendations:
            if "템플릿" in recommendation and "다양성" in recommendation:
                # 다양성을 위해 다른 템플릿 고려
                available_templates = list(template_analysis.keys())
                if len(available_templates) > 1:
                    # 기본 선택과 다른 템플릿 중에서 선택
                    alternative_templates = [t for t in available_templates if t != base_selection]
                    if alternative_templates:
                        print(f"  🎯 학습 인사이트 적용: 다양성을 위해 {alternative_templates[0]} 선택")
                        return alternative_templates[0]
            elif "템플릿" in recommendation and "신뢰도" in recommendation:
                # 신뢰도가 높은 템플릿 우선 선택
                high_confidence_templates = [
                    name for name, info in template_analysis.items()
                    if info.get('layout_confidence', 0) > 0.8
                ]
                if high_confidence_templates and base_selection not in high_confidence_templates:
                    print(f"  🎯 학습 인사이트 적용: 고신뢰도 템플릿 {high_confidence_templates[0]} 선택")
                    return high_confidence_templates[0]
        
        return base_selection

    def _clean_content_section_with_learning(self, content_section: Dict, learning_insights: Dict) -> Dict:
        """학습 인사이트를 적용한 콘텐츠 정제"""
        
        # 기본 정제
        clean_content = self._clean_content_section(content_section)
        
        # 학습 인사이트 적용
        key_insights = learning_insights.get('key_insights', [])
        for insight in key_insights:
            if "제목" in insight and "구체적" in insight:
                # 제목을 더 구체적으로 만들기
                title = clean_content.get('title', '')
                if len(title) < 15 and "여행" in title:
                    clean_content['title'] = title + " - 특별한 경험"
            elif "본문" in insight and "풍부" in insight:
                # 본문을 더 풍부하게 만들기
                body = clean_content.get('body', '')
                if len(body) < 500:
                    clean_content['body'] = body + "\n\n이 경험은 특별한 의미를 가지고 있습니다."
        
        return clean_content

    def _multi_agent_jsx_pipeline_with_learning(self, content: Dict, component_name: str, index: int, 
                                              total_sections: int, learning_insights: Dict) -> str:
        """4단계 다중 에이전트 파이프라인 (학습 기반)"""

        try:
            # 1단계: 콘텐츠 분석 (JSXContentAnalyzer) - 학습 적용
            print(f"  📊 1단계: 콘텐츠 분석 중... (학습 적용)")
            analysis_result = self.content_analyzer.analyze_content_for_jsx(
                content, index, total_sections
            )
            
            # 학습 인사이트를 분석 결과에 통합
            analysis_result = self._enhance_analysis_with_learning(analysis_result, learning_insights)

            # 2단계: 레이아웃 설계 (JSXLayoutDesigner) - 학습 적용
            print(f"  🎨 2단계: 레이아웃 설계 중... (학습 적용)")
            design_result = self.layout_designer.design_layout_structure(
                content, analysis_result, component_name
            )
            
            # 학습 인사이트를 설계 결과에 통합
            design_result = self._enhance_design_with_learning(design_result, learning_insights)

            # 3단계: JSX 코드 생성 (JSXCodeGenerator) - 학습 적용
            print(f"  💻 3단계: JSX 코드 생성 중... (학습 적용)")
            jsx_code = self.code_generator.generate_jsx_code(
                content, design_result, component_name
            )

            # 4단계: 코드 검증 및 수정 (학습 기반 검증)
            print(f"  🔍 4단계: 코드 검증 중... (학습 적용)")
            validated_jsx = self._validate_generated_jsx_with_learning(jsx_code, content, component_name, learning_insights)

            # 각 단계별 상호작용 로깅
            self.logger.log_agent_interaction(
                source_agent="MultiAgentPipeline",
                target_agent="JSXCreatorAgent",
                interaction_type="pipeline_completion",
                data_transferred={
                    "component_name": component_name,
                    "pipeline_steps": 4,
                    "learning_applied": True,
                    "jsx_length": len(validated_jsx)
                }
            )

            return validated_jsx

        except Exception as e:
            print(f"⚠️ 다중 에이전트 파이프라인 실패: {e}")
            # 폴백: 안전한 기본 JSX 생성 (학습 적용)
            return self._create_safe_fallback_jsx_with_learning(content, component_name, index, learning_insights)

    def _enhance_analysis_with_learning(self, analysis_result: Dict, learning_insights: Dict) -> Dict:
        """학습 인사이트로 분석 결과 강화"""
        
        enhanced_result = analysis_result.copy()
        
        # 학습 기반 레이아웃 추천 조정
        recommendations = learning_insights.get('recommendations', [])
        for recommendation in recommendations:
            if "레이아웃" in recommendation and "단순" in recommendation:
                if enhanced_result.get('recommended_layout') == 'complex':
                    enhanced_result['recommended_layout'] = 'grid'
                    enhanced_result['learning_adjustment'] = 'simplified_based_on_learning'
            elif "레이아웃" in recommendation and "혁신" in recommendation:
                if enhanced_result.get('recommended_layout') == 'minimal':
                    enhanced_result['recommended_layout'] = 'magazine'
                    enhanced_result['learning_adjustment'] = 'enhanced_based_on_learning'
        
        return enhanced_result

    def _enhance_design_with_learning(self, design_result: Dict, learning_insights: Dict) -> Dict:
        """학습 인사이트로 설계 결과 강화"""
        
        enhanced_result = design_result.copy()
        
        # 학습 기반 색상 스키마 조정
        performance_analysis = learning_insights.get('performance_analysis', {})
        if performance_analysis.get('performance_metrics'):
            # 이전 성능이 좋았던 색상 스키마 적용
            enhanced_result['color_scheme'] = {
                "primary": "#1e3a8a",  # 성능이 좋았던 블루 계열
                "secondary": "#f1f5f9"
            }
            enhanced_result['learning_enhancement'] = 'color_optimized'
        
        return enhanced_result

    def _validate_generated_jsx_with_learning(self, jsx_code: str, content: Dict, component_name: str, learning_insights: Dict) -> str:
        """학습 기반 JSX 검증"""
        
        # 기본 검증
        validated_jsx = self._validate_generated_jsx(jsx_code, content, component_name)
        
        # 학습 인사이트 기반 추가 검증
        key_insights = learning_insights.get('key_insights', [])
        for insight in key_insights:
            if "이미지" in insight and "포함" in insight:
                # 이미지 포함 확인 강화
                images = content.get('images', [])
                if images and '<img' not in validated_jsx:
                    # 강제로 이미지 추가
                    validated_jsx = self._force_add_images_to_jsx(validated_jsx, images)
            elif "콘텐츠" in insight and "완전" in insight:
                # 콘텐츠 완전성 확인 강화
                title = content.get('title', '')
                if title and title not in validated_jsx:
                    validated_jsx = self._force_add_content_to_jsx(validated_jsx, content)
        
        return validated_jsx

    def _create_safe_fallback_jsx_with_learning(self, content: Dict, component_name: str, index: int, learning_insights: Dict) -> str:
        """학습 기반 안전한 폴백 JSX 생성"""
        
        # 기본 폴백 생성
        base_jsx = self._create_safe_fallback_jsx(content, component_name, index)
        
        # 학습 인사이트 적용
        recommendations = learning_insights.get('recommendations', [])
        for recommendation in recommendations:
            if "색상" in recommendation and "따뜻한" in recommendation:
                # 따뜻한 색상으로 변경
                base_jsx = base_jsx.replace('#2c3e50', '#7c2d12')  # 따뜻한 브라운
                base_jsx = base_jsx.replace('#f5f7fa', '#fef7ed')  # 따뜻한 베이지
            elif "여백" in recommendation and "넓은" in recommendation:
                # 더 넓은 여백 적용
                base_jsx = base_jsx.replace('padding: 60px 20px', 'padding: 80px 40px')
        
        return base_jsx

    def _force_add_images_to_jsx(self, jsx_code: str, images: List[str]) -> str:
        """JSX에 이미지 강제 추가"""
        
        if not images:
            return jsx_code
        
        # Container 내부에 이미지 추가
        image_jsx = f'\n      <img src="{images[0]}" alt="Travel" style={{{{width: "100%", maxWidth: "600px", height: "300px", objectFit: "cover", borderRadius: "8px", margin: "20px 0"}}}} />'
        
        # return 문 내부에 추가
        jsx_code = jsx_code.replace(
            '<Container>',
            f'<Container>{image_jsx}'
        )
        
        return jsx_code

    def _force_add_content_to_jsx(self, jsx_code: str, content: Dict) -> str:
        """JSX에 콘텐츠 강제 추가"""
        
        title = content.get('title', '')
        body = content.get('body', '')
        
        if title and title not in jsx_code:
            jsx_code = jsx_code.replace('<Title>', f'<Title>{title}')
        
        if body and body not in jsx_code:
            jsx_code = jsx_code.replace('<Content>', f'<Content>{body}')
        
        return jsx_code

    # 기존 메서드들 유지 (변경 없음)
    def _validate_template_adapted_jsx(self, jsx_code: str, content: Dict, component_name: str) -> str:
        """템플릿 적용된 JSX 검증 및 수정"""

        # 1. 기본 구조 확인
        if 'import React' not in jsx_code:
            jsx_code = 'import React from "react";\n' + jsx_code

        if 'import styled' not in jsx_code:
            jsx_code = jsx_code.replace(
                'import React from "react";',
                'import React from "react";\nimport styled from "styled-components";'
            )

        # 2. 컴포넌트 이름 확인
        if f'export const {component_name}' not in jsx_code:
            jsx_code = re.sub(
                r'export const \w+',
                f'export const {component_name}',
                jsx_code
            )

        # 3. 실제 콘텐츠 포함 확인
        title = content.get('title', '')
        if title and title not in jsx_code:
            # Props를 실제 값으로 교체
            jsx_code = jsx_code.replace('{title}', title)
            jsx_code = jsx_code.replace('{subtitle}', content.get('subtitle', ''))
            jsx_code = jsx_code.replace('{body}', content.get('body', ''))
            jsx_code = jsx_code.replace('{tagline}', content.get('tagline', ''))

            # 이미지 URL 교체
            images = content.get('images', [])
            if images:
                for i, img_url in enumerate(images[:6]):
                    if img_url and img_url.strip():
                        jsx_code = jsx_code.replace(f'{{imageUrl{i+1}}}', img_url)
                        jsx_code = jsx_code.replace('{imageUrl}', img_url)

        # 4. 문법 오류 수정
        jsx_code = self._fix_basic_syntax_errors(jsx_code)

        return jsx_code

    def _validate_generated_jsx(self, jsx_code: str, content: Dict, component_name: str) -> str:
        """생성된 JSX 강화된 검증"""

        # 불완전한 구조 즉시 감지
        if jsx_code.count('return (') > jsx_code.count(');'):
            print(f"    ⚠️ 불완전한 return 문 감지 - 폴백 생성")
            return self._create_safe_fallback_jsx(content, component_name, 0)

        # 빈 JSX 내용 감지
        if 'return (' in jsx_code and jsx_code.split('return (')[1].strip() == '':
            print(f"    ⚠️ 빈 JSX 내용 감지 - 폴백 생성")
            return self._create_safe_fallback_jsx(content, component_name, 0)

        # 실제 콘텐츠 누락 감지
        title = content.get('title', '')
        if title and title not in jsx_code:
            print(f"    ⚠️ 실제 콘텐츠 누락 - 폴백 생성")
            return self._create_safe_fallback_jsx(content, component_name, 0)

        return jsx_code

    def _fix_basic_syntax_errors(self, jsx_code: str) -> str:
        """기본 문법 오류 수정"""

        # 1. 이중 중괄호 수정
        jsx_code = re.sub(r'\{\{([^}]+)\}\}', r'{\1}', jsx_code)

        # 2. className 수정
        jsx_code = jsx_code.replace('class=', 'className=')

        # 3. 빈 JSX 표현식 제거
        jsx_code = re.sub(r'\{\s*\}', '', jsx_code)

        # 4. 연속된 빈 줄 정리
        jsx_code = re.sub(r'\n\s*\n\s*\n', '\n\n', jsx_code)

        # 5. 마지막 }; 확인
        if not jsx_code.rstrip().endswith('};'):
            jsx_code = jsx_code.rstrip() + '\n};'

        return jsx_code

    def _create_safe_fallback_jsx(self, content: Dict, component_name: str, index: int) -> str:
        """확실히 작동하는 JSX 생성 (이미지 URL 포함)"""

        title = content.get('title', '도쿄 여행 이야기')
        subtitle = content.get('subtitle', '특별한 순간들')
        body = content.get('body', '여행의 아름다운 기억들이 마음 속에 깊이 새겨집니다.')
        images = content.get('images', [])
        tagline = content.get('tagline', 'TRAVEL & CULTURE')

        print(f"    📷 폴백 JSX에 {len(images)}개 이미지 포함")

        # 이미지 태그 생성
        image_tags = []
        for i, img_url in enumerate(images[:6]):
            if img_url and img_url.strip():
                image_tags.append(f'        <TravelImage src="{img_url}" alt="Travel {i+1}" />')

        image_jsx = '\n'.join(image_tags) if image_tags else '        <PlaceholderDiv>이미지 없음</PlaceholderDiv>'

        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 60px 20px;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  min-height: 100vh;
`;

const Header = styled.header`
  text-align: center;
  margin-bottom: 50px;
`;

const Title = styled.h1`
  font-size: 3em;
  color: #2c3e50;
  margin-bottom: 20px;
  font-weight: 300;
  letter-spacing: -1px;
`;

const Subtitle = styled.h2`
  font-size: 1.4em;
  color: #7f8c8d;
  margin-bottom: 30px;
  font-weight: 400;
`;

const Content = styled.div`
  font-size: 1.2em;
  line-height: 1.8;
  color: #34495e;
  text-align: justify;
  margin-bottom: 40px;
  max-width: 800px;
  margin-left: auto;
  margin-right: auto;
  white-space: pre-line;
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
  border-radius: 12px;
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
  transition: transform 0.3s ease;
  
  &:hover {{
    transform: translateY(-5px);
  }}
`;

const PlaceholderDiv = styled.div`
  width: 100%;
  height: 200px;
  background: #e9ecef;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6c757d;
  font-size: 1.1em;
`;

const Footer = styled.footer`
  text-align: center;
  margin-top: 50px;
  padding-top: 30px;
  border-top: 1px solid #ecf0f1;
`;

const Tagline = styled.div`
  font-size: 0.9em;
  color: #95a5a6;
  letter-spacing: 3px;
  text-transform: uppercase;
  font-weight: 600;
`;

export const {component_name} = () => {{
  return (
    <Container>
      <Header>
        <Title>{title}</Title>
        <Subtitle>{subtitle}</Subtitle>
      </Header>
      
      <Content>{body}</Content>
      
      <ImageGallery>
{image_jsx}
      </ImageGallery>
      
      <Footer>
        <Tagline>{tagline}</Tagline>
      </Footer>
    </Container>
  );
}};'''

    def _clean_content_section(self, content_section: Dict) -> Dict:
        """콘텐츠 섹션 정제"""

        title = content_section.get('title', '도쿄 여행 이야기')
        subtitle = content_section.get('subtitle', '특별한 순간들')
        body = content_section.get('body', '여행의 아름다운 기억들')
        images = content_section.get('images', [])
        tagline = content_section.get('tagline', 'TRAVEL & CULTURE')

        # 제목과 부제목에서 설명 텍스트 제거
        clean_title = self._clean_title_text(title)
        clean_subtitle = self._clean_subtitle_text(subtitle)
        clean_body = self._clean_body_text(body)

        return {
            'title': clean_title,
            'subtitle': clean_subtitle,
            'body': clean_body,
            'images': images,
            'tagline': tagline
        }

    def _clean_title_text(self, title: str) -> str:
        """제목에서 설명 텍스트 제거"""
        patterns_to_remove = [
            r'\(헤드라인\)', r'\(섹션 타이틀\)', r'및 부제목.*?배치되어 있음',
            r'필자 정보.*?배치되어 있음', r'포토 크레딧.*?배치되어 있음',
            r'계층적으로.*?배치되어 있음', r'과 본문의 배치 관계:',
            r'과 본문 배치:', r'배치:.*?배치되며', r'은 상단에.*?배치되며',
            r'혹은 좌상단에.*?줍니다', r'상단 혹은.*?강조합니다',
            r'없이 단일.*?집중시킵니다', r'과 소제목.*?있습니다',
            r'그 아래로.*?줄여줍니다', r'본문.*?구분할 수 있는.*?있습니다',
            r'콘텐츠의 각 요소.*?있습니다'
        ]

        clean_title = title
        for pattern in patterns_to_remove:
            clean_title = re.sub(pattern, '', clean_title, flags=re.IGNORECASE | re.DOTALL)

        clean_title = re.sub(r'\s+', ' ', clean_title)
        clean_title = re.sub(r'^[,\s]+|[,\s]+$', '', clean_title)

        return clean_title.strip() if clean_title.strip() else "도쿄 여행 이야기"

    def _clean_subtitle_text(self, subtitle: str) -> str:
        """부제목에서 설명 텍스트 제거"""
        patterns_to_remove = [
            r'필자 정보.*?배치되어 있음', r'포토 크레딧.*?배치되어 있음',
            r'계층적으로.*?배치되어 있음'
        ]

        clean_subtitle = subtitle
        for pattern in patterns_to_remove:
            clean_subtitle = re.sub(pattern, '', clean_subtitle, flags=re.IGNORECASE | re.DOTALL)

        clean_subtitle = re.sub(r'\s+', ' ', clean_subtitle)
        clean_subtitle = re.sub(r'^[,\s]+|[,\s]+$', '', clean_subtitle)

        return clean_subtitle.strip() if clean_subtitle.strip() else "특별한 순간들"

    def _clean_body_text(self, body: str) -> str:
        """본문에서 불필요한 설명 제거"""
        patterns_to_remove = [
            r'\*이 페이지에는.*?살렸습니다\.\*', r'블록은 균형.*?줄여줍니다',
            r'\(사진 캡션\)', r'시각적 리듬과.*?살렸습니다'
        ]

        clean_body = body
        for pattern in patterns_to_remove:
            clean_body = re.sub(pattern, '', clean_body, flags=re.IGNORECASE | re.DOTALL)

        return clean_body.strip()

    def _safe_parse_json(self, json_content: str) -> Dict:
        """JSON 콘텐츠를 안전하게 파싱"""
        try:
            parsed_data = json.loads(json_content)
            print("✅ JSON 직접 파싱 성공")
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON 직접 파싱 실패: {e}")
            try:
                cleaned_str = json_content.replace("'", '"').replace('True', 'true').replace('False', 'false').replace('None', 'null')
                parsed_data = json.loads(cleaned_str)
                print("✅ Python dict 문자열 변환 후 파싱 성공")
                return parsed_data
            except json.JSONDecodeError:
                try:
                    import ast
                    parsed_data = ast.literal_eval(json_content)
                    print("✅ ast.literal_eval 파싱 성공")
                    return parsed_data
                except (ValueError, SyntaxError):
                    print("❌ 모든 JSON 파싱 시도 실패")
                    return None

    def save_jsx_components(self, generated_components: List[Dict], components_folder: str) -> List[Dict]:
        """JSX 컴포넌트 저장 - 강화된 버전 (로깅 포함)"""

        # 폴더 존재 확인 및 생성
        if not os.path.exists(components_folder):
            os.makedirs(components_folder, exist_ok=True)
            print(f"✅ 컴포넌트 폴더 생성: {components_folder}")

        saved_components = []

        print(f"📁 저장 시작: {len(generated_components)}개 컴포넌트")

        for i, component in enumerate(generated_components):
            file_path = os.path.join(components_folder, component['file'])

            try:
                # JSX 코드 검증
                jsx_code = component.get('jsx_code', '')
                if not jsx_code or jsx_code.strip() == '':
                    print(f"⚠️ {component['file']}: JSX 코드가 비어있음")
                    continue

                # 파일 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(jsx_code)

                # 저장 확인
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    approach = component.get('approach', 'standard')
                    learning_applied = component.get('learning_insights_applied', False)
                    
                    print(f"✅ {component['file']} 저장 완료 (크기: {file_size} bytes, 방식: {approach}, 학습적용: {learning_applied})")
                    saved_components.append(component)
                    
                    # 개별 파일 저장 로깅
                    self.logger.log_agent_interaction(
                        source_agent="JSXCreatorAgent",
                        target_agent="FileSystem",
                        interaction_type="file_save",
                        data_transferred={
                            "file_name": component['file'],
                            "file_size": file_size,
                            "approach": approach,
                            "learning_applied": learning_applied
                        }
                    )
                else:
                    print(f"❌ {component['file']} 저장 실패: 파일이 생성되지 않음")

            except Exception as e:
                print(f"❌ {component['file']} 저장 실패: {e}")
                import traceback
                print(traceback.format_exc())

        # 전체 저장 과정 로깅
        self.logger.log_agent_decision(
            agent_name="JSXCreatorAgent",
            agent_role="JSX 파일 저장 관리자",
            input_data={
                "total_components": len(generated_components),
                "target_folder": components_folder
            },
            decision_process={
                "save_operation": "file_system_write",
                "validation_applied": True
            },
            output_result={
                "saved_components": len(saved_components),
                "failed_components": len(generated_components) - len(saved_components),
                "success_rate": len(saved_components) / max(len(generated_components), 1)
            },
            reasoning=f"JSX 파일 저장 완료: {len(saved_components)}/{len(generated_components)} 성공",
            confidence_score=0.95,
            performance_metrics={
                "save_efficiency": len(saved_components) / max(len(generated_components), 1),
                "file_system_success": True
            }
        )

        print(f"📁 저장 완료: {len(saved_components)}/{len(generated_components)}개 성공 (학습 기반)")
        return saved_components
