import asyncio
import json
import time
from typing import Dict, List
from custom_llm import get_azure_llm
from utils.hybridlogging import get_hybrid_logger
from utils.ai_search_isolation import AISearchIsolationManager
from utils.pdf_vector_manager import PDFVectorManager
from utils.session_isolation import SessionAwareMixin
from utils.agent_communication_isolation import InterAgentCommunicationMixin


class UnifiedJSXGenerator(SessionAwareMixin, InterAgentCommunicationMixin):
    """통합 JSX 생성기 - AI Search 벡터 데이터 기반 JSX 컴포넌트 생성"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        self.logger = get_hybrid_logger(self.__class__.__name__)
        self._setup_logging_system()
        # AI Search 격리 시스템 추가
        self.isolation_manager = AISearchIsolationManager()
        # PDF 벡터 매니저 추가 (격리 활성화)
        self.vector_manager = PDFVectorManager(isolation_enabled=True)
        self.__init_session_awareness__()
        self.__init_inter_agent_communication__()

    def _setup_logging_system(self):
        """로그 저장 시스템 설정"""
        self.log_enabled = True
        self.response_counter = 0

    async def _log_jsx_generation_response(self, jsx_result: Dict) -> str:
        """JSX 생성 결과 로그 저장"""
        if not self.log_enabled:
            return "logging_disabled"
        
        try:
            response_data = {
                "agent_name": "UnifiedJSXGenerator",
                "generation_type": "jsx_components",
                "total_components": len(jsx_result.get("jsx_components", [])),
                "multimodal_optimization": jsx_result.get("generation_metadata", {}).get("multimodal_optimization", False),
                "responsive_design": jsx_result.get("generation_metadata", {}).get("responsive_design", False),
                "ai_search_enhanced": jsx_result.get("generation_metadata", {}).get("ai_search_enhanced", False),
                "timestamp": time.time(),
                "session_id": self.current_session_id
            }
            
            response_id = f"JSXGeneration_{int(time.time() * 1000000)}"
            
            # 세션별 저장
            self.store_result(response_data)
            
            self.logger.info(f"📦 UnifiedJSXGenerator 응답 저장: {response_id}")
            return response_id
            
        except Exception as e:
            self.logger.error(f"로그 저장 실패: {e}")
            return "log_save_failed"
            
        
    async def generate_jsx_with_multimodal_context(self, template_data: Dict) -> Dict:
        """AI Search 벡터 데이터를 활용한 JSX 생성 (격리 시스템 적용)"""
        
        self.logger.info("=== 통합 JSX 생성 시작 (AI Search 통합) ===")
        
        # 입력 데이터 오염 검사
        if self.isolation_manager.is_contaminated(template_data, "jsx_input"):
            self.logger.warning("템플릿 데이터에서 AI Search 오염 감지, 정화 처리 중...")
            template_data = self.isolation_manager.filter_contaminated_data(template_data)
        
        # 1. AI Search 기반 템플릿 데이터 분석
        analyzed_data = await self._analyze_template_data_with_ai_search(template_data)
        
        # 2. 벡터 패턴 기반 JSX 컴포넌트 생성
        jsx_components = await self._generate_jsx_components_with_patterns(analyzed_data)
        
        # 3. AI Search 스타일 패턴 기반 최적화
        optimized_components = await self._optimize_jsx_styles_with_patterns(jsx_components)
        
        # 4. 벡터 데이터 기반 반응형 코드 적용
        responsive_components = await self._apply_responsive_jsx_with_ai_search(optimized_components)
        
        result = {
            "jsx_components": responsive_components,
            "generation_metadata": {
                "total_components": len(responsive_components),
                "multimodal_optimization": True,
                "responsive_design": True,
                "style_optimization": True,
                "ai_search_enhanced": True,
                "isolation_applied": True,
                "vector_patterns_used": True
            }
        }
        

        response_id = await self._log_jsx_generation_response(result)
        result["response_id"] = response_id
        

        self.store_result(result)
        
        return result
    
    async def _analyze_template_data_with_ai_search(self, template_data: Dict) -> Dict:
        """AI Search 벡터 데이터를 활용한 템플릿 데이터 분석"""
        
        content_sections = template_data.get("content_sections", [])
        optimized_layouts = template_data.get("optimized_layouts", [])
        
        analyzed_sections = []
        
        for i, section in enumerate(content_sections):
            # 섹션별 오염 검사
            if self.isolation_manager.is_contaminated(section, f"template_section_{i}"):
                self.logger.warning(f"템플릿 섹션 {i}에서 오염 감지, 정화 처리")
                section = self.isolation_manager.filter_contaminated_data(section)
            
            # 해당 섹션의 레이아웃 정보 찾기
            layout_info = next((layout for layout in optimized_layouts if layout.get("template") == section.get("template")), {})
            
            # AI Search에서 JSX 패턴 검색
            jsx_patterns = await self._search_jsx_patterns(section, layout_info)
            
            analyzed_section = {
                "section_index": i,
                "template_name": section.get("template", f"Section{i+1:02d}.jsx"),
                "content": section,
                "layout": layout_info,
                "jsx_requirements": await self._determine_jsx_requirements_with_patterns(section, layout_info, jsx_patterns),
                "jsx_patterns": jsx_patterns,
                "isolation_metadata": {
                    "section_cleaned": True,
                    "contamination_detected": False,
                    "patterns_found": len(jsx_patterns)
                }
            }
            
            analyzed_sections.append(analyzed_section)
        
        return {
            "analyzed_sections": analyzed_sections,
            "global_style_guide": await self._extract_global_style_guide_with_ai_search(optimized_layouts)
        }
    
    async def _search_jsx_patterns(self, section: Dict, layout_info: Dict) -> List[Dict]:
        """JSX 컴포넌트 생성을 위한 AI Search 패턴 검색"""
        
        try:
            # 섹션과 레이아웃 정보 기반 검색 쿼리 생성
            template_name = section.get("template", "")
            image_count = len(section.get("images", []))
            layout_type = layout_info.get("optimization_metadata", {}).get("optimization_level", "basic")
            
            search_query = f"react jsx component {template_name} {image_count} images {layout_type}"
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(search_query)
            
            # 벡터 검색 실행
            jsx_patterns = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "magazine_layout", top_k=8
                )
            )
            
            # 격리된 패턴만 반환
            isolated_patterns = self.isolation_manager.filter_contaminated_data(
                jsx_patterns, f"jsx_patterns_{template_name}"
            )
            
            self.logger.debug(f"JSX 패턴 검색 {template_name}: {len(jsx_patterns)} → {len(isolated_patterns)}개")
            return isolated_patterns
            
        except Exception as e:
            self.logger.error(f"JSX 패턴 검색 실패: {e}")
            return []
    
    async def _determine_jsx_requirements_with_patterns(self, section: Dict, layout_info: Dict, patterns: List[Dict]) -> Dict:
        """AI Search 패턴을 고려한 JSX 요구사항 결정"""
        
        image_count = len(section.get("images", []))
        text_length = len(section.get("body", ""))
        
        # 기본 요구사항
        requirements = {
            "component_type": "magazine_section",
            "layout_complexity": "simple",
            "image_handling": "static",
            "text_formatting": "basic",
            "responsive_priority": "medium"
        }
        
        # AI Search 패턴 기반 요구사항 개선
        if patterns:
            best_pattern = patterns[0]
            
            # 패턴에서 JSX 요구사항 추출
            pattern_requirements = {
                "component_type": best_pattern.get("component_type", "magazine_section"),
                "layout_complexity": best_pattern.get("complexity_level", "simple"),
                "image_handling": best_pattern.get("image_strategy", "static"),
                "text_formatting": best_pattern.get("text_format", "basic"),
                "responsive_priority": best_pattern.get("responsive_level", "medium"),
                "animation_level": best_pattern.get("animation", "none"),
                "interaction_level": best_pattern.get("interaction", "basic")
            }
            
            requirements.update(pattern_requirements)
            requirements["pattern_enhanced"] = True
        
        # 이미지 개수에 따른 요구사항 조정
        if image_count == 0:
            requirements["layout_complexity"] = "text_focused"
            requirements["text_formatting"] = "enhanced"
        elif image_count > 2:
            requirements["layout_complexity"] = "image_gallery"
            requirements["image_handling"] = "dynamic"
            requirements["responsive_priority"] = "high"
        
        # 텍스트 길이에 따른 요구사항 조정
        if text_length > 500:
            requirements["text_formatting"] = "multi_column"
            requirements["responsive_priority"] = "high"
        
        # 레이아웃 정보 반영
        layout_config = layout_info.get("layout_config", {})
        if layout_config.get("image_grid") == "갤러리":
            requirements["image_handling"] = "carousel"
        
        return requirements
    
    async def _extract_global_style_guide_with_ai_search(self, layouts: List[Dict]) -> Dict:
        """AI Search 패턴 기반 전역 스타일 가이드 추출"""
        
        try:
            # 전체 레이아웃 특성 분석
            total_layouts = len(layouts)
            total_images = sum(len(layout.get("images", [])) for layout in layouts)
            
            # AI Search에서 전역 스타일 패턴 검색
            search_query = f"global style guide {total_layouts} sections {total_images} images magazine"
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(search_query)
            
            style_patterns = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "magazine_layout", top_k=5
                )
            )
            
            # 격리된 패턴만 사용
            isolated_patterns = self.isolation_manager.filter_contaminated_data(
                style_patterns, "global_style_patterns"
            )
            
            # 기본 스타일 가이드
            style_guide = {
                "color_scheme": "warm_travel",
                "typography": {
                    "title_font": "Playfair Display",
                    "body_font": "Source Sans Pro",
                    "accent_font": "Montserrat"
                },
                "spacing": {
                    "section_gap": "4rem",
                    "element_gap": "2rem",
                    "text_line_height": "1.6"
                },
                "responsive": {
                    "mobile_breakpoint": "768px",
                    "tablet_breakpoint": "1024px"
                }
            }
            
            # AI Search 패턴 기반 스타일 개선
            if isolated_patterns:
                best_pattern = isolated_patterns[0]
                
                # 패턴에서 스타일 정보 업데이트
                pattern_style = {
                    "color_scheme": best_pattern.get("color_scheme", "warm_travel"),
                    "typography": {
                        "title_font": best_pattern.get("title_font", "Playfair Display"),
                        "body_font": best_pattern.get("body_font", "Source Sans Pro"),
                        "accent_font": best_pattern.get("accent_font", "Montserrat")
                    },
                    "spacing": {
                        "section_gap": best_pattern.get("section_gap", "4rem"),
                        "element_gap": best_pattern.get("element_gap", "2rem"),
                        "text_line_height": best_pattern.get("line_height", "1.6")
                    }
                }
                
                style_guide.update(pattern_style)
                style_guide["pattern_enhanced"] = True
                style_guide["pattern_source"] = best_pattern.get("pdf_name", "ai_search")
            
            return style_guide
            
        except Exception as e:
            self.logger.error(f"전역 스타일 가이드 추출 실패: {e}")
            return {
                "color_scheme": "warm_travel",
                "typography": {
                    "title_font": "Playfair Display",
                    "body_font": "Source Sans Pro",
                    "accent_font": "Montserrat"
                },
                "spacing": {
                    "section_gap": "4rem",
                    "element_gap": "2rem",
                    "text_line_height": "1.6"
                },
                "responsive": {
                    "mobile_breakpoint": "768px",
                    "tablet_breakpoint": "1024px"
                },
                "pattern_enhanced": False
            }
    
    async def _generate_jsx_components_with_patterns(self, analyzed_data: Dict) -> List[Dict]:
        """AI Search 패턴 기반 JSX 컴포넌트 생성"""
        
        jsx_components = []
        analyzed_sections = analyzed_data.get("analyzed_sections", [])
        global_style = analyzed_data.get("global_style_guide", {})
        
        self.logger.info(f"JSX 생성할 섹션 수: {len(analyzed_sections)}개")
        
        # ✅ 각 섹션별로 JSX 컴포넌트 생성
        for section_data in analyzed_sections:
            jsx_component = await self._create_jsx_component_with_patterns(section_data, global_style)
            jsx_components.append(jsx_component)
            self.logger.info(f"JSX 컴포넌트 생성: {jsx_component.get('template_name', 'Unknown')}")
        
        self.logger.info(f"총 {len(jsx_components)}개 JSX 컴포넌트 생성 완료")
        return jsx_components
    
    async def _create_jsx_component_with_patterns(self, section_data: Dict, global_style: Dict) -> Dict:
        """AI Search 패턴을 활용한 개별 JSX 컴포넌트 생성"""
        
        template_name = section_data.get("template_name", "Section01.jsx")
        content = section_data.get("content", {})
        layout = section_data.get("layout", {})
        requirements = section_data.get("jsx_requirements", {})
        jsx_patterns = section_data.get("jsx_patterns", [])
        
        # 콘텐츠 필터링
        filtered_title = self.isolation_manager.clean_query_from_azure_keywords(content.get("title", ""))
        filtered_subtitle = self.isolation_manager.clean_query_from_azure_keywords(content.get("subtitle", ""))
        filtered_body = self.isolation_manager.clean_query_from_azure_keywords(content.get("body", ""))
        
        # AI Search 패턴 컨텍스트 생성
        pattern_context = ""
        if jsx_patterns:
            pattern_info = []
            for pattern in jsx_patterns[:2]:  # 상위 2개 패턴 사용
                pattern_info.append({
                    "컴포넌트_구조": pattern.get("component_structure", "기본"),
                    "스타일_접근법": pattern.get("style_approach", "Tailwind"),
                    "반응형_전략": pattern.get("responsive_strategy", "모바일우선"),
                    "이미지_처리": pattern.get("image_handling", "lazy_loading"),
                    "애니메이션": pattern.get("animation_type", "none"),
                    "접근성_레벨": pattern.get("accessibility_level", "기본")
                })
            pattern_context = f"AI Search 참조 패턴: {json.dumps(pattern_info, ensure_ascii=False)}"
        
        jsx_prompt = f"""
다음 정보를 바탕으로 React JSX 컴포넌트를 생성하세요:

**컴포넌트명:** {template_name.replace('.jsx', '')}

**콘텐츠 정보:**
- 제목: {filtered_title}
- 부제목: {filtered_subtitle}
- 본문: {filtered_body[:300]}...
- 태그라인: {content.get("tagline", "")}
- 이미지: {content.get("images", [])}

**레이아웃 설정:**
{json.dumps(layout.get("layout_config", {}), ensure_ascii=False, indent=2)}

**JSX 요구사항:**
{json.dumps(requirements, ensure_ascii=False, indent=2)}

**글로벌 스타일:**
{json.dumps(global_style, ensure_ascii=False, indent=2)}

{pattern_context}

**생성 요구사항:**
1. AI Search 패턴을 참조한 현대적이고 반응형인 React 컴포넌트
2. Tailwind CSS 클래스 사용 (패턴 기반 최적화)
3. 이미지 최적화 및 lazy loading (패턴 참조)
4. 접근성 고려 (alt 태그, ARIA 레이블, 패턴 기반)
5. 성능 최적화 (memo, useMemo 활용)
6. 애니메이션 및 인터랙션 (패턴에 따라)

다음 형식으로 출력하세요:
import React, {{ memo, useMemo }} from 'react';
import Image from 'next/image';

const {template_name.replace('.jsx', '')} = memo(() => {{
// 컴포넌트 코드
}});

export default {template_name.replace('.jsx', '')};

text
"""
        
        try:
            response = await self.llm.ainvoke(jsx_prompt)
            jsx_code = response

            # JSX 코드 오염 검사
            if self.isolation_manager.is_contaminated(jsx_code, f"jsx_code_{template_name}"):
                self.logger.warning(f"JSX 코드 {template_name}에서 오염 감지, 폴백 사용")
                jsx_code = self._generate_fallback_jsx_with_patterns(template_name, content, jsx_patterns)

            # JSX 코드에서 실제 코드 부분 추출
            if "```jsx" in jsx_code:
                jsx_code = jsx_code.split("```jsx")[1].split("```")[0].strip()
            elif "```javascript" in jsx_code:
                jsx_code = jsx_code.split("```javascript")[1].split("```")[0].strip()
            elif "```javascript" in jsx_code:
                jsx_code = jsx_code.split("``````")[0].strip()
            
            return {
                "template_name": template_name,
                "jsx_code": jsx_code,
                "component_metadata": {
                    "complexity": requirements.get("layout_complexity", "simple"),
                    "image_count": len(content.get("images", [])),
                    "text_length": len(content.get("body", "")),
                    "responsive_optimized": True,
                    "accessibility_features": True,
                    "ai_search_patterns_used": len(jsx_patterns),
                    "pattern_enhanced": len(jsx_patterns) > 0,
                    "isolation_applied": True,
                    "contamination_detected": False
                }
            }
            
        except Exception as e:
            self.logger.error(f"JSX 컴포넌트 {template_name} 생성 실패: {e}")
            
            # 격리된 폴백 JSX 컴포넌트 반환
            fallback_jsx = self._generate_fallback_jsx_with_patterns(template_name, content, jsx_patterns)
            
            return {
                "template_name": template_name,
                "jsx_code": fallback_jsx,
                "component_metadata": {
                    "complexity": "simple",
                    "image_count": 0,
                    "text_length": len(content.get("body", "")),
                    "responsive_optimized": False,
                    "accessibility_features": True,
                    "ai_search_patterns_used": len(jsx_patterns),
                    "pattern_enhanced": False,
                    "isolation_applied": True,
                    "fallback_used": True
                }
            }
    
    def _generate_fallback_jsx_with_patterns(self, template_name: str, content: Dict, patterns: List[Dict]) -> str:
        """AI Search 패턴을 고려한 격리된 기본 JSX 컴포넌트 생성"""
        
        component_name = template_name.replace('.jsx', '')
        
        # 원본 데이터 사용 (AI Search 키워드 필터링 없이)
        title = content.get("title", "여행 이야기")
        subtitle = content.get("subtitle", "특별한 순간들")
        body = content.get("body", "멋진 여행 경험을 공유합니다.")
        
        # 패턴 기반 스타일 개선
        additional_classes = ""
        if patterns:
            best_pattern = patterns[0]
            if best_pattern.get("animation_type") == "fade":
                additional_classes = " transition-opacity duration-500"
            elif best_pattern.get("style_approach") == "modern":
                additional_classes = " backdrop-blur-sm bg-white/90"
        
        return f"""import React, {{ memo }} from 'react';

const {component_name} = memo(() => {{
  return (
    <section className="py-16 px-4 max-w-4xl mx-auto{additional_classes}">
      <div className="text-center mb-8">
        <h2 className="text-3xl md:text-4xl font-bold text-gray-800 mb-4">
          {title}
        </h2>
        <p className="text-lg text-gray-600 mb-6">
          {subtitle}
        </p>
      </div>
      
      <div className="prose prose-lg max-w-none">
        <p className="text-gray-700 leading-relaxed">
          {body[:200]}...
        </p>
      </div>
    </section>
  );
}});

export default {component_name};"""
    
    async def _optimize_jsx_styles_with_patterns(self, jsx_components: List[Dict]) -> List[Dict]:
        """AI Search 패턴 기반 JSX 스타일 최적화"""
        
        optimized_components = []
        
        for component in jsx_components:
            # AI Search에서 스타일 최적화 패턴 검색
            style_patterns = await self._search_jsx_style_patterns(component)
            
            optimized_component = await self._optimize_single_component_style_with_patterns(component, style_patterns)
            optimized_components.append(optimized_component)
        
        return optimized_components
    
    async def _search_jsx_style_patterns(self, component: Dict) -> List[Dict]:
        """JSX 스타일 최적화를 위한 AI Search 패턴 검색"""
        
        try:
            template_name = component.get("template_name", "")
            complexity = component.get("component_metadata", {}).get("complexity", "simple")
            image_count = component.get("component_metadata", {}).get("image_count", 0)
            
            search_query = f"jsx style optimization {template_name} {complexity} {image_count} images"
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(search_query)
            
            # 벡터 검색 실행
            style_patterns = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "magazine_layout", top_k=5
                )
            )
            
            # 격리된 패턴만 반환
            isolated_patterns = self.isolation_manager.filter_contaminated_data(
                style_patterns, f"jsx_style_patterns_{template_name}"
            )
            
            return isolated_patterns
            
        except Exception as e:
            self.logger.error(f"JSX 스타일 패턴 검색 실패: {e}")
            return []
    
    async def _optimize_single_component_style_with_patterns(self, component: Dict, patterns: List[Dict]) -> Dict:
        """AI Search 패턴을 활용한 개별 컴포넌트 스타일 최적화"""
        
        jsx_code = component.get("jsx_code", "")
        
        # 기본 스타일 최적화 규칙
        optimizations = [
            self._optimize_color_consistency_with_patterns,
            self._optimize_spacing_consistency_with_patterns,
            self._optimize_typography_consistency_with_patterns,
            self._optimize_responsive_classes_with_patterns
        ]
        
        optimized_code = jsx_code
        for optimization in optimizations:
            optimized_code = optimization(optimized_code, patterns)
        
        component["jsx_code"] = optimized_code
        component["component_metadata"]["style_optimized"] = True
        component["component_metadata"]["pattern_style_applied"] = len(patterns) > 0
        
        return component
    
    def _optimize_color_consistency_with_patterns(self, jsx_code: str, patterns: List[Dict]) -> str:
        """AI Search 패턴 기반 색상 일관성 최적화"""
        
        # 기본 색상 매핑
        color_mappings = {
            "text-gray-800": "text-slate-800",
            "text-gray-600": "text-slate-600",
            "text-gray-700": "text-slate-700",
            "bg-white": "bg-slate-50"
        }
        
        # AI Search 패턴 기반 색상 개선
        if patterns:
            best_pattern = patterns[0]
            pattern_colors = best_pattern.get("color_scheme", {})
            
            if pattern_colors.get("primary"):
                color_mappings["text-slate-800"] = f"text-{pattern_colors['primary']}-800"
            if pattern_colors.get("secondary"):
                color_mappings["text-slate-600"] = f"text-{pattern_colors['secondary']}-600"
        
        for old_color, new_color in color_mappings.items():
            jsx_code = jsx_code.replace(old_color, new_color)
        
        return jsx_code
    
    def _optimize_spacing_consistency_with_patterns(self, jsx_code: str, patterns: List[Dict]) -> str:
        """AI Search 패턴 기반 간격 일관성 최적화"""
        
        # 기본 간격 매핑
        spacing_mappings = {
            "py-8": "py-12",
            "mb-4": "mb-6",
            "px-4": "px-6"
        }
        
        # AI Search 패턴 기반 간격 개선
        if patterns:
            best_pattern = patterns[0]
            pattern_spacing = best_pattern.get("spacing_config", {})
            
            if pattern_spacing.get("section_padding"):
                spacing_mappings["py-12"] = f"py-{pattern_spacing['section_padding']}"
            if pattern_spacing.get("element_margin"):
                spacing_mappings["mb-6"] = f"mb-{pattern_spacing['element_margin']}"
        
        for old_spacing, new_spacing in spacing_mappings.items():
            jsx_code = jsx_code.replace(old_spacing, new_spacing)
        
        return jsx_code
    
    def _optimize_typography_consistency_with_patterns(self, jsx_code: str, patterns: List[Dict]) -> str:
        """AI Search 패턴 기반 타이포그래피 일관성 최적화"""
        
        # 기본 타이포그래피 매핑
        typography_mappings = {
            "text-3xl": "text-3xl font-serif",
            "text-lg": "text-lg font-sans",
            "prose": "prose prose-slate"
        }
        
        # AI Search 패턴 기반 타이포그래피 개선
        if patterns:
            best_pattern = patterns[0]
            pattern_typography = best_pattern.get("typography", {})
            
            if pattern_typography.get("title_style"):
                typography_mappings["text-3xl font-serif"] = f"text-3xl {pattern_typography['title_style']}"
            if pattern_typography.get("body_style"):
                typography_mappings["text-lg font-sans"] = f"text-lg {pattern_typography['body_style']}"
        
        for old_typo, new_typo in typography_mappings.items():
            jsx_code = jsx_code.replace(old_typo, new_typo)
        
        return jsx_code
    
    def _optimize_responsive_classes_with_patterns(self, jsx_code: str, patterns: List[Dict]) -> str:
        """AI Search 패턴 기반 반응형 클래스 최적화"""
        
        # 기본 반응형 매핑
        responsive_mappings = {
            "text-3xl md:text-4xl": "text-2xl sm:text-3xl lg:text-4xl",
            "max-w-4xl": "max-w-4xl lg:max-w-6xl",
            "py-16": "py-12 lg:py-16"
        }
        
        # AI Search 패턴 기반 반응형 개선
        if patterns:
            best_pattern = patterns[0]
            pattern_responsive = best_pattern.get("responsive_config", {})
            
            if pattern_responsive.get("breakpoints"):
                # 패턴의 브레이크포인트 정보 활용
                breakpoints = pattern_responsive["breakpoints"]
                if "mobile" in breakpoints and "tablet" in breakpoints:
                    responsive_mappings["text-2xl sm:text-3xl lg:text-4xl"] = f"text-xl {breakpoints['mobile']}:text-2xl {breakpoints['tablet']}:text-3xl lg:text-4xl"
        
        for old_responsive, new_responsive in responsive_mappings.items():
            jsx_code = jsx_code.replace(old_responsive, new_responsive)
        
        return jsx_code
    
    async def _apply_responsive_jsx_with_ai_search(self, jsx_components: List[Dict]) -> List[Dict]:
        """AI Search 패턴 기반 반응형 JSX 적용"""
        
        responsive_components = []
        
        for component in jsx_components:
            # AI Search에서 반응형 JSX 패턴 검색
            responsive_patterns = await self._search_responsive_jsx_patterns(component)
            
            responsive_component = await self._make_jsx_responsive_with_patterns(component, responsive_patterns)
            responsive_components.append(responsive_component)
        
        return responsive_components
    
    async def _search_responsive_jsx_patterns(self, component: Dict) -> List[Dict]:
        """반응형 JSX를 위한 AI Search 패턴 검색"""
        
        try:
            template_name = component.get("template_name", "")
            image_count = component.get("component_metadata", {}).get("image_count", 0)
            complexity = component.get("component_metadata", {}).get("complexity", "simple")
            
            search_query = f"responsive jsx {template_name} {image_count} images {complexity} mobile tablet desktop"
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(search_query)
            
            # 벡터 검색 실행
            responsive_patterns = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "magazine_layout", top_k=5
                )
            )
            
            # 격리된 패턴만 반환
            isolated_patterns = self.isolation_manager.filter_contaminated_data(
                responsive_patterns, f"responsive_jsx_patterns_{template_name}"
            )
            
            return isolated_patterns
            
        except Exception as e:
            self.logger.error(f"반응형 JSX 패턴 검색 실패: {e}")
            return []
    
    async def _make_jsx_responsive_with_patterns(self, component: Dict, patterns: List[Dict]) -> Dict:
        """AI Search 패턴을 활용한 JSX 컴포넌트 반응형 변환"""
        
        jsx_code = component.get("jsx_code", "")
        metadata = component.get("component_metadata", {})
        
        # 기본 반응형 처리
        if metadata.get("image_count", 0) > 2:
            jsx_code = self._add_image_carousel_responsive_with_patterns(jsx_code, patterns)
        
        if metadata.get("text_length", 0) > 500:
            jsx_code = self._add_multi_column_responsive_with_patterns(jsx_code, patterns)
        
        # AI Search 패턴 기반 추가 반응형 기능
        if patterns:
            jsx_code = self._apply_pattern_responsive_features(jsx_code, patterns)
        
        component["jsx_code"] = jsx_code
        component["component_metadata"]["responsive_enhanced"] = True
        component["component_metadata"]["pattern_responsive_applied"] = len(patterns) > 0
        
        return component
    
    def _add_image_carousel_responsive_with_patterns(self, jsx_code: str, patterns: List[Dict]) -> str:
        """AI Search 패턴 기반 이미지 캐러셀 반응형 처리"""
        
        # 기본 캐러셀 import 추가
        if "import React" in jsx_code and "useState" not in jsx_code:
            jsx_code = jsx_code.replace(
                "import React, { memo }",
                "import React, { memo, useState, useEffect }"
            )
        
        # AI Search 패턴 기반 캐러셀 기능 개선
        if patterns:
            best_pattern = patterns[0]
            carousel_type = best_pattern.get("carousel_type", "basic")
            
            if carousel_type == "swiper":
                # Swiper 기반 캐러셀 코드 추가
                jsx_code = jsx_code.replace(
                    "// 컴포넌트 코드",
                    """
  const [currentSlide, setCurrentSlide] = useState(0);
  
  const nextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % images.length);
  };
  
  const prevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + images.length) % images.length);
  };
  
  // 컴포넌트 코드
                    """
                )
        
        return jsx_code
    
    def _add_multi_column_responsive_with_patterns(self, jsx_code: str, patterns: List[Dict]) -> str:
        """AI Search 패턴 기반 다단 레이아웃 반응형 처리"""
        
        # 기본 다단 레이아웃 클래스 추가
        if "prose prose-lg" in jsx_code:
            jsx_code = jsx_code.replace(
                "prose prose-lg",
                "prose prose-lg lg:columns-2 lg:gap-8"
            )
        
        # AI Search 패턴 기반 다단 레이아웃 개선
        if patterns:
            best_pattern = patterns[0]
            column_config = best_pattern.get("column_config", {})
            
            if column_config.get("tablet_columns"):
                jsx_code = jsx_code.replace(
                    "lg:columns-2",
                    f"md:columns-{column_config['tablet_columns']} lg:columns-2"
                )
        
        return jsx_code
    
    def _apply_pattern_responsive_features(self, jsx_code: str, patterns: List[Dict]) -> str:
        """AI Search 패턴 기반 추가 반응형 기능 적용"""
        
        best_pattern = patterns[0]
        
        # 패턴 기반 반응형 기능 추가
        responsive_features = best_pattern.get("responsive_features", [])
        
        if "sticky_header" in responsive_features:
            jsx_code = jsx_code.replace(
                'className="text-center mb-8"',
                'className="text-center mb-8 sticky top-0 bg-white/90 backdrop-blur-sm z-10 md:relative md:bg-transparent"'
            )
        
        if "parallax_scroll" in responsive_features:
            jsx_code = jsx_code.replace(
                "import React, { memo",
                "import React, { memo, useEffect, useState"
            )
        
        if "lazy_loading" in responsive_features:
            jsx_code = jsx_code.replace(
                "loading=\"lazy\"",
                "loading=\"lazy\" placeholder=\"blur\""
            )
        
        return jsx_code
    
    def _generate_clean_jsx_fallback(self) -> Dict:
        """완전히 정화된 JSX 폴백 결과"""
        return {
            "jsx_components": [],
            "generation_metadata": {
                "total_components": 0,
                "multimodal_optimization": False,
                "responsive_design": False,
                "style_optimization": False,
                "ai_search_enhanced": False,
                "isolation_applied": True,
                "contamination_detected": True,
                "fallback_used": True
            }
        }