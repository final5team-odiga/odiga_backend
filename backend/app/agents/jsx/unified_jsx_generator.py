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
from utils.data.jsx_vector_manager import JSXVectorManager
from utils.data.pdf_vector_manager import PDFVectorManager

class UnifiedJSXGenerator(SessionAwareMixin, InterAgentCommunicationMixin):

    def __init__(self, logger: Any = None, vector_manager: PDFVectorManager = None):
        self.llm = get_azure_llm()
        self.logger = logger
        self._setup_logging_system()
        
        self.isolation_manager = AISearchIsolationManager()
        self.__init_session_awareness__()
        self.__init_inter_agent_communication__()
        
        # ✅ 필수 속성들 먼저 초기화
        self.external_vector_manager = vector_manager
        self.jsx_vector_manager = None
        self.jsx_vector_available = False
        
        # ✅ JSX 벡터 매니저 초기화 시도
        self._safe_initialize_jsx_vector_manager(vector_manager)

    def _safe_initialize_jsx_vector_manager(self, vector_manager: PDFVectorManager = None):
        """안전한 JSX 벡터 매니저 초기화"""
        try:
            
            if vector_manager:
                self.jsx_vector_manager = JSXVectorManager(vector_manager=vector_manager)
                self.logger.info("✅ 기존 PDFVectorManager로 JSXVectorManager 초기화")
            else:
                self.jsx_vector_manager = JSXVectorManager()
                self.logger.info("✅ 새로운 JSXVectorManager 초기화")
            
            # 가용성 확인
            if hasattr(self.jsx_vector_manager, 'initialize_jsx_search_index'):
                has_data = self.jsx_vector_manager.initialize_jsx_search_index()
                self.jsx_vector_available = has_data
            else:
                self.jsx_vector_available = self._is_jsx_vector_available()
            
            self.logger.info(f"✅ JSX 벡터 매니저 초기화 완료 (가용: {self.jsx_vector_available})")
            
        except ImportError as e:
            self.logger.warning(f"JSXVectorManager 임포트 실패: {e}")
            self.jsx_vector_manager = None
            self.jsx_vector_available = False
        except Exception as e:
            self.logger.error(f"JSX 벡터 매니저 초기화 실패: {e}")
            self.jsx_vector_manager = None
            self.jsx_vector_available = False

    def _is_jsx_vector_available(self) -> bool:
        """JSX 벡터 매니저 가용성 확인"""
        return (self.jsx_vector_manager is not None and 
                hasattr(self.jsx_vector_manager, 'pdf_vector_manager') and
                self.jsx_vector_manager.pdf_vector_manager is not None)

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
                agent_role="지능형 JSX 컴포넌트 생성기",
                task_description="벡터 기반 템플릿 분석 및 동적 JSX 생성",
                response_data=result,
                metadata={"vector_enhanced": True, "intelligent_binding": True}
            )
        
        return result

    async def generate_jsx_from_template(self, content_data: Dict, template_code: str) -> Dict:
        """
        ✅ 완전히 개선된 JSX 생성: 벡터 기반 지능형 템플릿 분석 + 동적 바인딩
        """
        try:
            title = content_data.get("title", "제목 없음")
            self.logger.info(f"지능형 JSX 생성 시작: {title}")
            
            # 하위 섹션 여부 확인
            is_subsection = content_data.get("metadata", {}).get("is_subsection", False)
            
            # ✅ 1단계: 콘텐츠 요구사항 분석
            content_requirements = self._analyze_content_requirements(content_data)
            
            # ✅ 2단계: 템플릿 구조 분석 및 적합성 평가
            if template_code and len(template_code.strip()) > 100:
                template_analysis = self._analyze_template_structure(template_code)
                template_suitability = self._evaluate_template_suitability(template_analysis, content_requirements)
                
                # ✅ 3단계: 템플릿 개선 필요성 판단
                if template_suitability["needs_enhancement"] and self.jsx_vector_available:
                    # 벡터 검색으로 템플릿 개선
                    enhanced_template = await self._enhance_template_with_vector_search(
                        template_code, template_analysis, content_requirements
                    )
                    jsx_result = self._apply_intelligent_data_binding(content_data, enhanced_template, template_analysis)
                    generation_method = "vector_enhanced_template"
                elif template_suitability["direct_usable"]:
                    # 기존 템플릿 직접 사용
                    jsx_result = self._apply_intelligent_data_binding(content_data, template_code, template_analysis)
                    generation_method = "intelligent_template_binding"
                else:
                    # LLM 기반 생성
                    jsx_result = await self._generate_intelligent_jsx(content_data, template_code)
                    generation_method = "llm_intelligent_generation"
            else:
                # ✅ 4단계: 템플릿이 없는 경우 벡터 기반 새 템플릿 생성
                jsx_result = await self._generate_jsx_from_vector_recommendations(content_data, content_requirements)
                generation_method = "vector_based_generation"
            
            # 메타데이터 추가
            jsx_result["metadata"] = {
                "template_applied": True,
                "generation_method": generation_method,
                "template_name": self._extract_template_name(template_code),
                "generation_timestamp": time.time(),
                "is_subsection": is_subsection,
                "content_requirements": content_requirements,
                "vector_enhanced": self.jsx_vector_available
            }
            
            # 하위 섹션 메타데이터
            if is_subsection:
                jsx_result["metadata"]["parent_section_id"] = content_data.get("metadata", {}).get("parent_section_id", "")
                jsx_result["metadata"]["parent_section_title"] = content_data.get("metadata", {}).get("parent_section_title", "")
            
            self.logger.info(f"지능형 JSX 생성 완료: {title} (방식: {generation_method})")
            return jsx_result
            
        except Exception as e:
            self.logger.error(f"지능형 JSX 생성 실패: {e}")
            return self._create_fallback_jsx(content_data, str(e))

    def _analyze_content_requirements(self, content_data: Dict) -> Dict:
        """✅ 콘텐츠 요구사항을 벡터 데이터와 매칭하여 분석"""
        
        images = content_data.get("images", [])
        content = content_data.get("content", "")
        title = content_data.get("title", "")
        
        # 기본 요구사항 분석
        requirements = {
            "image_count": len(images),
            "content_length": len(content),
            "title_length": len(title),
            "needs_more_images": len(images) > 3,
            "needs_content_area": len(content) > 500,
            "needs_title_emphasis": len(title) > 50,
            "content_type": self._classify_content_type(content),
            "layout_preference": "grid" if len(images) > 2 else "flex",
            "complexity_level": "complex" if len(content) > 1000 else "simple"
        }
        
        # ✅ 벡터 검색으로 유사한 콘텐츠 패턴 찾기
        if self.jsx_vector_available:
            try:
                similar_patterns = self.jsx_vector_manager.search_jsx_components(
                    query_text=f"content about {content[:100]}",
                    top_k=3
                )
                
                if similar_patterns:
                    # 패턴 분석을 통한 요구사항 개선
                    requirements["preferred_layout"] = self._infer_layout_from_patterns(similar_patterns)
                    requirements["style_preferences"] = self._extract_style_preferences(similar_patterns)
                    requirements["similar_components"] = [p["component_name"] for p in similar_patterns]
                    
            except Exception as e:
                self.logger.warning(f"벡터 기반 콘텐츠 분석 실패: {e}")
        
        return requirements

    def _classify_content_type(self, content: str) -> str:
        """콘텐츠 타입 분류"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["여행", "travel", "journey", "adventure"]):
            return "travel"
        elif any(word in content_lower for word in ["음식", "food", "restaurant", "recipe"]):
            return "food"
        elif any(word in content_lower for word in ["기술", "tech", "technology", "ai"]):
            return "technology"
        elif any(word in content_lower for word in ["문화", "culture", "art", "museum"]):
            return "culture"
        else:
            return "general"

    def _analyze_template_structure(self, template_code: str) -> Dict:
        """✅ 향상된 템플릿 구조 분석"""
        structure = {
            "has_main_title": False,
            "has_subtitle": False,
            "has_content_area": False,
            "has_images": False,
            "layout_type": "unknown",
            "title_patterns": [],
            "subtitle_patterns": [],
            "content_patterns": [],
            "image_patterns": [],
            "component_complexity": "simple",
            "responsive_design": False,
            "interactive_elements": False
        }
        
        # 제목 패턴 분석
        title_patterns = re.findall(r'<h1[^>]*>([^<]+)</h1>', template_code, re.IGNORECASE)
        if title_patterns:
            structure["has_main_title"] = True
            structure["title_patterns"] = title_patterns
        
        # 부제목 패턴 분석
        subtitle_patterns = re.findall(r'<h[2-3][^>]*>([^<]+)</h[2-3]>', template_code, re.IGNORECASE)
        if subtitle_patterns:
            structure["has_subtitle"] = True
            structure["subtitle_patterns"] = subtitle_patterns
        
        # 콘텐츠 영역 분석
        content_patterns = re.findall(r'<p[^>]*>([^<]+)</p>', template_code, re.IGNORECASE)
        div_content_patterns = re.findall(r'<div[^>]*>([^<]*(?:(?!</div>).)*[^<]*)</div>', template_code, re.IGNORECASE)
        if content_patterns or div_content_patterns:
            structure["has_content_area"] = True
            structure["content_patterns"] = content_patterns + div_content_patterns
        
        # 이미지 패턴 분석
        img_patterns = re.findall(r'<img[^>]*src="([^"]*)"[^>]*>', template_code, re.IGNORECASE)
        if img_patterns:
            structure["has_images"] = True
            structure["image_patterns"] = img_patterns
        
        # 레이아웃 타입 분석
        if "display: \"grid\"" in template_code or "gridTemplateColumns" in template_code:
            structure["layout_type"] = "grid"
        elif "display: \"flex\"" in template_code or "flexDirection" in template_code:
            structure["layout_type"] = "flex"
        else:
            structure["layout_type"] = "standard"
        
        # 복잡도 분석
        complexity_indicators = (
            template_code.count('<div') + 
            template_code.count('<img') * 2 + 
            template_code.count('style=') * 0.5
        )
        
        if complexity_indicators > 15:
            structure["component_complexity"] = "complex"
        elif complexity_indicators > 8:
            structure["component_complexity"] = "moderate"
        
        # 반응형 디자인 확인
        structure["responsive_design"] = "responsive" in template_code.lower() or "@media" in template_code
        
        # 인터랙티브 요소 확인
        structure["interactive_elements"] = "onClick" in template_code or "onHover" in template_code
        
        return structure

    def _evaluate_template_suitability(self, template_analysis: Dict, content_requirements: Dict) -> Dict:
        """템플릿과 콘텐츠 요구사항 간 적합성 평가"""
        
        suitability = {
            "direct_usable": True,
            "needs_enhancement": False,
            "compatibility_score": 0.0,
            "missing_features": [],
            "enhancement_suggestions": []
        }
        
        # 이미지 요구사항 매칭
        template_has_images = template_analysis["has_images"]
        content_needs_images = content_requirements["image_count"] > 0
        
        if content_needs_images and not template_has_images:
            suitability["missing_features"].append("image_support")
            suitability["enhancement_suggestions"].append("add_image_layout")
            suitability["needs_enhancement"] = True
        elif not content_needs_images and template_has_images:
            suitability["enhancement_suggestions"].append("remove_image_placeholders")
        
        # 콘텐츠 길이 매칭
        if content_requirements["needs_content_area"] and not template_analysis["has_content_area"]:
            suitability["missing_features"].append("content_area")
            suitability["enhancement_suggestions"].append("add_content_section")
            suitability["needs_enhancement"] = True
        
        # 복잡도 매칭
        template_complexity = template_analysis["component_complexity"]
        content_complexity = content_requirements["complexity_level"]
        
        if template_complexity == "simple" and content_complexity == "complex":
            suitability["enhancement_suggestions"].append("increase_template_complexity")
            suitability["needs_enhancement"] = True
        
        # 호환성 점수 계산
        compatibility_factors = []
        
        # 이미지 호환성
        if (template_has_images and content_needs_images) or (not template_has_images and not content_needs_images):
            compatibility_factors.append(0.3)
        
        # 콘텐츠 영역 호환성
        if template_analysis["has_content_area"] and content_requirements["needs_content_area"]:
            compatibility_factors.append(0.3)
        
        # 레이아웃 호환성
        if template_analysis["layout_type"] == content_requirements.get("layout_preference", "flex"):
            compatibility_factors.append(0.2)
        
        # 복잡도 호환성
        if template_complexity == content_complexity:
            compatibility_factors.append(0.2)
        
        suitability["compatibility_score"] = sum(compatibility_factors)
        
        # 직접 사용 가능 여부 결정
        if suitability["compatibility_score"] < 0.6 or len(suitability["missing_features"]) > 1:
            suitability["direct_usable"] = False
            suitability["needs_enhancement"] = True
        
        return suitability

    async def _enhance_template_with_vector_search(self, base_template: str, template_analysis: Dict, 
                                                 content_requirements: Dict) -> str:
        """✅ 벡터 검색으로 템플릿 구조 개선"""
        
        enhanced_template = base_template
        
        try:
            # 1. 이미지 레이아웃 개선
            if "add_image_layout" in content_requirements.get("enhancement_suggestions", []):
                image_layouts = self.jsx_vector_manager.search_jsx_components(
                    query_text="multiple images gallery grid layout responsive",
                    category="image_focused",
                    image_count=content_requirements["image_count"],
                    top_k=3
                )
                
                if image_layouts:
                    best_image_layout = self._extract_image_layout_pattern(image_layouts[0])
                    enhanced_template = self._merge_image_layout(enhanced_template, best_image_layout)
                    self.logger.info("벡터 검색으로 이미지 레이아웃 개선 완료")
            
            # 2. 콘텐츠 영역 개선
            if "add_content_section" in content_requirements.get("enhancement_suggestions", []):
                text_layouts = self.jsx_vector_manager.search_jsx_components(
                    query_text="text content paragraph layout typography",
                    category="text_focused",
                    top_k=3
                )
                
                if text_layouts:
                    best_text_layout = self._extract_text_layout_pattern(text_layouts[0])
                    enhanced_template = self._merge_text_layout(enhanced_template, best_text_layout)
                    self.logger.info("벡터 검색으로 텍스트 레이아웃 개선 완료")
            
            # 3. 스타일 최적화
            enhanced_template = await self._optimize_styles_with_vector_data(
                enhanced_template, content_requirements
            )
            
        except Exception as e:
            self.logger.error(f"벡터 기반 템플릿 개선 실패: {e}")
            return base_template
        
        return enhanced_template

    def _extract_image_layout_pattern(self, jsx_component: Dict) -> str:
        """JSX 컴포넌트에서 이미지 레이아웃 패턴 추출"""
        jsx_code = jsx_component.get("jsx_code", "")
        
        # 이미지 관련 JSX 패턴 추출
        image_section_match = re.search(
            r'<div[^>]*>.*?<img.*?</div>',
            jsx_code,
            re.DOTALL | re.IGNORECASE
        )
        
        if image_section_match:
            return image_section_match.group(0)
        
        # 기본 이미지 레이아웃 패턴 반환
        return '''
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "16px", marginTop: "20px" }}>
          <img src="{image_url}" alt="{image_alt}" style={{ width: "100%", height: "200px", objectFit: "cover", borderRadius: "8px" }} />
        </div>
        '''

    def _extract_text_layout_pattern(self, jsx_component: Dict) -> str:
        """JSX 컴포넌트에서 텍스트 레이아웃 패턴 추출"""
        jsx_code = jsx_component.get("jsx_code", "")
        
        # 텍스트 관련 JSX 패턴 추출
        text_section_match = re.search(
            r'<div[^>]*>.*?<p.*?</div>',
            jsx_code,
            re.DOTALL | re.IGNORECASE
        )
        
        if text_section_match:
            return text_section_match.group(0)
        
        # 기본 텍스트 레이아웃 패턴 반환
        return '''
        <div style={{ fontSize: "1rem", lineHeight: "1.7", marginBottom: "16px" }}>
          <p>{content_text}</p>
        </div>
        '''

    def _merge_image_layout(self, base_template: str, image_layout_pattern: str) -> str:
        """이미지 레이아웃 패턴을 기존 템플릿에 병합"""
        
        # 기존 이미지 영역 찾기
        image_section_match = re.search(r'<div[^>]*>.*?<img.*?</div>', base_template, re.DOTALL)
        
        if image_section_match:
            # 기존 이미지 영역을 새로운 패턴으로 교체
            return base_template.replace(image_section_match.group(0), image_layout_pattern)
        else:
            # 이미지 영역이 없으면 h1 태그 앞에 추가
            h1_match = re.search(r'<h1[^>]*>', base_template)
            if h1_match:
                insert_pos = h1_match.start()
                return (base_template[:insert_pos] + 
                       image_layout_pattern + "\n      " + 
                       base_template[insert_pos:])
        
        return base_template

    def _merge_text_layout(self, base_template: str, text_layout_pattern: str) -> str:
        """텍스트 레이아웃 패턴을 기존 템플릿에 병합"""
        
        # 기존 텍스트 영역 찾기
        text_section_match = re.search(r'<div[^>]*>.*?<p.*?</div>', base_template, re.DOTALL)
        
        if text_section_match:
            return base_template.replace(text_section_match.group(0), text_layout_pattern)
        else:
            # 텍스트 영역이 없으면 마지막 h 태그 뒤에 추가
            last_h_match = None
            for match in re.finditer(r'</h[1-6]>', base_template):
                last_h_match = match
            
            if last_h_match:
                insert_pos = last_h_match.end()
                return (base_template[:insert_pos] + 
                       "\n      " + text_layout_pattern + 
                       base_template[insert_pos:])
        
        return base_template

    async def _optimize_styles_with_vector_data(self, jsx_code: str, content_requirements: Dict) -> str:
        """✅ 벡터 데이터를 활용한 스타일 최적화"""
        
        try:
            # 콘텐츠 타입에 맞는 스타일 패턴 검색
            content_type = content_requirements.get("content_type", "general")
            
            style_patterns = self.jsx_vector_manager.search_jsx_components(
                query_text=f"{content_type} style design layout modern",
                top_k=3
            )
            
            if style_patterns:
                # 색상 팔레트 추출 및 적용
                for pattern in style_patterns:
                    jsx_code_sample = pattern.get("jsx_code", "")
                    
                    # 색상 추출
                    colors = self._extract_color_palette(jsx_code_sample)
                    jsx_code = self._apply_color_palette(jsx_code, colors)
                    
                    # 타이포그래피 스타일 추출
                    typography = self._extract_typography_styles(jsx_code_sample)
                    jsx_code = self._apply_typography_styles(jsx_code, typography)
                    
                    break  # 첫 번째 패턴만 적용
            
        except Exception as e:
            self.logger.warning(f"벡터 기반 스타일 최적화 실패: {e}")
        
        return jsx_code

    def _extract_color_palette(self, jsx_code: str) -> Dict:
        """JSX 코드에서 색상 팔레트 추출"""
        colors = {
            "background": "white",
            "text": "black",
            "accent": "#606060"
        }
        
        # 배경색 추출
        bg_match = re.search(r'backgroundColor:\s*["\']([^"\']+)["\']', jsx_code)
        if bg_match:
            colors["background"] = bg_match.group(1)
        
        # 텍스트 색상 추출
        color_match = re.search(r'color:\s*["\']([^"\']+)["\']', jsx_code)
        if color_match:
            colors["text"] = color_match.group(1)
        
        return colors

    def _apply_color_palette(self, jsx_code: str, colors: Dict) -> str:
        """색상 팔레트를 JSX 코드에 적용 (안전한 정규표현식)"""
        
        try:
            # ✅ 안전한 정규표현식 사용
            background_color = colors.get("background", "white")
            text_color = colors.get("text", "black")
            
            # 배경색 적용
            jsx_code = re.sub(
                r'(backgroundColor:\s*["\'])[^"\']*(["\'])',
                r'\g<1>' + background_color + r'\g<2>',
                jsx_code
            )
            
            # 텍스트 색상 적용
            jsx_code = re.sub(
                r'(color:\s*["\'])[^"\']*(["\'])',
                r'\g<1>' + text_color + r'\g<2>',
                jsx_code
            )
            
        except re.error as e:
            self.logger.warning(f"정규표현식 오류, 색상 적용 건너뜀: {e}")
        
        return jsx_code

    def _extract_typography_styles(self, jsx_code: str) -> Dict:
        """타이포그래피 스타일 추출"""
        typography = {
            "title_size": "2rem",
            "subtitle_size": "1.25rem",
            "content_size": "1rem",
            "line_height": "1.7"
        }
        
        # 제목 크기 추출
        title_size_match = re.search(r'<h1[^>]*style={{[^}]*fontSize:\s*["\']([^"\']+)["\']', jsx_code)
        if title_size_match:
            typography["title_size"] = title_size_match.group(1)
        
        return typography

    def _apply_typography_styles(self, jsx_code: str, typography: Dict) -> str:
        """타이포그래피 스타일을 JSX 코드에 적용 (정규표현식 수정)"""
        
        try:
            # ✅ 안전한 정규표현식 사용
            title_size = typography.get("title_size", "2rem")
            
            # h1 태그의 fontSize 속성 찾아서 교체
            jsx_code = re.sub(
                r'(<h1[^>]*style=\{[^}]*fontSize:\s*["\'])[^"\']*(["\'][^}]*\}[^>]*>)',
                r'\g<1>' + title_size + r'\g<2>',
                jsx_code
            )
            
            # h2 태그도 동일하게 처리
            subtitle_size = typography.get("subtitle_size", "1.25rem")
            jsx_code = re.sub(
                r'(<h2[^>]*style=\{[^}]*fontSize:\s*["\'])[^"\']*(["\'][^}]*\}[^>]*>)',
                r'\g<1>' + subtitle_size + r'\g<2>',
                jsx_code
            )
            
        except re.error as e:
            self.logger.warning(f"정규표현식 오류, 타이포그래피 적용 건너뜀: {e}")
        
        return jsx_code

    def _infer_layout_from_patterns(self, similar_patterns: List[Dict]) -> str:
        """유사한 패턴에서 레이아웃 추론"""
        layout_votes = {}
        
        for pattern in similar_patterns:
            layout_method = pattern.get("layout_method", "flex")
            layout_votes[layout_method] = layout_votes.get(layout_method, 0) + 1
        
        return max(layout_votes, key=layout_votes.get) if layout_votes else "flex"
    
    def _format_images_for_jsx(self, images: List[Dict]) -> List[Dict]:
        """✅ 실제 이미지 분석 데이터 구조에 맞춘 JSX 포맷팅"""
        formatted_images = []
        for i, img in enumerate(images):
            # ✅ 실제 데이터 구조에 맞는 키 사용
            image_url = img.get("image_url", "")  # 실제 키는 image_url
            
            # ✅ 간단하고 안전한 alt 텍스트 생성 (분석 텍스트 완전 제거)
            city = img.get("city", "")
            location = img.get("location", "")
            
            # 간단한 alt 텍스트 생성 (긴 분석 텍스트 사용 안함)
            if city and location:
                simple_alt = f"{city} {location[:30]}..."  # 30자로 제한
            else:
                simple_alt = f"베네치아 여행 이미지 {i+1}"
            
            # 따옴표 제거 (JSX 구문 오류 방지)
            safe_alt = simple_alt.replace('"', '').replace("'", "")
            
            formatted_img = {
                "image_url": image_url,
                "url": image_url,  # 호환성을 위해 둘 다 포함
                "description": safe_alt,
                "alt": safe_alt,
                "width": 800,
                "height": 600,
                "caption": safe_alt
            }
            formatted_images.append(formatted_img)

        return formatted_images
    
    def _extract_style_preferences(self, similar_patterns: List[Dict]) -> List[str]:
        """유사한 패턴에서 스타일 선호도 추출"""
        style_preferences = []
        
        for pattern in similar_patterns:
            category = pattern.get("category", "")
            complexity = pattern.get("complexity", "")
            
            if category:
                style_preferences.append(category)
            if complexity:
                style_preferences.append(complexity)
        
        return list(set(style_preferences))
    
    def _replace_images_with_safe_alt(self, jsx_code: str, images: List[Dict]) -> str:
        """✅ 실제 이미지 데이터 구조에 맞춘 안전한 이미지 교체"""
        
        # 기존 img 태그들을 찾아서 순차적으로 교체
        img_pattern = r'<img[^>]*src="[^"]*"[^>]*>'
        img_tags = re.findall(img_pattern, jsx_code)
        
        for i, old_img_tag in enumerate(img_tags):
            if i < len(images):
                real_img = images[i]
                # ✅ 실제 키 이름 사용
                real_img_url = real_img.get("image_url", "")
                
                # ✅ 안전한 alt 텍스트 생성 (실제 데이터 활용)
                city = real_img.get("city", "")
                if city:
                    safe_alt = f"{city} 여행 이미지 {i+1}"
                else:
                    safe_alt = f"베네치아 여행 이미지 {i+1}"
                
                # 따옴표 및 특수문자 제거
                safe_alt = re.sub(r'["\']', '', safe_alt)
                
                if real_img_url:
                    # 새로운 img 태그 생성 (안전한 속성만 포함)
                    new_img_tag = f'''<img
            src="{real_img_url}"
            alt="{safe_alt}"
            style={{{{ width: "100%", height: "auto", display: "block" }}}}
            />'''
                    
                    jsx_code = jsx_code.replace(old_img_tag, new_img_tag)
        
        return jsx_code

    async def _generate_jsx_from_vector_recommendations(self, content_data: Dict, 
                                                       content_requirements: Dict) -> Dict:
        """✅ 벡터 추천 기반 완전 새로운 JSX 생성"""
        
        try:
            if not self.jsx_vector_available:
                return self._generate_jsx_from_scratch(content_data)
            
            # 콘텐츠 요구사항에 맞는 템플릿 추천
            recommendations = self.jsx_vector_manager.get_jsx_recommendations(
                content_description=f"{content_requirements['content_type']} content with {content_requirements['image_count']} images",
                image_count=content_requirements["image_count"],
                layout_preference=content_requirements.get("layout_preference", "flex")
            )
            
            if recommendations:
                # 가장 적합한 추천 템플릿 사용
                best_recommendation = recommendations[0]
                recommended_template = best_recommendation.get("jsx_code", "")
                
                if recommended_template:
                    # 추천 템플릿 분석
                    template_analysis = self._analyze_template_structure(recommended_template)
                    
                    # 지능형 데이터 바인딩 적용
                    jsx_result = self._apply_intelligent_data_binding(
                        content_data, recommended_template, template_analysis
                    )
                    
                    jsx_result["metadata"] = jsx_result.get("metadata", {})
                    jsx_result["metadata"]["recommended_template"] = best_recommendation.get("component_name", "")
                    jsx_result["metadata"]["recommendation_score"] = best_recommendation.get("score", 0.0)
                    
                    return jsx_result
            
            # 추천이 없는 경우 기본 생성
            return self._generate_jsx_from_scratch(content_data)
            
        except Exception as e:
            self.logger.error(f"벡터 추천 기반 JSX 생성 실패: {e}")
            return self._generate_jsx_from_scratch(content_data)

    def _apply_intelligent_data_binding(self, content_data: Dict, template_code: str, 
                                      template_analysis: Dict) -> Dict:
        """✅ 지능형 데이터 바인딩 (구조 분석 기반)"""
        
        title = content_data.get("title", "제목 없음")
        subtitle = content_data.get("subtitle", "")
        content = content_data.get("content", "")
        images = content_data.get("images", [])
        
        # 템플릿 내용을 실제 데이터로 교체
        modified_jsx = self._replace_template_content_intelligent(
            template_code, 
            template_analysis, 
            title, 
            subtitle, 
            content, 
            images
        )
        
        # 컴포넌트 이름 업데이트
        component_name = self._generate_component_name(title)
        modified_jsx = self._update_component_name(modified_jsx, component_name)
        
        self.logger.info(f"지능형 데이터 바인딩 완료: {title} (이미지: {len(images)}개)")
        
        return {
            "title": title,
            "jsx_code": modified_jsx
        }

    def _remove_image_elements_safe(self, jsx_code: str) -> str:
        """✅ 이미지 요소 안전하게 제거"""
        # img 태그를 빈 div로 교체 (JSX 구문 오류 방지)
        jsx_code = re.sub(r'<img[^>]*>', '<div style={{ display: "none" }}></div>', jsx_code)
        
        # 빈 이미지 컨테이너 제거
        jsx_code = re.sub(r'<div[^>]*>\s*<div style=\{\{ display: "none" \}\}></div>\s*</div>', '', jsx_code)
        
        return jsx_code

    def _replace_template_content_intelligent(self, template_code: str, template_analysis: Dict, 
                                        title: str, subtitle: str, content: str, 
                                        images: List[Dict]) -> str:
        """✅ 완전한 하드코딩 텍스트 제거 및 매거진 데이터 삽입"""
        
        modified_jsx = template_code
        
        # ✅ 1. 모든 하드코딩된 제목 패턴을 매거진 제목으로 교체
        # 특정 문자열 매핑 대신 패턴 기반 교체
        modified_jsx = re.sub(
            r'(<h1[^>]*>)[^<]+(</h1>)',
            rf'\1{title}\2',
            modified_jsx,
            flags=re.IGNORECASE
        )
        
        # ✅ 2. 모든 하드코딩된 부제목을 매거진 부제목으로 교체
        if subtitle:
            modified_jsx = re.sub(
                r'(<h[2-3][^>]*>)[^<]+(</h[2-3]>)',
                rf'\1{subtitle}\2',
                modified_jsx,
                flags=re.IGNORECASE
            )
        
        # ✅ 3. 모든 하드코딩된 텍스트 콘텐츠를 매거진 콘텐츠로 교체
        # 영어 텍스트, 샘플 텍스트 모두 제거
        if content:
            # div 내부의 모든 텍스트 교체
            modified_jsx = re.sub(
                r'(<div[^>]*style=\{[^}]*flex:\s*["\']?2["\']?[^}]*\}[^>]*>)[^<]+(</div>)',
                rf'\1{content}\2',
                modified_jsx,
                flags=re.DOTALL
            )
            
            # p 태그 내부의 긴 텍스트 교체 (50자 이상)
            modified_jsx = re.sub(
                r'(<p[^>]*>)[^<]{50,}(</p>)',
                rf'\1{content[:500]}\2',
                modified_jsx,
                flags=re.IGNORECASE | re.DOTALL
            )
            
            # 여러 줄에 걸친 텍스트 콘텐츠 교체
            modified_jsx = re.sub(
                r'(<p[^>]*style=\{[^}]*fontSize[^}]*\}[^>]*>)\s*[A-Za-z][^<]*\.\s*(</p>)',
                rf'\1{content[:300]}\2',
                modified_jsx,
                flags=re.DOTALL | re.IGNORECASE
            )
        
        # ✅ 4. 이미지 교체 (안전한 alt 텍스트 사용)
        if images:
            modified_jsx = self._replace_images_with_safe_alt(modified_jsx, images)
        else:
            # 이미지가 없으면 이미지 관련 요소 제거
            modified_jsx = self._remove_image_elements_safe(modified_jsx)
        
        return modified_jsx

    def _replace_images_intelligent(self, jsx_code: str, image_patterns: List[str], 
                                  images: List[Dict]) -> str:
        """지능형 이미지 교체"""
        
        for i, img_url in enumerate(image_patterns):
            if i < len(images):
                real_img = images[i]
                real_img_url = real_img.get("image_url", real_img.get("url", ""))
                real_img_alt = real_img.get("description", real_img.get("alt", f"이미지 {i+1}"))
                
                if real_img_url:
                    # URL 교체
                    jsx_code = jsx_code.replace(f'src="{img_url}"', f'src="{real_img_url}"')
                    
                    # alt 속성 업데이트
                    alt_pattern = rf'(<img[^>]*src="{re.escape(real_img_url)}"[^>]*alt=")[^"]*(")'
                    jsx_code = re.sub(alt_pattern, rf'\1{real_img_alt}\2', jsx_code)
        
        return jsx_code

    def _remove_image_elements(self, jsx_code: str) -> str:
        """이미지 요소 제거"""
        # img 태그를 주석으로 교체
        jsx_code = re.sub(r'<img[^>]*>', '{/* 이미지 없음 */}', jsx_code)
        
        # 빈 이미지 컨테이너 제거
        jsx_code = re.sub(r'<div[^>]*>\s*{/\* 이미지 없음 \*/}\s*</div>', '', jsx_code)
        
        return jsx_code

    def _add_image_elements(self, jsx_code: str, images: List[Dict]) -> str:
        """이미지 요소 추가"""
        
        # 이미지 JSX 생성
        images_jsx = self._generate_images_jsx_for_template(images)
        
        # h1 태그 뒤에 이미지 추가
        h1_end_match = re.search(r'</h1>', jsx_code)
        if h1_end_match:
            insert_pos = h1_end_match.end()
            jsx_code = (jsx_code[:insert_pos] + 
                       "\n      " + images_jsx + 
                       jsx_code[insert_pos:])
        
        return jsx_code

    def _generate_jsx_from_scratch(self, content_data: Dict) -> Dict:
        """✅ 실제 이미지 데이터 구조에 맞춘 JSX 생성"""
        
        title = content_data.get("title", "제목 없음")
        subtitle = content_data.get("subtitle", "")
        content = content_data.get("content", "")
        images = content_data.get("images", [])
        
        # ✅ 실제 이미지 데이터 구조에 맞춘 이미지 JSX 생성
        image_jsx = ""
        if images:
            image_jsx = '''
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "16px", marginTop: "20px" }}>'''
            
            for i, img in enumerate(images[:5]):
                # ✅ 실제 키 이름 사용
                img_url = img.get("image_url", "")
                city = img.get("city", "")
                
                # 안전한 alt 텍스트
                if city:
                    safe_alt = f"{city} 여행 이미지 {i+1}"
                else:
                    safe_alt = f"베네치아 여행 이미지 {i+1}"
                
                # 특수문자 제거
                safe_alt = re.sub(r'["\']', '', safe_alt)
                
                if img_url:
                    image_jsx += f'''
            <div style={{ textAlign: "center" }}>
                <img 
                src="{img_url}" 
                alt="{safe_alt}" 
                style={{ width: "100%", height: "200px", objectFit: "cover", borderRadius: "8px" }}
                />
            </div>'''
            
            image_jsx += "\n        </div>"
        
        # 컴포넌트 이름 생성
        component_name = self._generate_component_name(title)
        
        jsx_code = f"""import React from "react";

    const {component_name} = () => {{
        return (
            <div style={{ backgroundColor: "white", color: "black", padding: "20px",fontFamily: "'Noto Sans KR', sans-serif" }}>
                <h1 style={{ fontSize: "2rem", marginBottom: "16px" }}>
                    {title}
                </h1>
                {f'<h2 style={{ fontSize: "1.2rem", marginBottom: "12px", color: "#606060" }}>{subtitle}</h2>' if subtitle else ''}
                <div style={{ fontSize: "1rem", lineHeight: "1.7", marginBottom: "16px" }}>
                    {content}
                </div>
                {image_jsx}
            </div>
        );
    }};

    export default {component_name};"""

        return {
            "title": title,
            "jsx_code": jsx_code
        }

    def _generate_images_jsx_for_template(self, images: List[Dict]) -> str:
        """템플릿용 이미지 JSX 생성 (alt 속성 제거)"""
        if not images:
            return ""
        
        images_jsx_parts = []
        for i, img in enumerate(images[:5]):
            img_url = img.get("image_url", img.get("url", ""))
            
            img_jsx = f"""
        <div key={i} style={{ textAlign: "center", marginBottom: "16px" }}>
            <img 
            src="{img_url}" 
            style={{ width: "100%", maxHeight: "300px", objectFit: "cover", borderRadius: "8px" }}
            onError={{(e) => {{ e.target.style.display = 'none'; }}}}
            />
        </div>"""
            
            images_jsx_parts.append(img_jsx)
        
        return "\n".join(images_jsx_parts)

    def _generate_component_name(self, title: str) -> str:
        """제목에서 유효한 React 컴포넌트 이름 생성"""
        # 특수문자 제거 및 PascalCase 변환
        clean_title = re.sub(r'[^a-zA-Z0-9가-힣\s]', '', title)
        words = clean_title.split()
        component_name = ''.join(word.capitalize() for word in words if word)
        
        # 유효하지 않은 경우 기본값 사용
        if not component_name or not component_name[0].isupper():
            component_name = "GeneratedSection"
        
        return component_name

    def _update_component_name(self, jsx_code: str, component_name: str) -> str:
        """JSX 코드의 컴포넌트 이름 업데이트"""
        # const ComponentName = () => { 패턴 찾기
        pattern = r'const\s+\w+\s*='
        replacement = f'const {component_name} ='
        jsx_code = re.sub(pattern, replacement, jsx_code)
        
        # export default ComponentName 패턴 찾기
        export_pattern = r'export\s+default\s+\w+;?'
        export_replacement = f'export default {component_name};'
        jsx_code = re.sub(export_pattern, export_replacement, jsx_code)
        
        return jsx_code

    def _extract_template_name(self, template_code: str) -> str:
        """템플릿 코드에서 템플릿 이름 추출"""
        match = re.search(r'const\s+(\w+)\s*=', template_code)
        if match:
            return match.group(1)
        return "UnknownTemplate"

    async def _generate_intelligent_jsx(self, content_data: Dict, template_code: str) -> Dict:
        """AI를 활용한 지능형 JSX 생성 (폴백용)"""
        title = content_data.get("title", "제목 없음")
        
        try:
            # 하위 섹션 정보 생성
            subsection_info = ""
            if content_data.get("metadata", {}).get("is_subsection", False):
                parent_section_id = content_data.get("metadata", {}).get("parent_section_id", "")
                parent_section_title = content_data.get("metadata", {}).get("parent_section_title", "")
                subsection_info = f"""
이 JSX 컴포넌트는 하위 섹션입니다:
- 상위 섹션 ID: {parent_section_id}
- 상위 섹션 제목: {parent_section_title}
"""
            
            prompt = self._create_jsx_generation_prompt(content_data, template_code, subsection_info)

            if not hasattr(self.llm, 'ainvoke'):
                raise AttributeError("LLM 객체에 ainvoke 메서드가 없습니다.")
            
            self.logger.info(f"'{title}' 섹션에 대한 지능형 JSX 생성을 시작합니다...")
            generated_code = await self.llm.ainvoke(prompt)
            
            if not generated_code or not isinstance(generated_code, str):
                raise ValueError("LLM으로부터 유효한 JSX 코드를 받지 못했습니다.")

            extracted_code = self._extract_jsx_code(generated_code)

            self.logger.info(f"'{title}' 섹션 JSX 생성 완료.")
            return {
                "title": title,
                "jsx_code": extracted_code
            }
        except Exception as e:
            self.logger.error(f"'{title}' 섹션의 지능형 JSX 생성 실패 (폴백 사용): {e}")
            fallback_result = self._simple_template_substitution(content_data, self._get_default_template())
            return fallback_result

    def _create_jsx_generation_prompt(self, content_data: Dict, template_code: str, subsection_info: str = "") -> str:
        """JSX 생성용 LLM 프롬프트를 구성합니다."""
        
        title = content_data.get("title", "제목 없음")
        pascal_case_title = ''.join(word.capitalize() for word in title.replace('-', ' ').replace('_', ' ').split())
        
        # ✅ 완전한 콘텐츠 사용 (잘리지 않음)
        full_content = content_data.get("content", "")
        image_data_json = self._format_image_data_for_prompt(content_data.get("images", []))
        
        return f"""# JSX 컴포넌트 생성 작업

## 기본 정보
- 컴포넌트 제목: {title}
- 컴포넌트 이름 (PascalCase): {pascal_case_title}
- 이미지 수: {len(content_data.get("images", []))}
{subsection_info}

## 완전한 콘텐츠 데이터
{{
  "title": "{title}",
  "subtitle": "{content_data.get("subtitle", "")}",
  "content": "{full_content}",
  "images": {image_data_json}
}}

## 기준 템플릿 코드
{template_code}

## 작업 지시사항
1. 위의 **완전한 콘텐츠**를 모두 포함하여 JSX 컴포넌트를 생성하세요.
2. 템플릿의 기본 구조와 스타일은 최대한 유지하되, 데이터에 맞게 내용을 채워 넣으세요.
3. 콘텐츠를 생략하거나 자르지 마세요. "...(생략)" 표현을 사용하지 마세요.
4. 이미지가 있다면, 템플릿 내의 적절한 위치에 배치하세요.
5. 완전하고 유효한 React 컴포넌트를 반환하세요.

## 출력 형식
완전한 JSX 코드만 제공하세요. 마크다운 코드 블록을 사용하지 마세요.
"""

    def _extract_jsx_code(self, response: str) -> str:
        """LLM의 응답에서 순수 JSX 코드만 추출합니다."""
        # 마크다운 코드 블록 제거
        if '```' in response:
            match = re.search(r'```(?:jsx)?\s*([\s\S]+?)\s*```', response)
            if match:
                return match.group(1).strip()
        
        # 코드 블록이 없는 경우, 불필요한 서론/결론 제거 시도
        lines = response.strip().split('\n')
        if lines and 'export default' in response:
            for i, line in enumerate(lines):
                if 'export default' in line:
                    return '\n'.join(lines[i:])
        
        return response.strip()

    def _simple_template_substitution(self, content_data: Dict, template_code: str) -> Dict:
        """✅ 실제 이미지 데이터 구조에 맞춘 템플릿 치환"""
        
        title = content_data.get("title", "제목 없음")
        subtitle = content_data.get("subtitle", "")
        content = content_data.get("content", "")
        images = content_data.get("images", [])
        
        # ✅ 실제 이미지 데이터 구조에 맞춘 이미지 JSX 생성
        image_jsx = ""
        if images:
            self.logger.info(f"이미지 {len(images)}개를 JSX에 포함")
            image_jsx = '''
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "16px", marginTop: "20px" }}>'''
            
            for i, img in enumerate(images[:5]):
                # ✅ 실제 키 이름 사용
                img_url = img.get("image_url", "")
                city = img.get("city", "")
                
                # 안전한 alt 텍스트
                if city:
                    safe_alt = f"{city} 여행 이미지 {i+1}"
                else:
                    safe_alt = f"베네치아 여행 이미지 {i+1}"
                
                # 특수문자 제거
                safe_alt = re.sub(r'["\']', '', safe_alt)
                
                if img_url:
                    image_jsx += f'''
            <div style={{ textAlign: "center" }}>
                <img 
                src="{img_url}" 
                alt="{safe_alt}" 
                style={{ width: "100%", height: "200px", objectFit: "cover", borderRadius: "8px" }}
                />
            </div>'''
            
            image_jsx += "\n        </div>"
        
        # 컴포넌트 이름 생성
        clean_title = re.sub(r'[^a-zA-Z0-9가-힣]', '', title.replace(' ', '').replace(':', ''))
        component_name = clean_title if clean_title else "DefaultSection"
        
        jsx_code = f"""import React from "react";

    const {component_name} = () => {{
    return (
        <div style={{ backgroundColor: "white", color: "black", padding: "20px" }}>
        <h1 style={{ fontSize: "2rem", marginBottom: "16px" }}>
            {title}
        </h1>
        {f'<h2 style={{ fontSize: "1.2rem", marginBottom: "12px", color: "#606060" }}>{subtitle}</h2>' if subtitle else ''}
        <div style={{ fontSize: "1rem", lineHeight: "1.7", marginBottom: "8px" }}>
            {content}
        </div>{image_jsx}
        </div>
    );
    }};

    export default {component_name};"""

        return {
            "title": title,
            "jsx_code": jsx_code
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
        
        try:
            return json.dumps(simplified_images, ensure_ascii=False)
        except:
            return "[]"
