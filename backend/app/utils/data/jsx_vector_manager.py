import re
from typing import List, Dict, Optional
from .pdf_vector_manager import PDFVectorManager

class JSXVectorManager:
    """
    JSX 컴포넌트 벡터 데이터 관리자 - PDFVectorManager 기반 JSX 특화 래퍼
    ✅ 기존 PDFVectorManager를 활용하면서 JSX 특화 기능 제공
    """
    
    def __init__(self, vector_manager: PDFVectorManager = None, isolation_enabled: bool = True):
        if vector_manager:
            self.pdf_vector_manager = vector_manager
        else:
            self.pdf_vector_manager = PDFVectorManager(
                isolation_enabled=isolation_enabled,
                default_index="jsx-component-vector-index"
            )
        
        # JSX 컴포넌트 카테고리 정의
        self.jsx_categories = {
            "image_focused": {
                "description": "이미지 중심 컴포넌트",
                "typical_image_count": [2, 3, 4, 5],
                "layout_methods": ["grid", "flexbox", "masonry"]
            },
            "text_focused": {
                "description": "텍스트 중심 컴포넌트", 
                "typical_image_count": [0, 1],
                "layout_methods": ["single_column", "multi_column"]
            },
            "mixed": {
                "description": "이미지와 텍스트 균형 컴포넌트",
                "typical_image_count": [1, 2, 3],
                "layout_methods": ["flexbox", "grid", "sidebar"]
            }
        }
        
        # 복잡도 레벨 정의
        self.complexity_levels = {
            "simple": {
                "description": "단순한 구조",
                "max_elements": 5,
                "layout_depth": 1
            },
            "moderate": {
                "description": "중간 복잡도",
                "max_elements": 10,
                "layout_depth": 2
            },
            "complex": {
                "description": "복잡한 구조",
                "max_elements": 20,
                "layout_depth": 3
            }
        }
        
        print("✅ JSXVectorManager 초기화 완료 (PDFVectorManager 기반)")

    def initialize_jsx_search_index(self) -> bool:
        """JSX 검색 인덱스 초기화 확인"""
        try:
            # jsx-component-vector-index 연결 테스트
            test_results = self.pdf_vector_manager.search_similar_layouts(
                "test jsx component", 
                "jsx-component-vector-index", 
                top_k=1
            )
            
            # 연결 성공 여부 확인 (결과가 있거나 오류가 없으면 성공)
            return True
            
        except Exception as e:
            print(f"❌ JSX 인덱스 초기화 실패: {e}")
            return False

    def search_jsx_components(self, query_text: str, category: str = None, 
                            image_count: int = None, complexity: str = None, 
                            top_k: int = 5) -> List[Dict]:
        """
        ✅ JSX 컴포넌트 특화 검색
        PDFVectorManager의 기본 검색을 활용하면서 JSX 특화 필터링 적용
        """
        try:
            # 1. 기본 벡터 검색 (PDFVectorManager 활용)
            base_query = self._enhance_jsx_query(query_text, category, complexity)
            
            raw_results = self.pdf_vector_manager.search_similar_layouts(
                query_text=base_query,
                index_name="jsx-component-vector-index",
                top_k=top_k * 3  # 필터링을 위해 더 많이 검색
            )
            
            # 2. JSX 특화 필터링 적용
            filtered_results = self._apply_jsx_filters(
                raw_results, category, image_count, complexity
            )
            
            # 3. JSX 특화 점수 조정
            scored_results = self._calculate_jsx_relevance_scores(
                filtered_results, query_text, category, image_count
            )
            
            # 4. 결과 정렬 및 반환
            sorted_results = sorted(scored_results, key=lambda x: x.get("jsx_relevance_score", 0), reverse=True)
            
            return sorted_results[:top_k]
            
        except Exception as e:
            print(f"❌ JSX 컴포넌트 검색 실패: {e}")
            return []

    def _enhance_jsx_query(self, base_query: str, category: str = None, complexity: str = None) -> str:
        """JSX 검색을 위한 쿼리 강화"""
        enhanced_parts = [base_query]
        
        # 카테고리 기반 키워드 추가
        if category and category in self.jsx_categories:
            category_info = self.jsx_categories[category]
            enhanced_parts.extend(category_info["layout_methods"])
            enhanced_parts.append(category_info["description"])
        
        # 복잡도 기반 키워드 추가
        if complexity and complexity in self.complexity_levels:
            complexity_info = self.complexity_levels[complexity]
            enhanced_parts.append(complexity_info["description"])
        
        # JSX 관련 키워드 추가
        enhanced_parts.extend(["react component", "jsx", "layout", "responsive"])
        
        return " ".join(enhanced_parts)

    def _apply_jsx_filters(self, results: List[Dict], category: str = None, 
                          image_count: int = None, complexity: str = None) -> List[Dict]:
        """JSX 특화 필터링 적용"""
        filtered = []
        
        for result in results:
            # 카테고리 필터
            if category:
                result_category = self._infer_category_from_result(result)
                if result_category != category:
                    continue
            
            # 이미지 수 필터
            if image_count is not None:
                result_image_count = result.get("image_count", 0)
                # 정확한 매칭 또는 ±1 범위 허용
                if abs(result_image_count - image_count) > 1:
                    continue
            
            # 복잡도 필터
            if complexity:
                result_complexity = self._infer_complexity_from_result(result)
                if result_complexity != complexity:
                    continue
            
            filtered.append(result)
        
        return filtered

    def _infer_category_from_result(self, result: Dict) -> str:
        """검색 결과에서 카테고리 추론"""
        image_count = result.get("image_count", 0)
        component_name = result.get("component_name", "").lower()
        jsx_code = result.get("jsx_code", "").lower()
        
        # 이미지 수 기반 추론
        if image_count == 0:
            return "text_focused"
        elif image_count >= 3:
            return "image_focused"
        else:
            # 컴포넌트 이름과 코드 분석으로 세분화
            if any(keyword in component_name for keyword in ["image", "gallery", "photo"]):
                return "image_focused"
            elif any(keyword in component_name for keyword in ["text", "article", "content"]):
                return "text_focused"
            else:
                return "mixed"

    def _infer_complexity_from_result(self, result: Dict) -> str:
        """검색 결과에서 복잡도 추론"""
        jsx_code = result.get("jsx_code", "")
        
        # JSX 코드 복잡도 분석
        div_count = jsx_code.count("<div")
        style_count = jsx_code.count("style=")
        component_count = jsx_code.count("<") - jsx_code.count("</")
        
        complexity_score = div_count + (style_count * 0.5) + (component_count * 0.3)
        
        if complexity_score < 5:
            return "simple"
        elif complexity_score < 15:
            return "moderate"
        else:
            return "complex"

    def _calculate_jsx_relevance_scores(self, results: List[Dict], query_text: str, 
                                      category: str = None, image_count: int = None) -> List[Dict]:
        """JSX 특화 관련성 점수 계산"""
        
        for result in results:
            base_score = result.get("score", 0.0)
            jsx_score = base_score
            
            # 카테고리 매칭 보너스
            if category:
                inferred_category = self._infer_category_from_result(result)
                if inferred_category == category:
                    jsx_score += 0.2
            
            # 이미지 수 매칭 보너스
            if image_count is not None:
                result_image_count = result.get("image_count", 0)
                if result_image_count == image_count:
                    jsx_score += 0.3
                elif abs(result_image_count - image_count) == 1:
                    jsx_score += 0.1
            
            # 컴포넌트 이름 관련성
            component_name = result.get("component_name", "").lower()
            query_words = query_text.lower().split()
            name_matches = sum(1 for word in query_words if word in component_name)
            jsx_score += (name_matches * 0.1)
            
            # JSX 코드 품질 점수
            jsx_code = result.get("jsx_code", "")
            if self._is_high_quality_jsx(jsx_code):
                jsx_score += 0.15
            
            result["jsx_relevance_score"] = jsx_score
        
        return results

    def _is_high_quality_jsx(self, jsx_code: str) -> bool:
        """JSX 코드 품질 평가"""
        quality_indicators = [
            "import React" in jsx_code,
            "export default" in jsx_code,
            "style={{" in jsx_code,
            len(jsx_code) > 200,  # 충분한 길이
            jsx_code.count("{") == jsx_code.count("}"),  # 괄호 균형
        ]
        
        return sum(quality_indicators) >= 3

    def get_jsx_recommendations(self, content_description: str, 
                              image_count: int = None, layout_preference: str = None) -> List[Dict]:
        """
        ✅ 콘텐츠 설명을 바탕으로 JSX 컴포넌트 추천
        """
        try:
            # 이미지 수에 따른 카테고리 결정
            if image_count is not None:
                if image_count == 0:
                    category = "text_focused"
                elif image_count <= 2:
                    category = "mixed"
                else:
                    category = "image_focused"
            else:
                category = None
            
            # 레이아웃 선호도를 쿼리에 포함
            search_query = f"{content_description} layout design component"
            if layout_preference:
                search_query += f" {layout_preference}"
            
            # JSX 컴포넌트 검색
            recommendations = self.search_jsx_components(
                query_text=search_query,
                category=category,
                image_count=image_count,
                top_k=5
            )
            
            # 추천 점수 추가
            for i, rec in enumerate(recommendations):
                rec["recommendation_rank"] = i + 1
                rec["recommendation_score"] = 1.0 - (i * 0.1)  # 순위에 따른 점수
            
            return recommendations
            
        except Exception as e:
            print(f"❌ JSX 추천 실패: {e}")
            return []

    def get_jsx_template_by_structure(self, layout_type: str, image_count: int, 
                                    complexity: str = "moderate") -> Optional[Dict]:
        """
        ✅ 구조적 요구사항에 맞는 JSX 템플릿 검색
        """
        try:
            # 구조적 쿼리 생성
            structure_query = f"{layout_type} layout {complexity} complexity {image_count} images"
            
            results = self.search_jsx_components(
                query_text=structure_query,
                image_count=image_count,
                complexity=complexity,
                top_k=1
            )
            
            return results[0] if results else None
            
        except Exception as e:
            print(f"❌ 구조적 JSX 템플릿 검색 실패: {e}")
            return None

    def analyze_jsx_component_structure(self, jsx_code: str) -> Dict:
        """
        ✅ JSX 컴포넌트 구조 분석
        """
        analysis = {
            "has_images": False,
            "image_count": 0,
            "layout_type": "unknown",
            "complexity": "simple",
            "component_elements": [],
            "style_properties": [],
            "responsive_design": False
        }
        
        try:
            # 이미지 분석
            img_tags = re.findall(r'<img[^>]*>', jsx_code, re.IGNORECASE)
            analysis["has_images"] = len(img_tags) > 0
            analysis["image_count"] = len(img_tags)
            
            # 레이아웃 타입 분석
            if "display: \"grid\"" in jsx_code or "gridTemplateColumns" in jsx_code:
                analysis["layout_type"] = "grid"
            elif "display: \"flex\"" in jsx_code or "flexDirection" in jsx_code:
                analysis["layout_type"] = "flex"
            else:
                analysis["layout_type"] = "standard"
            
            # 복잡도 분석
            analysis["complexity"] = self._infer_complexity_from_jsx(jsx_code)
            
            # 컴포넌트 요소 분석
            analysis["component_elements"] = self._extract_jsx_elements(jsx_code)
            
            # 스타일 속성 분석
            analysis["style_properties"] = self._extract_style_properties(jsx_code)
            
            # 반응형 디자인 확인
            analysis["responsive_design"] = self._check_responsive_design(jsx_code)
            
        except Exception as e:
            print(f"❌ JSX 구조 분석 실패: {e}")
        
        return analysis

    def _infer_complexity_from_jsx(self, jsx_code: str) -> str:
        """JSX 코드에서 복잡도 추론"""
        element_count = jsx_code.count("<") - jsx_code.count("</")
        style_count = jsx_code.count("style=")
        nesting_level = jsx_code.count("  <div") + jsx_code.count("    <div")
        
        complexity_score = element_count + (style_count * 0.5) + (nesting_level * 0.3)
        
        if complexity_score < 5:
            return "simple"
        elif complexity_score < 15:
            return "moderate"
        else:
            return "complex"

    def _extract_jsx_elements(self, jsx_code: str) -> List[str]:
        """JSX 코드에서 사용된 요소들 추출"""
        elements = re.findall(r'<(\w+)', jsx_code)
        return list(set(elements))

    def _extract_style_properties(self, jsx_code: str) -> List[str]:
        """JSX 코드에서 스타일 속성들 추출"""
        style_matches = re.findall(r'(\w+):\s*["\'][^"\']*["\']', jsx_code)
        return list(set(style_matches))

    def _check_responsive_design(self, jsx_code: str) -> bool:
        """반응형 디자인 요소 확인"""
        responsive_indicators = [
            "auto-fit" in jsx_code,
            "minmax" in jsx_code,
            "responsive" in jsx_code.lower(),
            "@media" in jsx_code,
            "100%" in jsx_code
        ]
        
        return sum(responsive_indicators) >= 2

    def get_jsx_statistics(self) -> Dict:
        """JSX 인덱스 통계 정보 반환"""
        try:
            # PDFVectorManager의 통계 기능 활용
            base_stats = self.pdf_vector_manager.get_index_statistics()
            jsx_stats = base_stats.get("jsx-component-vector-index", {})
            
            # JSX 특화 통계 추가
            jsx_enhanced_stats = {
                **jsx_stats,
                "supported_categories": list(self.jsx_categories.keys()),
                "complexity_levels": list(self.complexity_levels.keys()),
                "jsx_manager_version": "1.0.0"
            }
            
            return jsx_enhanced_stats
            
        except Exception as e:
            print(f"❌ JSX 통계 조회 실패: {e}")
            return {}

    def test_jsx_search_functionality(self) -> Dict[str, bool]:
        """JSX 검색 기능 테스트"""
        test_results = {}
        
        test_cases = [
            {
                "name": "basic_search",
                "query": "react component layout",
                "expected_results": True
            },
            {
                "name": "category_filter",
                "query": "image gallery",
                "category": "image_focused",
                "expected_results": True
            },
            {
                "name": "image_count_filter",
                "query": "component layout",
                "image_count": 2,
                "expected_results": True
            },
            {
                "name": "recommendation_system",
                "content_description": "travel magazine article",
                "image_count": 3,
                "expected_results": True
            }
        ]
        
        for test_case in test_cases:
            try:
                if test_case["name"] == "recommendation_system":
                    results = self.get_jsx_recommendations(
                        test_case["content_description"],
                        test_case.get("image_count")
                    )
                else:
                    results = self.search_jsx_components(
                        test_case["query"],
                        test_case.get("category"),
                        test_case.get("image_count"),
                        top_k=1
                    )
                
                test_results[test_case["name"]] = len(results) > 0
                print(f"✅ JSX 테스트 '{test_case['name']}': {'성공' if test_results[test_case['name']] else '실패'}")
                
            except Exception as e:
                test_results[test_case["name"]] = False
                print(f"❌ JSX 테스트 '{test_case['name']}' 실패: {e}")
        
        return test_results
