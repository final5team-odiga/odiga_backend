import os
from typing import Dict, List
from crewai import Agent, Task, Crew
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager

class BindingAgent:
    """PDF 벡터 데이터 기반 이미지 배치 에이전트"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        self.vector_manager = PDFVectorManager()
        
    def create_image_layout_agent(self):
        """이미지 레이아웃 에이전트"""
        return Agent(
            role="Magazine Image Layout Specialist",
            goal="PDF 벡터 데이터를 분석하여 이미지 배치에 최적화된 매거진 레이아웃 설계",
            backstory="""당신은 매거진 이미지 레이아웃 전문가입니다.
            실제 매거진 PDF에서 추출한 벡터 데이터를 분석하여
            이미지의 특성과 개수에 맞는 최적의 레이아웃을 찾아내고,
            독자의 시선을 사로잡는 전략적인 이미지 배치를 설계하는 전문성을 가지고 있습니다.""",
            llm=self.llm,
            verbose=True
        )
    
    def create_visual_coordinator_agent(self):
        """비주얼 코디네이터 에이전트"""
        return Agent(
            role="Visual Content Coordinator",
            goal="벡터 데이터 기반 레이아웃 분석을 바탕으로 이미지들을 전략적으로 배치 및 조합",
            backstory="""당신은 비주얼 콘텐츠 코디네이터입니다.
            실제 매거진의 이미지 배치 패턴을 분석한 결과를 바탕으로
            주어진 이미지들을 최적의 조합으로 배치하며,
            전체적인 매거진의 시각적 흐름과 임팩트를 극대화하는 전문성을 가지고 있습니다.""",
            llm=self.llm,
            verbose=True
        )
    
    def process_images(self, image_urls: List[str], image_locations: List[str], template_requirements: List[Dict]) -> Dict:
        """PDF 벡터 데이터 기반 이미지 처리"""
        
        print(f"BindingAgent: 처리할 이미지 {len(image_urls)}개, 템플릿 {len(template_requirements)}개")
        
        # 에이전트 생성
        layout_specialist = self.create_image_layout_agent()
        visual_coordinator = self.create_visual_coordinator_agent()
        
        # 이미지 개수별 벡터 검색으로 최적 레이아웃 찾기
        layout_recommendations = self._get_layout_recommendations_by_image_count(
            image_urls, template_requirements
        )
        
        # 템플릿별 이미지 배치 설계
        template_distributions = []
        
        for i, template_req in enumerate(template_requirements):
            template_name = template_req["template"]
            
            # 해당 템플릿에 할당할 이미지들 결정
            assigned_images = self._assign_images_to_template(
                image_urls, image_locations, i, len(template_requirements)
            )
            
            if not assigned_images:
                template_distributions.append({
                    "template": template_name,
                    "images": [],
                    "layout_strategy": "no_images"
                })
                continue
            
            print(f"🖼️ {template_name}: {len(assigned_images)}개 이미지 배치 설계 중...")
            
            # 해당 이미지 수에 맞는 레이아웃 추천 가져오기
            relevant_layouts = [
                layout for layout in layout_recommendations 
                if len(layout.get('image_info', [])) == len(assigned_images)
            ]
            
            if not relevant_layouts and layout_recommendations:
                # 가장 유사한 이미지 수의 레이아웃 선택
                relevant_layouts = [min(layout_recommendations, 
                                      key=lambda x: abs(len(x.get('image_info', [])) - len(assigned_images)))]
            
            # 1단계: 레이아웃 분석
            layout_analysis_task = Task(
                description=f"""
                다음 이미지들과 매거진 레이아웃 데이터를 분석하여 최적의 이미지 배치 전략을 수립하세요:
                
                **배치할 이미지들:**
                {self._format_image_data(assigned_images, image_locations)}
                
                **참고할 매거진 레이아웃 데이터:**
                {self._format_layout_recommendations(relevant_layouts)}
                
                **템플릿 정보:**
                - 템플릿명: {template_name}
                - 이미지 요구사항: {template_req.get('image_requirements', {})}
                
                **분석 요구사항:**
                1. **레이아웃 패턴 분석**
                   - 이미지 배치의 그리드 구조 및 비율
                   - 주요 이미지와 보조 이미지의 역할 분담
                   - 이미지 간 시각적 균형과 흐름
                
                2. **이미지 특성 매칭**
                   - 각 이미지의 특성과 레이아웃 위치의 적합성
                   - 이미지 크기와 중요도에 따른 배치 우선순위
                   - 색감과 구도의 조화를 고려한 배치
                
                3. **시각적 임팩트 최적화**
                   - 독자의 시선 흐름을 고려한 이미지 순서
                   - 스토리텔링을 강화하는 이미지 조합
                   - 매거진 전체의 시각적 일관성 유지
                
                **출력 형식:**
                레이아웃 전략: [선택된 레이아웃 패턴과 특징]
                주요 이미지: [메인으로 사용할 이미지와 배치 위치]
                보조 이미지: [서브로 사용할 이미지들과 역할]
                배치 순서: [이미지들의 최적 배치 순서]
                시각 효과: [기대되는 시각적 효과와 임팩트]
                """,
                agent=layout_specialist,
                expected_output="벡터 데이터 기반 이미지 배치 전략"
            )
            
            # 2단계: 이미지 배치 실행
            image_coordination_task = Task(
                description=f"""
                레이아웃 분석 결과를 바탕으로 이미지들을 최적으로 배치하고 조합하세요:
                
                **배치 지침:**
                1. 분석된 레이아웃 패턴에 따른 정확한 이미지 배치
                2. 각 이미지의 특성을 살린 최적 위치 선정
                3. 전체적인 시각적 균형과 조화 고려
                4. 독자의 감정적 몰입을 위한 스토리텔링 강화
                5. 매거진 브랜드 일관성 유지
                
                **품질 요구사항:**
                - 실제 매거진에서 볼 수 있는 수준의 전문적 배치
                - 이미지 간 시너지 효과 극대화
                - 독자의 시선을 자연스럽게 유도하는 배치
                - 콘텐츠와 이미지의 완벽한 조화
                
                **출력:** 최종 이미지 배치 결과 (이미지 URL과 배치 정보)
                """,
                agent=visual_coordinator,
                expected_output="최적화된 이미지 배치 결과",
                context=[layout_analysis_task]
            )
            
            # Crew 실행
            crew = Crew(
                agents=[layout_specialist, visual_coordinator],
                tasks=[layout_analysis_task, image_coordination_task],
                verbose=True
            )
            
            try:
                result = crew.kickoff()
                
                # 결과 파싱
                layout_strategy = str(layout_analysis_task.output) if hasattr(layout_analysis_task, 'output') else ""
                coordination_result = str(result.raw) if hasattr(result, 'raw') else str(result)
                
                template_distributions.append({
                    "template": template_name,
                    "images": assigned_images,
                    "layout_strategy": layout_strategy,
                    "coordination_result": coordination_result,
                    "layout_source": relevant_layouts[0].get("pdf_name", "default") if relevant_layouts else "default"
                })
                
                print(f"✅ {template_name} 이미지 배치 완료: {len(assigned_images)}개")
                
            except Exception as e:
                print(f"⚠️ {template_name} 이미지 배치 실패: {e}")
                # 폴백: 기본 배치
                template_distributions.append({
                    "template": template_name,
                    "images": assigned_images,
                    "layout_strategy": "기본 배치",
                    "coordination_result": "기본 순서 배치",
                    "layout_source": "default"
                })
        
        # 최종 이미지 분배 결과 생성
        final_distribution = self._create_final_distribution(template_distributions)
        
        print(f"✅ BindingAgent 완료: {len(image_urls)}개 이미지를 {len(template_requirements)}개 템플릿에 배치")
        
        return {
            "image_distribution": final_distribution,
            "template_distributions": template_distributions,
            "layout_recommendations": layout_recommendations,
            "vector_enhanced": True
        }
    
    def _get_layout_recommendations_by_image_count(self, image_urls: List[str], template_requirements: List[Dict]) -> List[Dict]:
        """이미지 개수별 레이아웃 추천 가져오기"""
        
        total_images = len(image_urls)
        
        # 이미지 개수에 따른 검색 쿼리
        if total_images <= 3:
            query = "minimal clean layout single image focus simple elegant"
        elif total_images <= 6:
            query = "multiple images grid layout balanced composition"
        elif total_images <= 10:
            query = "gallery style layout many images organized grid"
        else:
            query = "complex magazine layout multiple images rich visual content"
        
        # 벡터 검색으로 유사한 레이아웃 찾기
        recommendations = self.vector_manager.search_similar_layouts(
            query, "magazine_layout", top_k=5
        )
        
        print(f"📊 이미지 {total_images}개에 대한 레이아웃 추천 {len(recommendations)}개 획득")
        
        return recommendations
    
    def _assign_images_to_template(self, image_urls: List[str], image_locations: List[str], 
                                 template_index: int, total_templates: int) -> List[str]:
        """템플릿에 이미지 할당"""
        
        if not image_urls:
            return []
        
        # 기본 균등 분배
        images_per_template = len(image_urls) // total_templates
        remainder = len(image_urls) % total_templates
        
        # 시작 인덱스 계산
        start_idx = template_index * images_per_template
        
        # 나머지가 있으면 앞쪽 템플릿에 더 많이 할당
        if template_index < remainder:
            start_idx += template_index
            end_idx = start_idx + images_per_template + 1
        else:
            start_idx += remainder
            end_idx = start_idx + images_per_template
        
        return image_urls[start_idx:end_idx]
    
    def _format_image_data(self, image_urls: List[str], image_locations: List[str]) -> str:
        """이미지 데이터를 텍스트로 포맷팅"""
        if not image_urls:
            return "배치할 이미지 없음"
        
        formatted_data = []
        for i, url in enumerate(image_urls):
            location = image_locations[i] if i < len(image_locations) else f"위치 {i+1}"
            formatted_data.append(f"이미지 {i+1}: {url} (위치: {location})")
        
        return "\n".join(formatted_data)
    
    def _format_layout_recommendations(self, recommendations: List[Dict]) -> str:
        """레이아웃 추천 데이터를 텍스트로 포맷팅"""
        if not recommendations:
            return "참고할 레이아웃 데이터 없음"
        
        formatted_data = []
        for i, rec in enumerate(recommendations):
            image_count = len(rec.get('image_info', []))
            formatted_data.append(f"""
            레이아웃 {i+1} (유사도: {rec.get('score', 0):.2f}):
            - 출처: {rec.get('pdf_name', 'unknown')} (페이지 {rec.get('page_number', 0)})
            - 이미지 수: {image_count}개
            - 레이아웃 특징: {self._analyze_layout_structure(rec.get('layout_info', {}))}
            - 텍스트 샘플: {rec.get('text_content', '')[:150]}...
            """)
        
        return "\n".join(formatted_data)
    
    def _analyze_layout_structure(self, layout_info: Dict) -> str:
        """레이아웃 구조 분석"""
        text_blocks = layout_info.get('text_blocks', [])
        images = layout_info.get('images', [])
        tables = layout_info.get('tables', [])
        
        structure_analysis = []
        
        if len(images) == 1:
            structure_analysis.append("단일 이미지 중심")
        elif len(images) <= 3:
            structure_analysis.append("소수 이미지 균형 배치")
        elif len(images) <= 6:
            structure_analysis.append("다중 이미지 그리드")
        else:
            structure_analysis.append("갤러리 스타일")
        
        if len(text_blocks) > 5:
            structure_analysis.append("텍스트 중심")
        elif len(text_blocks) <= 2:
            structure_analysis.append("이미지 중심")
        else:
            structure_analysis.append("텍스트-이미지 균형")
        
        if tables:
            structure_analysis.append("정보 테이블 포함")
        
        return ", ".join(structure_analysis) if structure_analysis else "기본 레이아웃"
    
    def _create_final_distribution(self, template_distributions: List[Dict]) -> Dict:
        """최종 이미지 분배 결과 생성"""
        final_distribution = {}
        
        for dist in template_distributions:
            template_name = dist["template"]
            images = dist["images"]
            
            final_distribution[template_name] = images
        
        return final_distribution
