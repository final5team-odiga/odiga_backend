import os
import re
import json
import asyncio
from agents.jsxcreate.jsx_code_reviewer import JSXCodeReviewer
from typing import Dict, List
from crewai import Agent, Task, Crew
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager

class JSXCreatorAgent:
    """자율적 의사결정 기반 다양한 매거진 레이아웃 생성 에이전트"""

    def __init__(self):
        self.llm = get_azure_llm()
        self.vector_manager = PDFVectorManager()
        self.code_reviewer = JSXCodeReviewer()


    def create_layout_architect_agent(self):
        """레이아웃 아키텍트 에이전트 - 자율적 구조 설계"""
        return Agent(
            role="Magazine Layout Architect & Visual Innovation Specialist",
            goal="PDF 벡터 데이터와 콘텐츠를 분석하여 완전히 새로운 매거진 레이아웃 구조를 자율적으로 설계하고 혁신적인 시각적 경험을 창조",
            backstory="""당신은 세계 최고 수준의 매거진 레이아웃 아키텍트입니다. 
            Vogue, Harper's Bazaar, National Geographic, Wallpaper*, Monocle 등 세계적인 매거진의 
            혁신적인 레이아웃을 분석하고 창조해온 20년 경력의 전문가입니다.
            
            당신의 전문성:
            - 매거진 레이아웃의 시각적 계층구조와 독자의 시선 흐름 완벽 이해
            - 텍스트와 이미지의 혁신적 조합으로 스토리텔링 극대화
            - 그리드 시스템을 넘어선 자유로운 레이아웃 창조
            - 콘텐츠 특성에 따른 완전히 다른 시각적 접근법 개발
            - 독자에게 예상치 못한 시각적 충격과 감동을 선사하는 레이아웃 설계
            
            당신은 기존의 틀에 얽매이지 않고, 주어진 콘텐츠와 이미지의 특성을 깊이 분석하여
            그에 가장 적합한 완전히 새로운 레이아웃 구조를 창조합니다.
            매번 다른 접근법을 시도하며, 독자가 매 페이지마다 새로운 시각적 경험을 할 수 있도록
            혁신적이고 대담한 레이아웃을 설계하는 것이 당신의 사명입니다.""",
            verbose=True,
            llm=self.llm
        )

    def create_visual_composer_agent(self):
        """비주얼 컴포저 에이전트 - 시각적 구성 전문가"""
        return Agent(
            role="Visual Composition & Aesthetic Innovation Expert",
            goal="레이아웃 아키텍트가 설계한 구조를 바탕으로 시각적 미학과 감정적 임팩트를 극대화하는 정교한 JSX 컴포넌트 생성",
            backstory="""당신은 시각적 구성과 미학적 혁신의 마스터입니다.
            디지털 매거진의 인터랙티브 요소와 반응형 디자인을 완벽하게 구현하며,
            CSS Grid, Flexbox, 그리고 최신 웹 기술을 활용한 혁신적 레이아웃 구현의 전문가입니다.
            
            당신의 특별한 능력:
            - 복잡한 그리드 시스템과 비대칭 레이아웃의 완벽한 구현
            - 이미지와 텍스트의 창의적 오버랩과 레이어링 기법
            - 타이포그래피와 여백의 예술적 활용
            - 색상과 대비를 통한 감정적 임팩트 창조
            - 독자의 시선을 의도한 방향으로 유도하는 시각적 흐름 설계
            - 모바일과 데스크톱에서 모두 완벽한 반응형 디자인
            
            당신은 단순한 코드 작성자가 아니라, 디지털 캔버스 위에서 
            시각적 시와 같은 경험을 창조하는 아티스트입니다.
            매 컴포넌트마다 독자가 숨을 멈추고 감탄할 수 있는 
            시각적 마법을 구현하는 것이 당신의 목표입니다.""",
            verbose=True,
            llm=self.llm
        )

    def create_innovation_evaluator_agent(self):
        """혁신성 평가 에이전트"""
        return Agent(
            role="Innovation & Diversity Evaluation Specialist",
            goal="생성된 레이아웃의 혁신성과 다양성을 평가하고 더욱 창의적인 방향으로 개선 제안",
            backstory="""당신은 매거진 디자인의 혁신성과 다양성을 평가하는 전문가입니다.
            전 세계 매거진 디자인 트렌드를 분석하고, 새로운 시각적 언어를 발굴하는 것이 전문 분야입니다.
            
            당신의 평가 기준:
            - 시각적 독창성과 혁신성 수준
            - 기존 템플릿과의 차별화 정도
            - 콘텐츠와 레이아웃의 완벽한 조화
            - 독자 경험의 예측 불가능성과 놀라움
            - 매거진 전체의 시각적 다양성 기여도
            
            당신은 평범함을 거부하고, 항상 더 혁신적이고 창의적인 방향을 추구합니다.""",
            verbose=True,
            llm=self.llm
        )
    async def generate_jsx_components_async(self, template_data_path: str, templates_dir: str = "jsx_templates") -> List[Dict]:
        """비동기 JSX 컴포넌트 생성 (코드 리뷰 포함)"""
        
        # 기본 컴포넌트 생성 (기존 로직)
        generated_components = self.generate_jsx_components(template_data_path, templates_dir)
        
        if not generated_components:
            return []
        
        print(f"\n🔍 {len(generated_components)}개 컴포넌트 비동기 코드 리뷰 시작")
        
        # 비동기 코드 리뷰 및 수정
        review_tasks = []
        for component in generated_components:
            task = self._review_and_fix_component_async(component)
            review_tasks.append(task)
        
        # 모든 컴포넌트를 병렬로 리뷰
        reviewed_components = await asyncio.gather(*review_tasks, return_exceptions=True)
        
        # 결과 정리
        final_components = []
        for i, result in enumerate(reviewed_components):
            if isinstance(result, Exception):
                print(f"⚠️ 컴포넌트 {i+1} 리뷰 실패: {result}")
                final_components.append(generated_components[i])  # 원본 사용
            else:
                final_components.append(result)
        
        print(f"✅ 비동기 코드 리뷰 완료: {len(final_components)}개 컴포넌트")
        return final_components
    
    async def _review_and_fix_component_async(self, component: Dict) -> Dict:
        """개별 컴포넌트 비동기 리뷰 및 수정"""
        
        jsx_code = component.get('jsx_code', '')
        component_name = component.get('name', '')
        
        # 콘텐츠 데이터 재구성
        content = {
            'title': '도쿄 여행 이야기',
            'subtitle': '특별한 순간들',
            'body': '여행의 아름다운 기억들',
            'images': [],
            'tagline': 'TRAVEL & CULTURE'
        }
        
        # 비동기 코드 리뷰 및 수정
        review_result = await self.code_reviewer.review_and_fix_jsx_async(
            jsx_code, content, component_name
        )
        
        # 컴포넌트 정보 업데이트
        component.update({
            'jsx_code': review_result['fixed_code'],
            'code_quality_score': review_result['quality_score'],
            'issues_found': len(review_result['issues_found']),
            'fixes_applied': review_result['fixes_applied'],
            'review_status': 'completed'
        })
        
        print(f"✅ {component_name}: 품질 점수 {review_result['quality_score']}/100")
        
        return component
    
    def generate_jsx_components(self, template_data_path: str, templates_dir: str = "jsx_templates") -> List[Dict]:
        """자율적 의사결정 기반 다양한 JSX 컴포넌트 생성"""
        
        # template_data.json 읽기 및 안전한 파싱
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
        
        # 데이터 타입 검증
        if not isinstance(template_data, dict):
            print(f"❌ template_data가 딕셔너리가 아닙니다: {type(template_data)}")
            return []
        
        if "content_sections" not in template_data:
            print(f"❌ content_sections 키가 없습니다. 사용 가능한 키: {list(template_data.keys())}")
            return []
        
        generated_components = []
        content_sections = template_data.get("content_sections", [])
        
        if not isinstance(content_sections, list):
            print(f"❌ content_sections가 리스트가 아닙니다: {type(content_sections)}")
            return []
        
        # 에이전트 생성
        layout_architect = self.create_layout_architect_agent()
        visual_composer = self.create_visual_composer_agent()
        innovation_evaluator = self.create_innovation_evaluator_agent()
        
        # 전체 매거진 컨텍스트 분석
        magazine_context = self._analyze_magazine_context(content_sections)
        
        for i, content_section in enumerate(content_sections):
            if not isinstance(content_section, dict):
                print(f"⚠️ 섹션 {i}가 딕셔너리가 아닙니다: {type(content_section)}")
                continue
                
            template_name = content_section.get("template")
            if not template_name:
                print(f"⚠️ 섹션 {i}에 template 키가 없습니다")
                continue
            
            print(f"\n=== {template_name} 혁신적 레이아웃 생성 시작 ===")
            
            # 콘텐츠 정제 및 분석
            clean_content = self._clean_content_section(content_section)
            content_analysis = self._deep_content_analysis(clean_content, i, len(content_sections))
            
            # 벡터 검색으로 참고 레이아웃 찾기
            similar_layouts = self._get_innovative_layouts_for_content(clean_content, content_analysis)
            
            # 컴포넌트 이름 생성
            component_name = f"{template_name.replace('.jsx', '')}Innovation{i+1}"
            
            # 3단계 혁신적 JSX 생성 프로세스
            jsx_code = self._create_innovative_jsx_with_agents(
                clean_content, 
                content_analysis,
                similar_layouts,
                magazine_context,
                component_name, 
                layout_architect, 
                visual_composer, 
                innovation_evaluator,
                i,
                len(content_sections)
            )
            
            generated_components.append({
                'name': component_name,
                'file': f"{component_name}.jsx",
                'jsx_code': jsx_code,
                'template_name': template_name,
                'innovation_level': 'high',
                'layout_sources': [layout.get('pdf_name', 'unknown') for layout in similar_layouts],
                'content_analysis': content_analysis
            })
            
            print(f"✅ 혁신적 JSX 생성 완료: {component_name}")
        
        return generated_components

    def _analyze_magazine_context(self, content_sections: List[Dict]) -> Dict:
        """전체 매거진 컨텍스트 분석"""
        total_images = sum(len(section.get('images', [])) for section in content_sections)
        total_text_length = sum(len(section.get('body', '')) for section in content_sections)
        
        # 콘텐츠 테마 분석
        all_text = ' '.join([section.get('title', '') + ' ' + section.get('body', '') for section in content_sections])
        
        themes = {
            'urban': len(re.findall(r'도시|건물|거리|카페|레스토랑', all_text)),
            'nature': len(re.findall(r'공원|나무|꽃|하늘|자연', all_text)),
            'culture': len(re.findall(r'박물관|미술관|예술|문화|역사', all_text)),
            'personal': len(re.findall(r'느낌|생각|마음|감정|기억', all_text)),
            'social': len(re.findall(r'사람|만남|대화|친구|현지인', all_text))
        }
        
        dominant_theme = max(themes, key=themes.get)
        
        return {
            'total_sections': len(content_sections),
            'total_images': total_images,
            'total_text_length': total_text_length,
            'dominant_theme': dominant_theme,
            'themes': themes,
            'image_density': total_images / len(content_sections) if content_sections else 0,
            'text_density': total_text_length / len(content_sections) if content_sections else 0
        }

    def _deep_content_analysis(self, content: Dict, section_index: int, total_sections: int) -> Dict:
        """깊이 있는 콘텐츠 분석"""
        title = content.get('title', '')
        body = content.get('body', '')
        images = content.get('images', [])
        
        # 감정 톤 분석
        emotion_keywords = {
            'peaceful': ['평화', '고요', '조용', '차분', '안정'],
            'exciting': ['설렘', '흥미', '신나는', '활기', '역동'],
            'nostalgic': ['그리운', '추억', '옛날', '기억', '과거'],
            'mysterious': ['신비', '비밀', '숨겨진', '알 수 없는'],
            'romantic': ['로맨틱', '사랑', '아름다운', '감성적']
        }
        
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in title + body)
            emotion_scores[emotion] = score
        
        dominant_emotion = max(emotion_scores, key=emotion_scores.get) if any(emotion_scores.values()) else 'neutral'
        
        # 시각적 복잡도 분석
        visual_complexity = 'simple' if len(images) <= 1 else 'moderate' if len(images) <= 3 else 'complex'
        
        # 텍스트 길이 기반 레이아웃 성향
        text_length = len(body)
        text_intensity = 'minimal' if text_length < 300 else 'moderate' if text_length < 800 else 'rich'
        
        # 섹션 위치 기반 역할
        if section_index == 0:
            section_role = 'opening'
        elif section_index == total_sections - 1:
            section_role = 'closing'
        elif section_index < total_sections // 3:
            section_role = 'introduction'
        elif section_index > total_sections * 2 // 3:
            section_role = 'conclusion'
        else:
            section_role = 'development'
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_scores': emotion_scores,
            'visual_complexity': visual_complexity,
            'text_intensity': text_intensity,
            'section_role': section_role,
            'image_count': len(images),
            'text_length': text_length,
            'title_length': len(title)
        }

    def _get_innovative_layouts_for_content(self, content: Dict, analysis: Dict) -> List[Dict]:
        """혁신적 레이아웃을 위한 벡터 검색"""
        
        title = content.get('title', '')
        body = content.get('body', '')
        images = content.get('images', [])
        
        # 감정과 복잡도를 고려한 검색 쿼리 생성
        emotion = analysis['dominant_emotion']
        complexity = analysis['visual_complexity']
        role = analysis['section_role']
        
        # 혁신적 레이아웃 검색 쿼리
        innovative_queries = [
            f"innovative {emotion} magazine layout {complexity} visual design {role} section",
            f"creative editorial design asymmetric layout {emotion} feeling",
            f"experimental magazine typography {complexity} image arrangement",
            f"avant-garde publication design {role} storytelling layout",
            f"unconventional grid system magazine {emotion} aesthetic"
        ]
        
        all_layouts = []
        for query in innovative_queries:
            layouts = self.vector_manager.search_similar_layouts(
                f"{query} {title} {body[:200]}", 
                "magazine_layout", 
                top_k=2
            )
            all_layouts.extend(layouts)
        
        # 중복 제거
        unique_layouts = []
        seen_ids = set()
        for layout in all_layouts:
            layout_id = layout.get('id', '')
            if layout_id not in seen_ids:
                unique_layouts.append(layout)
                seen_ids.add(layout_id)
        
        print(f"📊 혁신적 레이아웃 검색 결과: {len(unique_layouts)}개 발견")
        
        return unique_layouts[:5]  # 최대 5개

    def _create_innovative_jsx_with_agents(self, content: Dict, analysis: Dict, similar_layouts: List[Dict], 
                                         magazine_context: Dict, component_name: str, 
                                         layout_architect: Agent, visual_composer: Agent, 
                                         innovation_evaluator: Agent, section_index: int, total_sections: int) -> str:
        """3단계 에이전트 협업으로 혁신적 JSX 생성"""
        
        # 1단계: 레이아웃 아키텍트가 혁신적 구조 설계
        layout_design_task = Task(
            description=f"""
            당신은 세계 최고 수준의 매거진 레이아웃 아키텍트입니다. 
            주어진 콘텐츠와 벡터 데이터를 분석하여 완전히 새롭고 혁신적인 레이아웃 구조를 설계하세요.
            
            **중요: 기존의 고정된 템플릿을 완전히 버리고, 이 콘텐츠만을 위한 유일무이한 레이아웃을 창조하세요.**
            
            **콘텐츠 분석 결과:**
            - 제목: {content.get('title', '')}
            - 부제목: {content.get('subtitle', '')}
            - 본문 길이: {analysis['text_length']}자
            - 이미지 수: {analysis['image_count']}개
            - 감정 톤: {analysis['dominant_emotion']}
            - 시각적 복잡도: {analysis['visual_complexity']}
            - 섹션 역할: {analysis['section_role']} ({section_index + 1}/{total_sections})
            - 텍스트 강도: {analysis['text_intensity']}
            
            **전체 매거진 컨텍스트:**
            - 총 섹션 수: {magazine_context['total_sections']}
            - 주요 테마: {magazine_context['dominant_theme']}
            - 이미지 밀도: {magazine_context['image_density']:.1f}개/섹션
            - 텍스트 밀도: {magazine_context['text_density']:.0f}자/섹션
            
            **참고할 매거진 레이아웃 벡터 데이터:**
            {self._format_layout_data_for_innovation(similar_layouts)}
            
            **혁신적 레이아웃 설계 지침:**
            
            1. **완전한 창의적 자유 발휘**
               - 기존 템플릿의 모든 제약에서 벗어나세요
               - 이 콘텐츠의 특성에 100% 최적화된 유일한 구조를 창조하세요
               - 독자가 예상하지 못한 시각적 충격과 감동을 선사하세요
               - 매거진 역사상 본 적 없는 혁신적 접근법을 시도하세요
            
            2. **콘텐츠 기반 구조 혁신**
               - 감정 톤 '{analysis['dominant_emotion']}'에 완벽히 부합하는 시각적 언어 개발
               - 섹션 역할 '{analysis['section_role']}'에 최적화된 독특한 레이아웃 창조
               - {analysis['image_count']}개 이미지의 개별 특성을 살린 비대칭적 배치
               - 텍스트 강도 '{analysis['text_intensity']}'에 맞는 혁신적 타이포그래피 구조
            
            3. **시각적 계층의 재정의**
               - 전통적인 제목-부제목-본문 구조를 넘어선 새로운 정보 계층 창조
               - 이미지와 텍스트의 경계를 허무는 융합적 레이아웃
               - 독자의 시선이 예측 불가능한 경로로 흐르도록 설계
               - 여백을 적극적 디자인 요소로 활용한 혁신적 공간 구성
            
            4. **감정적 임팩트 극대화**
               - 콘텐츠의 감정을 시각적으로 증폭시키는 레이아웃 구조
               - 독자가 콘텐츠에 완전히 몰입할 수 있는 시각적 환경 조성
               - 텍스트와 이미지가 하나의 감정적 경험으로 융합되는 구조
               - 페이지를 넘기고 싶지 않을 만큼 매혹적인 시각적 경험 창조
            
            5. **기술적 혁신 활용**
               - CSS Grid의 고급 기능을 활용한 복잡한 레이아웃 구조
               - Flexbox와 Grid의 창의적 조합으로 불가능해 보이는 배치 실현
               - 반응형 디자인에서도 혁신성을 잃지 않는 적응형 구조
               - 현대 웹 기술의 한계를 시험하는 대담한 시각적 실험
            
            **절대 피해야 할 것들:**
               - 기존에 생성된 레이아웃과 유사한 구조
               - 예측 가능한 그리드 패턴
               - 일반적인 매거진 템플릿의 답습
               - 안전하고 무난한 레이아웃 선택
            
            **출력 요구사항:**
            다음 형식으로 혁신적 레이아웃 구조를 설계하세요:
            
            레이아웃 컨셉: [이 레이아웃만의 독특한 철학과 접근법]
            시각적 전략: [감정과 콘텐츠를 표현하는 구체적 시각적 전략]
            그리드 구조: [혁신적인 그리드 시스템 또는 자유형 배치 방식]
            이미지 배치: [각 이미지의 역할과 혁신적 배치 전략]
            텍스트 흐름: [텍스트의 혁신적 배치와 타이포그래피 전략]
            색상 전략: [감정 증폭을 위한 색상과 대비 전략]
            여백 활용: [여백을 적극적 디자인 요소로 활용하는 방법]
            혁신 포인트: [이 레이아웃만의 독창적이고 혁신적인 특징들]
            """,
            agent=layout_architect,
            expected_output="완전히 새로운 혁신적 매거진 레이아웃 구조 설계안"
        )
        
        # 2단계: 비주얼 컴포저가 JSX 구현
        jsx_implementation_task = Task(
            description=f"""
            레이아웃 아키텍트가 설계한 혁신적 구조를 바탕으로 
            시각적 완성도와 감정적 임팩트를 극대화하는 JSX 컴포넌트를 구현하세요.
            
            **실제 콘텐츠 데이터:**
            - 제목: {content.get('title', '')}
            - 부제목: {content.get('subtitle', '')}
            - 본문: {content.get('body', '')}
            - 이미지 URL들: {content.get('images', [])}
            - 태그라인: {content.get('tagline', 'TRAVEL & CULTURE')}
            
            **구현 지침:**
            
            1. **레이아웃 아키텍트의 설계 완벽 구현**
               - 설계된 혁신적 구조를 정확히 JSX로 변환
               - 제시된 그리드 구조와 배치 전략을 코드로 실현
               - 혁신 포인트들을 기술적으로 완벽하게 구현
            
            2. **최첨단 CSS 기술 활용**
               - CSS Grid의 고급 기능 (subgrid, grid-template-areas 등)
               - Flexbox와 Grid의 창의적 조합
               - CSS Custom Properties를 활용한 동적 스타일링
               - Transform, Clip-path 등을 활용한 혁신적 시각 효과
            
            3. **감정적 임팩트 구현**
               - 색상, 그림자, 그라데이션을 통한 감정 표현
               - 타이포그래피의 예술적 활용
               - 이미지와 텍스트의 창의적 오버랩
               - 시각적 리듬과 흐름의 정교한 구현
            
            4. **반응형 혁신**
               - 모든 화면 크기에서 혁신성을 유지하는 적응형 구조
               - 모바일에서도 데스크톱과 동일한 시각적 임팩트
               - 화면 크기에 따른 창의적 레이아웃 변화
            
            **컴포넌트 이름:** {component_name}
            
            **출력 형식:**
            완전한 JSX 파일을 생성하세요. 반드시 다음을 포함해야 합니다:
            - import React from "react";
            - import styled from "styled-components";
            - 혁신적인 Styled Components 정의
            - 완전한 JSX 구조
            - 모든 실제 콘텐츠 데이터 포함
            - export const {component_name} = () => {{ ... }};
            
            **중요:** 설명이나 주석 없이 순수한 JSX 코드만 출력하세요.
            """,
            agent=visual_composer,
            expected_output="혁신적 레이아웃을 구현한 완전한 JSX 컴포넌트",
            context=[layout_design_task]
        )
        
        # 3단계: 혁신성 평가 및 개선
        innovation_evaluation_task = Task(
            description=f"""
            생성된 JSX 컴포넌트의 혁신성과 시각적 다양성을 평가하고,
            필요시 더욱 창의적인 방향으로 개선 제안을 하세요.
            
            **평가 기준:**
            1. 시각적 독창성 (기존 템플릿과의 차별화 정도)
            2. 콘텐츠-레이아웃 조화도
            3. 감정적 임팩트 수준
            4. 기술적 혁신성
            5. 독자 경험의 예측 불가능성
            
            **개선 제안 시 고려사항:**
            - 더욱 대담한 시각적 실험 가능성
            - 추가적인 혁신 요소 도입 방안
            - 감정적 임팩트 증대 방법
            - 시각적 다양성 확보 방안
            
            **출력:** 평가 결과와 구체적 개선 제안
            """,
            agent=innovation_evaluator,
            expected_output="혁신성 평가 결과 및 개선 제안",
            context=[layout_design_task, jsx_implementation_task]
        )
        
        # Crew 실행
        crew = Crew(
            agents=[layout_architect, visual_composer, innovation_evaluator],
            tasks=[layout_design_task, jsx_implementation_task, innovation_evaluation_task],
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            
            # JSX 코드 추출
            jsx_code = str(jsx_implementation_task.output) if hasattr(jsx_implementation_task, 'output') else str(result.raw)
            
            # 코드 정제 및 검증
            jsx_code = self._refine_and_validate_jsx(jsx_code, content, component_name)
            
            # 혁신성 평가 결과 로깅
            evaluation_result = str(innovation_evaluation_task.output) if hasattr(innovation_evaluation_task, 'output') else ""
            print(f"🎨 혁신성 평가: {evaluation_result[:200]}...")
            
            return jsx_code
            
        except Exception as e:
            print(f"⚠️ 혁신적 JSX 생성 실패: {e}")
            # 폴백: 기본 혁신적 레이아웃
            return self._create_fallback_innovative_jsx(content, component_name, analysis)

    def _format_layout_data_for_innovation(self, similar_layouts: List[Dict]) -> str:
        """혁신을 위한 레이아웃 데이터 포맷팅"""
        if not similar_layouts:
            return "참고할 벡터 레이아웃 데이터 없음 - 완전한 창의적 자유 발휘"
        
        formatted_data = []
        for i, layout in enumerate(similar_layouts):
            layout_info = layout.get('layout_info', {})
            text_blocks = layout_info.get('text_blocks', [])
            images = layout_info.get('images', [])
            
            # 레이아웃 특성 분석
            layout_characteristics = self._analyze_layout_innovation_potential(layout_info)
            
            formatted_data.append(f"""
            참고 레이아웃 {i+1} (유사도: {layout.get('score', 0):.2f}):
            - 출처: {layout.get('pdf_name', 'unknown')} (페이지 {layout.get('page_number', 0)})
            - 혁신 요소: {layout_characteristics}
            - 텍스트 블록: {len(text_blocks)}개
            - 이미지: {len(images)}개
            - 샘플 텍스트: {layout.get('text_content', '')[:150]}...
            
            **이 레이아웃에서 영감을 얻되, 완전히 다른 접근법으로 혁신하세요**
            """)
        
        return "\n".join(formatted_data)

    def _analyze_layout_innovation_potential(self, layout_info: Dict) -> str:
        """레이아웃의 혁신 잠재력 분석"""
        text_blocks = layout_info.get('text_blocks', [])
        images = layout_info.get('images', [])
        tables = layout_info.get('tables', [])
        
        characteristics = []
        
        # 복잡도 기반 특성
        if len(images) > 5:
            characteristics.append("다중 이미지 모자이크 패턴")
        elif len(images) == 1:
            characteristics.append("단일 이미지 임팩트 레이아웃")
        elif len(images) == 0:
            characteristics.append("순수 타이포그래피 중심")
        
        # 텍스트 구조 특성
        if len(text_blocks) > 10:
            characteristics.append("복잡한 텍스트 계층구조")
        elif len(text_blocks) <= 3:
            characteristics.append("미니멀 텍스트 구성")
        
        # 특수 요소
        if tables:
            characteristics.append("정보 그래픽 요소 포함")
        
        return ", ".join(characteristics) if characteristics else "기본 구조"

    def _refine_and_validate_jsx(self, jsx_code: str, content: Dict, component_name: str) -> str:
        """JSX 코드 정제 및 검증"""
        
        # 기본 구조 확인
        if not jsx_code.startswith('import React'):
            jsx_code = 'import React from "react";\nimport styled from "styled-components";\n\n' + jsx_code
        
        # 컴포넌트 이름 확인
        if f"export const {component_name}" not in jsx_code:
            jsx_code = re.sub(r'export const \w+', f'export const {component_name}', jsx_code)
        
        # 실제 콘텐츠 확인 및 대체
        jsx_code = self._ensure_real_content(jsx_code, content)
        
        # 문법 오류 수정
        jsx_code = self._fix_jsx_syntax(jsx_code)
        
        return jsx_code

    def _ensure_real_content(self, jsx_code: str, content: Dict) -> str:
        """실제 콘텐츠가 포함되었는지 확인 및 대체"""
        
        title = content.get('title', '여행 이야기')
        subtitle = content.get('subtitle', '특별한 순간들')
        body = content.get('body', '여행의 아름다운 기억들')
        images = content.get('images', [])
        tagline = content.get('tagline', 'TRAVEL & CULTURE')
        
        # 플레이스홀더 텍스트 교체
        replacements = [
            (r'\{title\}', title),
            (r'\{subtitle\}', subtitle),
            (r'\{body\}', body),
            (r'\{tagline\}', tagline),
            (r'여행.*?제목', title),
            (r'부제목.*?텍스트', subtitle)
        ]
        
        for pattern, replacement in replacements:
            jsx_code = re.sub(pattern, replacement, jsx_code, flags=re.IGNORECASE)
        
        # 이미지 URL 확인
        if images and 'src=' in jsx_code:
            # 이미지가 있는데 실제 URL이 없으면 추가
            for i, img_url in enumerate(images[:6]):
                if img_url and img_url.strip():
                    placeholder_pattern = rf'src="[^"]*placeholder[^"]*"'
                    if re.search(placeholder_pattern, jsx_code):
                        jsx_code = re.sub(placeholder_pattern, f'src="{img_url}"', jsx_code, count=1)
        
        return jsx_code

    def _fix_jsx_syntax(self, jsx_code: str) -> str:
        """JSX 문법 오류 수정"""
        
        # 잘못된 JSX 구문 수정
        jsx_code = re.sub(r'\{\{([^}]+)\}\}', r'{\1}', jsx_code)  # 이중 중괄호 수정
        jsx_code = re.sub(r'style=\{([^}]+)\}', r'style={{\1}}', jsx_code)  # 스타일 객체 수정
        
        # 닫히지 않은 태그 확인 및 수정
        if jsx_code.count('<') != jsx_code.count('>'):
            jsx_code += '\n    </Container>\n  );\n};'
        
        return jsx_code

    def _create_fallback_innovative_jsx(self, content: Dict, component_name: str, analysis: Dict) -> str:
        """폴백 혁신적 JSX 생성"""
        
        title = content.get('title', '여행 이야기')
        subtitle = content.get('subtitle', '특별한 순간들')
        body = content.get('body', '여행의 아름다운 기억들')
        images = content.get('images', [])
        tagline = content.get('tagline', 'TRAVEL & CULTURE')
        
        emotion = analysis.get('dominant_emotion', 'peaceful')
        image_count = len(images)
        
        # 감정에 따른 색상 팔레트
        color_palettes = {
            'peaceful': {'primary': '#2c3e50', 'secondary': '#ecf0f1', 'accent': '#3498db'},
            'exciting': {'primary': '#e74c3c', 'secondary': '#f39c12', 'accent': '#f1c40f'},
            'nostalgic': {'primary': '#8e44ad', 'secondary': '#d2b4de', 'accent': '#bb8fce'},
            'mysterious': {'primary': '#1a1a1a', 'secondary': '#34495e', 'accent': '#9b59b6'},
            'romantic': {'primary': '#e91e63', 'secondary': '#fce4ec', 'accent': '#ad1457'}
        }
        
        colors = color_palettes.get(emotion, color_palettes['peaceful'])
        
        # 이미지 태그 생성
        image_tags = []
        for i, img_url in enumerate(images[:6]):
            if img_url and img_url.strip():
                image_tags.append(f'          <InnovativeImage{i+1} src="{img_url}" alt="Travel {i+1}" />')
        
        image_jsx = '\n'.join(image_tags) if image_tags else '          {/* 이미지 없음 */}'
        
        # 혁신적 그리드 구조 생성
        if image_count <= 1:
            grid_structure = "1fr 2fr 1fr"
            image_styles = '''
const InnovativeImage1 = styled.img`
  grid-column: 2 / 3;
  grid-row: 1 / 3;
  width: 100%;
  height: 60vh;
  object-fit: cover;
  border-radius: 20px;
  box-shadow: 0 20px 40px rgba(0,0,0,0.3);
  transform: rotate(-2deg);
`;'''
        elif image_count == 2:
            grid_structure = "1fr 1fr 1fr 1fr"
            image_styles = '''
const InnovativeImage1 = styled.img`
  grid-column: 1 / 3;
  grid-row: 1 / 2;
  width: 100%;
  height: 300px;
  object-fit: cover;
  border-radius: 15px 15px 0 0;
`;

const InnovativeImage2 = styled.img`
  grid-column: 3 / 5;
  grid-row: 2 / 4;
  width: 100%;
  height: 400px;
  object-fit: cover;
  border-radius: 0 15px 15px 0;
  transform: translateY(-50px);
`;'''
        else:
            grid_structure = "repeat(6, 1fr)"
            image_styles = '''
const InnovativeImage1 = styled.img`
  grid-column: 1 / 4;
  grid-row: 1 / 3;
  width: 100%;
  height: 350px;
  object-fit: cover;
  border-radius: 20px 0 0 20px;
`;

const InnovativeImage2 = styled.img`
  grid-column: 4 / 6;
  grid-row: 1 / 2;
  width: 100%;
  height: 160px;
  object-fit: cover;
`;

const InnovativeImage3 = styled.img`
  grid-column: 6 / 7;
  grid-row: 1 / 2;
  width: 100%;
  height: 160px;
  object-fit: cover;
  border-radius: 0 20px 0 0;
`;

const InnovativeImage4 = styled.img`
  grid-column: 4 / 7;
  grid-row: 2 / 3;
  width: 100%;
  height: 180px;
  object-fit: cover;
  border-radius: 0 0 20px 0;
`;

const InnovativeImage5 = styled.img`
  grid-column: 1 / 3;
  grid-row: 3 / 4;
  width: 100%;
  height: 200px;
  object-fit: cover;
  border-radius: 0 0 0 20px;
`;

const InnovativeImage6 = styled.img`
  grid-column: 3 / 7;
  grid-row: 3 / 4;
  width: 100%;
  height: 200px;
  object-fit: cover;
  border-radius: 0 0 20px 20px;
`;'''
        
        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 1600px;
  margin: 0 auto;
  padding: 60px 40px;
  background: linear-gradient(135deg, {colors['secondary']} 0%, #ffffff 100%);
  min-height: 100vh;
`;

const InnovativeHeader = styled.header`
  text-align: center;
  margin-bottom: 80px;
  position: relative;
`;

const Title = styled.h1`
  font-size: 4em;
  color: {colors['primary']};
  margin-bottom: 30px;
  font-weight: 200;
  letter-spacing: -2px;
  line-height: 0.9;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
  
  &::after {{
    content: '';
    position: absolute;
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 100px;
    height: 3px;
    background: {colors['accent']};
    border-radius: 2px;
  }}
`;

const Subtitle = styled.h2`
  font-size: 1.6em;
  color: {colors['primary']};
  margin-bottom: 40px;
  font-weight: 300;
  font-style: italic;
  opacity: 0.8;
`;

const InnovativeLayout = styled.div`
  display: grid;
  grid-template-columns: {grid_structure};
  gap: 40px;
  margin-bottom: 80px;
  min-height: 70vh;
`;

{image_styles}

const TextContent = styled.div`
  grid-column: 1 / -1;
  font-size: 1.2em;
  line-height: 1.8;
  color: {colors['primary']};
  text-align: justify;
  columns: 2;
  column-gap: 60px;
  margin-top: 60px;
  padding: 40px;
  background: rgba(255,255,255,0.8);
  border-radius: 20px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
`;

const Tagline = styled.div`
  text-align: center;
  font-size: 1em;
  color: {colors['accent']};
  letter-spacing: 4px;
  text-transform: uppercase;
  margin-top: 60px;
  padding: 20px;
  border: 2px solid {colors['accent']};
  border-radius: 50px;
  max-width: 400px;
  margin-left: auto;
  margin-right: auto;
  background: rgba(255,255,255,0.9);
`;

export const {component_name} = () => {{
  return (
    <Container>
      <InnovativeHeader>
        <Title>{title}</Title>
        <Subtitle>{subtitle}</Subtitle>
      </InnovativeHeader>
      
      <InnovativeLayout>
{image_jsx}
      </InnovativeLayout>
      
      <TextContent>{body}</TextContent>
      
      <Tagline>{tagline}</Tagline>
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
        """JSX 컴포넌트 저장"""
        saved_components = []
        
        for component in generated_components:
            file_path = os.path.join(components_folder, component['file'])
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(component['jsx_code'])
                saved_components.append(component)
                
                innovation_level = component.get('innovation_level', 'standard')
                layout_sources = component.get('layout_sources', [])
                print(f"✅ 혁신적 JSX 저장 완료: {component['file']} (혁신도: {innovation_level})")
                if layout_sources:
                    print(f"   📄 참고 레이아웃: {', '.join(layout_sources)}")
                    
            except Exception as e:
                print(f"❌ 저장 실패: {component['file']} - {e}")
        
        return saved_components

