import os
import json
import re
from typing import Dict, List
from crewai import Agent, Task, Crew
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager

class OrgAgent:
    """PDF 벡터 데이터 기반 텍스트 배치 에이전트"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        self.vector_manager = PDFVectorManager()
        
    def create_layout_analyzer_agent(self):
        """레이아웃 분석 에이전트"""
        return Agent(
            role="Magazine Layout Analyzer",
            goal="PDF 벡터 데이터를 분석하여 텍스트 콘텐츠에 최적화된 매거진 레이아웃 추천",
            backstory="""당신은 매거진 레이아웃 분석 전문가입니다.
            실제 매거진 PDF에서 추출한 벡터 데이터를 분석하여 
            주어진 텍스트 콘텐츠에 가장 적합한 레이아웃 구조를 찾아내고,
            전문적인 매거진 수준의 텍스트 배치를 설계하는 전문성을 가지고 있습니다.""",
            llm=self.llm,
            verbose=True
        )
    
    def create_content_editor_agent(self):
        """콘텐츠 편집 에이전트"""
        return Agent(
            role="Magazine Content Editor",
            goal="벡터 데이터 기반 레이아웃 분석 결과를 바탕으로 텍스트 콘텐츠를 매거진 스타일로 편집",
            backstory="""당신은 매거진 콘텐츠 편집 전문가입니다.
            실제 매거진의 레이아웃 패턴을 분석한 결과를 바탕으로
            텍스트 콘텐츠를 해당 레이아웃에 최적화하여 편집하며,
            독자의 시선을 사로잡는 매력적인 매거진 콘텐츠를 만드는 전문성을 가지고 있습니다.
            특히 설명 텍스트나 지시사항을 포함하지 않고 순수한 콘텐츠만 생성합니다.""",
            llm=self.llm,
            verbose=True
        )
    
    def process_content(self, magazine_content, available_templates: List[str]) -> Dict:
        """PDF 벡터 데이터 기반 콘텐츠 처리"""
        
        # 텍스트 추출 및 전처리
        all_content = self._extract_all_text(magazine_content)
        content_sections = self._analyze_content_structure(all_content)
        
        print(f"OrgAgent: 처리할 콘텐츠 - {len(all_content)}자, {len(content_sections)}개 섹션")
        
        # 에이전트 생성
        layout_analyzer = self.create_layout_analyzer_agent()
        content_editor = self.create_content_editor_agent()
        
        # 각 섹션별로 벡터 기반 레이아웃 분석 및 편집
        refined_sections = []
        
        for i, section_content in enumerate(content_sections):
            if len(section_content.strip()) < 50:
                continue
            
            print(f"📄 섹션 {i+1} 처리 중...")
            
            # 1단계: 벡터 검색으로 유사한 레이아웃 찾기
            similar_layouts = self.vector_manager.search_similar_layouts(
                section_content[:500],  # 처음 500자로 검색
                "magazine_layout",
                top_k=3
            )
            
            # 2단계: 레이아웃 분석
            layout_analysis_task = Task(
                description=f"""
                다음 텍스트 콘텐츠와 유사한 매거진 레이아웃을 분석하여 최적의 텍스트 배치 전략을 수립하세요:
                
                **분석할 콘텐츠:**
                {section_content}
                
                **유사한 매거진 레이아웃 데이터:**
                {self._format_layout_data(similar_layouts)}
                
                **분석 요구사항:**
                1. **레이아웃 패턴 분석**
                   - 텍스트 블록의 위치와 크기 패턴
                   - 제목과 본문의 배치 관계
                   - 여백과 간격의 활용 방식
                
                2. **콘텐츠 적합성 평가**
                   - 현재 콘텐츠와 레이아웃의 매칭도
                   - 텍스트 길이와 레이아웃 용량의 적합성
                   - 콘텐츠 성격에 맞는 레이아웃 스타일
                
                3. **편집 전략 수립**
                   - 매력적인 제목 생성 방향
                   - 본문 텍스트 분할 및 구조화 방안
                   - 독자 몰입도 향상을 위한 텍스트 배치
                
                **출력 형식:**
                제목: [구체적이고 매력적인 제목]
                부제목: [간결하고 흥미로운 부제목]
                편집방향: [전체적인 편집 방향성]
                """,
                agent=layout_analyzer,
                expected_output="벡터 데이터 기반 레이아웃 분석 및 편집 전략"
            )
            
            # 3단계: 콘텐츠 편집
            content_editing_task = Task(
                description=f"""
                레이아웃 분석 결과를 바탕으로 다음 콘텐츠를 전문 매거진 수준으로 편집하세요:
                
                **원본 콘텐츠:**
                {section_content}
                
                **매거진 스타일 편집 지침:**
                1. **시각적 계층 구조**: 이미지 크기와 배치에 맞는 텍스트 구조 생성
                2. **다이나믹한 레이아웃**: 대형/중형/소형 이미지와 조화되는 텍스트 배치
                3. **매거진 특유의 리듬**: 긴 문단과 짧은 문단의 조화로 시각적 리듬 생성
                4. **이미지와 텍스트 상호작용**: 이미지 주변에 배치될 텍스트의 톤과 길이 조절
                5. **편집 디자인 고려**: 실제 매거진처럼 텍스트가 이미지와 자연스럽게 어우러지도록
                
                **벡터 데이터 기반 최적화:**
                - 검색된 매거진 레이아웃의 텍스트 배치 패턴 적용
                - 이미지 크기별 텍스트 분량과 스타일 조절
                - 매거진 특유의 비대칭 균형감 반영
                
                **출력:** 매거진 레이아웃에 최적화된 편집 콘텐츠
                """,
                agent=content_editor,
                expected_output="매거진 스타일 레이아웃에 최적화된 전문 콘텐츠",
                context=[layout_analysis_task]
            )

            
            # Crew 실행
            crew = Crew(
                agents=[layout_analyzer, content_editor],
                tasks=[layout_analysis_task, content_editing_task],
                verbose=True
            )
            
            try:
                result = crew.kickoff()
                
                # 결과 파싱
                analysis_result = str(layout_analysis_task.output) if hasattr(layout_analysis_task, 'output') else ""
                edited_content = str(result.raw) if hasattr(result, 'raw') else str(result)
                
                # 제목과 부제목 추출
                title, subtitle = self._extract_clean_title_subtitle(analysis_result, i)
                
                # 편집된 콘텐츠에서 설명 텍스트 제거
                clean_content = self._remove_meta_descriptions(edited_content)
                
                refined_sections.append({
                    "title": title,
                    "subtitle": subtitle,
                    "content": clean_content,
                    "layout_info": similar_layouts[0] if similar_layouts else {},
                    "original_length": len(section_content),
                    "refined_length": len(clean_content)
                })
                
                print(f"✅ 섹션 {i+1} 편집 완료: {len(section_content)}자 → {len(clean_content)}자")
                
            except Exception as e:
                print(f"⚠️ 섹션 {i+1} 편집 실패: {e}")
                # 폴백: 기본 처리
                refined_sections.append({
                    "title": f"도쿄 여행 이야기 {i+1}",
                    "subtitle": "특별한 순간들",
                    "content": section_content,
                    "layout_info": {},
                    "original_length": len(section_content),
                    "refined_length": len(section_content)
                })
        
        # 템플릿 매핑
        text_mapping = self._map_to_templates(refined_sections, available_templates)
        
        total_refined_length = sum(section["refined_length"] for section in refined_sections)
        print(f"✅ OrgAgent 완료: {len(refined_sections)}개 섹션, 총 {total_refined_length}자")
        
        return {
            "text_mapping": text_mapping,
            "refined_sections": refined_sections,
            "total_sections": len(refined_sections),
            "total_content_length": total_refined_length,
            "vector_enhanced": True
        }
    
    def _extract_clean_title_subtitle(self, analysis_result: str, index: int) -> tuple:
        """분석 결과에서 깨끗한 제목과 부제목 추출"""
        title_pattern = r'제목[:\s]*([^\n]+)'
        subtitle_pattern = r'부제목[:\s]*([^\n]+)'
        
        title_match = re.search(title_pattern, analysis_result)
        subtitle_match = re.search(subtitle_pattern, analysis_result)
        
        title = title_match.group(1).strip() if title_match else f"도쿄 여행 이야기 {index + 1}"
        subtitle = subtitle_match.group(1).strip() if subtitle_match else "특별한 순간들"
        
        # 설명 텍스트 제거
        title = self._clean_title_from_descriptions(title)
        subtitle = self._clean_title_from_descriptions(subtitle)
        
        # 제목 길이 조정
        if len(title) > 40:
            title = title[:37] + "..."
        if len(subtitle) > 30:
            subtitle = subtitle[:27] + "..."
        
        return title, subtitle
    
    def _clean_title_from_descriptions(self, text: str) -> str:
        """제목에서 설명 텍스트 제거"""
        patterns_to_remove = [
            r'\(헤드라인\)',
            r'\(섹션 타이틀\)',
            r'및 부.*?배치.*?있음',
            r'필자 정보.*?있음',
            r'포토 크레딧.*?있음',
            r'계층적.*?있음',
            r'과 본문.*?관계',
            r'배치.*?관계',
            r'상단.*?배치',
            r'좌상단.*?배치',
            r'혹은.*?배치',
            r'없이.*?집중',
            r'그 아래로.*?있습니다'
        ]
        
        clean_text = text
        for pattern in patterns_to_remove:
            clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE | re.DOTALL)
        
        # 연속된 공백과 특수문자 정리
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = re.sub(r'^[,\s:]+|[,\s:]+$', '', clean_text)
        
        return clean_text.strip() if clean_text.strip() else "도쿄 여행 이야기"
    
    def _remove_meta_descriptions(self, content: str) -> str:
        """콘텐츠에서 메타 설명 제거"""
        patterns_to_remove = [
            r'\*이 페이지에는.*?살렸습니다\.\*',
            r'블록은 균형.*?줄여줍니다',
            r'\(사진 캡션\)',
            r'시각적 리듬과.*?살렸습니다',
            r'충분한 여백.*?완성합니다',
            r'사진은 본문.*?완성합니다',
            r'이 콘텐츠는.*?디자인되었습니다'
        ]
        
        clean_content = content
        for pattern in patterns_to_remove:
            clean_content = re.sub(pattern, '', clean_content, flags=re.IGNORECASE | re.DOTALL)
        
        return clean_content.strip()
    
    def _format_layout_data(self, similar_layouts: List[Dict]) -> str:
        """레이아웃 데이터를 텍스트로 포맷팅"""
        if not similar_layouts:
            return "유사한 레이아웃 데이터 없음"
        
        formatted_data = []
        for i, layout in enumerate(similar_layouts):
            formatted_data.append(f"""
            레이아웃 {i+1} (유사도: {layout.get('score', 0):.2f}):
            - 출처: {layout.get('pdf_name', 'unknown')} (페이지 {layout.get('page_number', 0)})
            - 텍스트 샘플: {layout.get('text_content', '')[:200]}...
            - 이미지 수: {len(layout.get('image_info', []))}개
            - 레이아웃 특징: {self._summarize_layout_info(layout.get('layout_info', {}))}
            """)
        
        return "\n".join(formatted_data)
    
    def _summarize_layout_info(self, layout_info: Dict) -> str:
        """레이아웃 정보 요약"""
        text_blocks = layout_info.get('text_blocks', [])
        images = layout_info.get('images', [])
        tables = layout_info.get('tables', [])
        
        summary = []
        if text_blocks:
            summary.append(f"텍스트 블록 {len(text_blocks)}개")
        if images:
            summary.append(f"이미지 {len(images)}개")
        if tables:
            summary.append(f"테이블 {len(tables)}개")
        
        return ", ".join(summary) if summary else "기본 레이아웃"
    
    def _extract_all_text(self, magazine_content) -> str:
        """모든 텍스트 추출"""
        if isinstance(magazine_content, dict):
            all_text = ""
            
            # 우선순위에 따른 텍스트 추출
            priority_fields = [
                "integrated_content", "essay_content", "interview_content", 
                "sections", "content", "body", "text"
            ]
            
            for field in priority_fields:
                if field in magazine_content:
                    value = magazine_content[field]
                    if isinstance(value, str) and value.strip():
                        all_text += value + "\n\n"
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, str) and sub_value.strip():
                                all_text += sub_value + "\n\n"
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                for sub_key, sub_value in item.items():
                                    if isinstance(sub_value, str) and sub_value.strip():
                                        all_text += sub_value + "\n\n"
                            elif isinstance(item, str) and item.strip():
                                all_text += item + "\n\n"
            
            return all_text.strip()
        else:
            return str(magazine_content)
    
    def _analyze_content_structure(self, content: str) -> List[str]:
        """콘텐츠 구조 분석 및 지능적 분할"""
        if not content:
            return []
        
        sections = []
        
        # 1. 헤더 기반 분할
        header_sections = self._split_by_headers(content)
        if len(header_sections) >= 3:
            sections.extend(header_sections)
        
        # 2. 문단 기반 분할
        if len(sections) < 5:
            paragraph_sections = self._split_by_paragraphs(content)
            sections.extend(paragraph_sections)
        
        # 3. 의미 기반 분할
        if len(sections) < 6:
            semantic_sections = self._split_by_semantics(content)
            sections.extend(semantic_sections)
        
        # 중복 제거 및 길이 필터링
        unique_sections = []
        seen_content = set()
        
        for section in sections:
            section_clean = re.sub(r'\s+', ' ', section.strip())
            if len(section_clean) >= 100 and section_clean not in seen_content:
                unique_sections.append(section)
                seen_content.add(section_clean)
        
        return unique_sections[:8]  # 최대 8개 섹션
    
    def _split_by_headers(self, content: str) -> List[str]:
        """헤더 기반 분할"""
        sections = []
        header_pattern = r'^(#{1,3})\s+(.+?)$'
        current_section = []
        
        lines = content.split('\n')
        for line in lines:
            if re.match(header_pattern, line.strip()):
                if current_section:
                    section_content = '\n'.join(current_section).strip()
                    if len(section_content) > 50:
                        sections.append(section_content)
                current_section = [line]
            else:
                current_section.append(line)
        
        # 마지막 섹션 추가
        if current_section:
            section_content = '\n'.join(current_section).strip()
            if len(section_content) > 50:
                sections.append(section_content)
        
        return sections
    
    def _split_by_paragraphs(self, content: str) -> List[str]:
        """문단 기반 분할"""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 100]
        
        # 문단을 그룹화하여 적절한 길이의 섹션 생성
        sections = []
        current_group = []
        current_length = 0
        
        for paragraph in paragraphs:
            if current_length + len(paragraph) > 800 and current_group:
                sections.append('\n\n'.join(current_group))
                current_group = [paragraph]
                current_length = len(paragraph)
            else:
                current_group.append(paragraph)
                current_length += len(paragraph)
        
        if current_group:
            sections.append('\n\n'.join(current_group))
        
        return sections
    
    def _split_by_semantics(self, content: str) -> List[str]:
        """의미 기반 분할"""
        # 여행 관련 키워드 그룹
        keyword_groups = {
            "arrival": ["도착", "공항", "첫인상", "시작", "출발"],
            "exploration": ["탐험", "걷기", "발견", "거리", "구경"],
            "culture": ["문화", "역사", "전통", "예술", "박물관"],
            "food": ["음식", "맛", "레스토랑", "카페", "먹"],
            "people": ["사람", "만남", "대화", "친구", "현지인"],
            "reflection": ["생각", "느낌", "감정", "의미", "마무리"]
        }
        
        sentences = [s.strip() + '.' for s in content.split('.') if s.strip() and len(s.strip()) > 30]
        sections = {group: [] for group in keyword_groups}
        unclassified = []
        
        for sentence in sentences:
            classified = False
            for group, keywords in keyword_groups.items():
                if any(keyword in sentence for keyword in keywords):
                    sections[group].append(sentence)
                    classified = True
                    break
            
            if not classified:
                unclassified.append(sentence)
        
        # 분류된 섹션들을 텍스트로 변환
        result_sections = []
        for group, group_sentences in sections.items():
            if group_sentences:
                section_text = ' '.join(group_sentences)
                if len(section_text) > 100:
                    result_sections.append(section_text)
        
        # 분류되지 않은 문장들도 추가
        if unclassified:
            unclassified_text = ' '.join(unclassified)
            if len(unclassified_text) > 100:
                result_sections.append(unclassified_text)
        
        return result_sections
    
    def _map_to_templates(self, refined_sections: List[Dict], available_templates: List[str]) -> List[Dict]:
        """정제된 섹션을 템플릿에 매핑"""
        text_mapping = []
        
        for i, section in enumerate(refined_sections):
            template_name = available_templates[i] if i < len(available_templates) else f"Section{i+1:02d}.jsx"
            
            text_mapping.append({
                "template": template_name,
                "title": section["title"],
                "subtitle": section["subtitle"],
                "body": section["content"],
                "tagline": "",
                "content_length": section["refined_length"],
                "layout_source": section.get("layout_info", {}).get("pdf_name", "default")
            })
        
        return text_mapping
