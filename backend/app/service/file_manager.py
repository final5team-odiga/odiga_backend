import os
import json
from typing import List, Dict
import aiofiles
import asyncio

class FileManager:
    def __init__(self, output_folder="./output"):
        self.output_folder = output_folder

    def create_project_folder(self, project_name):
        """프로젝트 폴더 생성"""
        project_path = os.path.join(self.output_folder, project_name)
        os.makedirs(project_path, exist_ok=True)
        return project_path

    def save_content(self, content, file_path):
        """콘텐츠를 파일로 저장"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path

    async def load_json_async(self, file_path: str) -> Dict:
        """JSON 파일 비동기 로드"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            return json.loads(content)
        except Exception as e:
            print(f"❌ JSON 파일 로드 실패: {e}")
            return {}

    async def save_json_async(self, file_path: str, data: Dict) -> str:
        """JSON 파일 비동기 저장"""
        try:
            # 구조적 메타데이터 추가 (기존 로직 유지)
            if isinstance(data, dict) and 'metadata' not in data:
                data['metadata'] = {
                    "creation_date": self._get_current_timestamp(),
                    "file_manager_version": "2.0",
                    "agent_enhanced": True
                }

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))
            print(f"✅ JSON 파일 비동기 저장 성공: {file_path}")
            return file_path
        except Exception as e:
            print(f"❌ JSON 파일 비동기 저장 실패: {e}")
            return file_path

    async def save_text_async(self, file_path: str, content: str) -> str:
        """텍스트 파일 비동기 저장"""
        try:
            # 디렉토리 생성 코드 제거
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            print(f"✅ 텍스트 파일 비동기 저장 성공: {file_path}")
            return file_path
        except Exception as e:
            print(f"❌ 텍스트 파일 비동기 저장 실패: {e}")
            return file_path

    def save_magazine_content_json(self, magazine_content, file_path):
        """매거진 콘텐츠를 JSON 형식으로 저장 (구조적 데이터 포함)"""
        try:
            # magazine_content가 이미 딕셔너리인지 확인
            if isinstance(magazine_content, dict):
                content_json = magazine_content
            elif isinstance(magazine_content, str):
                # 문자열인 경우 JSON 파싱 시도
                try:
                    content_json = json.loads(magazine_content)
                except json.JSONDecodeError:
                    # JSON이 아닌 일반 텍스트인 경우 구조화
                    content_json = {
                        "magazine_title": "여행 매거진",
                        "content_type": "integrated_magazine",
                        "sections": self._parse_text_to_sections(magazine_content),
                        "raw_content": magazine_content,
                        "layout_structure": self._generate_default_layout_structure(),
                        "metadata": {
                            "content_length": len(magazine_content),
                            "creation_date": self._get_current_timestamp(),
                            "format": "json",
                            "agent_enhanced": True
                        }
                    }
            else:
                # 기타 타입인 경우 문자열로 변환 후 처리
                content_json = {
                    "content_type": "unknown",
                    "raw_content": str(magazine_content),
                    "layout_structure": self._generate_default_layout_structure(),
                    "metadata": {
                        "original_type": str(type(magazine_content)),
                        "creation_date": self._get_current_timestamp(),
                        "agent_enhanced": True
                    }
                }

            # JSON 파일로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(content_json, f, ensure_ascii=False, indent=2)
            print(f"✅ 매거진 콘텐츠 JSON 저장 성공: {file_path}")
            return file_path
        except Exception as e:
            print(f"❌ 매거진 콘텐츠 JSON 저장 실패: {e}")
            # 폴백: 텍스트로 저장
            with open(file_path.replace('.json', '.txt'), 'w', encoding='utf-8') as f:
                f.write(str(magazine_content))
            return file_path

    def _generate_default_layout_structure(self) -> Dict:
        """기본 레이아웃 구조 생성"""
        return {
            "page_grid": {
                "columns": 12,
                "gutter": "20px",
                "margin": "40px"
            },
            "text_areas": [
                {"id": "title", "position": {"x": 0, "y": 0, "width": 12, "height": 2}},
                {"id": "subtitle", "position": {"x": 0, "y": 2, "width": 12, "height": 1}},
                {"id": "body", "position": {"x": 0, "y": 3, "width": 8, "height": 6}}
            ],
            "image_areas": [
                {"id": "hero", "position": {"x": 8, "y": 3, "width": 4, "height": 6}}
            ],
            "layout_type": "grid",
            "responsive_breakpoints": {
                "mobile": "768px",
                "tablet": "1024px",
                "desktop": "1200px"
            }
        }

    def _parse_text_to_sections(self, content: str) -> List[Dict]:
        """텍스트 콘텐츠를 섹션별로 파싱 (구조적 정보 포함)"""
        import re
        sections = []
        
        # 헤더 기반 분할 (## 또는 ###)
        header_pattern = r'^(#{1,3})\s+(.+?)$'
        current_section = {"title": "", "content": "", "level": 1}
        current_content = []
        
        lines = content.split('\n')
        for line in lines:
            header_match = re.match(header_pattern, line.strip())
            if header_match:
                # 이전 섹션 저장
                if current_content or current_section["title"]:
                    current_section["content"] = '\n'.join(current_content).strip()
                    if current_section["content"] or current_section["title"]:
                        # 구조적 정보 추가
                        current_section["layout_info"] = {
                            "estimated_reading_time": len(current_section["content"]) // 200,
                            "content_density": "high" if len(current_section["content"]) > 1000 else "medium",
                            "suggested_layout": "grid" if len(current_section["content"]) > 500 else "minimal"
                        }
                        sections.append(current_section.copy())
                
                # 새 섹션 시작
                header_level = len(header_match.group(1))
                header_text = header_match.group(2).strip()
                current_section = {
                    "title": header_text,
                    "content": "",
                    "level": header_level,
                    "type": "header_section"
                }
                current_content = []
            else:
                # 내용 추가
                if line.strip():
                    current_content.append(line)
        
        # 마지막 섹션 저장
        if current_content or current_section["title"]:
            current_section["content"] = '\n'.join(current_content).strip()
            if current_section["content"] or current_section["title"]:
                current_section["layout_info"] = {
                    "estimated_reading_time": len(current_section["content"]) // 200,
                    "content_density": "high" if len(current_section["content"]) > 1000 else "medium",
                    "suggested_layout": "grid" if len(current_section["content"]) > 500 else "minimal"
                }
                sections.append(current_section)
        
        # 섹션이 없으면 전체를 하나의 섹션으로
        if not sections:
            sections.append({
                "title": "여행 이야기",
                "content": content,
                "level": 1,
                "type": "full_content",
                "layout_info": {
                    "estimated_reading_time": len(content) // 200,
                    "content_density": "high" if len(content) > 1000 else "medium",
                    "suggested_layout": "magazine"
                }
            })
        
        return sections

    def _get_current_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()

    def save_json(self, data, file_path):
        """데이터를 JSON 파일로 저장 - 안전한 처리 (구조적 데이터 지원)"""
        try:
            # 데이터가 문자열인 경우 파싱 시도
            if isinstance(data, str):
                try:
                    # JSON 파싱 시도
                    parsed_data = json.loads(data)
                    data = parsed_data
                except json.JSONDecodeError:
                    # Python dict 문자열 변환 시도
                    try:
                        import ast
                        cleaned_str = data.replace("'", '"').replace('True', 'true').replace('False', 'false').replace('None', 'null')
                        parsed_data = json.loads(cleaned_str)
                        data = parsed_data
                    except:
                        try:
                            parsed_data = ast.literal_eval(data)
                            data = parsed_data
                        except:
                            print(f"⚠️ JSON 저장 실패: 문자열을 파싱할 수 없습니다")
                            return file_path

            # 구조적 메타데이터 추가
            if isinstance(data, dict) and 'metadata' not in data:
                data['metadata'] = {
                    "creation_date": self._get_current_timestamp(),
                    "file_manager_version": "2.0",
                    "agent_enhanced": True
                }

            # JSON 파일로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ JSON 파일 저장 성공: {file_path}")
            return file_path
        except Exception as e:
            print(f"❌ JSON 파일 저장 중 오류: {e}")
            # 폴백: 문자열로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
            return file_path

    # ✅ Template Data 처리 메서드들 (핵심)
    async def save_template_data_async(self, template_data: Dict, file_path: str) -> str:
        """Template Data 구조화하여 비동기 저장"""
        try:
            # Template Data 구조 분석 및 향상
            enhanced_template_data = self._enhance_template_data_structure(template_data)
            
            # 비동기 저장
            await self.save_json_async(file_path, enhanced_template_data)
            print(f"✅ Template Data 저장 완료: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"❌ Template Data 저장 실패: {e}")
            return file_path
    
    def _enhance_template_data_structure(self, template_data: Dict) -> Dict:
        """Template Data 구조 향상 (File Manager 전용)"""
        
        # 기본 구조 분석
        content_sections = template_data.get("content_sections", [])
        selected_templates = template_data.get("selected_templates", [])
        
        enhanced_data = {
            "magazine_metadata": {
                "title": "AI 생성 여행 매거진",
                "subtitle": "멀티모달 AI 기반 자동 생성",
                "creation_timestamp": self._get_current_timestamp(),
                "total_sections": len(content_sections),
                "template_count": len(selected_templates),
                "file_manager_version": "2.0"
            },
            "template_mapping": {
                "selected_templates": selected_templates,
                "template_structure": self._analyze_template_structure(content_sections)
            },
            "content_sections": self._process_content_sections(content_sections),
            "layout_configuration": {
                "responsive_design": True,
                "ai_search_enhanced": template_data.get("integration_metadata", {}).get("ai_search_enhanced", False),
                "multimodal_optimized": template_data.get("integration_metadata", {}).get("multimodal_processing", False)
            },
            "file_structure": {
                "components_folder": "./components",
                "app_folder": "./app",
                "public_folder": "./public"
            },
            "build_configuration": {
                "framework": "Next.js",
                "styling": "Tailwind CSS + Styled Components",
                "build_tool": "Next.js",
                "target_browsers": ["Chrome", "Firefox", "Safari", "Edge"]
            }
        }
        
        return enhanced_data
    
    def _analyze_template_structure(self, content_sections: List[Dict]) -> Dict:
        """템플릿 구조 분석"""
        
        structure_analysis = {
            "section_types": {},
            "image_distribution": {},
            "text_complexity": {},
            "layout_patterns": []
        }
        
        for i, section in enumerate(content_sections):
            template_name = section.get("template", f"Section{i+1:02d}.jsx")
            
            # 섹션 타입 분석
            images = section.get("images", [])
            body_length = len(section.get("body", ""))
            
            if len(images) > 2:
                section_type = "image_gallery"
            elif len(images) == 1:
                section_type = "image_text_balanced"
            elif len(images) == 0:
                section_type = "text_only"
            else:
                section_type = "multi_image"
            
            structure_analysis["section_types"][template_name] = section_type
            structure_analysis["image_distribution"][template_name] = len(images)
            
            # 텍스트 복잡도 분석
            if body_length > 500:
                complexity = "high"
            elif body_length > 200:
                complexity = "medium"
            else:
                complexity = "low"
            
            structure_analysis["text_complexity"][template_name] = complexity
            
            # 레이아웃 패턴 추가
            layout_config = section.get("layout_config", {})
            if layout_config:
                structure_analysis["layout_patterns"].append({
                    "template": template_name,
                    "pattern": layout_config
                })
        
        return structure_analysis
    
    def _process_content_sections(self, content_sections: List[Dict]) -> List[Dict]:
        """콘텐츠 섹션 처리 및 구조화"""
        
        processed_sections = []
        
        for i, section in enumerate(content_sections):
            processed_section = {
                "section_id": f"section_{i+1:03d}",
                "template_info": {
                    "name": section.get("template", f"Section{i+1:02d}.jsx"),
                    "type": self._determine_section_type(section),
                    "complexity": self._calculate_section_complexity(section)
                },
                "content_data": {
                    "title": section.get("title", f"여행 이야기 {i+1}"),
                    "subtitle": section.get("subtitle", "특별한 순간들"),
                    "body": section.get("body", "멋진 여행 경험을 공유합니다."),
                    "tagline": section.get("tagline", "TRAVEL & CULTURE")
                },
                "media_assets": {
                    "images": section.get("images", []),
                    "image_count": len(section.get("images", [])),
                    "primary_image": section.get("images", [None])[0] if section.get("images") else None
                },
                "layout_specifications": {
                    "config": section.get("layout_config", {}),
                    "responsive": True,
                    "accessibility": True
                },
                "metadata": {
                    "ai_enhanced": section.get("metadata", {}).get("ai_search_enhanced", False),
                    "processing_timestamp": self._get_current_timestamp(),
                    "ready_for_jsx": True
                }
            }
            
            processed_sections.append(processed_section)
        
        return processed_sections
    
    def _determine_section_type(self, section: Dict) -> str:
        """섹션 타입 결정"""
        images = section.get("images", [])
        body_length = len(section.get("body", ""))
        
        if len(images) == 0:
            return "text_focused"
        elif len(images) == 1:
            return "balanced_layout"
        elif len(images) <= 3:
            return "image_rich"
        else:
            return "gallery_style"
    
    def _calculate_section_complexity(self, section: Dict) -> str:
        """섹션 복잡도 계산"""
        body_length = len(section.get("body", ""))
        image_count = len(section.get("images", []))
        
        complexity_score = (body_length / 100) + (image_count * 2)
        
        if complexity_score > 10:
            return "high"
        elif complexity_score > 5:
            return "medium"
        else:
            return "low"
    
    # ✅ JSX 컴포넌트 처리 메서드들 (핵심)
    async def save_jsx_components_async(self, jsx_components: List[Dict]) -> None:
        """JSX 컴포넌트들을 구조화하여 저장"""
        try:
            # 1. 컴포넌트 폴더 생성
            components_folder = os.path.join(self.output_folder, "components")
            os.makedirs(components_folder, exist_ok=True)
            
            # 2. 개별 JSX 파일 저장
            saved_components = []
            for component in jsx_components:
                template_name = component.get("template_name", "Unknown.jsx")
                jsx_code = component.get("jsx_code", "")
                
                if jsx_code and template_name:
                    component_path = os.path.join(components_folder, template_name)
                    await self.save_text_async(component_path, jsx_code)
                    
                    saved_components.append({
                        "template_name": template_name,
                        "file_path": component_path,
                        "component_metadata": component.get("component_metadata", {}),
                        "save_timestamp": self._get_current_timestamp()
                    })
            
            # 3. 컴포넌트 인덱스 파일 생성
            await self._create_component_index_async(saved_components, components_folder)
            
            # 4. 컴포넌트 매니페스트 저장
            manifest_data = {
                "total_components": len(saved_components),
                "components": saved_components,
                "creation_timestamp": self._get_current_timestamp(),
                "file_manager_version": "2.0"
            }
            
            manifest_path = os.path.join(components_folder, "component_manifest.json")
            await self.save_json_async(manifest_path, manifest_data)
            
            print(f"✅ JSX 컴포넌트 저장 완료: {len(saved_components)}개")
            
        except Exception as e:
            print(f"❌ JSX 컴포넌트 저장 실패: {e}")
