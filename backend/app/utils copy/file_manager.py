import os
import json
from typing import List, Dict
import aiofiles
import asyncio

class FileManager:
    def __init__(self, output_folder="./output"):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

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
            # 디렉토리 생성
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
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
    
    async def _create_component_index_async(self, saved_components: List[Dict], components_folder: str) -> None:
        """컴포넌트 인덱스 파일 생성"""
        
        # index.js 생성
        index_content = "// 자동 생성된 컴포넌트 인덱스 파일\n"
        index_content += f"// 생성 시간: {self._get_current_timestamp()}\n\n"
        
        # Import 문 생성
        for component in saved_components:
            template_name = component.get("template_name", "").replace(".jsx", "")
            if template_name:
                index_content += f"import {template_name} from './{template_name}';\n"
        
        index_content += "\n// 컴포넌트 배열 Export\n"
        index_content += "export const MagazineComponents = [\n"
        
        for component in saved_components:
            template_name = component.get("template_name", "").replace(".jsx", "")
            if template_name:
                index_content += f"  {template_name},\n"
        
        index_content += "];\n\n"
        
        # 개별 Export
        index_content += "// 개별 컴포넌트 Export\n"
        for component in saved_components:
            template_name = component.get("template_name", "").replace(".jsx", "")
            if template_name:
                index_content += f"export {{ default as {template_name} }} from './{template_name}';\n"
        
        # 메타데이터 Export
        index_content += f"\n// 컴포넌트 메타데이터\n"
        index_content += f"export const ComponentMetadata = {{\n"
        index_content += f"  totalComponents: {len(saved_components)},\n"
        index_content += f"  generationTime: '{self._get_current_timestamp()}',\n"
        index_content += f"  aiGenerated: true\n"
        index_content += f"}};\n"
        
        index_path = os.path.join(components_folder, "index.js")
        await self.save_text_async(index_path, index_content)
        print("✅ 컴포넌트 인덱스 파일 생성: index.js")

    # ✅ Next.js 앱 생성 메서드 (핵심)
    def create_magazine_react_app(self, project_folder, saved_components, template_data):
        """Next.js 기반 매거진 앱 생성 (다중 컴포넌트 캐러셀 뷰어)"""
        print(f"📱 Next.js 매거진 앱 생성 시작: {project_folder}")

        # ✅ Next.js 구조 생성
        app_folder = os.path.join(project_folder, "app")
        components_folder = os.path.join(project_folder, "components")
        public_folder = os.path.join(project_folder, "public")
        
        os.makedirs(app_folder, exist_ok=True)
        os.makedirs(components_folder, exist_ok=True)
        os.makedirs(public_folder, exist_ok=True)

        # ✅ 1. Next.js package.json 생성
        package_json = {
            "name": "ai-magazine-viewer",
            "version": "0.1.0",
            "private": True,
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "next": "^14.0.0",
                "styled-components": "^6.0.0",
                "react-multi-carousel": "^2.8.4"
            },
            "scripts": {
                "dev": "next dev",
                "build": "next build",
                "start": "next start",
                "lint": "next lint"
            },
            "devDependencies": {
                "eslint": "^8",
                "eslint-config-next": "14.0.0"
            }
        }

        with open(os.path.join(project_folder, "package.json"), 'w', encoding='utf-8') as f:
            json.dump(package_json, f, indent=2)

        # ✅ 2. Next.js layout.js 생성
        layout_js = """export const metadata = {
  title: 'AI 매거진 뷰어',
  description: 'AI가 생성한 매거진 컴포넌트 뷰어',
}

export default function RootLayout({ children }) {
  return (
    <html lang="ko">
      <body style={{
        margin: 0,
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        minHeight: '100vh'
      }}>
        {children}
      </body>
    </html>
  )
}"""

        with open(os.path.join(app_folder, "layout.js"), 'w', encoding='utf-8') as f:
            f.write(layout_js)

        # ✅ 3. 생성된 컴포넌트들을 components 폴더에 저장 및 import 정보 생성
        component_imports = []
        component_list = []
        
        for i, component_data in enumerate(saved_components):
            # JSX 컴포넌트 데이터 구조에 맞게 수정
            if isinstance(component_data, dict):
                # 새로운 구조: template_name과 jsx_code 사용
                template_name = component_data.get('template_name', f'Component{i+1}.jsx')
                jsx_code = component_data.get('jsx_code', '')
                component_name = template_name.replace('.jsx', '')
            else:
                # 기존 구조 호환성 유지
                component_name = f'Component{i+1}'
                template_name = f'{component_name}.jsx'
                jsx_code = str(component_data)
            
            if jsx_code:
                # ✅ Next.js 컴포넌트 파일 저장 (JSX 그대로 사용)
                component_path = os.path.join(components_folder, template_name)
                with open(component_path, 'w', encoding='utf-8') as f:
                    f.write(jsx_code)
                
                # import 문과 컴포넌트 리스트에 추가
                component_imports.append(f"import {component_name} from '../components/{component_name}';")
                component_list.append({
                    'name': component_name,
                    'component': component_name,
                    'title': f"컴포넌트 {i+1}: {component_name}",
                    'description': f"AI가 생성한 매거진 컴포넌트 #{i+1}"
                })

        # ✅ 4. Next.js page.js 생성 (캐러셀 뷰어)
        page_js = f"""'use client'

import React, {{ useState }} from 'react';
import Carousel from 'react-multi-carousel';
import 'react-multi-carousel/lib/styles.css';
import styled from 'styled-components';
{chr(10).join(component_imports)}

const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 30px;
  color: white;
`;

const Title = styled.h1`
  font-size: 3rem;
  margin: 0 0 10px 0;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
  font-weight: 700;
`;

const Subtitle = styled.p`
  font-size: 1.2rem;
  margin: 0;
  opacity: 0.9;
`;

const CarouselContainer = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 20px;
  padding: 30px;
  backdrop-filter: blur(10px);
  box-shadow: 0 20px 40px rgba(0,0,0,0.1);
`;

const ComponentWrapper = styled.div`
  margin: 0 15px;
  background: white;
  border-radius: 15px;
  overflow: hidden;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  
  &:hover {{
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
  }}
`;

const ComponentHeader = styled.div`
  background: linear-gradient(45deg, #667eea, #764ba2);
  color: white;
  padding: 20px;
  text-align: center;
`;

const ComponentTitle = styled.h3`
  margin: 0 0 5px 0;
  font-size: 1.4rem;
  font-weight: 600;
`;

const ComponentDescription = styled.p`
  margin: 0;
  opacity: 0.9;
  font-size: 0.9rem;
`;

const ComponentContent = styled.div`
  max-height: 80vh;
  overflow-y: auto;
  padding: 0;
  
  /* 스크롤바 스타일링 */
  &::-webkit-scrollbar {{
    width: 8px;
  }}
  
  &::-webkit-scrollbar-track {{
    background: #f1f1f1;
  }}
  
  &::-webkit-scrollbar-thumb {{
    background: #888;
    border-radius: 4px;
  }}
  
  &::-webkit-scrollbar-thumb:hover {{
    background: #555;
  }}
`;

const NavigationInfo = styled.div`
  text-align: center;
  margin-bottom: 20px;
  color: white;
  font-size: 1.1rem;
  opacity: 0.8;
`;

const ComponentCounter = styled.div`
  text-align: center;
  margin-top: 20px;
  color: white;
  font-size: 1rem;
  opacity: 0.7;
`;

const responsive = {{
  desktop: {{
    breakpoint: {{ max: 3000, min: 1024 }},
    items: 1,
    slidesToSlide: 1
  }},
  tablet: {{
    breakpoint: {{ max: 1024, min: 464 }},
    items: 1,
    slidesToSlide: 1
  }},
  mobile: {{
    breakpoint: {{ max: 464, min: 0 }},
    items: 1,
    slidesToSlide: 1
  }}
}};

const CustomButtonGroup = ({{ next, previous, goToSlide, ...rest }}) => {{
  const {{ carouselState: {{ currentSlide, totalItems }} }} = rest;
  
  return (
    <div style={{{{ 
      textAlign: 'center', 
      marginTop: '20px',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      gap: '20px'
    }}}}>
      <button
        onClick={{() => previous()}}
        disabled={{currentSlide === 0}}
        style={{{{
          background: currentSlide === 0 ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.8)',
          border: 'none',
          borderRadius: '50%',
          width: '50px',
          height: '50px',
          cursor: currentSlide === 0 ? 'not-allowed' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '20px',
          transition: 'all 0.3s ease',
          color: currentSlide === 0 ? 'rgba(0,0,0,0.3)' : '#333'
        }}}}
      >
        ←
      </button>
      
      <span style={{{{ color: 'white', fontSize: '16px', fontWeight: '500' }}}}>
        {{currentSlide + 1}} / {{totalItems}}
      </span>
      
      <button
        onClick={{() => next()}}
        disabled={{currentSlide >= totalItems - 1}}
        style={{{{
          background: currentSlide >= totalItems - 1 ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.8)',
          border: 'none',
          borderRadius: '50%',
          width: '50px',
          height: '50px',
          cursor: currentSlide >= totalItems - 1 ? 'not-allowed' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '20px',
          transition: 'all 0.3s ease',
          color: currentSlide >= totalItems - 1 ? 'rgba(0,0,0,0.3)' : '#333'
        }}}}
      >
        →
      </button>
    </div>
  );
}};

const components = [
{chr(10).join([f"  {{ name: '{comp['name']}', component: {comp['component']}, title: '{comp['title']}', description: '{comp['description']}' }}," for comp in component_list])}
];

export default function Home() {{
  const [currentSlide, setCurrentSlide] = useState(0);

  const handleSlideChange = (previousSlide, {{ currentSlide, onMove }}) => {{
    setCurrentSlide(currentSlide);
  }};

  return (
    <AppContainer>
      <Header>
        <Title>🎨 AI 매거진 컴포넌트 뷰어</Title>
        <Subtitle>생성된 {len(component_list)}개의 매거진 컴포넌트를 확인해보세요</Subtitle>
      </Header>
      
      <CarouselContainer>
        <NavigationInfo>
          ← → 화살표 버튼을 사용하여 컴포넌트를 탐색하세요
        </NavigationInfo>
        
        <Carousel
          responsive={{responsive}}
          infinite={{false}}
          autoPlay={{false}}
          keyBoardControl={{true}}
          customTransition="transform 300ms ease-in-out"
          transitionDuration={{300}}
          containerClass="carousel-container"
          removeArrowOnDeviceType={{[]}}
          arrows={{false}}
          renderButtonGroupOutside={{true}}
          customButtonGroup={{<CustomButtonGroup />}}
          beforeChange={{handleSlideChange}}
          showDots={{false}}
        >
          {{components.map((comp, index) => {{
            const ComponentToRender = comp.component;
            return (
              <ComponentWrapper key={{index}}>
                <ComponentHeader>
                  <ComponentTitle>{{comp.title}}</ComponentTitle>
                  <ComponentDescription>{{comp.description}}</ComponentDescription>
                </ComponentHeader>
                <ComponentContent>
                  <ComponentToRender />
                </ComponentContent>
              </ComponentWrapper>
            );
          }})}}
        </Carousel>
        
        <ComponentCounter>
          현재 보고 있는 컴포넌트: {{currentSlide + 1}} / {len(component_list)}
        </ComponentCounter>
      </CarouselContainer>
    </AppContainer>
  );
}}"""

        with open(os.path.join(app_folder, "page.js"), 'w', encoding='utf-8') as f:
            f.write(page_js)

        # ✅ 5. Next.js 설정 파일들 생성
        # next.config.js
        next_config = """/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  compiler: {
    styledComponents: true,
  },
  images: {
    domains: ['localhost'],
  },
}

module.exports = nextConfig"""

        with open(os.path.join(project_folder, "next.config.js"), 'w', encoding='utf-8') as f:
            f.write(next_config)

        # ✅ 6. 실행 스크립트 생성
        # Windows 배치 파일
        windows_script = f"""@echo off
echo 📱 AI 매거진 Next.js 앱 시작 중...
echo.
cd /d "{project_folder}"
echo 📦 의존성 설치 중...
call npm install
echo.
echo 🚀 개발 서버 시작 중...
echo 브라우저에서 http://localhost:3000 을 확인하세요.
call npm run dev
pause
"""

        with open(os.path.join(project_folder, "start_nextjs_app.bat"), 'w', encoding='utf-8') as f:
            f.write(windows_script)

        # Unix 스크립트
        unix_script = f"""#!/bin/bash
echo "📱 AI 매거진 Next.js 앱 시작 중..."
echo
cd "{project_folder}"
echo "📦 의존성 설치 중..."
npm install
echo
echo "🚀 개발 서버 시작 중..."
echo "브라우저에서 http://localhost:3000 을 확인하세요."
npm run dev
"""

        with open(os.path.join(project_folder, "start_nextjs_app.sh"), 'w', encoding='utf-8') as f:
            f.write(unix_script)

        print(f"✅ Next.js 매거진 앱 생성 완료: {project_folder}")
        print(f"📱 실행 방법: cd {project_folder} && npm install && npm run dev")
