import os
import json
import shutil
from pathlib import Path
from typing import List, Dict

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
    
    def save_magazine_content_json(self, magazine_content, file_path):
        """매거진 콘텐츠를 JSON 형식으로 저장"""
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
                        "metadata": {
                            "content_length": len(magazine_content),
                            "creation_date": self._get_current_timestamp(),
                            "format": "json"
                        }
                    }
            else:
                # 기타 타입인 경우 문자열로 변환 후 처리
                content_json = {
                    "content_type": "unknown",
                    "raw_content": str(magazine_content),
                    "metadata": {
                        "original_type": str(type(magazine_content)),
                        "creation_date": self._get_current_timestamp()
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

    def _parse_text_to_sections(self, content: str) -> List[Dict]:
        """텍스트 콘텐츠를 섹션별로 파싱"""
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
                sections.append(current_section)
        
        # 섹션이 없으면 전체를 하나의 섹션으로
        if not sections:
            sections.append({
                "title": "여행 이야기",
                "content": content,
                "level": 1,
                "type": "full_content"
            })
        
        return sections

    def _get_current_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()


    def save_json(self, data, file_path):
        """데이터를 JSON 파일로 저장 - 안전한 처리"""
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

    def create_react_app(self, project_folder):
        """React 앱 생성 및 필요한 파일 설정"""
        # 기본 구조 생성
        src_folder = os.path.join(project_folder, "src")
        components_folder = os.path.join(src_folder, "components")
        public_folder = os.path.join(project_folder, "public")
        
        os.makedirs(components_folder, exist_ok=True)
        os.makedirs(public_folder, exist_ok=True)

        # index.html 생성 (public 폴더에)
        index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="CrewAI로 생성된 여행 매거진" />
    <title>여행 매거진</title>
</head>
<body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
</body>
</html>"""
        
        self.save_content(index_html, os.path.join(public_folder, "index.html"))

        # index.js 생성 (src 폴더에)
        index_js = """import React from 'react';
import ReactDOM from 'react-dom/client';
import './App.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
"""
        self.save_content(index_js, os.path.join(src_folder, "index.js"))

        # App.css 생성
        app_css = """.App {
  text-align: center;
}

.magazine-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.magazine-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 40px 20px;
  border-radius: 10px;
  margin-bottom: 30px;
}

.magazine-header h1 {
  font-size: 3em;
  margin-bottom: 10px;
}

.magazine-footer {
  background: #f8f9fa;
  padding: 30px;
  border-radius: 10px;
  margin-top: 40px;
  text-align: center;
  color: #6c757d;
}

.content-credits {
  display: flex;
  justify-content: center;
  gap: 15px;
  margin-top: 15px;
  flex-wrap: wrap;
}

.content-credits span {
  background: rgba(255,255,255,0.2);
  padding: 5px 10px;
  border-radius: 15px;
  font-size: 0.9em;
}

.generation-info {
  margin-top: 15px;
  font-size: 0.9em;
}
"""
        self.save_content(app_css, os.path.join(src_folder, "App.css"))

        # package.json 생성
        package_json = {
            "name": "travel-magazine",
            "version": "0.1.0",
            "private": True,
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "react-scripts": "5.0.1",
                "styled-components": "^6.0.7"
            },
            "scripts": {
                "start": "react-scripts start",
                "build": "react-scripts build",
                "test": "react-scripts test",
                "eject": "react-scripts eject"
            },
            "eslintConfig": {
                "extends": [
                    "react-app",
                    "react-app/jest"
                ]
            },
            "browserslist": {
                "production": [
                    ">0.2%",
                    "not dead",
                    "not op_mini all"
                ],
                "development": [
                    "last 1 chrome version",
                    "last 1 firefox version",
                    "last 1 safari version"
                ]
            }
        }
        
        self.save_json(package_json, os.path.join(project_folder, "package.json"))

        return src_folder, components_folder
