import asyncio
import aiofiles
from typing import List, Dict
from pathlib import Path
from utils.hybridlogging import get_hybrid_logger

class TemplateScanner:
    """JSX 템플릿 동적 스캔 및 관리"""
    
    def __init__(self, template_folder: str = "./jsx_templates"):
        self.template_folder = Path(template_folder)
        self.logger = get_hybrid_logger(self.__class__.__name__)
        
    async def scan_jsx_templates(self) -> List[str]:
        """jsx_templates 폴더에서 JSX 파일 동적 스캔"""
        
        self.logger.info(f"JSX 템플릿 스캔 시작: {self.template_folder}")
        
        try:
            # 폴더 존재 확인
            if not self.template_folder.exists():
                self.logger.warning(f"템플릿 폴더가 존재하지 않습니다: {self.template_folder}")
                await self._create_template_folder()
                return []
            
            # JSX 파일 스캔
            jsx_files = []
            
            # 동기적 스캔 (빠른 처리)
            for file_path in self.template_folder.glob("*.jsx"):
                if file_path.is_file():
                    jsx_files.append(file_path.name)
            
            # 파일명 정렬 (Section01.jsx, Section02.jsx 순서)
            jsx_files.sort(key=self._extract_section_number)
            
            # 템플릿 유효성 검증
            valid_templates = await self._validate_templates(jsx_files)
            
            self.logger.info(f"유효한 JSX 템플릿 {len(valid_templates)}개 발견")
            
            return valid_templates
            
        except Exception as e:
            self.logger.error(f"JSX 템플릿 스캔 실패: {e}")
            return []
    
    def _extract_section_number(self, filename: str) -> int:
        """파일명에서 섹션 번호 추출 (정렬용)"""
        try:
            # Section01.jsx -> 1, Section02.jsx -> 2
            if filename.startswith("Section") and filename.endswith(".jsx"):
                number_part = filename[7:-4]  # "Section"과 ".jsx" 제거
                return int(number_part)
            return 999  # 패턴에 맞지 않는 파일은 뒤로
        except:
            return 999
    
    async def _validate_templates(self, jsx_files: List[str]) -> List[str]:
        """템플릿 파일 유효성 검증"""
        
        valid_templates = []
        
        for jsx_file in jsx_files:
            file_path = self.template_folder / jsx_file
            
            try:
                # 파일 내용 검증
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                
                # 기본 React 컴포넌트 구조 확인
                if self._is_valid_react_component(content, jsx_file):
                    valid_templates.append(jsx_file)
                    self.logger.debug(f"유효한 템플릿: {jsx_file}")
                else:
                    self.logger.warning(f"유효하지 않은 템플릿: {jsx_file}")
                    
            except Exception as e:
                self.logger.error(f"템플릿 검증 실패 {jsx_file}: {e}")
        
        return valid_templates
    
    def _is_valid_react_component(self, content: str, filename: str) -> bool:
        """React 컴포넌트 유효성 검사"""
        
        component_name = filename.replace('.jsx', '')
        
        # 필수 요소 확인
        required_elements = [
            "import React",
            f"const {component_name}",
            "export default",
            "return"
        ]
        
        return all(element in content for element in required_elements)
    
    async def _create_template_folder(self):
        """템플릿 폴더 생성"""
        try:
            self.template_folder.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"템플릿 폴더 생성: {self.template_folder}")
        except Exception as e:
            self.logger.error(f"템플릿 폴더 생성 실패: {e}")
    
    async def create_default_templates(self) -> List[str]:
        """기본 템플릿 생성"""
        
        self.logger.info("기본 JSX 템플릿 생성 시작")
        
        default_templates = []
        
        for i in range(1, 13):
            template_name = f"Section{i:02d}.jsx"
            template_content = self._generate_default_template_content(template_name)
            
            file_path = self.template_folder / template_name
            
            try:
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(template_content)
                
                default_templates.append(template_name)
                self.logger.debug(f"기본 템플릿 생성: {template_name}")
                
            except Exception as e:
                self.logger.error(f"기본 템플릿 생성 실패 {template_name}: {e}")
        
        self.logger.info(f"기본 템플릿 {len(default_templates)}개 생성 완료")
        return default_templates
    
    def _generate_default_template_content(self, template_name: str) -> str:
        """기본 템플릿 내용 생성"""
        
        component_name = template_name.replace('.jsx', '')
        section_number = component_name.replace('Section', '')
        
        return f"""import React, {{ memo }} from 'react';

const {component_name} = memo(() => {{
  return (
    <section className="py-16 px-4 max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl md:text-4xl font-bold text-gray-800 mb-4">
          여행 이야기 {section_number}
        </h2>
        <p className="text-lg text-gray-600 mb-6">
          특별한 순간들
        </p>
      </div>
      
      <div className="prose prose-lg max-w-none">
        <p className="text-gray-700 leading-relaxed">
          멋진 여행 경험을 공유합니다.
        </p>
      </div>
    </section>
  );
}});

export default {component_name};"""
    
    async def get_template_metadata(self) -> Dict:
        """템플릿 메타데이터 수집"""
        
        templates = await self.scan_jsx_templates()
        
        metadata = {
            "total_templates": len(templates),
            "template_list": templates,
            "template_folder": str(self.template_folder),
            "scan_timestamp": asyncio.get_event_loop().time(),
            "template_details": []
        }
        
        # 각 템플릿의 상세 정보 수집
        for template in templates:
            file_path = self.template_folder / template
            try:
                stat = file_path.stat()
                metadata["template_details"].append({
                    "name": template,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "component_name": template.replace('.jsx', '')
                })
            except Exception as e:
                self.logger.error(f"템플릿 메타데이터 수집 실패 {template}: {e}")
        
        return metadata
    
    async def watch_template_changes(self, callback=None):
        """템플릿 폴더 변경 감지 (선택적 기능)"""
        
        self.logger.info("템플릿 폴더 변경 감지 시작")
        
        last_templates = await self.scan_jsx_templates()
        
        while True:
            await asyncio.sleep(5)  # 5초마다 체크
            
            current_templates = await self.scan_jsx_templates()
            
            if current_templates != last_templates:
                self.logger.info("템플릿 변경 감지!")
                self.logger.info(f"이전: {last_templates}")
                self.logger.info(f"현재: {current_templates}")
                
                if callback:
                    await callback(current_templates)
                
                last_templates = current_templates
