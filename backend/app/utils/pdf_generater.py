import os
import glob
import asyncio
from typing import List
import logging

class PDFGenerationService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def find_magazine_components(self, base_path: str = None) -> str:
        """ 내의 components 경로 찾기"""
        if base_path is None:
            base_path = os.path.abspath("output")
            
        pattern = os.path.join(base_path, "**/magazine_app_*/components")
        all_component_dirs = glob.glob(pattern, recursive=True)
        
        if not all_component_dirs:
            # pdf_components 폴더도 확인
            pdf_components = os.path.abspath("pdf_components")
            if os.path.exists(pdf_components):
                return pdf_components
            raise FileNotFoundError("컴포넌트 폴더를 찾을 수 없습니다.")
        
        # 가장 최근 폴더 선택
        latest_dir = max(all_component_dirs, key=os.path.getmtime)
        return latest_dir

    def collect_jsx_files(self, components_path: str) -> List[str]:
        """JSX 파일들을 수집하고 정렬"""
        jsx_patterns = [
            os.path.join(components_path, "*.jsx"),
        ]
        
        jsx_files = []
        for pattern in jsx_patterns:
            jsx_files.extend(glob.glob(pattern))
        
        # 중복 제거 및 정렬
        jsx_files = list(set(jsx_files))
        jsx_files.sort(key=lambda x: os.path.basename(x))
        
        if not jsx_files:
            raise FileNotFoundError(f"JSX 파일을 찾을 수 없습니다: {components_path}")
            
        return [os.path.abspath(p) for p in jsx_files]

    async def preprocess_components(self, jsx_files: List[str]) -> List[str]:
        """컴포넌트 전처리 (이미지 URL 검증 등)"""
        processed_files = []
        
        for jsx_file in jsx_files:
            try:
                with open(jsx_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 외부 이미지 URL을 플레이스홀더로 대체하는 로직 추가 가능
                # 현재는 원본 파일 경로 그대로 사용
                processed_files.append(jsx_file)
                
            except Exception as e:
                self.logger.warning(f"컴포넌트 전처리 실패: {jsx_file} - {e}")
                continue
                
        return processed_files

    async def generate_pdf_async(self, output_pdf_path: str = "magazine_result.pdf"):
        """비동기 PDF 생성"""
        try:
            # 1. 컴포넌트 폴더 찾기
            components_path = self.find_magazine_components()
            self.logger.info(f"컴포넌트 폴더 발견: {components_path}")
            
            # 2. JSX 파일 수집
            jsx_files = self.collect_jsx_files(components_path)
            self.logger.info(f"JSX 파일 {len(jsx_files)}개 발견")
            
            # 3. 컴포넌트 전처리
            processed_files = await self.preprocess_components(jsx_files)
            
            if not processed_files:
                raise FileNotFoundError("처리 가능한 JSX 파일이 없습니다.")
            
            # 4. export_pdf.js 실행
            script_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../pdf_service/export_pdf.js")
            )
            
            cmd = [
                "node",
                script_path,
                "--files", *processed_files,
                "--output", os.path.abspath(output_pdf_path)
            ]
            
            self.logger.info("PDF 생성 시작...")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"✅ PDF 생성 완료: {output_pdf_path}")
                return True
            else:
                self.logger.error(f"❌ PDF 생성 실패: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"PDF 생성 중 오류: {e}")
            return False

    def generate_pdf(self, output_pdf_path: str = "magazine_result.pdf"):
        """동기 PDF 생성 (기존 인터페이스 유지)"""
        return asyncio.run(self.generate_pdf_async(output_pdf_path))
