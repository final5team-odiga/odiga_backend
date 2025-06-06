import os
import glob
import asyncio
import tempfile
import shutil
from typing import List, Dict, Optional
import logging

# 추가: Cosmos DB 유틸리티 가져오기
from db.cosmos_connection import jsx_container
from db.db_utils import get_from_cosmos

class PDFGenerationService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def find_magazine_components(self, base_path: str = None) -> str:
        """ 내의 components 경로 찾기"""
        if base_path is None:
            base_path = os.path.abspath("output")
            
        # output/components 폴더를 직접 확인
        components_path = os.path.join(base_path, "components")
        
        if os.path.exists(components_path) and os.path.isdir(components_path):
            return components_path
        
        # 폴백: pdf_components 폴더도 확인
            pdf_components = os.path.abspath("pdf_components")
            if os.path.exists(pdf_components):
                return pdf_components
            
            raise FileNotFoundError("컴포넌트 폴더를 찾을 수 없습니다.")

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
                os.path.join(os.path.dirname(__file__), "export_pdf.js")
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
            
            stdout, stderr = await process.communicate()
            
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

    async def generate_pdf_from_cosmosdb(self, magazine_id: str, output_pdf_path: str = "magazine_result.pdf") -> bool:
        """
        Cosmos DB에 저장된 JSX 컴포넌트로부터 PDF를 생성합니다.
        
        Args:
            magazine_id: PDF로 생성할 매거진의 ID
            output_pdf_path: 생성할 PDF 파일 경로
            
        Returns:
            bool: PDF 생성 성공 여부
        """
        temp_dir = None
        try:
            # 1. Cosmos DB에서 JSX 컴포넌트 조회
            self.logger.info(f"매거진 ID {magazine_id}의 JSX 컴포넌트 조회 중...")
            
            # 쿼리로 매거진에 속한 모든 JSX 컴포넌트 검색
            query = f"SELECT * FROM c WHERE c.magazine_id = '{magazine_id}' ORDER BY c.order_index"
            items = list(jsx_container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            
            if not items:
                self.logger.error(f"매거진 ID {magazine_id}에 대한 JSX 컴포넌트를 찾을 수 없습니다.")
                return False
                
            self.logger.info(f"{len(items)}개의 JSX 컴포넌트를 발견했습니다.")
            
            # 2. 임시 디렉토리 생성 및 JSX 파일 저장
            temp_dir = tempfile.mkdtemp(prefix="jsx_pdf_")
            self.logger.info(f"임시 디렉토리 생성: {temp_dir}")
            
            jsx_files = []
            for i, item in enumerate(items):
                jsx_code = item.get('jsx_code')
                if not jsx_code:
                    self.logger.warning(f"항목 {item.get('id')}에 JSX 코드가 없습니다. 건너뜁니다.")
                    continue
                    
                # 순서에 따라 파일명 생성 (order_index가 있으면 사용, 없으면 쿼리 순서 사용)
                order_index = item.get('order_index', i)
                filename = f"Section{order_index+1:02d}.jsx"
                file_path = os.path.join(temp_dir, filename)
                
                # JSX 코드를 파일로 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(jsx_code)
                
                jsx_files.append(file_path)
                self.logger.info(f"JSX 컴포넌트 {i+1}/{len(items)} 저장: {filename}")
                
            # 3. 컴포넌트 전처리
            processed_files = await self.preprocess_components(jsx_files)
            
            if not processed_files:
                raise FileNotFoundError("처리 가능한 JSX 파일이 없습니다.")
                
            # 4. export_pdf.js 실행
            script_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "export_pdf.js")
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
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"✅ PDF 생성 완료: {output_pdf_path}")
                return True
            else:
                self.logger.error(f"❌ PDF 생성 실패: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Cosmos DB로부터 PDF 생성 중 오류: {e}")
            return False
            
        finally:
            # 임시 파일 정리
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                self.logger.info(f"임시 디렉토리 삭제: {temp_dir}")
                
    def generate_pdf_from_db(self, magazine_id: str, output_pdf_path: str = "magazine_result.pdf") -> bool:
        """
        Cosmos DB에서 JSX 컴포넌트를 가져와 PDF를 생성하는 동기 메서드 (편의를 위한 인터페이스)
        """
        return asyncio.run(self.generate_pdf_from_cosmosdb(magazine_id, output_pdf_path))
