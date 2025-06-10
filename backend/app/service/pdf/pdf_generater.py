import os
import asyncio
import tempfile
import shutil
import logging

# 추가: Cosmos DB 유틸리티 가져오기
from db.cosmos_connection import jsx_container
from db.db_utils import get_from_cosmos

class PDFGenerationService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
 
    async def generate_pdf_from_cosmosdb(self, magazine_id: str, output_pdf_path: str = "magazine_result.pdf") -> bool:
        """Cosmos DB에서 JSX 컴포넌트를 가져와 PDF 생성 (프로젝트 루트 기반)"""
        temp_dir = None
        try:
            # 1. Cosmos DB에서 JSX 컴포넌트 조회
            query = f"SELECT * FROM c WHERE c.magazine_id = '{magazine_id}' ORDER BY c.order_index"
            items = list(jsx_container.query_items(query=query, enable_cross_partition_query=True))
            
            if not items:
                self.logger.error(f"매거진 ID {magazine_id}에 대한 JSX 컴포넌트를 찾을 수 없습니다.")
                return False
            
            # 2. 프로젝트 루트 경로 확인
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            self.logger.info(f"프로젝트 루트: {project_root}")
            
            # ✅ 3. 프로젝트 루트에 필요한 파일들이 있는지 확인
            package_json_path = os.path.join(project_root, "package.json")
            node_modules_path = os.path.join(project_root, "node_modules")
            
            if not os.path.exists(package_json_path):
                raise FileNotFoundError(f"package.json이 없습니다: {package_json_path}")
            if not os.path.exists(node_modules_path):
                raise FileNotFoundError(f"node_modules가 없습니다: {node_modules_path}")
            
            self.logger.info("✅ package.json과 node_modules 확인 완료")
            
            # 4. 임시 디렉토리 생성 (프로젝트 루트 하위에)
            temp_dir = tempfile.mkdtemp(prefix="jsx_pdf_", dir=project_root)
            self.logger.info(f"임시 디렉토리 생성: {temp_dir}")
            
            # 5. JSX 파일 저장
            jsx_files = []
            for i, item in enumerate(items):
                jsx_code = item.get('jsx_code')
                if not jsx_code:
                    continue
                    
                order_index = item.get('order_index', i)
                filename = f"Section{order_index+1:02d}.jsx"
                file_path = os.path.join(temp_dir, filename)
                
                # ✅ JSX 코드 정리 (이스케이프 문자 처리)
                cleaned_jsx_code = jsx_code.replace('\\n', '\n').replace('\\"', '"')
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_jsx_code)
                
                jsx_files.append(file_path)
                self.logger.info(f"JSX 파일 저장: {filename}")
            
            # 6. export_pdf.js 실행
            script_path = os.path.join(project_root, "service", "export_pdf.js")
            
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"export_pdf.js를 찾을 수 없습니다: {script_path}")
            
            cmd = [
                "node",
                script_path,
                "--files", *jsx_files,
                "--output", os.path.abspath(output_pdf_path)
            ]
            
            self.logger.info("PDF 생성 시작...")
            
            # ✅ 핵심: 프로젝트 루트를 작업 디렉토리로 설정
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root  # ✅ 프로젝트 루트에서 실행
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
            
        finally:
            # 임시 파일 정리
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    self.logger.info(f"임시 디렉토리 삭제: {temp_dir}")
                except Exception as e:
                    self.logger.warning(f"임시 디렉토리 삭제 실패: {e}")

                
    def generate_pdf_from_db(self, magazine_id: str, output_pdf_path: str = "magazine_result.pdf") -> bool:
        """
        Cosmos DB에서 JSX 컴포넌트를 가져와 PDF를 생성하는 동기 메서드 (편의를 위한 인터페이스)
        """
        return asyncio.run(self.generate_pdf_from_cosmosdb(magazine_id, output_pdf_path))
