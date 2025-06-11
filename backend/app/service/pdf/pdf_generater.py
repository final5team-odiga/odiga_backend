import os
import asyncio
import tempfile
import shutil
import logging
import re


from ...db.cosmos_connection import jsx_container

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
            
            
            current_file = os.path.abspath(__file__)
            # 2. 프로젝트 루트 경로 확인
            project_root = os.path.dirname(os.path.dirname(current_file))
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
            
            # ✅ 5. JSX 파일 저장 시 템플릿별 스타일 적용
            jsx_files = []
            for i, item in enumerate(items):
                jsx_code = item.get('jsx_code')
                if not jsx_code:
                    continue
                    
                order_index = item.get('order_index', i)
                
                # ✅ 템플릿 이름 추출
                template_name = self._extract_template_name_from_jsx(jsx_code)
                
                # ✅ 템플릿별 스타일 적용 (여기서 호출!)
                styled_jsx_code = self._apply_template_specific_styles(jsx_code, template_name)
                
                filename = f"Section{order_index+1:02d}.jsx"
                file_path = os.path.join(temp_dir, filename)
                
                # JSX 코드 정리 (이스케이프 문자 처리)
                cleaned_jsx_code = styled_jsx_code.replace('\\n', '\n').replace('\\"', '"')
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_jsx_code)
                
                jsx_files.append(file_path)
                self.logger.info(f"JSX 파일 저장: {filename} (템플릿: {template_name})")
            
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

    def _extract_template_name_from_jsx(self, jsx_code: str) -> str:
        """✅ JSX 코드에서 템플릿 이름 추출"""
        
        # const 컴포넌트명 패턴 찾기
        match = re.search(r'const\s+(\w+)\s*=', jsx_code)
        if match:
            component_name = match.group(1)
            self.logger.debug(f"추출된 컴포넌트 이름: {component_name}")
            return component_name
        
        # export default 패턴 찾기
        match = re.search(r'export\s+default\s+(\w+)', jsx_code)
        if match:
            component_name = match.group(1)
            self.logger.debug(f"추출된 컴포넌트 이름 (export): {component_name}")
            return component_name
        
        # 기본값
        return "Unknown"


    def _apply_template_specific_styles(self, jsx_code: str, template_name: str) -> str:
        """✅ 흰색 배경, 검정 텍스트 고정 조건에서 템플릿별 타이포그래피 적용"""
        
        # 템플릿별 스타일 매핑 (색상 제외, 타이포그래피 중심)
        template_styles = {
            "MixedMagazine07": {
                "container_class": "template-mixed-07 layout-elegant template-elegant",
                "title_class": "title-elegant",
                "subtitle_class": "subtitle-elegant", 
                "text_class": "text-elegant",
                "image_class": "image-elegant",
                "spacing_class": "spacing-loose"
            },
            "MixedMagazine08": {
                "container_class": "template-mixed-08 layout-gallery template-modern",
                "title_class": "title-modern",
                "subtitle_class": "subtitle-modern",
                "text_class": "text-modern", 
                "image_class": "image-small",
                "spacing_class": "spacing-normal"
            },
            "MixedMagazine09": {
                "container_class": "layout-magazine template-classic",
                "title_class": "title-classic",
                "subtitle_class": "subtitle-classic",
                "text_class": "text-classic",
                "image_class": "image-classic",
                "spacing_class": "spacing-normal"
            },
            "MixedMagazine10": {
                "container_class": "template-mixed-10 layout-modern template-modern",
                "title_class": "title-modern",
                "subtitle_class": "subtitle-modern",
                "text_class": "text-modern",
                "image_class": "image-modern",
                "spacing_class": "spacing-normal"
            },
            "MixedMagazine11": {
                "container_class": "layout-elegant template-elegant content-box-elegant",
                "title_class": "title-elegant",
                "subtitle_class": "subtitle-elegant",
                "text_class": "text-elegant",
                "image_class": "image-large",
                "spacing_class": "spacing-loose"
            },
            "MixedMagazine12": {
                "container_class": "layout-modern template-modern content-box-modern",
                "title_class": "title-modern",
                "subtitle_class": "subtitle-modern",
                "text_class": "text-modern",
                "image_class": "image-modern",
                "spacing_class": "spacing-normal"
            },
            "MixedMagazine13": {
                "container_class": "template-mixed-13 layout-gallery template-modern",
                "title_class": "title-modern",
                "subtitle_class": "subtitle-modern",
                "text_class": "text-modern",
                "image_class": "image-small",
                "spacing_class": "spacing-tight"
            },
            "MixedMagazine14": {
                "container_class": "layout-elegant template-elegant",
                "title_class": "title-elegant",
                "subtitle_class": "subtitle-elegant",
                "text_class": "text-elegant",
                "image_class": "image-elegant",
                "spacing_class": "spacing-loose"
            },
            "MixedMagazine15": {
                "container_class": "layout-magazine template-modern content-box-modern",
                "title_class": "title-modern",
                "subtitle_class": "subtitle-modern",
                "text_class": "text-modern",
                "image_class": "image-large",
                "spacing_class": "spacing-normal"
            },
            "MixedMagazine16": {
                "container_class": "layout-classic template-classic content-box-minimal",
                "title_class": "title-minimal",
                "subtitle_class": "subtitle-classic",
                "text_class": "text-classic",
                "image_class": "image-classic",
                "spacing_class": "spacing-normal"
            }
        }
        
        # 기본 스타일
        default_style = {
            "container_class": "layout-classic template-classic",
            "title_class": "title-classic",
            "subtitle_class": "subtitle-classic",
            "text_class": "text-classic",
            "image_class": "image-classic",
            "spacing_class": "spacing-normal"
        }
        
        # 템플릿 스타일 선택
        style = template_styles.get(template_name, default_style)
        
        # ✅ 컨테이너 클래스 적용 (배경색 흰색 강제)
        jsx_code = jsx_code.replace(
            'className="bg-white text-black p-8',
            f'className="bg-white text-black p-8 {style["container_class"]} {style["spacing_class"]}'
        )
        jsx_code = jsx_code.replace(
            'style={{ backgroundColor: "white", color: "black"',
            f'style={{ backgroundColor: "white", color: "black"'
        )
        
        # ✅ 제목 스타일 적용
        jsx_code = jsx_code.replace(
            'className="text-6xl font-bold',
            f'className="text-6xl font-bold {style["title_class"]}'
        )
        jsx_code = jsx_code.replace(
            'className="text-5xl font-bold',
            f'className="text-5xl font-bold {style["title_class"]}'
        )
        jsx_code = jsx_code.replace(
            'className="text-4xl font-bold',
            f'className="text-4xl font-bold {style["title_class"]}'
        )
        
        # ✅ 부제목 스타일 적용
        jsx_code = jsx_code.replace(
            'className="text-2xl',
            f'className="text-2xl {style["subtitle_class"]}'
        )
        jsx_code = jsx_code.replace(
            'className="text-xl',
            f'className="text-xl {style["subtitle_class"]}'
        )
        jsx_code = jsx_code.replace(
            'className="text-lg',
            f'className="text-lg {style["subtitle_class"]}'
        )
        
        # ✅ 본문 텍스트 스타일 적용
        jsx_code = jsx_code.replace(
            '<p className="',
            f'<p className="{style["text_class"]} '
        )
        jsx_code = jsx_code.replace(
            '<div className="text-',
            f'<div className="{style["text_class"]} text-'
        )
        
        # ✅ 이미지 스타일 적용
        jsx_code = jsx_code.replace(
            '<img',
            f'<img className="{style["image_class"]}"'
        )
        
        return jsx_code