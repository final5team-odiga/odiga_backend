import os
import glob
import subprocess
from typing import List


class PDFGenerationService:
    def find_latest_magazine_folder(self, output_root: str) -> str:
        """
        output/ 경로 내에서 가장 최신 magazine_app_ 폴더 경로 반환
        """
        pattern = os.path.join(output_root, "**", "magazine_app_*")
        all_folders = glob.glob(pattern, recursive=True)

        print("🔍 찾은 폴더 목록:", all_folders)  # 디버깅용

        if not all_folders:
            raise FileNotFoundError("magazine_app_ 폴더가 존재하지 않습니다.")

        latest_folder = max(all_folders, key=os.path.getmtime)
        return latest_folder

    def collect_jsx_files(self, magazine_folder: str) -> List[str]:
        components_path = os.path.join(magazine_folder, "components")
        jsx_files = glob.glob(os.path.join(components_path, "*.jsx"))

        if not jsx_files:
            raise FileNotFoundError("JSX 컴포넌트 파일을 찾을 수 없습니다.")

        return [os.path.abspath(p) for p in jsx_files]

    def generate_pdf(self, output_pdf_path: str = "magazine_result.pdf"):
        # ✅ pdf_components 폴더에 있는 Section JSX 파일만 수집
        component_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../pdf_components")
        )
        jsx_paths = sorted(glob.glob(os.path.join(component_dir, "Section*.jsx")))

        # 🧱 JSX 파일이 없다면 generatePdfComponents.js 자동 실행
        if not jsx_paths:
            print("📂 JSX 파일 없음 → generatePdfComponents.js 자동 실행 중...")
            script_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../scripts/generatePdfComponents.js")
            )
            try:
                subprocess.run(["node", script_path], check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"❌ JSX 생성 실패: {e}")

            # 다시 jsx_paths 재로드
            jsx_paths = sorted(glob.glob(os.path.join(component_dir, "Section*.jsx")))
            if not jsx_paths:
                raise FileNotFoundError("❌ 자동 생성 후에도 JSX 파일이 없습니다.")

        # ✅ export_pdf.js 실행
        script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../pdf_service/export_pdf.js")
        )

        try:
            subprocess.run([
                "node",
                script_path,
                "--files", *jsx_paths,
                "--output", os.path.abspath(output_pdf_path)
            ], check=True)
            print(f"✅ PDF 저장 완료: {output_pdf_path}")
        except FileNotFoundError:
            print("❌ Node.js가 설치되어 있는지 확인하세요 (node 명령어 없음)")
        except subprocess.CalledProcessError as e:
            print(f"❌ PDF 생성 실패: {str(e)}")
