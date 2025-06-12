
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

import aiohttp

# 프로젝트 고유 모듈
from ...db.cosmos_connection import jsx_container

logger = logging.getLogger(__name__)

_CHEVRON_FIX_RE = re.compile(r'<{2,}\s*([A-Za-z\/])')

class PDFGenerationService:
    # ─────────────────────────────── 퍼블릭 API ────────────────────────────────
    async def generate_pdf_from_cosmosdb(
        self, magazine_id: str, output_pdf_path: str
    ) -> bool:
        """SystemCoordinator 가 await 하는 비동기 진입점"""
        return await self._generate_pdf_from_cosmosdb(magazine_id, output_pdf_path)

    def generate_pdf_from_db(
        self, magazine_id: str, output_pdf_path: str = "magazine_result.pdf"
    ) -> bool:
        """백오퍼드 호환용 동기 래퍼"""
        return asyncio.run(
            self._generate_pdf_from_cosmosdb(magazine_id, output_pdf_path)
        )

    # ─────────────────────── 내부: CosmosDB → PDF(비동기) ────────────────────────
    async def _generate_pdf_from_cosmosdb(
        self, magazine_id: str, output_pdf_path: str
    ) -> bool:
        temp_dir: str | None = None

        try:
            # 1) JSX 목록 조회
            query = (
                f"SELECT * FROM c WHERE c.magazine_id = '{magazine_id}' "
                "ORDER BY c.order_index"
            )
            items: List[Dict] = list(
                jsx_container.query_items(
                    query=query, enable_cross_partition_query=True
                )
            )
            if not items:
                logger.error(f"JSX 없음 – magazine_id={magazine_id}")
                return False

            # 2) 깨진 이미지 URL 사전 필터링
            items = await self._prefilter_images(items)

            # 3) Node 프로젝트 루트 경로 확인
            current = Path(__file__).resolve()
            project_root = current.parent.parent.parent.parent.parent
            if not (project_root / "package.json").exists():
                raise FileNotFoundError("package.json 누락")
            if not (project_root / "node_modules").exists():
                raise FileNotFoundError("node_modules 누락")
            logger.info("✅ Node 의존성 확인 완료")

            # 4) 임시 디렉터리 생성
            temp_dir = tempfile.mkdtemp(prefix="jsx_pdf_", dir=project_root)
            logger.info(f"임시 디렉터리: {temp_dir}")

            # 5) JSX 파일 저장 + 구문 검증
            jsx_files: list[str] = []
            for idx, item in enumerate(items, start=1):
                jsx_code: str = item.get("jsx_code", "")
                if not jsx_code.strip():
                    continue

                jsx_code = self._fix_double_chevrons(jsx_code)

                try:
                    self._assert_valid_jsx(jsx_code)
                except ValueError as err:
                    logger.error(f"⚠️ JSX 구문 오류, 제외: Section{idx:02d}.jsx – {err}")
                    continue

                file_path = Path(temp_dir) / f"Section{idx:02d}.jsx"
                file_path.write_text(
                    jsx_code.replace("\\n", "\n").replace('\\"', '"'),
                    encoding="utf-8",
                )
                jsx_files.append(str(file_path))
                logger.debug(f"저장 완료: {file_path.name}")

            if not jsx_files:
                logger.error("유효한 JSX 파일이 없어서 PDF를 생성할 수 없습니다.")
                return False

            # 6) PDF 생성 스크립트 실행
            script = project_root / "service" / "export_pdf.js"
            if not script.exists():
                raise FileNotFoundError("export_pdf.js 누락")

            cmd = [
                "node",
                str(script),
                "--files",
                *jsx_files,
                "--output",
                str(Path(output_pdf_path).resolve()),
            ]
            logger.info("📄 PDF 생성 시작")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                logger.info(f"✅ PDF 완료: {output_pdf_path}")
                logger.debug(stdout.decode())
                return True

            logger.error(f"❌ PDF 실패\n{stderr.decode()}")
            return False

        except Exception as exc:
            logger.exception(f"PDF 생성 중 예외: {exc}")
            return False

        finally:
            if temp_dir and Path(temp_dir).is_dir():
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"임시 디렉터리 삭제: {temp_dir}")
                except Exception as exc:
                    logger.warning(f"임시 삭제 실패: {exc}")

    # ─────────────────────────── 헬퍼: JSX 구문 검증 ────────────────────────────
    def _assert_valid_jsx(self, code: str) -> None:
        """
        Babel Parser(@babel/parser)로 JSX를 파싱해 문법 오류가 있으면 ValueError 발생.
        ✅ 상세 오류 로깅 추가
        """
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsx", encoding="utf-8") as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            # ✅ stderr 캡처를 위해 run 사용
            result = subprocess.run(
                [
                    "node",
                    "-e",
                    (
                        "const p=require('@babel/parser');"
                        f"p.parse(require('fs').readFileSync('{tmp_path}','utf8'),"
                        "{sourceType:'module',plugins:['jsx']});"
                    ),
                ],
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
            )
        except subprocess.CalledProcessError as err:
            # ✅ 상세 오류 메시지 로깅
            error_detail = err.stderr.strip() if err.stderr else "알 수 없는 오류"
            logger.error(f"⚠️ JSX 구문 오류 상세: {error_detail}")
            logger.error(f"⚠️ 문제가 된 JSX 코드 일부: {code[:200]}...")
            raise ValueError(f"Babel-JSX 파싱 실패: {error_detail}") from err
        finally:
            Path(tmp_path).unlink(missing_ok=True)


    def _fix_double_chevrons(self, code: str) -> str:
        """이중 꺾쇠 패턴 수정 강화"""
        # 기존 패턴
        code = _CHEVRON_FIX_RE.sub(r'<\1', code)
        
        # ✅ 추가 패턴: <<h1, <<div 등 직접 수정
        code = re.sub(r'<<([a-zA-Z][a-zA-Z0-9]*)', r'<\1', code)
        
        # ✅ 중첩된 꺾쇠 모든 경우 처리
        while '<<' in code:
            code = code.replace('<<', '<')
        
        return code
    

    # ─────────────────────── 헬퍼: 깨진 이미지 URL 교체 ────────────────────────
    async def _head_ok(self, session: aiohttp.ClientSession, url: str) -> bool:
        """HEAD 200 이면 True, 그 외/예외 시 False"""
        try:
            async with session.head(url, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _prefilter_images(self, items: List[Dict]) -> List[Dict]:
        """
        JSX 코드에 포함된 이미지 URL 중 HEAD 200이 아닌 것은
        FALLBACK_URL 로 교체해 PDF 렌더링 시 빈 프레임이 남지 않도록 한다.
        """
        FALLBACK_URL = "https://static.example.com/fallback.png"
        async with aiohttp.ClientSession() as sess:
            for item in items:
                jsx = item.get("jsx_code", "")
                for bad_url in re.findall(r'src="(https?://[^"]+)"', jsx):
                    if not await self._head_ok(sess, bad_url):
                        jsx = jsx.replace(bad_url, FALLBACK_URL)
                item["jsx_code"] = jsx
        return items
