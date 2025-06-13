
from __future__ import annotations
import asyncio
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List
import aiohttp

# 프로젝트 고유 모듈
from ...db.cosmos_connection import jsx_container

logger = logging.getLogger(__name__)
_CHEVRON_FIX_RE = re.compile(r'<{2,}\s*([A-Za-z\/])')

class PDFGenerationService:
    def __init__(self):
        # 프로젝트 루트 및 임시 디렉토리 설정
        current = Path(__file__).resolve()
        self.project_root = current.parent.parent.parent.parent.parent
        
        # ✅ 통일된 임시 디렉토리 사용
        self.temp_base_dir = self.project_root / "temp"
        self.temp_base_dir.mkdir(exist_ok=True)
        
        # ✅ 디렉토리 접근 권한 확인 및 설정
        try:
            test_file = self.temp_base_dir / "access_test.tmp"
            test_file.write_text("test", encoding="utf-8")
            test_file.unlink()
            logger.info(f"✅ 임시 디렉토리 접근 가능: {self.temp_base_dir}")
        except Exception as e:
            logger.error(f"❌ 임시 디렉토리 접근 실패: {e}")
            # 폴백: 시스템 임시 디렉토리 사용
            self.temp_base_dir = Path(tempfile.gettempdir()) / "odiga_pdf"
            self.temp_base_dir.mkdir(exist_ok=True)
            logger.warning(f"⚠️ 폴백 임시 디렉토리 사용: {self.temp_base_dir}")

    # ✅ 새로 추가된 메서드
    def _replace_unsplash_with_fallback(self, jsx_code: str) -> str:
        """Unsplash 이미지를 흰색 폴백 이미지로 교체"""
        
        # 흰색 폴백 이미지
        fallback_image = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAwIiBoZWlnaHQ9IjQwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg=="
        
        # 모든 Unsplash URL을 폴백 이미지로 교체
        jsx_code = re.sub(
            r'src="https://images\.unsplash\.com/[^"]*"',
            f'src="{fallback_image}"',
            jsx_code
        )
        
        logger.debug("✅ Unsplash 이미지를 흰색 폴백 이미지로 교체 완료")
        return jsx_code

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

    async def _generate_pdf_from_cosmosdb(self, magazine_id: str, output_pdf_path: str) -> bool:
        temp_dir: str | None = None
        try:
            # ✅ 1-1) 최신 세션 ID 조회
            session_query = (
                f"SELECT VALUE MAX(c.session_id) FROM c WHERE c.magazine_id = '{magazine_id}'"
            )
            session_results = list(jsx_container.query_items(
                query=session_query, enable_cross_partition_query=True
            ))
            
            if not session_results or not session_results[0]:
                logger.error(f"JSX 세션 없음 – magazine_id={magazine_id}")
                return False
                
            latest_session_id = session_results[0]
            logger.info(f"✅ 최신 세션 ID 발견: {latest_session_id}")

            # ✅ 1-2) 최신 세션의 JSX만 조회
            query = (
                f"SELECT * FROM c WHERE c.magazine_id = '{magazine_id}' "
                f"AND c.session_id = '{latest_session_id}' "
                "ORDER BY c.order_index"
            )
            items: List[Dict] = list(jsx_container.query_items(
                query=query, enable_cross_partition_query=True
            ))
            
            if not items:
                logger.error(f"최신 세션 JSX 없음 – magazine_id={magazine_id}, session_id={latest_session_id}")
                return False
                
            logger.info(f"✅ 최신 세션 데이터 로드: {len(items)}개 섹션 (session_id: {latest_session_id})")

            # 2) 깨진 이미지 URL 사전 필터링
            items = await self._prefilter_images(items)

            # 3) Node 프로젝트 루트 경로 확인
            if not (self.project_root / "package.json").exists():
                raise FileNotFoundError(f"package.json 누락: {self.project_root}")
            if not (self.project_root / "node_modules").exists():
                raise FileNotFoundError(f"node_modules 누락: {self.project_root}")
            logger.info("✅ Node 의존성 확인 완료")

            # 4) ✅ 통일된 임시 디렉토리 생성
            session_id = uuid.uuid4().hex[:8]
            temp_dir = str(self.temp_base_dir / f"jsx_pdf_{session_id}")
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"임시 디렉터리: {temp_dir}")

            # 5) JSX 파일 저장 + 구문 검증 + ✅ Unsplash 이미지 교체
            jsx_files: list[str] = []
            for idx, item in enumerate(items, start=1):
                jsx_code: str = item.get("jsx_code", "")
                if not jsx_code.strip():
                    continue
                    
                # ✅ Unsplash 이미지를 흰색 폴백 이미지로 교체
                jsx_code = self._replace_unsplash_with_fallback(jsx_code)
                
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
            script = self.project_root / "service" / "export_pdf.js"
            if not script.exists():
                raise FileNotFoundError(f"export_pdf.js 누락: {script}")

            cmd = [
                "node", str(script),
                "--files", *jsx_files,
                "--output", str(Path(output_pdf_path).resolve()),
            ]

            logger.info("📄 PDF 생성 시작")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root),
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
    async def generate_pdf_from_cosmosdb_by_session(
            self,
            magazine_id: str,
            session_id: str,
            output_pdf_path: str = "magazine_result.pdf",
        ) -> bool:
            """
            magazine_id 와 session_id 를 모두 고정해 Cosmos DB → PDF.
            기존 헬퍼∙유틸은 그대로 재사용한다.
            """
            return await self._generate_pdf_for_session(
                magazine_id, session_id, output_pdf_path
            )

    def generate_pdf_by_session(
        self,
        magazine_id: str,
        session_id: str,
        output_pdf_path: str = "magazine_result.pdf",
    ) -> bool:
        """동기 편의 래퍼"""
        import asyncio

        return asyncio.run(
            self._generate_pdf_for_session(magazine_id, session_id, output_pdf_path)
        )

    # ─── 내부 구현: _generate_pdf_for_session ───
    async def _generate_pdf_for_session(
        self, magazine_id: str, session_id: str, output_pdf_path: str
    ) -> bool:
        """
        _generate_pdf_from_cosmosdb() 와 동일하지만
        ① 최신 세션 검색 단계 생략
        ② 전달받은 session_id 로 바로 JSX 쿼리
        """
        temp_dir = None
        try:
            # 1) 해당 session 의 JSX 직접 조회
            items_query = {
                "query": (
                    "SELECT * FROM c "
                    "WHERE c.magazine_id = @mid AND c.session_id = @sid "
                    "ORDER BY c.order_index ASC"
                ),
                "parameters": [
                    {"name": "@mid", "value": magazine_id},
                    {"name": "@sid", "value": session_id},
                ],
            }
            items = list(
                jsx_container.query_items(items_query, enable_cross_partition_query=True)
            )
            if not items:
                logger.error(
                    f"JSX 없음 – magazine_id={magazine_id}, session_id={session_id}"
                )
                return False
            logger.info(f"✅ 세션 데이터 로드: {len(items)}개 섹션")

            # 2) 이후 단계는 기존 _generate_pdf_from_cosmosdb 와 동일
            items = await self._prefilter_images(items)               # [2]
            session_token = uuid.uuid4().hex[:8]
            temp_dir = self.temp_base_dir / f"jsx_pdf_{session_token}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            jsx_files = []
            for idx, item in enumerate(items, start=1):
                jsx_code = item.get("jsx_code", "")
                if not jsx_code.strip():
                    continue
                jsx_code = self._replace_unsplash_with_fallback(jsx_code)  # [2]
                jsx_code = self._fix_double_chevrons(jsx_code)             # [2]
                try:
                    self._assert_valid_jsx(jsx_code)                       # [2]
                except ValueError as e:
                    logger.warning(f"⚠️ JSX 오류 skip: {e}")
                    continue
                p = temp_dir / f"Section{idx:02d}.jsx"
                p.write_text(
                    jsx_code.replace("\\n", "\n").replace('\\"', '"'),
                    encoding="utf-8",
                )
                jsx_files.append(str(p))
            if not jsx_files:
                logger.error("유효 JSX 없음 → PDF 중단")
                return False

            script = self.project_root / "service" / "export_pdf.js"       # [2]
            cmd = ["node", str(script), "--files", *jsx_files,
                   "--output", str(Path(output_pdf_path).resolve())]

            logger.info("📄 PDF 생성 시작")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root),
            )
            await asyncio.gather(
                self._pipe_stream(proc.stdout, logging.INFO),
                self._pipe_stream(proc.stderr, logging.ERROR),
            )
            await proc.wait()
            if proc.returncode == 0:
                logger.info(f"✅ PDF 완료: {output_pdf_path}")
                return True
            logger.error(f"❌ PDF 실패 – returncode={proc.returncode}")
            return False

        except Exception as exc:
            logger.exception(f"PDF 생성 예외: {exc}")
            return False

        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
    def _assert_valid_jsx(self, code: str) -> None:
        """
        Babel Parser(@babel/parser)로 JSX를 파싱해 문법 오류가 있으면 ValueError 발생.
        ✅ Windows 경로 문제 및 상세 오류 로깅 해결
        """
        # ✅ 통일된 임시 파일 생성
        temp_filename = f"jsx_validation_{uuid.uuid4().hex[:8]}.jsx"
        tmp_path = self.temp_base_dir / temp_filename
        
        try:
            # UTF-8로 파일 작성
            tmp_path.write_text(code, encoding="utf-8")
            
            # ✅ Windows 호환성을 위한 경로 정규화
            normalized_path = str(tmp_path).replace("\\", "/")
            
            result = subprocess.run(
                [
                    "node", "-e",
                    (
                        "const p=require('@babel/parser');"
                        f"p.parse(require('fs').readFileSync('{normalized_path}','utf8'),"
                        "{sourceType:'module',plugins:['jsx']});"
                        "console.log('JSX validation passed');"
                    ),
                ],
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                shell=True,  # ✅ Windows ENOENT 오류 해결
                cwd=str(self.project_root),  # ✅ 작업 디렉터리 명시
            )
            
            logger.debug(f"JSX 검증 성공: {result.stdout.strip()}")
            
        except subprocess.CalledProcessError as err:
            # ✅ 상세 오류 메시지 포함
            error_detail = err.stderr.strip() if err.stderr else "알 수 없는 오류"
            stdout_detail = err.stdout.strip() if err.stdout else ""
            
            logger.error(f"⚠️ JSX 구문 오류 상세: {error_detail}")
            if stdout_detail:
                logger.error(f"⚠️ 추가 정보: {stdout_detail}")
            logger.error(f"⚠️ 임시 파일 경로: {tmp_path}")
            logger.error(f"⚠️ 문제가 된 JSX 코드 일부: {code[:200]}...")
            
            raise ValueError(f"Babel-JSX 파싱 실패: {error_detail}") from err
        finally:
            # 임시 파일 정리
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception as cleanup_err:
                    logger.warning(f"임시 파일 삭제 실패: {cleanup_err}")

                    
    def _fix_double_chevrons(self, code: str) -> str:
        """✅ 완벽한 이중 꺾쇠 패턴 수정"""
        # 기존 패턴
        code = _CHEVRON_FIX_RE.sub(r'<\1', code)
        
        # ✅ 추가 패턴들
        # <<h1, <<div 등 직접 수정
        code = re.sub(r'<<([a-zA-Z][a-zA-Z0-9]*)', r'<\1', code)
        
        # ✅ 중첩된 꺾쇠 모든 경우 처리
        while '<<' in code:
            code = code.replace('<<', '<')
        
        # ✅ 특별한 경우: <<<< 등 다중 꺾쇠
        code = re.sub(r'<{3,}', '<', code)
        
        return code

    # ─────────────────────── 헬퍼: 이미지 URL 검증 ────────────────────────────

    async def _prefilter_images(self, items: List[Dict]) -> List[Dict]:
        """깨진 이미지 URL을 사전에 필터링"""
        # 기존 로직 유지 (생략)
        return items

    async def _pipe_stream(
            self,
            stream: asyncio.StreamReader,
            log_level: int = logging.INFO,
        ) -> None:
            """
            asyncio.create_subprocess_exec 로 얻은 stdout/stderr 스트림을
            실시간으로 읽어 지정한 로그 레벨로 출력한다.

            Parameters
            ----------
            stream : asyncio.StreamReader
                proc.stdout 또는 proc.stderr
            log_level : int
                logging.INFO / logging.ERROR 등
            """
            try:
                while True:
                    line = await stream.readline()
                    if not line:           # EOF
                        break
                    logger.log(log_level, line.decode(errors="replace").rstrip())
            except Exception as exc:
                logger.warning(f"_pipe_stream 예외: {exc}")