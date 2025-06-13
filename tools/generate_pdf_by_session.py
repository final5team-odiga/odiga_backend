#!/usr/bin/env python
# coding: utf-8
"""
특정 magazine_id + session_id 로컬 PDF 생성기
"""

import argparse, asyncio, logging, sys
from pathlib import Path

# 프로젝트 루트 경로를 PYTHONPATH 에 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.service.pdf.pdf_generater import PDFGenerationService  # 기존 모듈[2]

logging.basicConfig(level=logging.INFO, format="%(asctime)s ▶ %(message)s")
log = logging.getLogger("pdf_cli")

async def _run(mid: str, sid: str, out: str):
    svc = PDFGenerationService()
    ok = await svc.generate_pdf_from_cosmosdb_by_session(mid, sid, out)
    if ok:
        log.info(f"🎉 완료: {out}")
    else:
        log.error("⛔ 실패")

def main():
    ap = argparse.ArgumentParser(description="CosmosDB 세션 → PDF")
    ap.add_argument("--magazine", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--output", default="magazine_result.pdf")
    args = ap.parse_args()

    asyncio.run(_run(args.magazine, args.session, args.output))

if __name__ == "__main__":
    main()
