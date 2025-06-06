# main.py (PDF 생성 부분만 수정)
import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import asyncio
from agents.system_coordinator import SystemCoordinator
from utils.log.hybridlogging import get_hybrid_logger
from utils.data.pdf_vector_manager import PDFVectorManager
from service.pdf_generater import PDFGenerationService

if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

async def main():
    """완전 통합 멀티모달 매거진 생성"""
    logger = get_hybrid_logger("Main")
    logger.info("=== 통합 멀티모달 매거진 생성 시스템 시작 ===")

    try:
        system_coordinator = SystemCoordinator()
        
        # PDFVectorManager를 초기화하고 인덱스 연결 상태 확인
        pdf_vector_manager = PDFVectorManager()
        # 매거진 생성 실행
        final_result = await system_coordinator.coordinate_complete_magazine_generation()

        print("🧪 final_result =", final_result)

        # 결과 요약 출력
        processing_summary = final_result.get("processing_summary", {})
        logger.info(f"""
=== 매거진 생성 완료 ===
- 총 섹션 수: {processing_summary.get('total_sections', 0)}
- JSX 컴포넌트 수: {processing_summary.get('total_jsx_components', 0)}
- 사용된 템플릿: {processing_summary.get('templates_used', 0)}개
- 의미적 신뢰도: {processing_summary.get('semantic_confidence', 0.0):.2f}
- 멀티모달 최적화: {processing_summary.get('multimodal_optimization', False)}
- 반응형 디자인: {processing_summary.get('responsive_design', False)}
""")
        
        logger.info("JSX 파일 기반으로 PDF 생성 시작...")
        pdf_service = PDFGenerationService()
        output_pdf_path = os.path.abspath("magazine_result.pdf")
        
        success = await pdf_service.generate_pdf_async(output_pdf_path=output_pdf_path)
        
        if success:
            logger.info(f"PDF 생성 완료: {output_pdf_path}")
        else:
            logger.error("PDF 생성 실패")

        logger.info("=== 통합 멀티모달 매거진 생성 시스템 완료 ===")

    except Exception as e:
        logger.error(f"매거진 생성 실패: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
