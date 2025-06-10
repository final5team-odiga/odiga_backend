import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import asyncio
from agents.system_coordinator import SystemCoordinator
from utils.log.hybridlogging import get_hybrid_logger
from service.pdf_generater import PDFGenerationService

if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

async def main():
    """완전 통합 멀티모달 매거진 생성"""
    logger = get_hybrid_logger("Main")
    logger.info("=== 통합 멀티모달 매거진 생성 시스템 시작 ===")
    pdf_service = PDFGenerationService()
    
    try:
        system_coordinator = SystemCoordinator()
        
        # 매거진 생성 실행
        final_result = await system_coordinator.coordinate_complete_magazine_generation()
        
        magazine_id = final_result.get("magazine_id")
        if not magazine_id:
            # result 내부에서 magazine_id 찾기
            result_data = final_result.get("result", {})
            magazine_id = result_data.get("magazine_id")
        
        if magazine_id:
            pdf_service = PDFGenerationService()
            output_pdf_path = os.path.abspath("magazine_result.pdf")
            
            # ✅ magazine_id를 사용하여 Cosmos DB에서 JSX 컴포넌트 조회 후 PDF 생성
            success = await pdf_service.generate_pdf_from_cosmosdb(
                magazine_id=magazine_id,
                output_pdf_path=output_pdf_path
            )
            
            if success:
                logger.info(f"PDF 생성 완료: {output_pdf_path}")
            else:
                logger.error("PDF 생성 실패")
        else:
            logger.error("Magazine ID를 찾을 수 없어 PDF 생성을 건너뜁니다.")

        logger.info("=== 통합 멀티모달 매거진 생성 시스템 완료 ===")

    except Exception as e:
        logger.error(f"매거진 생성 실패: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
