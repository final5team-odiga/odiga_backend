import os
import sys

import asyncio
from agents.system_coordinator import SystemCoordinator
from utils.hybridlogging import get_hybrid_logger
from utils.template_scanner import TemplateScanner


if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

async def main():
    """완전 통합 멀티모달 매거진 생성"""
    
    logger = get_hybrid_logger("Main")
    logger.info("=== 통합 멀티모달 매거진 생성 시스템 시작 ===")
    
    try:
        # 통합 시스템 조율자 초기화
        system_coordinator = SystemCoordinator()
        
        # 동적 템플릿 스캔
        template_scanner = TemplateScanner()
        available_templates = await template_scanner.scan_jsx_templates()
        
        logger.info(f"발견된 JSX 템플릿: {len(available_templates)}개")
        logger.info(f"템플릿 목록: {available_templates}")
        
        # 템플릿이 없는 경우 처리
        if not available_templates:
            logger.warning("JSX 템플릿을 찾을 수 없습니다. 기본 템플릿을 생성합니다.")
            available_templates = await template_scanner.create_default_templates()
        
        # 단일 통합 처리
        final_result = await system_coordinator.coordinate_complete_magazine_generation(
            available_templates=available_templates
        )
        
        # 결과 요약 출력
        processing_summary = final_result.get("processing_summary", {})
        logger.info(f"""
=== 매거진 생성 완료 ===
- 총 섹션 수: {processing_summary.get('total_sections', 0)}
- JSX 컴포넌트 수: {processing_summary.get('total_jsx_components', 0)}
- 사용된 템플릿: {len(available_templates)}개
- 의미적 신뢰도: {processing_summary.get('semantic_confidence', 0.0):.2f}
- 멀티모달 최적화: {processing_summary.get('multimodal_optimization', False)}
- 반응형 디자인: {processing_summary.get('responsive_design', False)}
        """)
        
        logger.info("=== 통합 멀티모달 매거진 생성 시스템 완료 ===")
        
    except Exception as e:
        logger.error(f"매거진 생성 실패: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
