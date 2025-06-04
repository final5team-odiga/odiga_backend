import asyncio
import json
import time
from typing import Dict, Any, List
from utils.agent_decision_logger import get_agent_logger

class LoggingManager:
    """에이전트 응답 로그 전문 관리자 - 각 에이전트의 실제 응답 데이터만 수집"""
    
    def __init__(self):
        self.logger = get_agent_logger()
        self.agent_responses = {}  # 에이전트별 응답 저장소
        self.response_counter = 0
        
    async def log_agent_response(self, agent_name: str, agent_role: str, 
                                task_description: str, response_data: Any,
                                metadata: Dict = None) -> str:
        """에이전트 응답 로그 저장 (핵심 메서드)"""
        try:
            response_id = f"{agent_name}_{int(time.time() * 1000000)}"
            
            # 응답 데이터 정제
            processed_response = self._process_response_data(response_data)
            
            log_entry = {
                "response_id": response_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "task_description": task_description,
                "response_data": processed_response,
                "response_length": len(str(processed_response)),
                "timestamp": time.time(),
                "metadata": metadata or {}
            }
            
            # 에이전트별 응답 저장
            if agent_name not in self.agent_responses:
                self.agent_responses[agent_name] = []
            
            self.agent_responses[agent_name].append(log_entry)
            
            # agent_decision_logger에 저장
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.logger.log_agent_real_output(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    task_description=task_description,
                    final_answer=processed_response,
                    reasoning_process=f"에이전트 응답 로그 수집: {len(str(processed_response))}자",
                    performance_metrics={
                        "response_length": len(str(processed_response)),
                        "response_type": type(response_data).__name__,
                        "processing_success": True,
                        "log_entry_id": response_id
                    }
                )
            )
            
            self.response_counter += 1
            return response_id
            
        except Exception as e:
            print(f"에이전트 응답 로그 저장 실패 {agent_name}: {e}")
            return "log_save_failed"
    
    def _process_response_data(self, response_data: Any) -> str:
        """응답 데이터 정제 및 표준화"""
        try:
            if isinstance(response_data, str):
                return response_data
            elif isinstance(response_data, dict):
                return json.dumps(response_data, ensure_ascii=False, indent=2)
            elif isinstance(response_data, list):
                return json.dumps(response_data, ensure_ascii=False, indent=2)
            else:
                return str(response_data)
        except Exception as e:
            return f"응답 데이터 처리 실패: {str(e)}"
    
    # ✅ 기존 메서드들을 새로운 방식으로 통합
    async def log_image_analysis_completion(self, images_count: int, results_count: int):
        """이미지 분석 완료 로깅"""
        await self.log_agent_response(
            agent_name="ImageAnalyzerAgent",
            agent_role="이미지 분석 에이전트",
            task_description=f"{images_count}개 이미지 분석 완료",
            response_data={
                "images_processed": images_count,
                "results_generated": results_count,
                "success_rate": results_count / images_count if images_count > 0 else 0
            },
            metadata={"stage": "image_analysis", "async_processing": True}
        )
    
    async def log_content_creation_completion(self, texts_count: int, images_count: int, content_length: int):
        """콘텐츠 생성 완료 로깅"""
        await self.log_agent_response(
            agent_name="ContentCreatorV2Agent",
            agent_role="콘텐츠 생성 에이전트",
            task_description=f"{texts_count}개 텍스트와 {images_count}개 이미지로 콘텐츠 생성",
            response_data={
                "source_texts": texts_count,
                "source_images": images_count,
                "content_length": content_length,
                "content_richness": content_length / texts_count if texts_count > 0 else 0
            },
            metadata={"stage": "content_creation", "async_processing": True}
        )
    
    async def log_semantic_analysis_completion(self, analysis_result: Dict):
        """의미적 분석 완료 로깅"""
        await self.log_agent_response(
            agent_name="SemanticAnalysisEngine",
            agent_role="의미적 분석 엔진",
            task_description="텍스트-이미지 의미적 분석 완료",
            response_data=analysis_result,
            metadata={"stage": "semantic_analysis", "async_processing": True}
        )
    
    async def log_layout_generation_completion(self, layout_result: Dict):
        """레이아웃 생성 완료 로깅"""
        await self.log_agent_response(
            agent_name="RealtimeLayoutGenerator",
            agent_role="실시간 레이아웃 생성기",
            task_description="AI Search 기반 레이아웃 생성 완료",
            response_data=layout_result,
            metadata={"stage": "layout_generation", "async_processing": True}
        )
    
    async def log_diversity_optimization_completion(self, optimization_result: Dict):
        """이미지 다양성 최적화 완료 로깅"""
        await self.log_agent_response(
            agent_name="ImageDiversityManager",
            agent_role="이미지 다양성 관리자",
            task_description="이미지 다양성 최적화 완료",
            response_data=optimization_result,
            metadata={"stage": "diversity_optimization", "async_processing": True}
        )
    
    async def log_multimodal_processing_completion(self, processing_result: Dict):
        """멀티모달 처리 완료 로깅"""
        await self.log_agent_response(
            agent_name="UnifiedMultimodalAgent",
            agent_role="통합 멀티모달 에이전트",
            task_description="AI Search 통합 멀티모달 처리 완료",
            response_data=processing_result,
            metadata={"stage": "multimodal_processing", "async_processing": True}
        )
    
    async def log_jsx_generation_completion(self, generated_count: int, jsx_results: Dict):
        """JSX 생성 완료 로깅"""
        await self.log_agent_response(
            agent_name="UnifiedJSXGenerator",
            agent_role="통합 JSX 생성기",
            task_description=f"{generated_count}개 JSX 컴포넌트 생성 완료",
            response_data=jsx_results,
            metadata={"stage": "jsx_generation", "async_processing": True}
        )
    
    # ✅ 에이전트 응답 조회 메서드
    def get_agent_responses(self, agent_name: str = None) -> Dict:
        """에이전트별 응답 로그 조회"""
        if agent_name:
            return self.agent_responses.get(agent_name, [])
        return self.agent_responses
    
    def get_latest_response(self, agent_name: str) -> Dict:
        """특정 에이전트의 최신 응답 조회"""
        responses = self.agent_responses.get(agent_name, [])
        return responses[-1] if responses else {}
    
    def get_response_summary(self) -> Dict:
        """전체 응답 로그 요약"""
        summary = {
            "total_agents": len(self.agent_responses),
            "total_responses": sum(len(responses) for responses in self.agent_responses.values()),
            "agents_summary": {}
        }
        
        for agent_name, responses in self.agent_responses.items():
            summary["agents_summary"][agent_name] = {
                "response_count": len(responses),
                "latest_response_time": responses[-1]["timestamp"] if responses else 0,
                "total_response_length": sum(resp["response_length"] for resp in responses)
            }
        
        return summary
    
    # 동기 버전 메서드들 (호환성 유지)
    def log_image_analysis_completion_sync(self, images_count: int, results_count: int):
        return asyncio.run(self.log_image_analysis_completion(images_count, results_count))
    
    def log_content_creation_completion_sync(self, texts_count: int, images_count: int, content_length: int):
        return asyncio.run(self.log_content_creation_completion(texts_count, images_count, content_length))
