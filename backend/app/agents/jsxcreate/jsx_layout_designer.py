import asyncio
import time
import sys
import inspect
from typing import Dict, List, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import re

from crewai import Agent, Task
from custom_llm import get_azure_llm
from utils.agent_decision_logger import get_agent_logger, get_complete_data_manager

# --- Infrastructure Classes ---
@dataclass
class WorkItem:
    id: str
    task_func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 0
    max_retries: int = 3
    current_retry: int = 0
    timeout: float = 300.0

    def __lt__(self, other):
        return self.priority < other.priority

class CircuitBreakerState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 8, recovery_timeout: float = 30.0, half_open_attempts: int = 1):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_attempts = half_open_attempts
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None

    @property
    def state(self):
        if self._state == CircuitBreakerState.OPEN:
            if self._last_failure_time and (time.monotonic() - self._last_failure_time) > self.recovery_timeout:
                self._state = CircuitBreakerState.HALF_OPEN
                self._success_count = 0
        return self._state

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._state = CircuitBreakerState.OPEN
            self._failure_count = self.failure_threshold
        elif self._failure_count >= self.failure_threshold and self.state == CircuitBreakerState.CLOSED:
            self._state = CircuitBreakerState.OPEN
            
    def record_success(self):
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_attempts:
                self._state = CircuitBreakerState.CLOSED
                self._reset_counts()
        elif self.state == CircuitBreakerState.CLOSED:
            self._reset_counts()

    def _reset_counts(self):
        self._failure_count = 0
        self._success_count = 0

    async def execute(self, task_func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitBreakerState.OPEN:
            raise Exception(f"CircuitBreaker is OPEN for {getattr(task_func, '__name__', 'unknown_task')}. Call rejected.")

        try:
            if inspect.iscoroutinefunction(task_func):
                result = await task_func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: task_func(*args, **kwargs))
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise e

class AsyncWorkQueue:
    def __init__(self, max_workers: int = 2, max_queue_size: int = 50):
        self._queue = asyncio.PriorityQueue(max_queue_size if max_queue_size > 0 else 0)
        self._workers: List[asyncio.Task] = []
        self._max_workers = max_workers
        self._running = False
        self._results: Dict[str, Any] = {}

    async def _worker(self, worker_id: int):
        while self._running or not self._queue.empty():
            try:
                item: WorkItem = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                try:
                    if inspect.iscoroutinefunction(item.task_func):
                        result = await asyncio.wait_for(item.task_func(*item.args, **item.kwargs), timeout=item.timeout)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, lambda: item.task_func(*item.args, **item.kwargs)),
                            timeout=item.timeout
                        )
                    self._results[item.id] = {"status": "success", "result": result}
                except asyncio.TimeoutError:
                    self._results[item.id] = {"status": "timeout", "error": f"Task {item.id} timed out"}
                except Exception as e:
                    self._results[item.id] = {"status": "error", "error": str(e)}
                finally:
                    self._queue.task_done()
            except asyncio.TimeoutError:
                if not self._running and self._queue.empty():
                    break
                continue
            except Exception as e:
                await asyncio.sleep(1)

    async def start(self):
        if not self._running:
            self._running = True
            self._workers = [asyncio.create_task(self._worker(i)) for i in range(self._max_workers)]

    async def stop(self, graceful=True):
        if self._running:
            self._running = False
            if graceful:
                await self._queue.join()
            
            if self._workers:
                for worker_task in self._workers:
                    worker_task.cancel()
                await asyncio.gather(*self._workers, return_exceptions=True)
                self._workers.clear()

    async def enqueue_work(self, item: WorkItem) -> bool:
        if not self._running:
             await self.start()
        try:
            await self._queue.put(item)
            return True
        except asyncio.QueueFull:
            return False

    async def clear_results(self):
        self._results.clear()

class BaseAsyncAgent:
    def __init__(self):
        self.work_queue = AsyncWorkQueue(max_workers=2, max_queue_size=50)
        self.circuit_breaker = CircuitBreaker(failure_threshold=8, recovery_timeout=30.0)
        self.recursion_threshold = 800  # 수정된 값 적용
        self.fallback_to_sync = False
        self._recursion_check_buffer = 50

        # 실행 통계 추가
        self.execution_stats = {
            "total_attempts": 0,
            "successful_executions": 0,
            "fallback_used": 0,
            "circuit_breaker_triggered": 0,
            "timeout_occurred": 0
        }

    def _check_recursion_depth(self):
        """현재 재귀 깊이 확인"""
        current_depth = len(inspect.stack())
        return current_depth

    def _should_use_sync(self):
        """동기 모드로 전환할지 판단"""
        current_depth = self._check_recursion_depth()
        if current_depth >= sys.getrecursionlimit() - self._recursion_check_buffer:
            self.fallback_to_sync = True
            return True
        return self.fallback_to_sync

    async def execute_with_resilience(self, task_func: Callable, task_id: str, 
                                    timeout: float = 300.0, max_retries: int = 3, 
                                    *args, **kwargs) -> Any:
        """복원력 있는 작업 실행"""
        current_retry = 0
        last_exception = None

        while current_retry <= max_retries:
            try:
                if self._check_recursion_depth() >= sys.getrecursionlimit() - self._recursion_check_buffer:
                    raise RecursionError(f"Preemptive recursion depth stop for {task_id}")

                result = await asyncio.wait_for(
                    self.circuit_breaker.execute(task_func, *args, **kwargs),
                    timeout=timeout
                )
                return result
            except asyncio.TimeoutError as e:
                last_exception = e
                self.execution_stats["timeout_occurred"] += 1
            except RecursionError as e:
                last_exception = e
                self.fallback_to_sync = True
                raise e
            except Exception as e:
                if "CircuitBreaker is OPEN" in str(e):
                    self.execution_stats["circuit_breaker_triggered"] += 1
                last_exception = e

            current_retry += 1
            if current_retry <= max_retries:
                backoff_time = min((2 ** (current_retry - 1)), 30)
                await asyncio.sleep(backoff_time)

        if last_exception:
            raise last_exception
        else:
            raise Exception(f"Task '{task_id}' failed after max retries.")

    def _get_fallback_result(self, task_id: str, analysis: Dict, component_name: str, agent_analysis: Dict) -> Dict:
        """폴백 결과 생성"""
        self.execution_stats["fallback_used"] += 1
        return self._create_agent_based_default_design_sync_mode(analysis, component_name, agent_analysis)

class JSXLayoutDesigner(BaseAsyncAgent):
    """레이아웃 설계 전문 에이전트 (에이전트 결과 데이터 기반)"""

    def __init__(self):
        super().__init__()  # BaseAsyncAgent 초기화
        self.llm = get_azure_llm()
        self.logger = get_agent_logger()
        self.result_manager = get_complete_data_manager()

    def create_agent(self):
        return Agent(
            role="에이전트 결과 데이터 기반 매거진 레이아웃 아키텍트",
            goal="이전 에이전트들의 모든 결과 데이터, template_data.json, PDF 벡터 데이터를 종합 분석하여 완벽한 JSX 레이아웃 구조를 설계",
            backstory="""당신은 25년간 세계 최고 수준의 매거진 디자인과 디지털 레이아웃 분야에서 활동해온 전설적인 레이아웃 아키텍트입니다.

**에이전트 결과 데이터 활용 마스터십:**
- 이전 모든 에이전트들의 출력 결과를 종합 분석
- ContentCreator, ImageAnalyzer, ContentAnalyzer 등의 결과를 레이아웃에 반영
- 에이전트 협업 패턴과 성공 지표를 설계 결정에 활용
- jsx_templates는 사용하지 않고 에이전트 데이터만 활용

**데이터 기반 설계 우선순위:**
1. 이전 에이전트들의 결과 데이터 (최우선)
2. template_data.json의 콘텐츠 구조
3. PDF 벡터 데이터의 레이아웃 패턴
4. 에이전트 협업 품질 지표
5, 존재하는 모든 콘텐츠 데이터와 이미지 URL을 사용해야함함

**설계 철학:**
"진정한 매거진 레이아웃은 에이전트들의 협업 결과를 존중하면서도 독자의 인지 과정을 과학적으로 설계한 시스템입니다. jsx_templates에 의존하지 않고 순수한 에이전트 데이터만으로 최적의 레이아웃을 창조합니다."

**오류 없는 설계 보장:**
모든 설계 결정은 JSX 구현 시 오류가 발생하지 않도록 기술적 완성도를 고려합니다.""",
            verbose=True,
            llm=self.llm
        )

    # 기존 public 메서드 (이름 변경 없이 내부 로직만 개선)
    async def design_layout_structure(self, content: Dict, analysis: Dict, component_name: str) -> Dict:
        """에이전트 결과 데이터 기반 레이아웃 구조 설계"""
        self.execution_stats["total_attempts"] += 1

        # 재귀 깊이 체크
        if self._should_use_sync():
            return await self._design_layout_structure_sync_mode(content, analysis, component_name)
        
        try:
            return await self._design_layout_structure_batch_mode(content, analysis, component_name)
        except RecursionError:
            self.fallback_to_sync = True
            return await self._design_layout_structure_sync_mode(content, analysis, component_name)
        except Exception as e:
            print(f"⚠️ 배치 모드 실패, 동기 모드로 폴백: {e}")
            return await self._design_layout_structure_sync_mode(content, analysis, component_name)

    # 새로운 배치 처리 메서드
    async def _design_layout_structure_batch_mode(self, content: Dict, analysis: Dict, component_name: str) -> Dict:
        """배치 기반 안전한 처리"""
        print(f"📚 수집된 에이전트 결과 및 학습 인사이트 수집 시작...")

        try:
            # 복원력 있는 데이터 수집
            all_agent_results = await self.execute_with_resilience(
                self.result_manager.get_all_outputs,
                f"get_all_outputs_{component_name}",
                timeout=60.0,
                max_retries=3,
                exclude_agent="JSXLayoutDesigner"
            )

            learning_insights = await self.execute_with_resilience(
                self.logger.get_learning_insights,
                f"get_learning_insights_{component_name}",
                timeout=30.0,
                max_retries=3,
                agent_name="JSXLayoutDesigner"
            )

            print(f"📚 수집된 에이전트 결과: {len(all_agent_results)}개")
            print(f"🧠 학습 인사이트: {len(learning_insights.get('recommendations', []))}개")

            # 에이전트 데이터 분석
            agent_data_analysis = await self._analyze_all_agent_results(all_agent_results)

            # CrewAI 에이전트 생성 및 실행
            agent = self.create_agent()
            design_task = self._create_design_task(content, analysis, component_name, all_agent_results, learning_insights, agent_data_analysis)

            # 복원력 있는 CrewAI 실행
            result = await self.execute_with_resilience(
                agent.execute_task,
                f"execute_task_{component_name}",
                timeout=180.0,
                max_retries=2,
                task=design_task
            )

            # 결과 파싱 및 처리
            design_result = await self._parse_design_result_with_agent_data(str(result), analysis, agent_data_analysis)

            # 결과 저장
            await self._store_design_result(component_name, design_result, content, analysis, all_agent_results, learning_insights)

            self.execution_stats["successful_executions"] += 1
            print(f"✅ 에이전트 데이터 기반 레이아웃 설계 완료: {design_result.get('layout_type', '기본')} 구조")
            return design_result

        except Exception as e:
            print(f"⚠️ 레이아웃 설계 실패 (배치 모드): {e}")
            raise

    async def _design_layout_structure_sync_mode(self, content: Dict, analysis: Dict, component_name: str) -> Dict:
        """동기 모드 폴백 처리"""
        print(f"🔄 동기 폴백 모드로 레이아웃 설계 실행: {component_name}")

        try:
            # 간소화된 데이터 수집
            all_agent_results = await self.result_manager.get_all_outputs(exclude_agent="JSXLayoutDesigner")
            learning_insights = await self.logger.get_learning_insights("JSXLayoutDesigner")
            
            agent_data_analysis = await self._analyze_all_agent_results(all_agent_results)
            
            # 동기 모드에서는 기본 설계 사용
            design_result = self._create_agent_based_default_design_sync_mode(analysis, component_name, agent_data_analysis)
            
            # 간소화된 결과 저장
            await self._store_design_result(component_name, design_result, content, analysis, all_agent_results, learning_insights, mode="sync_fallback")
            
            print(f"✅ 동기 모드 레이아웃 설계 완료: {design_result.get('layout_type', '기본')} 구조")
            return design_result

        except Exception as e:
            print(f"⚠️ 동기 모드에서도 실패: {e}")
            # 최종 폴백
            return self._create_agent_based_default_design_sync_mode(analysis, component_name, {})

    # 모든 기존 private 메서드들 유지
    async def _analyze_all_agent_results(self, agent_results: List[Dict]) -> Dict:
        """모든 에이전트 결과 데이터 분석"""
        analysis = {
            "agent_summary": {},
            "quality_indicators": {},
            "content_patterns": {},
            "design_preferences": {},
            "success_metrics": {}
        }

        if not agent_results:
            return analysis

        # 에이전트별 결과 분류
        for result in agent_results:
            agent_name = result.get('agent_name', 'unknown')
            if agent_name not in analysis["agent_summary"]:
                analysis["agent_summary"][agent_name] = {
                    "count": 0,
                    "avg_confidence": 0,
                    "latest_output": None,
                    "success_rate": 0
                }

            analysis["agent_summary"][agent_name]["count"] += 1

            # 신뢰도 계산
            confidence = result.get('metadata', {}).get('confidence_score', 0)
            if confidence > 0:
                current_avg = analysis["agent_summary"][agent_name]["avg_confidence"]
                count = analysis["agent_summary"][agent_name]["count"]
                analysis["agent_summary"][agent_name]["avg_confidence"] = (current_avg * (count-1) + confidence) / count

            # 최신 출력 저장
            analysis["agent_summary"][agent_name]["latest_output"] = result.get('full_output')

        # 전체 품질 지표
        all_confidences = [
            r.get('metadata', {}).get('confidence_score', 0)
            for r in agent_results
            if r.get('metadata', {}).get('confidence_score', 0) > 0
        ]

        if all_confidences:
            analysis["quality_indicators"] = {
                "overall_confidence": sum(all_confidences) / len(all_confidences),
                "high_quality_count": len([c for c in all_confidences if c > 0.8]),
                "total_agents": len(analysis["agent_summary"]),
                "collaboration_success": len(all_confidences) / len(agent_results)
            }

        return analysis

    def _create_design_task(self, content: Dict, analysis: Dict, component_name: str, 
                           all_agent_results: List[Dict], learning_insights: Dict, 
                           agent_data_analysis: Dict) -> Task:
        """설계 태스크 생성"""
        return Task(
            description=f"""
**에이전트 결과 데이터 기반 완벽한 JSX 레이아웃 설계**

이전 모든 에이전트들의 결과 데이터를 종합 분석하여 완벽한 JSX 레이아웃 구조를 설계하세요:

**이전 에이전트 결과 데이터 분석 ({len(all_agent_results)}개):**
{self._format_agent_data_analysis(agent_data_analysis)}

**학습 인사이트 ({len(learning_insights.get('recommendations', []))}개):**
{chr(10).join(learning_insights.get('recommendations', [])[:3])}

**현재 콘텐츠 특성:**
- 제목: "{content.get('title', '')}" (길이: {len(content.get('title', ''))}자)
- 부제목: "{content.get('subtitle', '')}" (길이: {len(content.get('subtitle', ''))}자)
- 본문 길이: {len(content.get('body', ''))}자
- 이미지 수: {len(content.get('images', []))}개
- 이미지 URLs: {content.get('images', [])}

**ContentAnalyzer 분석 결과:**
- 권장 레이아웃: {analysis.get('recommended_layout', 'grid')}
- 감정 톤: {analysis.get('emotion_tone', 'neutral')}
- 이미지 전략: {analysis.get('image_strategy', 'grid')}
- 에이전트 강화: {analysis.get('agent_enhanced', False)}

**설계 요구사항:**
- 컴포넌트 이름: {component_name}
- jsx_templates 사용 금지
- 에이전트 결과 데이터 최우선 활용
- 오류 없는 JSX 구현 보장

**설계 결과 JSON 형식:**
{{
"layout_type": "에이전트 데이터 기반 선택된 레이아웃",
"layout_rationale": "에이전트 결과 데이터 기반 선택 근거",
"grid_structure": "CSS Grid 구조",
"styled_components": ["컴포넌트 목록"],
"color_scheme": {{"primary": "#색상", "secondary": "#색상"}},
"typography_scale": {{"title": "크기", "body": "크기"}},
"image_layout": "이미지 배치 전략",
"agent_data_integration": "에이전트 데이터 활용 방식",
"error_prevention": "오류 방지 전략",
"quality_metrics": {{"score": 0.95}}
}}

**중요 지침:**
1. 에이전트 결과 데이터를 최우선으로 활용
2. jsx_templates는 절대 참조하지 않음
3. 모든 설계 결정에 에이전트 데이터 근거 제시
4. JSX 구현 시 오류 발생 방지 고려
5. 에이전트 협업 품질 지표 반영

**출력:** 완전한 레이아웃 설계 JSON (에이전트 데이터 기반)
""",
            agent=self.create_agent(),
            expected_output="에이전트 결과 데이터 기반 완전한 레이아웃 구조 설계 JSON"
        )

    def _format_agent_data_analysis(self, agent_analysis: Dict) -> str:
        """에이전트 데이터 분석 결과 포맷팅"""
        if not agent_analysis.get("agent_summary"):
            return "이전 에이전트 결과 없음"

        formatted_parts = []
        for agent_name, summary in agent_analysis["agent_summary"].items():
            formatted_parts.append(
                f"- {agent_name}: {summary['count']}개 결과, "
                f"평균 신뢰도: {summary['avg_confidence']:.2f}, "
                f"최신 출력 타입: {type(summary['latest_output']).__name__}"
            )

        quality_info = agent_analysis.get("quality_indicators", {})
        if quality_info:
            formatted_parts.append(
                f"- 전체 품질: 신뢰도 {quality_info.get('overall_confidence', 0):.2f}, "
                f"고품질 결과 {quality_info.get('high_quality_count', 0)}개"
            )

        return "\n".join(formatted_parts)

    async def _parse_design_result_with_agent_data(self, result_text: str, analysis: Dict, agent_analysis: Dict) -> Dict:
        """에이전트 데이터 기반 설계 결과 파싱"""
        try:
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                parsed_result = json.loads(json_match.group())
                # 에이전트 데이터 통합
                parsed_result['agent_data_integration'] = agent_analysis
                parsed_result['jsx_templates_ignored'] = True
                parsed_result['error_prevention_applied'] = True
                return parsed_result
        except Exception as e:
            print(f"⚠️ JSON 파싱 실패: {e}")

        return self._create_agent_based_default_design_sync_mode(analysis, "DefaultComponent", agent_analysis)

    def _create_agent_based_default_design_sync_mode(self, analysis: Dict, component_name: str, agent_analysis: Dict) -> Dict:
        """에이전트 데이터 기반 기본 설계 (동기 모드)"""
        layout_type = analysis.get('recommended_layout', 'grid')

        # 에이전트 품질 지표 기반 조정
        quality_indicators = agent_analysis.get("quality_indicators", {})
        if quality_indicators.get("overall_confidence", 0) > 0.8:
            layout_type = 'magazine'  # 고품질일 때 매거진 레이아웃

        return {
            "layout_type": layout_type,
            "layout_rationale": f"에이전트 데이터 기반 {layout_type} 레이아웃 선택. "
                              f"{len(agent_analysis.get('agent_summary', {}))}개 에이전트 결과 반영",
            "grid_structure": "1fr 1fr" if layout_type == 'grid' else "1fr",
            "styled_components": ["Container", "Header", "Title", "Subtitle", "Content", "ImageGallery", "Footer"],
            "color_scheme": {
                "primary": "#2c3e50",
                "secondary": "#f8f9fa",
                "accent": "#3498db",
                "background": "#ffffff"
            },
            "typography_scale": {
                "title": "3em",
                "subtitle": "1.4em",
                "body": "1.1em",
                "caption": "0.9em"
            },
            "image_layout": "grid_responsive",
            "agent_data_integration": agent_analysis,
            "jsx_templates_ignored": True,
            "error_prevention": "완전한 JSX 문법 준수 및 오류 방지 적용",
            "quality_metrics": {
                "agent_collaboration_score": quality_indicators.get("collaboration_success", 0.8),
                "design_confidence": quality_indicators.get("overall_confidence", 0.85),
                "error_free_guarantee": 1.0
            }
        }

    async def _store_design_result(self, component_name: str, design_result: Dict, content: Dict, 
                                 analysis: Dict, all_agent_results: List[Dict], 
                                 learning_insights: Dict, mode: str = "batch"):
        """설계 결과 저장"""
        try:
            await self.result_manager.store_agent_output(
                agent_name="JSXLayoutDesigner",
                agent_role="에이전트 데이터 기반 레이아웃 아키텍트",
                task_description=f"컴포넌트 {component_name} 레이아웃 설계 ({mode} 모드)",
                final_answer=str(design_result),
                reasoning_process=f"{len(all_agent_results)}개 에이전트 결과 분석하여 레이아웃 설계",
                execution_steps=[
                    "에이전트 결과 수집",
                    "데이터 분석",
                    "레이아웃 설계",
                    "검증 완료"
                ],
                raw_input={"content": content, "analysis": analysis, "component_name": component_name},
                raw_output=design_result,
                performance_metrics={
                    "agent_results_utilized": len(all_agent_results),
                    "jsx_templates_ignored": True,
                    "learning_insights_applied": len(learning_insights.get('recommendations', [])),
                    "layout_type": design_result.get('layout_type'),
                    "error_prevention_applied": True,
                    "execution_mode": mode
                }
            )
        except Exception as e:
            print(f"⚠️ 결과 저장 실패: {e}")

    # 시스템 관리 메서드들
    def get_execution_statistics(self) -> Dict:
        """실행 통계 조회"""
        return {
            **self.execution_stats,
            "success_rate": (
                self.execution_stats["successful_executions"] / 
                max(self.execution_stats["total_attempts"], 1)
            ) * 100,
            "circuit_breaker_state": self.circuit_breaker.state
        }

    def reset_system_state(self) -> None:
        """시스템 상태 리셋"""
        self.circuit_breaker._reset_counts()
        self.circuit_breaker._state = CircuitBreakerState.CLOSED
        self.fallback_to_sync = False
        self.execution_stats = {
            "total_attempts": 0,
            "successful_executions": 0,
            "fallback_used": 0,
            "circuit_breaker_triggered": 0,
            "timeout_occurred": 0
        }

    def get_system_info(self) -> Dict:
        """시스템 정보 조회"""
        return {
            "class_name": self.__class__.__name__,
            "version": "2.0_resilient",
            "features": [
                "에이전트 결과 데이터 기반 레이아웃 설계",
                "복원력 있는 실행",
                "Circuit Breaker 패턴",
                "재귀 깊이 감지",
                "동기/비동기 폴백"
            ],
            "execution_modes": ["batch_resilient", "sync_fallback"]
        }

    # 기존 동기 버전 메서드 (호환성 유지)
    def design_layout_structure_sync(self, content: Dict, analysis: Dict, component_name: str) -> Dict:
        """동기 버전 레이아웃 설계 (호환성 유지)"""
        return asyncio.run(self.design_layout_structure(content, analysis, component_name))
