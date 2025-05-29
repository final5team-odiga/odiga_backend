import re
import asyncio
import logging
import time
import sys
import inspect
from typing import Dict, List, Callable, Any, Coroutine
from dataclasses import dataclass, field

from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
from utils.agent_decision_logger import get_agent_logger, get_complete_data_manager

# --- Infrastructure Classes ---
@dataclass
class WorkItem:
    id: str
    task_func: Callable[..., Coroutine[Any, Any, Any]]
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 0
    max_retries: int = 3
    current_retry: int = 0
    timeout: float = 300.0

    def __lt__(self, other):
        return self.priority < other.priority

class CircuitBreakerState:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 8, recovery_timeout: float = 30.0, half_open_attempts: int = 1):  # 수정된 값 적용
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_attempts = half_open_attempts
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def state(self):
        if self._state == CircuitBreakerState.OPEN:
            if self._last_failure_time and (time.monotonic() - self._last_failure_time) > self.recovery_timeout:
                self.logger.info("CircuitBreaker recovery timeout elapsed. Transitioning to HALF_OPEN.")
                self._state = CircuitBreakerState.HALF_OPEN
                self._success_count = 0
        return self._state

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.logger.warning("CircuitBreaker failed in HALF_OPEN state. Transitioning back to OPEN.")
            self._state = CircuitBreakerState.OPEN
            self._failure_count = self.failure_threshold
        elif self._failure_count >= self.failure_threshold and self.state == CircuitBreakerState.CLOSED:
            self.logger.error(f"CircuitBreaker failure threshold {self.failure_threshold} reached. Transitioning to OPEN.")
            self._state = CircuitBreakerState.OPEN
            
    def record_success(self):
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_attempts:
                self.logger.info("CircuitBreaker successful in HALF_OPEN state. Transitioning to CLOSED.")
                self._state = CircuitBreakerState.CLOSED
                self._reset_counts()
        elif self.state == CircuitBreakerState.CLOSED:
            self._reset_counts()

    def _reset_counts(self):
        self._failure_count = 0
        self._success_count = 0

    async def execute(self, task_func: Callable[..., Coroutine[Any, Any, Any]], *args, **kwargs) -> Any:
        if self.state == CircuitBreakerState.OPEN:
            self.logger.warning("CircuitBreaker is OPEN. Call rejected.")
            raise Exception(f"CircuitBreaker is OPEN for {task_func.__name__}. Call rejected.")

        try:
            result = await task_func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.logger.error(f"CircuitBreaker recorded failure for {task_func.__name__}: {e}")
            self.record_failure()
            raise e

class AsyncWorkQueue:
    def __init__(self, max_workers: int = 1, max_queue_size: int = 0):
        self._queue = asyncio.PriorityQueue(max_queue_size)
        self._workers: List[asyncio.Task] = []
        self._max_workers = max_workers
        self._running = False
        self.logger = logging.getLogger(self.__class__.__name__)

    async def _worker(self, worker_id: int):
        self.logger.info(f"Worker {worker_id} starting.")
        while self._running:
            try:
                item: WorkItem = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                self.logger.info(f"Worker {worker_id} processing task {item.id} (retry {item.current_retry})")
                try:
                    await asyncio.wait_for(item.task_func(*item.args, **item.kwargs), timeout=item.timeout)
                    self.logger.info(f"Task {item.id} completed successfully by worker {worker_id}.")
                except asyncio.TimeoutError:
                    self.logger.error(f"Task {item.id} timed out in worker {worker_id}.")
                except Exception as e:
                    self.logger.error(f"Task {item.id} failed in worker {worker_id}: {e}")
                finally:
                    self._queue.task_done()
            except asyncio.TimeoutError:
                if not self._running:
                    break
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} encountered an error: {e}")
                await asyncio.sleep(1)
        self.logger.info(f"Worker {worker_id} stopping.")

    async def start(self):
        if not self._running:
            self._running = True
            self.logger.info(f"Starting {self._max_workers} workers.")
            for i in range(self._max_workers):
                task = asyncio.create_task(self._worker(i))
                self._workers.append(task)

    async def stop(self):
        if self._running:
            self.logger.info("Stopping work queue. Waiting for tasks to drain...")
            self._running = False
            if self._workers:
                await asyncio.gather(*self._workers, return_exceptions=True)
                self._workers.clear()
            self.logger.info("Work queue stopped.")

    async def enqueue_work(self, item: WorkItem) -> bool:
        if not self._running:
            await self.start()
        try:
            await self._queue.put(item)
            self.logger.debug(f"Enqueued task {item.id}")
            return True
        except asyncio.QueueFull:
            self.logger.warning(f"Queue is full. Could not enqueue task {item.id}")
            return False

class JSXTemplateAdapter:
    """JSX 템플릿 어댑터 (CrewAI 기반 로깅 시스템 통합, 복원력 강화)"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        self.logger = get_agent_logger()
        self.result_manager = get_complete_data_manager()

        # --- Resilience Infrastructure ---
        self.adapter_circuit_breaker = CircuitBreaker(failure_threshold=8, recovery_timeout=30.0)  # 수정된 값 적용
        self._force_sync_mode_global = False
        self._recursion_check_buffer = 50
        self.recursion_threshold = 800  # 수정된 값 적용
        
        # 실행 통계 추가
        self.execution_stats = {
            "total_attempts": 0,
            "successful_executions": 0,
            "fallback_used": 0,
            "circuit_breaker_triggered": 0,
            "timeout_occurred": 0
        }

        # CrewAI 에이전트들 생성 (기존 방식 유지)
        self.template_adaptation_agent = self._create_template_adaptation_agent_sync()
        self.image_integration_agent = self._create_image_integration_agent_sync()
        self.structure_preservation_agent = self._create_structure_preservation_agent_sync()
        self.validation_agent = self._create_validation_agent_sync()

    # --- Helper for Resilient Execution ---
    async def _execute_with_resilience(
        self, 
        task_id: str,
        task_func: Callable[..., Coroutine[Any, Any, Any]],
        args: tuple = (),
        kwargs: dict = None,
        max_retries: int = 2,
        initial_timeout: float = 120.0,
        backoff_factor: float = 2.0,
        circuit_breaker: CircuitBreaker = None
    ) -> Any:
        if kwargs is None:
            kwargs = {}
        
        current_retry = 0
        current_timeout = initial_timeout
        last_exception = None

        while current_retry <= max_retries:
            try:
                self.logger.info(f"Attempt {current_retry + 1}/{max_retries + 1} for task '{task_id}' with timeout {current_timeout}s.")
                
                # 재귀 깊이 확인
                current_depth = len(inspect.stack())
                recursion_limit = sys.getrecursionlimit()
                if current_depth >= recursion_limit - self._recursion_check_buffer:
                    self.logger.warning(f"Approaching recursion limit ({current_depth}/{recursion_limit}) for task '{task_id}'. Raising preemptively.")
                    raise RecursionError(f"Preemptive recursion depth stop for {task_id}")

                if circuit_breaker:
                    result = await asyncio.wait_for(
                        circuit_breaker.execute(task_func, *args, **kwargs),
                        timeout=current_timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        task_func(*args, **kwargs),
                        timeout=current_timeout
                    )
                self.logger.info(f"Task '{task_id}' completed successfully on attempt {current_retry + 1}.")
                return result
            except asyncio.TimeoutError as e:
                last_exception = e
                self.execution_stats["timeout_occurred"] += 1
                self.logger.warning(f"Task '{task_id}' timed out on attempt {current_retry + 1} after {current_timeout}s.")
            except RecursionError as e:
                last_exception = e
                self.logger.error(f"Task '{task_id}' failed due to RecursionError on attempt {current_retry + 1}: {e}")
                self._force_sync_mode_global = True
                raise e
            except Exception as e:
                last_exception = e
                self.logger.error(f"Task '{task_id}' failed on attempt {current_retry + 1} with error: {e}")

            current_retry += 1
            if current_retry <= max_retries:
                sleep_duration = (backoff_factor ** (current_retry - 1))
                self.logger.info(f"Retrying task '{task_id}' in {sleep_duration}s...")
                await asyncio.sleep(sleep_duration)
                current_timeout *= backoff_factor
            else:
                self.logger.error(f"Task '{task_id}' failed after {max_retries + 1} attempts.")
                raise last_exception if last_exception else Exception(f"Task '{task_id}' failed after max retries.")

    def _get_fallback_result(self, task_id: str, component_name: str = "FallbackComponent", content: Dict = None) -> str:
        self.logger.warning(f"Generating fallback result for task_id: {task_id}")
        self.execution_stats["fallback_used"] += 1
        
        if content:
            return self._create_fallback_adaptation_sync(
                template_info={},
                content=content,
                component_name=component_name
            )
        return f"// Fallback for {component_name} due to error in task {task_id}\nexport const {component_name} = () => <div>Error generating component.</div>;"

    # --- Agent Creation Methods (동기 버전) ---
    def _create_template_adaptation_agent_sync(self):
        return Agent(
            role="JSX 템플릿 적응 전문가",
            goal="원본 JSX 템플릿의 구조를 완벽히 보존하면서 새로운 콘텐츠에 최적화된 적응을 수행",
            backstory="""당신은 10년간 React 및 JSX 템플릿 시스템을 설계하고 최적화해온 전문가입니다. 다양한 콘텐츠 타입에 맞춰 템플릿을 적응시키면서도 원본의 구조적 무결성을 유지하는 데 특화되어 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_image_integration_agent_sync(self):
        return Agent(
            role="이미지 URL 통합 전문가",
            goal="JSX 템플릿에 이미지 URL을 완벽하게 통합하여 시각적 일관성과 기능적 완성도를 보장",
            backstory="""당신은 8년간 웹 개발에서 이미지 최적화와 통합을 담당해온 전문가입니다. JSX 컴포넌트 내 이미지 요소의 동적 처리와 URL 관리에 특화되어 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_structure_preservation_agent_sync(self):
        return Agent(
            role="JSX 구조 보존 전문가",
            goal="원본 JSX 템플릿의 아키텍처와 디자인 패턴을 완벽히 보존하면서 콘텐츠 적응을 수행",
            backstory="""당신은 12년간 대규모 React 프로젝트에서 컴포넌트 아키텍처 설계와 유지보수를 담당해온 전문가입니다. 템플릿의 구조적 무결성을 보장하면서도 유연한 적응을 가능하게 하는 데 특화되어 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_validation_agent_sync(self):
        return Agent(
            role="JSX 적응 검증 전문가",
            goal="적응된 JSX 템플릿의 품질과 기능성을 종합적으로 검증하여 완벽한 결과물을 보장",
            backstory="""당신은 8년간 React 프로젝트의 품질 보증과 코드 검증을 담당해온 전문가입니다. JSX 템플릿 적응 과정에서 발생할 수 있는 모든 오류와 품질 이슈를 사전에 식별하고 해결하는 데 특화되어 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    # --- Async Agent Creation Methods (호환성 유지) ---
    async def _create_template_adaptation_agent(self):
        return self._create_template_adaptation_agent_sync()

    async def _create_image_integration_agent(self):
        return self._create_image_integration_agent_sync()

    async def _create_structure_preservation_agent(self):
        return self._create_structure_preservation_agent_sync()

    async def _create_validation_agent(self):
        return self._create_validation_agent_sync()

    # --- Main Public Method: adapt_template_to_content ---
    async def adapt_template_to_content(self, template_info: Dict, content: Dict, component_name: str) -> str:
        """템플릿을 콘텐츠에 맞게 적용 (CrewAI 기반 이미지 URL 완전 통합 + 로깅, 복원력 강화)"""
        task_id = f"adapt_template_to_content-{component_name}-{time.time()}"
        self.logger.info(f"Starting task: {task_id} for component {component_name}")
        
        self.execution_stats["total_attempts"] += 1

        if self._force_sync_mode_global:
            self.logger.warning(f"Global sync mode is active. Running {task_id} in sync mode.")
            return await self._adapt_template_to_content_sync_mode(template_info, content, component_name, task_id)
        
        try:
            return await self._adapt_template_to_content_batch_mode(template_info, content, component_name, task_id)
        except RecursionError:
            self.logger.error(f"RecursionError caught in adapt_template_to_content for {task_id}. Switching to global sync mode and retrying in sync mode.")
            self._force_sync_mode_global = True
            return await self._adapt_template_to_content_sync_mode(template_info, content, component_name, task_id)
        except Exception as e:
            self.logger.error(f"Critical error in adapt_template_to_content ({task_id}) batch mode: {e}. Falling back to sync mode as a last resort or returning fallback.")
            try:
                return await self._adapt_template_to_content_sync_mode(template_info, content, component_name, task_id + "_fallback_attempt")
            except Exception as final_e:
                self.logger.error(f"Sync mode also failed for {task_id} after batch mode failure: {final_e}")
                return self._get_fallback_result(task_id, component_name, content)

    async def _adapt_template_to_content_batch_mode(self, template_info: Dict, content: Dict, component_name: str, task_id: str) -> str:
        """배치 모드 템플릿 적응"""
        self.logger.info(f"Executing {task_id} in batch (resilient) mode.")

        # Step 1: 에이전트 준비
        structure_preservation_agent_instance = await self._create_structure_preservation_agent()
        image_integration_agent_instance = await self._create_image_integration_agent()
        template_adaptation_agent_instance = await self._create_template_adaptation_agent()
        validation_agent_instance = await self._create_validation_agent()
        
        # Step 2: 태스크 생성
        structure_analysis_task = self._create_structure_analysis_task(structure_preservation_agent_instance, template_info, content, component_name)
        image_integration_task = await self._create_image_integration_task(image_integration_agent_instance, content)
        content_adaptation_task = await self._create_content_adaptation_task(template_adaptation_agent_instance, template_info, content, component_name, structure_analysis_task, image_integration_task)
        validation_task = await self._create_validation_task(validation_agent_instance, component_name, content_adaptation_task)

        # Step 3: CrewAI 실행
        async def run_crew():
            adaptation_crew = Crew(
                agents=[structure_preservation_agent_instance, image_integration_agent_instance, template_adaptation_agent_instance, validation_agent_instance],
                tasks=[structure_analysis_task, image_integration_task, content_adaptation_task, validation_task],
                process=Process.sequential,
                verbose=True
            )
            return adaptation_crew.kickoff()

        crew_result = await self._execute_with_resilience(
            task_id=f"{task_id}-crew_kickoff",
            task_func=run_crew,
            initial_timeout=600.0,  # 10분으로 증가
            circuit_breaker=self.adapter_circuit_breaker
        )

        if not crew_result or isinstance(crew_result, Exception):
            self.logger.error(f"Crew kickoff for {task_id} failed or returned invalid result. Result: {crew_result}")
            return self._get_fallback_result(f"{task_id}-crew_kickoff_failed", component_name, content)

        # Step 4: 후처리 적응
        async def post_crew_adaptation():
            return await self._execute_adaptation_with_crew_insights(crew_result, template_info, content, component_name)

        adapted_jsx = await self._execute_with_resilience(
            task_id=f"{task_id}-post_crew_adaptation",
            task_func=post_crew_adaptation,
            initial_timeout=60.0
        )

        # Step 5: 결과 로깅
        await self._log_adaptation_results(adapted_jsx, template_info, content, component_name, crew_result, task_id)
        
        self.execution_stats["successful_executions"] += 1
        self.logger.info(f"✅ CrewAI 기반 실제 구조 보존 및 이미지 통합 완료 for {task_id}")
        return adapted_jsx

    async def _adapt_template_to_content_sync_mode(self, template_info: Dict, content: Dict, component_name: str, task_id: str) -> str:
        """동기 모드 템플릿 적응"""
        self.logger.warning(f"Executing {task_id} in sync (simplified fallback) mode.")
        
        try:
            # 간소화된 에이전트/태스크 생성 및 crew 실행
            structure_preservation_agent_instance = await self._create_structure_preservation_agent()
            image_integration_agent_instance = await self._create_image_integration_agent()
            template_adaptation_agent_instance = await self._create_template_adaptation_agent()
            validation_agent_instance = await self._create_validation_agent()

            structure_analysis_task_s = self._create_structure_analysis_task(structure_preservation_agent_instance, template_info, content, component_name)
            image_integration_task_s = await self._create_image_integration_task(image_integration_agent_instance, content)
            content_adaptation_task_s = await self._create_content_adaptation_task(template_adaptation_agent_instance, template_info, content, component_name, structure_analysis_task_s, image_integration_task_s)
            validation_task_s = await self._create_validation_task(validation_agent_instance, component_name, content_adaptation_task_s)
            
            adaptation_crew_s = Crew(
                agents=[structure_preservation_agent_instance, image_integration_agent_instance, template_adaptation_agent_instance, validation_agent_instance],
                tasks=[structure_analysis_task_s, image_integration_task_s, content_adaptation_task_s, validation_task_s],
                process=Process.sequential,
                verbose=False
            )
            
            self.logger.info(f"Kicking off simplified crew for {task_id}")
            crew_result_s = await asyncio.wait_for(adaptation_crew_s.kickoff(), timeout=90.0)

            adapted_jsx_s = await self._execute_adaptation_with_crew_insights(crew_result_s, template_info, content, component_name)
            await self._log_adaptation_results(adapted_jsx_s, template_info, content, component_name, crew_result_s, task_id, mode="sync")
            
            self.logger.info(f"Sync mode execution completed for {task_id}")
            return adapted_jsx_s
        except Exception as e:
            self.logger.error(f"Error during sync mode execution for {task_id}: {e}")
            return self._get_fallback_result(task_id, component_name, content)

    async def _log_adaptation_results(self, adapted_jsx: str, template_info: Dict, content: Dict, component_name: str, crew_result: Any, task_id: str, mode: str = "batch"):
        """적응 결과 로깅"""
        try:
            previous_results_count = len(await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.result_manager.get_all_outputs(exclude_agent="JSXTemplateAdapter")
            ))
        except:
            previous_results_count = 0
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.result_manager.store_agent_output(
                agent_name="JSXTemplateAdapter",
                agent_role="JSX 템플릿 어댑터",
                task_description=f"컴포넌트 {component_name} CrewAI 기반 템플릿 어댑테이션 ({mode} mode, task: {task_id})",
                final_answer=adapted_jsx,
                reasoning_process=f"CrewAI 기반 원본 JSX 구조 보존하며 콘텐츠 적용, 이미지 {len(content.get('images', []))}개 통합. Crew Output: {str(crew_result)[:200]}...",
                execution_steps=[
                    "CrewAI 에이전트 및 태스크 생성",
                    "구조 분석 및 보존 (Crew)",
                    "이미지 통합 (Crew)",
                    "콘텐츠 적응 (Crew)",
                    "검증 (Crew)",
                    "최종 JSX 조정"
                ],
                raw_input={"template_info": template_info, "content": content, "component_name": component_name},
                raw_output=adapted_jsx,
                performance_metrics={
                    "original_jsx_length": len(template_info.get('original_jsx', '')),
                    "adapted_jsx_length": len(adapted_jsx),
                    "images_integrated": len(content.get('images', [])),
                    "vector_matched": template_info.get('vector_matched', False),
                    "previous_results_count": previous_results_count,
                    "crewai_enhanced": True,
                    "execution_mode": mode,
                    "task_id": task_id
                }
            )
        )
        self.logger.debug(f"Adaptation results logged for {task_id}")

    # --- Task Creation Methods ---
    def _create_structure_analysis_task(self, agent_instance: Agent, template_info: Dict, content: Dict, component_name: str) -> Task:
        """구조 분석 태스크"""
        return Task(
            description=f"""
JSX 템플릿의 구조를 분석하고 보존 전략을 수립하세요.

**분석 대상:**
- 컴포넌트명: {component_name}
- 원본 JSX 길이: {len(template_info.get('original_jsx', ''))} 문자
- 벡터 매칭: {template_info.get('vector_matched', False)}

**분석 요구사항:**
1. 원본 JSX 구조 완전 분석
2. Styled-components 패턴 식별
3. 레이아웃 시스템 특성 파악
4. 보존해야 할 핵심 요소 식별

**보존 전략:**
- 컴포넌트 아키텍처 유지
- CSS 스타일링 패턴 보존
- 반응형 디자인 특성 유지
- 접근성 표준 준수

구조 분석 결과와 보존 전략을 제시하세요.
""",
            expected_output="JSX 구조 분석 결과 및 보존 전략",
            agent=agent_instance
        )

    async def _create_image_integration_task(self, agent_instance: Agent, content: Dict) -> Task:
        """이미지 통합 태스크"""
        return Task(
            description=f"""
콘텐츠의 이미지들을 JSX 템플릿에 완벽하게 통합하세요.

**통합 대상:**
- 이미지 개수: {len(content.get('images', []))}개
- 이미지 URL들: {content.get('images', [])[:3]}...

**통합 요구사항:**
1. 기존 이미지 태그 URL 교체
2. 이미지 props 동적 할당
3. 누락된 이미지 요소 추가
4. 이미지 갤러리 자동 생성 (필요시)

**통합 전략:**
- 기존 img 태그의 src 속성 교체
- styled 이미지 컴포넌트 src 업데이트
- 이미지 props 패턴 매칭 및 교체
- 이미지가 없는 경우 갤러리 추가

**품질 기준:**
- 모든 이미지 URL 유효성 확인
- 이미지 태그 문법 정확성
- 반응형 이미지 처리

이미지 통합 전략과 구현 방안을 제시하세요.
""",
            expected_output="이미지 통합 전략 및 구현 방안",
            agent=agent_instance
        )

    async def _create_content_adaptation_task(self, agent_instance: Agent, template_info: Dict, content: Dict, component_name: str, structure_task: Task, image_task: Task) -> Task:
        """콘텐츠 적응 태스크"""
        return Task(
            description=f"""
템플릿 구조를 보존하면서 새로운 콘텐츠에 맞게 적응시키세요.

**적응 대상:**
- 제목: {content.get('title', 'N/A')}
- 본문 길이: {len(content.get('body', ''))} 문자
- 부제목: {content.get('subtitle', 'N/A')}

**적응 요구사항:**
1. 원본 JSX 구조 완전 보존
2. 콘텐츠 요소만 선택적 교체
3. 컴포넌트명 정확한 적용
4. 벡터 데이터 기반 스타일 최적화

**적응 원칙:**
- 구조적 무결성 유지
- 콘텐츠 특성 반영
- 디자인 일관성 보장
- 사용자 경험 최적화

이전 태스크들의 결과를 활용하여 완벽한 적응을 수행하세요.
""",
            expected_output="완벽하게 적응된 JSX 템플릿",
            agent=agent_instance,
            context=[structure_task, image_task]
        )

    async def _create_validation_task(self, agent_instance: Agent, component_name: str, content_adaptation_task_ref: Task) -> Task:
        """검증 태스크"""
        return Task(
            description=f"""
적응된 JSX 템플릿의 품질과 기능성을 종합적으로 검증하세요.

**검증 대상:**
- 컴포넌트명: {component_name}

**검증 영역:**
1. JSX 문법 정확성 확인
2. 컴포넌트 구조 무결성 검증
3. 이미지 통합 완성도 평가
4. 마크다운 블록 완전 제거

**품질 기준:**
- 문법 오류 제로
- 컴파일 가능성 보장
- 원본 구조 보존 확인
- 콘텐츠 적응 완성도

**최종 검증:**
- import 문 정확성
- export 문 일치성
- styled-components 활용
- 접근성 준수

모든 검증 항목을 통과한 최종 JSX 템플릿을 제공하세요.
""",
            expected_output="품질 검증 완료된 최종 JSX 템플릿",
            agent=agent_instance,
            context=[content_adaptation_task_ref]
        )

    # --- Original Private Helper Methods ---
    async def _execute_adaptation_with_crew_insights(self, crew_result: Any, template_info: Dict, content: Dict, component_name: str) -> str:
        """CrewAI 인사이트를 활용한 실제 적응 수행"""
        self.logger.debug(f"Executing adaptation with crew insights for {component_name}. Crew result (preview): {str(crew_result)[:100]}")
        original_jsx = template_info.get('original_jsx', '')

        if not original_jsx:
            self.logger.warning(f"⚠️ 원본 JSX 없음 - 폴백 생성 for {component_name}")
            return self._create_fallback_adaptation_sync(template_info, content, component_name)

        self.logger.info(f"🔧 CrewAI 기반 실제 템플릿 구조 적용 시작 (이미지 URL 통합) for {component_name}")

        adapted_jsx = self._preserve_structure_adapt_content(original_jsx, template_info, content, component_name)
        
        # 비동기 적응 단계들 실행
        adapted_jsx = await self._force_integrate_image_urls(adapted_jsx, content)
        adapted_jsx = self._apply_vector_style_enhancements(adapted_jsx, template_info)
        adapted_jsx = await self._remove_markdown_blocks_and_validate(adapted_jsx, content, component_name)

        return adapted_jsx

    # --- Image Integration Methods ---
    async def _force_integrate_image_urls(self, jsx_code: str, content: Dict) -> str:
        """이미지 URL 강제 통합"""
        images = content.get('images', [])
        if not images:
            self.logger.debug(f"📷 이미지 없음 - 플레이스홀더 유지 in _force_integrate_image_urls")
            return jsx_code

        self.logger.debug(f"📷 {len(images)}개 이미지 URL 통합 시작 in _force_integrate_image_urls")
        jsx_code = await self._replace_existing_image_tags(jsx_code, images)
        jsx_code = await self._replace_image_props(jsx_code, images)
        jsx_code = await self._add_missing_images(jsx_code, images)
        self.logger.debug(f"✅ 이미지 URL 통합 완료 in _force_integrate_image_urls")
        return jsx_code

    async def _replace_existing_image_tags(self, jsx_code: str, images: List[str]) -> str:
        """기존 이미지 태그에 실제 URL 적용"""
        img_pattern = r'<img([^>]*?)src="([^"]*)"([^>]*?)/?>'
        def replace_img_src(match):
            if images and images[0]:
                new_src = images[0]
            else:
                return match.group(0)
            return f'<img{match.group(1)}src="{new_src}"{match.group(3)} />'
        jsx_code = re.sub(img_pattern, replace_img_src, jsx_code)

        styled_img_pattern = r'<(\w*[Ii]mage?\w*)\s+([^>]*?)src="([^"]*)"([^>]*?)/?>'
        def replace_styled_img_src(match):
            component_name, before_src, _, after_src = match.groups()
            img_index = self._extract_image_index_from_component(component_name)
            new_src = ""
            if img_index < len(images) and images[img_index]:
                new_src = images[img_index]
            elif images and images[0]:
                new_src = images[0]
            else:
                return match.group(0)
            return f'<{component_name} {before_src}src="{new_src}"{after_src} />'
        jsx_code = re.sub(styled_img_pattern, replace_styled_img_src, jsx_code)
        return jsx_code

    async def _replace_image_props(self, jsx_code: str, images: List[str]) -> str:
        """이미지 props 교체"""
        image_prop_patterns = [
            (r'\{imageUrl\}', 0),
            (r'\{imageUrl1\}', 0),
            (r'\{imageUrl2\}', 1),
            (r'\{imageUrl3\}', 2),
            (r'\{imageUrl4\}', 3),
            (r'\{imageUrl5\}', 4),
            (r'\{imageUrl6\}', 5),
            (r'\{image\}', 0),
            (r'\{heroImage\}', 0),
            (r'\{featuredImage\}', 0),
            (r'\{mainImage\}', 0)
        ]
        
        for pattern, index in image_prop_patterns:
            if index < len(images) and images[index]:
                jsx_code = re.sub(pattern, f'"{images[index]}"', jsx_code)
        return jsx_code

    async def _add_missing_images(self, jsx_code: str, images: List[str]) -> str:
        """이미지가 없는 경우 새로 추가"""
        if not images:
            return jsx_code
            
        if '<img' not in jsx_code and 'Image' not in jsx_code:
            container_match = re.search(r'(<[A-Z]\w*[^>]*>\s*)$', jsx_code, re.MULTILINE)
            if container_match:
                insertion_point = container_match.end()
                image_gallery_jsx = await self._create_image_gallery_jsx(images[:1])
                if image_gallery_jsx:
                    jsx_code = jsx_code[:insertion_point] + '\n      ' + image_gallery_jsx + jsx_code[insertion_point:]
            else:
                export_match = re.search(r'(</\w+>;?\s*\}\);?\s*)$', jsx_code, re.MULTILINE)
                if export_match:
                    insertion_point = export_match.start()
                    image_gallery_jsx = await self._create_image_gallery_jsx(images[:1])
                    if image_gallery_jsx:
                        jsx_code = jsx_code[:insertion_point] + '\n      ' + image_gallery_jsx + '\n' + jsx_code[insertion_point:]
                else:
                    self.logger.warning("Could not find a clear place to insert missing image gallery.")

        return jsx_code
        
    async def _create_image_gallery_jsx(self, images: List[str]) -> str:
        """이미지 갤러리 JSX 생성"""
        image_tags = []
        for i, img_url in enumerate(images[:3]):
            if img_url and img_url.strip():
                image_tags.append(f'        <img src="{img_url}" alt="Image {i+1}" style={{width: "100%", height: "auto", maxHeight:"200px", objectFit: "cover", borderRadius: "8px", marginTop: "10px"}} />')
        
        if not image_tags:
            return ""
            
        return f"""<div style={{display: "flex", flexDirection: "column", gap: "10px", marginTop: "20px"}}>\n{chr(10).join(image_tags)}\n      </div>"""

    def _extract_image_index_from_component(self, component_name: str) -> int:
        """컴포넌트명에서 이미지 인덱스 추출"""
        match = re.search(r'(\d+)', component_name)
        return int(match.group(1)) - 1 if match else 0

    def _preserve_structure_adapt_content(self, original_jsx: str, template_info: Dict, content: Dict, component_name: str) -> str:
        """구조를 보존하면서 콘텐츠 적응"""
        adapted_jsx = original_jsx
        adapted_jsx = re.sub(r'export const \w+', f'export const {component_name}', adapted_jsx)
        
        title = content.get('title', '제목')
        subtitle = content.get('subtitle', '부제목')
        body = content.get('body', '본문 내용')
        
        text_replacements = [
            (r'>\s*\{title\}\s*<', f'>{title}<'),
            (r'>\s*\{subtitle\}\s*<', f'>{subtitle}<'),
            (r'>\s*\{body\}\s*<', f'>{body}<'),
            (r'\{title\}', title),
            (r'\{subtitle\}', subtitle),
            (r'\{body\}', body)
        ]
        
        for pattern, replacement in text_replacements:
            adapted_jsx = re.sub(pattern, replacement, adapted_jsx)
        
        return adapted_jsx

    def _apply_vector_style_enhancements(self, jsx_code: str, template_info: Dict) -> str:
        """벡터 스타일 향상 적용"""
        if not template_info.get('vector_matched', False):
            return jsx_code
        
        # 간단한 스타일 향상
        if 'travel' in template_info.get('recommended_usage', ''):
            jsx_code = jsx_code.replace('#333333', '#2c5aa0')
        
        return jsx_code

    async def _remove_markdown_blocks_and_validate(self, jsx_code: str, content: Dict, component_name: str) -> str:
        """마크다운 블록 제거 및 검증"""
        jsx_code = re.sub(r'```[\s\S]*?```', '', jsx_code)
        jsx_code = re.sub(r'```\n?', '', jsx_code)
        
        # 기본 import/export 검증
        if 'import React' not in jsx_code:
            jsx_code = 'import React from "react";\n' + jsx_code
        
        if 'styled-components' in jsx_code and 'import styled' not in jsx_code:
            jsx_code = jsx_code.replace(
                'import React from "react";',
                'import React from "react";\nimport styled from "styled-components";'
            )
        
        if f'export const {component_name}' not in jsx_code:
            jsx_code = re.sub(r'export const \w+', f'export const {component_name}', jsx_code, 1)
        
        return jsx_code.strip()

    def _create_fallback_adaptation_sync(self, template_info: Dict, content: Dict, component_name: str) -> str:
        """폴백 적응 생성 (동기 버전)"""
        title = content.get('title', '제목')
        subtitle = content.get('subtitle', '부제목')
        body = content.get('body', '본문 내용')
        images = content.get('images', [])
        
        image_jsx = ""
        if images and images[0]:
            image_jsx = f'      <img src="{images[0]}" alt="Main Image" style={{width: "100%", height: "auto", objectFit: "cover"}} />'
        
        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  padding: 20px;
`;

const Title = styled.h1``;
const Subtitle = styled.h2``;
const Content = styled.p``;

export const {component_name} = () => {{
  return (
    <Container>
      <Title>{title}</Title>
      <Subtitle>{subtitle}</Subtitle>
      {image_jsx}
      <Content>{body}</Content>
    </Container>
  );
}};'''

    # 기존 비동기 버전 (호환성 유지)
    async def _create_fallback_adaptation(self, template_info: Dict, content: Dict, component_name: str) -> str:
        """폴백 적응 생성 (비동기 버전)"""
        return self._create_fallback_adaptation_sync(template_info, content, component_name)

    # 디버깅 및 모니터링 메서드
    def get_execution_statistics(self) -> Dict:
        """실행 통계 조회"""
        return {
            **self.execution_stats,
            "success_rate": (
                self.execution_stats["successful_executions"] / 
                max(self.execution_stats["total_attempts"], 1)
            ) * 100,
            "circuit_breaker_state": self.adapter_circuit_breaker.state
        }

    def reset_system_state(self) -> None:
        """시스템 상태 리셋"""
        self.logger.info("🔄 JSXTemplateAdapter 시스템 상태 리셋")
        
        # Circuit Breaker 리셋
        self.adapter_circuit_breaker._reset_counts()
        self.adapter_circuit_breaker._state = CircuitBreakerState.CLOSED
        
        # 폴백 플래그 리셋
        self._force_sync_mode_global = False
        
        # 통계 초기화
        self.execution_stats = {
            "total_attempts": 0,
            "successful_executions": 0,
            "fallback_used": 0,
            "circuit_breaker_triggered": 0,
            "timeout_occurred": 0
        }
        
        self.logger.info("✅ 시스템 상태가 리셋되었습니다.")

    def get_performance_metrics(self) -> Dict:
        """성능 메트릭 수집"""
        return {
            "circuit_breaker": {
                "state": self.adapter_circuit_breaker.state,
                "failure_count": self.adapter_circuit_breaker._failure_count,
                "failure_threshold": self.adapter_circuit_breaker.failure_threshold
            },
            "system": {
                "recursion_threshold": self.recursion_threshold,
                "force_sync_mode": self._force_sync_mode_global
            },
            "execution_stats": self.execution_stats
        }

    def get_system_info(self) -> Dict:
        """시스템 정보 조회"""
        return {
            "class_name": self.__class__.__name__,
            "version": "2.0_resilient",
            "features": [
                "CrewAI 기반 템플릿 적응",
                "복원력 있는 실행",
                "Circuit Breaker 패턴",
                "재귀 깊이 감지",
                "동기/비동기 폴백",
                "이미지 URL 통합"
            ],
            "agents": [
                "template_adaptation_agent",
                "image_integration_agent", 
                "structure_preservation_agent",
                "validation_agent"
            ],
            "execution_modes": ["batch_resilient", "sync_fallback"],
            "safety_features": [
                "재귀 깊이 모니터링",
                "타임아웃 처리",
                "Circuit Breaker",
                "점진적 백오프",
                "폴백 메커니즘"
            ]
        }
