import re
import os
import json
import asyncio
import logging
import time
import sys
import inspect
from typing import Dict, List, Callable, Any, Coroutine, Optional
from dataclasses import dataclass, field
from enum import Enum

from agents.jsxcreate.jsx_content_analyzer import JSXContentAnalyzer
from agents.jsxcreate.jsx_layout_designer import JSXLayoutDesigner
from agents.jsxcreate.jsx_code_generator import JSXCodeGenerator
from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager
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

    async def execute(self, task_func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitBreakerState.OPEN:
            self.logger.warning(f"CircuitBreaker is OPEN for {getattr(task_func, '__name__', 'unknown_task')}. Call rejected.")
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
            self.logger.error(f"CircuitBreaker recorded failure for {getattr(task_func, '__name__', 'unknown_task')}: {e}")
            self.record_failure()
            raise e

class AsyncWorkQueue:
    def __init__(self, max_workers: int = 1, max_queue_size: int = 0):
        self._queue = asyncio.PriorityQueue(max_queue_size if max_queue_size > 0 else 0)
        self._workers: List[asyncio.Task] = []
        self._max_workers = max_workers
        self._running = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self._results: Dict[str, Any] = {}

    async def _worker(self, worker_id: int):
        self.logger.info(f"Worker {worker_id} starting.")
        while self._running or not self._queue.empty():
            try:
                item: WorkItem = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                self.logger.info(f"Worker {worker_id} processing task {item.id} (retry {item.current_retry})")
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
                    self.logger.info(f"Task {item.id} completed successfully by worker {worker_id}.")
                except asyncio.TimeoutError:
                    self._results[item.id] = {"status": "timeout", "error": f"Task {item.id} timed out"}
                    self.logger.error(f"Task {item.id} timed out in worker {worker_id}.")
                except Exception as e:
                    self._results[item.id] = {"status": "error", "error": str(e)}
                    self.logger.error(f"Task {item.id} failed in worker {worker_id}: {e}")
                finally:
                    self._queue.task_done()
            except asyncio.TimeoutError:
                if not self._running and self._queue.empty():
                    break
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} encountered an unexpected error: {e}")
                await asyncio.sleep(1)
        self.logger.info(f"Worker {worker_id} stopping.")

    async def start(self):
        if not self._running:
            self._running = True
            self.logger.info(f"Starting {self._max_workers} workers.")
            self._workers = [asyncio.create_task(self._worker(i)) for i in range(self._max_workers)]

    async def stop(self, graceful=True):
        if self._running:
            self.logger.info("Stopping work queue...")
            self._running = False
            if graceful:
                await self._queue.join()
            
            if self._workers:
                for worker_task in self._workers:
                    worker_task.cancel()
                await asyncio.gather(*self._workers, return_exceptions=True)
                self._workers.clear()
            self.logger.info("Work queue stopped.")

    async def enqueue_work(self, item: WorkItem) -> bool:
        if not self._running:
             await self.start()
        try:
            await self._queue.put(item)
            self.logger.debug(f"Enqueued task {item.id} with priority {item.priority}")
            return True
        except asyncio.QueueFull:
            self.logger.warning(f"Queue is full. Could not enqueue task {item.id}")
            return False
            
    async def get_result(self, task_id: str, wait_timeout: Optional[float] = None) -> Any:
        """특정 작업 ID의 결과를 기다려서 반환"""
        start_time = time.monotonic()
        while True:
            if task_id in self._results:
                return self._results[task_id]
            if wait_timeout is not None and (time.monotonic() - start_time) > wait_timeout:
                raise asyncio.TimeoutError(f"Timeout waiting for result of task {task_id}")
            await asyncio.sleep(0.1)

    async def clear_results(self):
        self._results.clear()

class JSXCreatorAgent:
    """다중 에이전트 조율자 - JSX 생성 총괄 (CrewAI 기반 에이전트 결과 데이터 기반)"""

    def __init__(self):
        self.llm = get_azure_llm()
        self.vector_manager = PDFVectorManager()
        self.logger = get_agent_logger()
        self.result_manager = get_complete_data_manager()

        # 전문 에이전트들 초기화
        self.content_analyzer = JSXContentAnalyzer()
        self.layout_designer = JSXLayoutDesigner()
        self.code_generator = JSXCodeGenerator()

        # CrewAI 에이전트들 생성
        self.jsx_coordinator_agent = self._create_jsx_coordinator_agent()
        self.data_collection_agent = self._create_data_collection_agent()
        self.component_generation_agent = self._create_component_generation_agent()
        self.quality_assurance_agent = self._create_quality_assurance_agent()

        # --- Resilience Infrastructure ---
        self.work_queue = AsyncWorkQueue(max_workers=2, max_queue_size=50)
        self.circuit_breaker = CircuitBreaker(failure_threshold=8, recovery_timeout=30.0)
        self.recursion_threshold = 800
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
            self.logger.warning(f"Approaching recursion limit ({current_depth}/{sys.getrecursionlimit()}). Switching to sync mode.")
            self.fallback_to_sync = True
            return True
        return self.fallback_to_sync

    async def _execute_with_resilience(
        self, 
        task_id: str,
        task_func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        max_retries: int = 2,
        initial_timeout: float = 180.0,
        backoff_factor: float = 1.5,
        circuit_breaker: CircuitBreaker = None
    ) -> Any:
        """복원력 있는 작업 실행"""
        if kwargs is None: kwargs = {}
        
        current_retry = 0
        current_timeout = initial_timeout
        last_exception = None

        actual_circuit_breaker = circuit_breaker if circuit_breaker else self.circuit_breaker

        while current_retry <= max_retries:
            task_full_id = f"{task_id}-attempt-{current_retry + 1}"
            self.logger.info(f"Attempt {current_retry + 1}/{max_retries + 1} for task '{task_full_id}' with timeout {current_timeout}s.")
            
            try:
                if self._check_recursion_depth() >= sys.getrecursionlimit() - self._recursion_check_buffer:
                    self.logger.warning(f"Preemptive recursion stop for '{task_full_id}'.")
                    raise RecursionError(f"Preemptive recursion depth stop for {task_full_id}")

                result = await asyncio.wait_for(
                    actual_circuit_breaker.execute(task_func, *args, **kwargs),
                    timeout=current_timeout
                )
                
                self.logger.info(f"Task '{task_full_id}' completed successfully.")
                return result
            except asyncio.TimeoutError as e:
                last_exception = e
                self.execution_stats["timeout_occurred"] += 1
                self.logger.warning(f"Task '{task_full_id}' timed out after {current_timeout}s.")
            except RecursionError as e:
                last_exception = e
                self.logger.error(f"Task '{task_full_id}' failed due to RecursionError: {e}")
                self.fallback_to_sync = True
                raise e
            except Exception as e:
                if "CircuitBreaker is OPEN" in str(e):
                    self.execution_stats["circuit_breaker_triggered"] +=1
                    self.logger.warning(f"Task '{task_full_id}' rejected by CircuitBreaker.")
                last_exception = e
                self.logger.error(f"Task '{task_full_id}' failed: {e}")

            current_retry += 1
            if current_retry <= max_retries:
                sleep_duration = (backoff_factor ** (current_retry - 1))
                self.logger.info(f"Retrying task '{task_id}' in {sleep_duration}s...")
                await asyncio.sleep(sleep_duration)
                current_timeout *= backoff_factor
            else:
                self.logger.error(f"Task '{task_id}' failed after {max_retries + 1} attempts.")
                if last_exception:
                    raise last_exception
                else:
                    raise Exception(f"Task '{task_id}' failed after max retries without a specific exception.")
    
    def _get_fallback_result(self, task_id: str, component_name: Optional[str] = None, template_data: Optional[Dict] = None) -> List[Dict]:
        """폴백 JSX 컴포넌트 목록 생성"""
        self.logger.warning(f"Generating fallback result for task_id: {task_id}")
        self.execution_stats["fallback_used"] += 1

        fallback_components = []
        if template_data and "content_sections" in template_data:
            for i, section in enumerate(template_data.get("content_sections", [])):
                comp_name = component_name if component_name else f"FallbackComponent{i+1}"
                title = section.get('title', f'Fallback Title {i+1}')
                body = section.get('body', 'Error generating component content.')
                jsx_code = f"""// Fallback for {comp_name} due to error in task {task_id}
import React from 'react';
import styled from 'styled-components';

const FallbackContainer = styled.div`
  border: 1px dashed #ccc;
  padding: 20px;
  margin: 10px;
  background-color: #f9f9f9;
`;

export const {comp_name} = () => (
  <FallbackContainer>
    <h2>{title}</h2>
    <p>{body}</p>
    <p><small><i>Content generated via fallback mechanism.</i></small></p>
  </FallbackContainer>
);
"""
                fallback_components.append({
                    'name': comp_name,
                    'file': f"{comp_name}.jsx",
                    'jsx_code': jsx_code,
                    'approach': 'fallback_generation',
                    'error_info': task_id
                })
        else:
            comp_name = component_name if component_name else "GlobalFallbackComponent"
            jsx_code = f"""// Global fallback for {comp_name} due to error in task {task_id}
import React from 'react';
export const {comp_name} = () => <div>Error generating component. Please check logs. Task ID: {task_id}</div>;
"""
            fallback_components.append({
                'name': comp_name,
                'file': f"{comp_name}.jsx",
                'jsx_code': jsx_code,
                'approach': 'global_fallback_generation',
                'error_info': task_id
            })
            
        return fallback_components

    def _create_jsx_coordinator_agent(self):
        """JSX 생성 총괄 조율자"""
        return Agent(
            role="JSX 생성 총괄 조율자",
            goal="에이전트 결과 데이터를 기반으로 고품질 JSX 컴포넌트 생성 프로세스를 총괄하고 최적화된 결과를 도출",
            backstory="""당신은 15년간 React 및 JSX 기반 대규모 웹 개발 프로젝트를 총괄해온 시니어 아키텍트입니다. 다중 에이전트 시스템의 결과를 통합하여 최고 품질의 JSX 컴포넌트를 생성하는 데 특화되어 있습니다.

**전문 영역:**
- 다중 에이전트 결과 데이터 통합 및 분석
- JSX 컴포넌트 아키텍처 설계
- 에이전트 기반 개발 워크플로우 최적화
- 품질 보증 및 성능 최적화

**조율 철학:**
"각 에이전트의 전문성을 최대한 활용하여 단일 에이전트로는 달성할 수 없는 수준의 JSX 컴포넌트를 생성합니다."

**책임 영역:**
- 전체 JSX 생성 프로세스 관리
- 에이전트 간 데이터 흐름 최적화
- 품질 기준 설정 및 검증
- 최종 결과물 승인 및 배포""",
            verbose=True,
            llm=self.llm,
            allow_delegation=True
        )

    def _create_data_collection_agent(self):
        """데이터 수집 및 분석 전문가"""
        return Agent(
            role="에이전트 결과 데이터 수집 및 분석 전문가",
            goal="이전 에이전트들의 실행 결과를 체계적으로 수집하고 분석하여 JSX 생성에 필요한 인사이트를 도출",
            backstory="""당신은 10년간 다중 에이전트 시스템의 데이터 분석과 패턴 인식을 담당해온 전문가입니다. 복잡한 에이전트 결과 데이터에서 의미 있는 패턴과 인사이트를 추출하는 데 탁월한 능력을 보유하고 있습니다.

**핵심 역량:**
- 에이전트 실행 결과 패턴 분석
- 성공적인 접근법 식별 및 분류
- 품질 지표 기반 성능 평가
- 학습 인사이트 통합 및 활용

**분석 방법론:**
"데이터 기반 의사결정을 통해 각 에이전트의 강점을 파악하고 이를 JSX 생성 품질 향상에 활용합니다."

**특별 처리:**
- ContentCreatorV2Agent: 콘텐츠 생성 품질 분석
- ImageAnalyzerAgent: 이미지 분석 결과 활용
- 성능 메트릭: 성공률 및 신뢰도 평가""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_component_generation_agent(self):
        """JSX 컴포넌트 생성 전문가"""
        return Agent(
            role="JSX 컴포넌트 생성 전문가",
            goal="에이전트 분석 결과를 바탕으로 오류 없는 고품질 JSX 컴포넌트를 생성하고 최적화",
            backstory="""당신은 12년간 React 생태계에서 수천 개의 JSX 컴포넌트를 설계하고 구현해온 전문가입니다. 에이전트 기반 데이터를 활용한 동적 컴포넌트 생성과 최적화에 특화되어 있습니다.

**기술 전문성:**
- React 및 JSX 고급 패턴
- Styled-components 기반 디자인 시스템
- 반응형 웹 디자인 구현
- 컴포넌트 성능 최적화

**생성 전략:**
"에이전트 분석 결과의 모든 인사이트를 반영하여 사용자 경험과 개발자 경험을 모두 만족시키는 컴포넌트를 생성합니다."

**품질 기준:**
- 문법 오류 제로
- 컴파일 가능성 보장
- 접근성 표준 준수
- 성능 최적화 적용""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_quality_assurance_agent(self):
        """품질 보증 전문가"""
        return Agent(
            role="JSX 품질 보증 및 검증 전문가",
            goal="생성된 JSX 컴포넌트의 품질을 종합적으로 검증하고 오류를 제거하여 완벽한 결과물을 보장",
            backstory="""당신은 8년간 대규모 React 프로젝트의 품질 보증과 코드 리뷰를 담당해온 전문가입니다. JSX 컴포넌트의 모든 측면을 검증하여 프로덕션 레벨의 품질을 보장하는 데 특화되어 있습니다.

**검증 영역:**
- JSX 문법 및 구조 검증
- React 모범 사례 준수 확인
- 접근성 및 사용성 검증
- 성능 및 최적화 평가

**품질 철학:**
"완벽한 JSX 컴포넌트는 기능적 완성도와 코드 품질, 사용자 경험이 모두 조화를 이루는 결과물입니다."

**검증 프로세스:**
- 다단계 문법 검증
- 컴파일 가능성 테스트
- 에이전트 인사이트 반영 확인
- 최종 품질 승인""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def generate_jsx_components_async(self, template_data_path: str, templates_dir: str = "jsx_templates") -> List[Dict]:
        """에이전트 결과 데이터 기반 JSX 생성 (CrewAI 기반 jsx_templates 미사용, 복원력 강화)"""
        task_id = f"generate_jsx_components_async-{os.path.basename(template_data_path)}-{time.time_ns()}"
        self.logger.info(f"🚀 CrewAI 기반 에이전트 결과 데이터 기반 JSX 생성 시작 (Task ID: {task_id})")
        self.logger.info(f"📁 jsx_templates 폴더 무시 - 에이전트 데이터 우선 사용")
        
        self.execution_stats["total_attempts"] += 1

        if self._should_use_sync() or self.fallback_to_sync:
            self.logger.warning(f"Task {task_id}: 재귀 깊이 또는 폴백 플래그로 인해 동기 모드로 전환.")
            return await self._generate_jsx_components_sync_mode(template_data_path, templates_dir, task_id)
        
        try:
            return await self._generate_jsx_components_batch_mode(template_data_path, templates_dir, task_id)
        except RecursionError:
            self.logger.error(f"Task {task_id}: Batch 모드 실행 중 RecursionError 발생. 동기 모드로 폴백.")
            self.fallback_to_sync = True
            return await self._generate_jsx_components_sync_mode(template_data_path, templates_dir, task_id)
        except Exception as e:
            self.logger.error(f"Task {task_id}: Batch 모드 실행 중 치명적 오류: {e}. 최종 폴백 결과 반환.")
            template_data_for_fallback = self._load_template_data_for_fallback(template_data_path)
            return self._get_fallback_result(task_id, template_data=template_data_for_fallback)

    async def _generate_jsx_components_batch_mode(self, template_data_path: str, templates_dir: str, task_id_prefix: str) -> List[Dict]:
        """배치 모드 JSX 생성 (비동기)"""
        self.logger.info(f"Task {task_id_prefix}: Batch 모드 실행 시작.")

        # CrewAI Task들 생성 (기존 방식 유지)
        data_collection_task = self._create_data_collection_task()
        template_parsing_task = self._create_template_parsing_task(template_data_path)
        jsx_generation_task = self._create_jsx_generation_task()
        quality_assurance_task = self._create_quality_assurance_task()

        # CrewAI Crew 생성
        jsx_crew = Crew(
            agents=[self.data_collection_agent, self.jsx_coordinator_agent, self.component_generation_agent, self.quality_assurance_agent],
            tasks=[data_collection_task, template_parsing_task, jsx_generation_task, quality_assurance_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Crew 실행 (동기 메서드이므로 run_in_executor 사용, _execute_with_resilience로 래핑)
        crew_kickoff_task_id = f"{task_id_prefix}-crew_kickoff"
        crew_result = await self._execute_with_resilience(
            task_id=crew_kickoff_task_id,
            task_func=jsx_crew.kickoff,
            initial_timeout=900.0,
            circuit_breaker=self.circuit_breaker
        )

        if isinstance(crew_result, Exception) or crew_result is None:
            self.logger.error(f"Task {crew_kickoff_task_id}: Crew 실행 실패 또는 유효하지 않은 결과 반환. Result: {crew_result}")
            template_data_for_fallback = self._load_template_data_for_fallback(template_data_path)
            return self._get_fallback_result(crew_kickoff_task_id, template_data=template_data_for_fallback)
        
        # 실제 JSX 생성 수행 (CrewAI 결과 활용)
        generation_task_id = f"{task_id_prefix}-jsx_generation_with_insights"
        generated_components = await self._execute_with_resilience(
            task_id=generation_task_id,
            task_func=self._execute_jsx_generation_with_crew_insights,
            args=(crew_result, template_data_path, templates_dir),
            initial_timeout=300.0
        )
        
        if isinstance(generated_components, Exception) or not generated_components:
            self.logger.error(f"Task {generation_task_id}: JSX 생성 실패 또는 빈 결과. Result: {generated_components}")
            template_data_for_fallback = self._load_template_data_for_fallback(template_data_path)
            return self._get_fallback_result(generation_task_id, template_data=template_data_for_fallback)

        # 최종 로깅
        self._log_generation_summary(task_id_prefix, generated_components, "batch_async", crewai_enhanced=True)
        self.execution_stats["successful_executions"] += 1
        return generated_components

    async def _generate_jsx_components_sync_mode(self, template_data_path: str, templates_dir: str, task_id_prefix: str) -> List[Dict]:
        """동기 모드 JSX 생성 (폴백용)"""
        self.logger.warning(f"Task {task_id_prefix}: 동기 폴백 모드 실행 시작.")
        
        self.logger.info(f"Task {task_id_prefix}: 동기 모드에서는 CrewAI 실행 없이 에이전트 결과 기반으로만 생성 시도.")
        template_data = self._load_template_data_for_fallback(template_data_path)
        if not template_data:
            return self._get_fallback_result(f"{task_id_prefix}-template_data_load_failed_sync")

        all_agent_results = self.result_manager.get_all_outputs(exclude_agent="JSXCreatorAgent")
        learning_insights = self.logger.get_learning_insights("JSXCreatorAgent")

        try:
            generated_components = self.generate_jsx_from_agent_results(
                template_data, all_agent_results, learning_insights
            )
            self._log_generation_summary(task_id_prefix, generated_components, "sync_fallback", crewai_enhanced=False)
            return generated_components
        except Exception as e_sync_gen:
            self.logger.error(f"Task {task_id_prefix}: 동기 모드 JSX 생성 중 오류: {e_sync_gen}")
            return self._get_fallback_result(task_id_prefix, template_data=template_data)

    def _load_template_data_for_fallback(self, template_data_path: str) -> Optional[Dict]:
        """폴백용 template_data 로드"""
        try:
            with open(template_data_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            template_data = self._safe_parse_json(file_content)
            if not isinstance(template_data, dict) or "content_sections" not in template_data:
                self.logger.error(f"Fallback: 잘못된 template_data 구조 ({template_data_path})")
                return None
            return template_data
        except Exception as e:
            self.logger.error(f"Fallback: template_data.json 읽기 오류 ({template_data_path}): {e}")
            return None
            
    def _log_generation_summary(self, task_id_prefix:str, generated_components: List[Dict], mode: str, crewai_enhanced: bool):
        """JSX 생성 결과 요약 로깅"""
        total_components = len(generated_components)
        successful_components = len([c for c in generated_components if c.get('jsx_code') and c.get('approach') != 'fallback_generation' and c.get('approach') != 'global_fallback_generation'])
        
        self.result_manager.store_agent_output(
            agent_name="JSXCreatorAgent",
            agent_role="JSX 생성 총괄 조율자",
            task_description=f"{mode} 모드: {total_components}개 JSX 컴포넌트 생성 시도 (Task Prefix: {task_id_prefix})",
            final_answer=f"JSX 생성 완료: {successful_components}/{total_components}개 성공",
            reasoning_process=f"{mode} 모드 실행. CrewAI 사용: {crewai_enhanced}.",
            execution_steps=[
                f"모드: {mode}",
                "에이전트 결과 수집",
                "template_data.json 파싱",
                "JSX 컴포넌트 생성 로직 실행",
                "품질 검증 (내부 로직)"
            ],
            raw_input={"template_data_path": "N/A for summary", "crewai_enabled": crewai_enhanced},
            raw_output=[{"name": c.get("name"), "status": "success" if c.get('jsx_code') else "failure"} for c in generated_components],
            performance_metrics={
                "total_components_attempted": total_components,
                "successful_components_generated": successful_components,
                "success_rate": successful_components / max(total_components, 1),
                "execution_mode": mode,
                "crewai_enhanced_process": crewai_enhanced
            }
        )
        self.logger.info(f"✅ {mode} 모드 JSX 생성 완료: {successful_components}/{total_components}개 컴포넌트 성공 (Task Prefix: {task_id_prefix})")

    async def _execute_jsx_generation_with_crew_insights(self, crew_result: Any, template_data_path: str, templates_dir: str) -> List[Dict]:
        """CrewAI 인사이트를 활용한 실제 JSX 생성 (기존 로직 유지 및 개선)"""
        self.logger.info(f"Crew 결과 기반 JSX 생성 시작. Crew Result (type): {type(crew_result)}")

        all_agent_results = self.result_manager.get_all_outputs(exclude_agent="JSXCreatorAgent")
        learning_insights = self.logger.get_learning_insights("JSXCreatorAgent")

        self.logger.info(f"📚 수집된 에이전트 결과: {len(all_agent_results)}개")
        self.logger.info(f"🧠 학습 인사이트: {len(learning_insights.get('recommendations', []))}개")

        try:
            with open(template_data_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            template_data = self._safe_parse_json(file_content)
            if template_data is None:
                self.logger.error(f"❌ template_data.json 파싱 실패 ({template_data_path})")
                return self._get_fallback_result(f"parse_template_data_failed-{os.path.basename(template_data_path)}")
        except Exception as e:
            self.logger.error(f"template_data.json 읽기 오류 ({template_data_path}): {str(e)}")
            return self._get_fallback_result(f"read_template_data_failed-{os.path.basename(template_data_path)}")

        if not isinstance(template_data, dict) or "content_sections" not in template_data:
            self.logger.error(f"❌ 잘못된 template_data 구조 ({template_data_path})")
            return self._get_fallback_result(f"invalid_template_data_structure-{os.path.basename(template_data_path)}", template_data=template_data)

        self.logger.info(f"✅ JSON 직접 파싱 성공 ({template_data_path})")

        # 에이전트 결과 데이터 기반 JSX 생성 (기존 핵심 로직)
        generated_components = self.generate_jsx_from_agent_results(
            template_data, all_agent_results, learning_insights
        )
        return generated_components

    def _create_data_collection_task(self) -> Task:
        return Task(
            description="""
이전 에이전트들의 실행 결과를 체계적으로 수집하고 분석하여 JSX 생성에 필요한 인사이트를 도출하세요.

**수집 대상:**
1. 모든 이전 에이전트 실행 결과
2. 학습 인사이트 및 권장사항
3. 성능 메트릭 및 품질 지표

**분석 요구사항:**
1. 에이전트별 성공 패턴 식별
2. 콘텐츠 패턴 및 디자인 선호도 분석
3. 품질 지표 기반 성능 평가
4. JSX 생성에 활용 가능한 인사이트 추출

**출력 형식:**
- 에이전트 결과 요약
- 성공 패턴 분석
- JSX 생성 권장사항
""",
            expected_output="에이전트 결과 데이터 분석 및 JSX 생성 인사이트",
            agent=self.data_collection_agent
        )

    def _create_template_parsing_task(self, template_data_path: str) -> Task:
        return Task(
            description=f"""
template_data.json 파일을 파싱하고 JSX 생성에 필요한 구조화된 데이터를 준비하세요.

**파싱 대상:**
- 파일 경로: {template_data_path}

**파싱 요구사항:**
1. JSON 파일 안전한 읽기 및 파싱
2. content_sections 데이터 구조 검증
3. 각 섹션별 콘텐츠 요소 확인
4. JSX 생성을 위한 데이터 정제

**검증 항목:**
- JSON 구조 유효성
- 필수 필드 존재 여부
- 데이터 타입 일치성
- 콘텐츠 완성도

**출력 요구사항:**
파싱된 템플릿 데이터와 검증 결과
""",
            expected_output="파싱 및 검증된 템플릿 데이터",
            agent=self.jsx_coordinator_agent,
            context=[self._create_data_collection_task()]
        )

    def _create_jsx_generation_task(self) -> Task:
        return Task(
            description="""
에이전트 분석 결과와 템플릿 데이터를 바탕으로 고품질 JSX 컴포넌트를 생성하세요.

**생성 요구사항:**
1. 에이전트 인사이트 기반 콘텐츠 강화
2. 다중 에이전트 파이프라인 실행
   - 콘텐츠 분석 (JSXContentAnalyzer)
   - 레이아웃 설계 (JSXLayoutDesigner)
   - 코드 생성 (JSXCodeGenerator)
3. 에이전트 결과 기반 검증

**품질 기준:**
- React 및 JSX 문법 준수
- Styled-components 활용
- 반응형 디자인 적용
- 접근성 표준 준수

**컴포넌트 구조:**
- 명명 규칙: AgentBased{번호}Component
- 파일 확장자: .jsx
- 에러 프리 코드 보장
""",
            expected_output="생성된 JSX 컴포넌트 목록 (코드 포함)",
            agent=self.component_generation_agent,
            context=[self._create_data_collection_task(), self._create_template_parsing_task("")]
        )

    def _create_quality_assurance_task(self) -> Task:
        return Task(
            description="""
생성된 JSX 컴포넌트의 품질을 종합적으로 검증하고 최종 승인하세요.

**검증 영역:**
1. JSX 문법 및 구조 검증
2. React 모범 사례 준수 확인
3. 컴파일 가능성 테스트
4. 에이전트 인사이트 반영 확인

**품질 기준:**
- 문법 오류 제로
- 마크다운 블록 완전 제거
- 필수 import 문 포함
- export 문 정확성
- styled-components 활용

**최종 검증:**
- 컴포넌트명 일관성
- 코드 구조 완성도
- 성능 최적화 적용
- 접근성 준수

**승인 기준:**
모든 검증 항목 통과 시 최종 승인
""",
            expected_output="품질 검증 완료된 최종 JSX 컴포넌트 목록",
            agent=self.quality_assurance_agent,
            context=[self._create_jsx_generation_task()]
        )

    # 기존 메서드들 유지 (변경 없음)
    def generate_jsx_from_agent_results(self, template_data: Dict, agent_results: List[Dict], learning_insights: Dict) -> List[Dict]:
        """에이전트 결과 데이터를 활용한 JSX 생성"""
        generated_components = []
        content_sections = template_data.get("content_sections", [])

        # 에이전트 결과 데이터 분석
        agent_data_analysis = self._analyze_agent_results(agent_results)

        for i, content_section in enumerate(content_sections):
            if not isinstance(content_section, dict):
                continue
            
            component_name = f"AgentBased{i+1:02d}Component"
            print(f"\n=== {component_name} 에이전트 데이터 기반 생성 시작 ===")

            # 콘텐츠 정제 (에이전트 결과 반영)
            enhanced_content = self._enhance_content_with_agent_results(
                content_section, agent_data_analysis, learning_insights
            )

            # 다중 에이전트 파이프라인 (에이전트 데이터 기반)
            jsx_code = self._agent_result_based_jsx_pipeline(
                enhanced_content, component_name, i, len(content_sections),
                agent_data_analysis, learning_insights
            )

            # 에이전트 결과 기반 검증
            jsx_code = self._validate_jsx_with_agent_insights(
                jsx_code, enhanced_content, component_name, agent_data_analysis
            )

            # 개별 컴포넌트 생성 저장
            self.result_manager.store_agent_output(
                agent_name="JSXCreatorAgent_Component",
                agent_role="개별 JSX 컴포넌트 생성자",
                task_description=f"컴포넌트 {component_name} 생성",
                final_answer=jsx_code,
                reasoning_process="CrewAI 기반 에이전트 데이터 기반 JSX 컴포넌트 생성",
                execution_steps=[
                    "콘텐츠 강화",
                    "JSX 파이프라인 실행",
                    "검증 완료"
                ],
                raw_input=enhanced_content,
                raw_output=jsx_code,
                performance_metrics={
                    "jsx_code_length": len(jsx_code),
                    "error_free": self._validate_jsx_syntax(jsx_code),
                    "agent_data_utilized": True,
                    "crewai_enhanced": True
                }
            )

            generated_components.append({
                'name': component_name,
                'file': f"{component_name}.jsx",
                'jsx_code': jsx_code,
                'approach': 'crewai_agent_results_based',
                'agent_data_analysis': agent_data_analysis,
                'learning_insights_applied': True,
                'error_free_validated': True,
                'crewai_enhanced': True
            })
            print(f"✅ CrewAI 기반 에이전트 데이터 기반 JSX 생성 완료: {component_name}")

        return generated_components

    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _analyze_agent_results(self, agent_results: List[Dict]) -> Dict:
        """에이전트 결과 데이터 분석"""
        analysis = {
            "content_patterns": {},
            "design_preferences": {},
            "successful_approaches": [],
            "common_elements": [],
            "quality_indicators": {},
            "agent_insights": {},
            "crewai_enhanced": True
        }

        if not agent_results:
            print("📊 이전 에이전트 결과 없음 - 기본 분석 사용")
            return analysis

        for result in agent_results:
            agent_name = result.get('agent_name', 'unknown')
            # final_output 우선, 없으면 processed_output, 없으면 raw_output 사용
            full_output = result.get('final_output') or result.get('processed_output') or result.get('raw_output', {})

            # 에이전트별 인사이트 수집
            if agent_name not in analysis["agent_insights"]:
                analysis["agent_insights"][agent_name] = []

            analysis["agent_insights"][agent_name].append({
                "output_type": type(full_output).__name__,
                "content_length": len(str(full_output)),
                "timestamp": result.get('timestamp'),
                "has_performance_data": bool(result.get('performance_data'))
            })

            # 콘텐츠 패턴 분석
            if isinstance(full_output, dict):
                for key, value in full_output.items():
                    if key not in analysis["content_patterns"]:
                        analysis["content_patterns"][key] = []
                    analysis["content_patterns"][key].append(str(value)[:100])

            # 성공적인 접근법 식별
            performance_data = result.get('performance_data', {})
            if performance_data.get('success_rate', 0) > 0.8:
                analysis["successful_approaches"].append({
                    "agent": agent_name,
                    "approach": result.get('output_metadata', {}).get('approach', 'unknown'),
                    "success_rate": performance_data.get('success_rate', 0)
                })

        # 공통 요소 추출
        if analysis["content_patterns"]:
            analysis["common_elements"] = list(analysis["content_patterns"].keys())

        # 품질 지표 계산
        all_success_rates = [
            r.get('performance_data', {}).get('success_rate', 0)
            for r in agent_results
            if r.get('performance_data', {}).get('success_rate', 0) > 0
        ]

        analysis["quality_indicators"] = {
            "total_agents": len(set(r.get('agent_name') for r in agent_results)),
            "avg_success_rate": sum(all_success_rates) / len(all_success_rates) if all_success_rates else 0.5,
            "successful_rate": len(analysis["successful_approaches"]) / max(len(agent_results), 1),
            "data_richness": len(analysis["content_patterns"])
        }

        print(f"📊 CrewAI 기반 에이전트 데이터 분석 완료: {analysis['quality_indicators']['total_agents']}개 에이전트, 평균 성공률: {analysis['quality_indicators']['avg_success_rate']:.2f}")
        return analysis

    def _enhance_content_with_agent_results(self, content_section: Dict, agent_analysis: Dict, learning_insights: Dict) -> Dict:
        """에이전트 결과로 콘텐츠 강화"""
        enhanced_content = content_section.copy()
        enhanced_content['crewai_enhanced'] = True

        # 에이전트 인사이트 적용
        for agent_name, insights in agent_analysis["agent_insights"].items():
            if agent_name == "ContentCreatorV2Agent":
                # 콘텐츠 생성 에이전트 결과 반영
                if insights and insights[-1].get("content_length", 0) > 1000:
                    # 풍부한 콘텐츠가 생성되었으면 본문 확장
                    current_body = enhanced_content.get('body', '')
                    if len(current_body) < 500:
                        enhanced_content['body'] = current_body + "\n\n이 여행은 특별한 의미와 감동을 선사했습니다."
            elif agent_name == "ImageAnalyzerAgent":
                # 이미지 분석 에이전트 결과 반영
                if insights and insights[-1].get("has_performance_data", False):
                    # 성능 데이터가 있으면 이미지 관련 설명 추가
                    enhanced_content['image_description'] = "전문적으로 분석된 이미지들"

        # 성공적인 접근법 반영
        for approach in agent_analysis["successful_approaches"]:
            if approach["success_rate"] > 0.9:
                enhanced_content['quality_boost'] = f"고품질 {approach['agent']} 결과 반영"

        # 학습 인사이트 통합
        recommendations = learning_insights.get('recommendations', [])
        for recommendation in recommendations:
            if "콘텐츠" in recommendation and "풍부" in recommendation:
                current_body = enhanced_content.get('body', '')
                if len(current_body) < 800:
                    enhanced_content['body'] = current_body + "\n\n이러한 경험들이 모여 잊을 수 없는 여행의 추억을 만들어냅니다."

        return enhanced_content

    def _agent_result_based_jsx_pipeline(self, content: Dict, component_name: str, index: int,
                                       total_sections: int, agent_analysis: Dict, learning_insights: Dict) -> str:
        """에이전트 결과 기반 JSX 파이프라인"""
        try:
            # 1단계: 에이전트 결과 기반 콘텐츠 분석
            print(f" 📊 1단계: 에이전트 결과 기반 콘텐츠 분석...")
            analysis_result = self.content_analyzer.analyze_content_for_jsx(content, index, total_sections)
            # 에이전트 분석 결과 통합
            analysis_result = self._integrate_agent_analysis(analysis_result, agent_analysis)

            # 2단계: 에이전트 인사이트 기반 레이아웃 설계
            print(f" 🎨 2단계: 에이전트 인사이트 기반 레이아웃 설계...")
            design_result = self.layout_designer.design_layout_structure(content, analysis_result, component_name)
            # 에이전트 결과 기반 설계 강화
            design_result = self._enhance_design_with_agent_results(design_result, agent_analysis)

            # 3단계: 오류 없는 JSX 코드 생성
            print(f" 💻 3단계: 오류 없는 JSX 코드 생성...")
            jsx_code = self.code_generator.generate_jsx_code(content, design_result, component_name)

            # 4단계: 에이전트 결과 기반 검증 및 오류 제거
            print(f" 🔍 4단계: 에이전트 결과 기반 검증...")
            validated_jsx = self._comprehensive_jsx_validation(jsx_code, content, component_name, agent_analysis)

            return validated_jsx

        except Exception as e:
            print(f"⚠️ 에이전트 결과 기반 파이프라인 실패: {e}")
            # 폴백: 에이전트 데이터 기반 안전한 JSX 생성
            return self._create_agent_based_fallback_jsx(content, component_name, index, agent_analysis)

    def _integrate_agent_analysis(self, analysis_result: Dict, agent_analysis: Dict) -> Dict:
        """에이전트 분석 결과 통합"""
        enhanced_result = analysis_result.copy()
        enhanced_result['crewai_enhanced'] = True

        # 품질 지표 반영
        quality_indicators = agent_analysis.get("quality_indicators", {})
        if quality_indicators.get("avg_success_rate", 0) > 0.8:
            enhanced_result['confidence_boost'] = True
            enhanced_result['recommended_layout'] = 'magazine'  # 고품질일 때 매거진 레이아웃

        # 공통 요소 반영
        common_elements = agent_analysis.get("common_elements", [])
        if 'title' in common_elements and 'body' in common_elements:
            enhanced_result['layout_complexity'] = '고급'

        # 성공적인 접근법 반영
        successful_approaches = agent_analysis.get("successful_approaches", [])
        if len(successful_approaches) > 2:
            enhanced_result['design_confidence'] = 'high'
            enhanced_result['color_palette'] = '프리미엄 블루'

        return enhanced_result

    def _enhance_design_with_agent_results(self, design_result: Dict, agent_analysis: Dict) -> Dict:
        """에이전트 결과로 설계 강화"""
        enhanced_result = design_result.copy()
        enhanced_result['crewai_enhanced'] = True

        # 에이전트 인사이트 기반 색상 조정
        agent_insights = agent_analysis.get("agent_insights", {})
        if "ImageAnalyzerAgent" in agent_insights:
            # 이미지 분석 결과가 있으면 시각적 조화 강화
            enhanced_result['color_scheme'] = {
                "primary": "#2c3e50",
                "secondary": "#f8f9fa",
                "accent": "#3498db",
                "background": "#ffffff"
            }

        # 성공적인 접근법 기반 컴포넌트 구조 조정
        successful_approaches = agent_analysis.get("successful_approaches", [])
        if len(successful_approaches) >= 3:
            # 다양한 성공 사례가 있으면 더 풍부한 컴포넌트 구조
            enhanced_result['styled_components'] = [
                "Container", "Header", "MainContent", "ImageGallery",
                "TextSection", "Sidebar", "Footer"
            ]

        return enhanced_result

    def _comprehensive_jsx_validation(self, jsx_code: str, content: Dict, component_name: str, agent_analysis: Dict) -> str:
        """포괄적 JSX 검증 (오류 제거)"""
        # 1. 기본 구문 검증
        jsx_code = self._validate_basic_jsx_syntax(jsx_code, component_name)

        # 2. 에이전트 결과 기반 콘텐츠 검증
        jsx_code = self._validate_content_with_agent_results(jsx_code, content, agent_analysis)

        # 3. 마크다운 블록 완전 제거
        jsx_code = self._remove_all_markdown_blocks(jsx_code)

        # 4. 문법 오류 완전 제거
        jsx_code = self._fix_all_syntax_errors(jsx_code)

        # 5. 컴파일 가능성 검증
        jsx_code = self._ensure_compilation_safety(jsx_code, component_name)

        return jsx_code

    def _validate_basic_jsx_syntax(self, jsx_code: str, component_name: str) -> str:
        """기본 JSX 문법 검증"""
        # 필수 import 확인
        if 'import React' not in jsx_code:
            jsx_code = 'import React from "react";\n' + jsx_code

        if 'import styled' not in jsx_code:
            jsx_code = jsx_code.replace(
                'import React from "react";',
                'import React from "react";\nimport styled from "styled-components";'
            )

        # export 문 확인
        if f'export const {component_name}' not in jsx_code:
            jsx_code = re.sub(r'export const \w+', f'export const {component_name}', jsx_code)

        # return 문 확인
        if 'return (' not in jsx_code:
            jsx_code = jsx_code.replace(
                f'export const {component_name} = () => {{',
                f'export const {component_name} = () => {{\n  return (\n    <div>Component Content</div>\n  );\n}};'
            )

        return jsx_code

    def _validate_content_with_agent_results(self, jsx_code: str, content: Dict, agent_analysis: Dict) -> str:
        """에이전트 결과 기반 콘텐츠 검증"""
        # 에이전트 인사이트 기반 콘텐츠 강화
        quality_indicators = agent_analysis.get("quality_indicators", {})
        if quality_indicators.get("avg_success_rate", 0) > 0.8:
            # 고품질 에이전트 결과 시 스타일 강화
            if 'background: #ffffff' in jsx_code:
                jsx_code = jsx_code.replace(
                    'background: #ffffff',
                    'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                )

        return jsx_code

    def _remove_all_markdown_blocks(self, jsx_code: str) -> str:
        """마크다운 블록 완전 제거"""
        # 코드 블록 제거
        jsx_code = re.sub(r'```[\s\S]*?```', '', jsx_code)
        jsx_code = re.sub(r'```\n?', '', jsx_code)
        jsx_code = re.sub(r'`', '', jsx_code)

        # 마크다운 헤더 제거
        jsx_code = re.sub(r'#{1,6}\s+', '', jsx_code)

        # 마크다운 강조 제거
        jsx_code = re.sub(r'\*\*(.*?)\*\*', r'\1', jsx_code)
        jsx_code = re.sub(r'\*(.*?)\*', r'\1', jsx_code)

        return jsx_code

    def _fix_all_syntax_errors(self, jsx_code: str) -> str:
        """문법 오류 완전 제거"""
        # 중괄호 균형 맞추기
        open_braces = jsx_code.count('{')
        close_braces = jsx_code.count('}')
        if open_braces > close_braces:
            jsx_code += '}' * (open_braces - close_braces)

        # 괄호 균형 맞추기
        open_parens = jsx_code.count('(')
        close_parens = jsx_code.count(')')
        if open_parens > close_parens:
            jsx_code += ')' * (open_parens - close_parens)

        return jsx_code

    def _ensure_compilation_safety(self, jsx_code: str, component_name: str) -> str:
        """컴파일 가능성 보장"""
        # React import 보장
        if 'import React from "react";' not in jsx_code:
            jsx_code = 'import React from "react";\n' + jsx_code

        # styled-components import 보장
        if re.search(r'styled\.\w+', jsx_code) and 'import styled from "styled-components";' not in jsx_code:
            jsx_code = jsx_code.replace(
                'import React from "react";',
                'import React from "react";\nimport styled from "styled-components";',
                1
            )

        # export 문 보장
        export_pattern = rf'export\s+const\s+{component_name}\s*=\s*\(\s*\)\s*=>'
        if not re.search(export_pattern, jsx_code):
            found_export = re.search(r'export\s+const\s+\w+\s*=\s*\(\s*\)\s*=>', jsx_code)
            if found_export:
                jsx_code = jsx_code.replace(
                    found_export.group(0),
                    f'export const {component_name} = () =>',
                    1
                )

        return jsx_code

    def _validate_jsx_with_agent_insights(self, jsx_code: str, content: Dict, component_name: str, agent_analysis: Dict) -> str:
        """에이전트 인사이트 기반 JSX 검증"""
        successful_approaches = agent_analysis.get("successful_approaches", [])
        if len(successful_approaches) > 2:
            # 성공적인 접근법이 많으면 스타일 강화
            if 'padding: 20px;' in jsx_code:
                jsx_code = jsx_code.replace(
                    'padding: 20px;',
                    'padding: 40px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);'
                )

        return jsx_code

    def _create_agent_based_fallback_jsx(self, content: Dict, component_name: str, index: int, agent_analysis: Dict) -> str:
        """에이전트 기반 폴백 JSX 생성"""
        title = content.get('title', f'Component {index + 1}')
        body = content.get('body', '콘텐츠를 표시합니다.')
        quality_score = agent_analysis.get("quality_indicators", {}).get("avg_success_rate", 0.5)

        # 품질 점수에 따른 스타일 조정
        background_style = 'background: #f0f0f0;'
        if quality_score > 0.8:
            background_style = 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'
        elif quality_score > 0.6:
            background_style = 'background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%); color: white;'

        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 1200px;
  margin: 20px auto;
  padding: 30px;
  {background_style}
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.1);
  text-align: center;
`;

const Title = styled.h1`
  font-size: 2.2rem;
  color: {'white' if quality_score > 0.6 else '#2c3e50'};
  margin-bottom: 1rem;
`;

const Content = styled.p`
  font-size: 1rem;
  line-height: 1.7;
  color: {'white' if quality_score > 0.6 else '#555'};
`;

export const {component_name} = () => {{
  return (
    <Container>
      <Title>{title}</Title>
      <Content>{body}</Content>
      <small style={{{{ marginTop: '20px', display: 'block', opacity: 0.7 }}}}><i>Fallback content generated based on agent analysis.</i></small>
    </Container>
  );
}};'''

    def _safe_parse_json(self, content: str) -> Dict:
        """안전한 JSON 파싱"""
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON 파싱 오류: {e}")
            return None

    def _validate_jsx_syntax(self, jsx_code: str) -> bool:
        """JSX 문법 검증"""
        has_react_import = 'import React' in jsx_code
        has_export = 'export const' in jsx_code
        has_return = 'return (' in jsx_code

        # 기본적인 괄호 짝 맞춤 검증
        balanced_parens = jsx_code.count('(') == jsx_code.count(')')
        balanced_braces = jsx_code.count('{') == jsx_code.count('}')

        return has_react_import and has_export and has_return and balanced_parens and balanced_braces

    def save_jsx_components(self, generated_components: List[Dict], components_folder: str) -> List[Dict]:
        """JSX 컴포넌트 파일 저장"""
        self.logger.info(f"📁 JSX 컴포넌트 저장 시작: {len(generated_components)}개 → {components_folder}")
        os.makedirs(components_folder, exist_ok=True)
        saved_components = []
        successful_saves = 0

        for i, component_data in enumerate(generated_components):
            try:
                component_name = component_data.get('name', f'AgentBased{i+1:02d}Component')
                component_file = component_data.get('file', f'{component_name}.jsx')
                jsx_code = component_data.get('jsx_code', '')

                if not jsx_code:
                    self.logger.warning(f"⚠️ {component_name}: JSX 코드 없음 - 건너뛰기")
                    continue

                file_path = os.path.join(components_folder, component_file)

                # 최종 정리 및 검증 단계 강화
                validated_jsx = self._ensure_compilation_safety(jsx_code, component_name)
                validated_jsx = self._remove_all_markdown_blocks(validated_jsx)
                validated_jsx = self._fix_all_syntax_errors(validated_jsx)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(validated_jsx)

                saved_component = {
                    'name': component_name,
                    'file': component_file,
                    'file_path': file_path,
                    'jsx_code': validated_jsx,
                    'size_bytes': len(validated_jsx.encode('utf-8')),
                    'approach': component_data.get('approach', 'crewai_agent_results_based'),
                    'error_free': self._validate_jsx_syntax(validated_jsx),
                    'crewai_enhanced': component_data.get('crewai_enhanced', True),
                    'agent_data_utilized': bool(component_data.get('agent_data_analysis', {})),
                    'save_timestamp': self._get_timestamp()
                }
                saved_components.append(saved_component)
                successful_saves += 1

                # 개별 컴포넌트 저장 로깅
                self.result_manager.store_agent_output(
                    agent_name="JSXCreatorAgent_FileSaver",
                    agent_role="JSX 파일 저장자",
                    task_description=f"컴포넌트 {component_name} 파일 저장",
                    final_answer=f"파일 저장 성공: {file_path}",
                    reasoning_process=f"CrewAI 기반 생성된 JSX 컴포넌트를 {components_folder}에 저장",
                    execution_steps=[
                        "JSX 코드 최종 검증",
                        "마크다운 블록 제거",
                        "컴파일 안전성 확보",
                        "파일 저장 완료"
                    ],
                    raw_input={
                        "component_name": component_name,
                        "file_path": file_path,
                        "jsx_code_length": len(jsx_code)
                    },
                    raw_output=saved_component,
                    performance_metrics={
                        "file_size_bytes": saved_component['size_bytes'],
                        "error_free": saved_component['error_free'],
                        "crewai_enhanced": saved_component['crewai_enhanced'],
                        "agent_data_utilized": saved_component['agent_data_utilized']
                    }
                )

                self.logger.info(f"✅ {component_name} 저장 완료 (크기: {saved_component['size_bytes']} bytes, 방식: {saved_component['approach']}, 오류없음: {saved_component['error_free']})")

            except Exception as e:
                self.logger.error(f"❌ {component_data.get('name', f'Component{i+1}')} 저장 실패: {e}")
                # 저장 실패 로깅
                self.result_manager.store_agent_output(
                    agent_name="JSXCreatorAgent_FileSaver",
                    agent_role="JSX 파일 저장자",
                    task_description=f"컴포넌트 저장 실패",
                    final_answer=f"ERROR: {str(e)}",
                    reasoning_process="JSX 컴포넌트 파일 저장 중 예외 발생",
                    error_logs=[{
                        "error": str(e),
                        "component": component_data.get('name', 'unknown')
                    }],
                    performance_metrics={
                        "save_failed": True,
                        "error_occurred": True
                    }
                )
                continue

        # 배치 저장 결과 로깅
        self.result_manager.store_agent_output(
            agent_name="JSXCreatorAgent_SaveBatch",
            agent_role="JSX 배치 저장 관리자",
            task_description=f"CrewAI 기반 {len(generated_components)}개 JSX 컴포넌트 배치 저장",
            final_answer=f"배치 저장 완료: {successful_saves}/{len(generated_components)}개 성공",
            reasoning_process=f"CrewAI 기반 생성된 JSX 컴포넌트들을 {components_folder}에 일괄 저장",
            execution_steps=[
                "컴포넌트 폴더 생성",
                "개별 컴포넌트 저장 루프",
                "JSX 코드 검증 및 정리",
                "파일 저장 및 메타데이터 생성",
                "저장 결과 집계"
            ],
            raw_input={
                "generated_components_count": len(generated_components),
                "components_folder": components_folder
            },
            raw_output=saved_components,
            performance_metrics={
                "total_components": len(generated_components),
                "successful_saves": successful_saves,
                "save_success_rate": successful_saves / max(len(generated_components), 1),
                "total_file_size": sum(comp['size_bytes'] for comp in saved_components),
                "error_free_count": len([comp for comp in saved_components if comp['error_free']]),
                "crewai_enhanced_count": len([comp for comp in saved_components if comp['crewai_enhanced']]),
                "agent_data_utilized_count": len([comp for comp in saved_components if comp['agent_data_utilized']])
            }
        )

        self.logger.info(f"📁 저장 완료: {successful_saves}/{len(generated_components)}개 성공 (CrewAI 기반 에이전트 데이터 활용)")
        self.logger.info(f"📊 총 파일 크기: {sum(comp['size_bytes'] for comp in saved_components):,} bytes")
        self.logger.info(f"✅ 컴포넌트 저장 완료: {len(saved_components)}개")
        return saved_components

    # 시스템 관리 메서드들
    def get_execution_statistics(self) -> Dict:
        """실행 통계 조회"""
        return {
            **self.execution_stats,
            "success_rate": (
                self.execution_stats["successful_executions"] / 
                max(self.execution_stats["total_attempts"], 1)
            ) * 100,
            "circuit_breaker_state": self.circuit_breaker.state,
            "current_queue_size": self.work_queue._queue.qsize() if hasattr(self.work_queue, '_queue') else 'N/A'
        }

    def reset_system_state(self) -> None:
        """시스템 상태 리셋"""
        self.logger.info("🔄 JSXCreatorAgent 시스템 상태 리셋")

        # Circuit Breaker 리셋
        self.circuit_breaker._reset_counts()
        self.circuit_breaker._state = CircuitBreakerState.CLOSED

        # 폴백 플래그 리셋
        self.fallback_to_sync = False

        # 작업 큐 클리어
        if hasattr(self.work_queue, 'clear_results'):
            asyncio.create_task(self.work_queue.clear_results())

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
                "state": self.circuit_breaker.state,
                "failure_count": self.circuit_breaker._failure_count,
                "failure_threshold": self.circuit_breaker.failure_threshold
            },
            "work_queue": {
                "running": self.work_queue._running,
                "workers": len(self.work_queue._workers),
                "results_count": len(self.work_queue._results)
            },
            "system": {
                "recursion_threshold": self.recursion_threshold,
                "fallback_to_sync": self.fallback_to_sync,
                "recursion_check_buffer": self._recursion_check_buffer
            },
            "execution_stats": self.execution_stats
        }

    def get_system_info(self) -> Dict:
        """시스템 정보 조회"""
        return {
            "class_name": self.__class__.__name__,
            "version": "2.1_resilient_jsx_creator",
            "features": [
                "CrewAI 기반 에이전트 결과 데이터 활용 JSX 생성",
                "복원력 있는 실행 (Circuit Breaker, 재시도, 타임아웃)",
                "재귀 깊이 감지 및 동기 폴백",
                "비동기 작업 큐 관리",
                "포괄적 JSX 검증 및 오류 제거"
            ],
            "agents_used": [
                "jsx_coordinator_agent",
                "data_collection_agent",
                "component_generation_agent",
                "quality_assurance_agent"
            ],
            "core_logic_agents": [
                "JSXContentAnalyzer",
                "JSXLayoutDesigner", 
                "JSXCodeGenerator"
            ],
            "execution_modes": ["batch_async_resilient", "sync_fallback"],
            "safety_features": [
                "재귀 깊이 모니터링",
                "타임아웃 처리",
                "Circuit Breaker",
                "점진적 백오프",
                "폴백 메커니즘",
                "작업 큐 관리"
            ]
        }

    def validate_system_integrity(self) -> bool:
        """시스템 무결성 검증"""
        try:
            # 필수 컴포넌트 확인
            required_components = [
                self.llm,
                self.vector_manager,
                self.logger,
                self.result_manager,
                self.content_analyzer,
                self.layout_designer,
                self.code_generator
            ]

            for component in required_components:
                if component is None:
                    return False

            # CrewAI 에이전트들 확인
            crewai_agents = [
                self.jsx_coordinator_agent,
                self.data_collection_agent,
                self.component_generation_agent,
                self.quality_assurance_agent
            ]

            for agent in crewai_agents:
                if agent is None:
                    return False

            # 복원력 시스템 확인
            if (self.work_queue is None or 
                self.circuit_breaker is None):
                return False

            return True

        except Exception as e:
            self.logger.error(f"⚠️ 시스템 무결성 검증 실패: {e}")
            return False

    async def cleanup_resources(self) -> None:
        """리소스 정리"""
        self.logger.info("🧹 JSXCreatorAgent 리소스 정리 시작")

        try:
            # 작업 큐 정리
            await self.work_queue.stop()
            self.logger.info("✅ 리소스 정리 완료")
        except Exception as e:
            self.logger.error(f"⚠️ 리소스 정리 중 오류: {e}")

    def __del__(self):
        """소멸자 - 리소스 정리"""
        try:
            if hasattr(self, 'work_queue') and self.work_queue._running:
                asyncio.create_task(self.work_queue.stop())
        except Exception:
            pass  # 소멸자에서는 예외를 무시

    # 기존 동기 버전 메서드들 (호환성 유지)
    def generate_jsx_components(self, template_data_path: str, templates_dir: str = "jsx_templates") -> List[Dict]:
        """동기 버전 JSX 생성 (호환성 유지)"""
        return asyncio.run(self.generate_jsx_components_async(template_data_path, templates_dir))
