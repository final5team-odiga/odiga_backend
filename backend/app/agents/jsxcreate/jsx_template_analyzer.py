import os
import re
import asyncio
import time
import sys
import concurrent.futures
from typing import Dict, List, Callable, Any, Optional
from collections import deque
from dataclasses import dataclass
from enum import Enum

from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager
from utils.agent_decision_logger import get_agent_logger, get_complete_data_manager

# --- Infrastructure Classes ---
@dataclass
class WorkItem:
    id: str
    task_func: Callable
    args: tuple
    kwargs: dict
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
    def __init__(self, failure_threshold: int = 8, recovery_timeout: float = 30.0, half_open_attempts: int = 1):  # 수정된 값 적용
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
            raise Exception(f"CircuitBreaker is OPEN for {task_func.__name__}. Call rejected.")

        try:
            result = await task_func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise e

class AsyncWorkQueue:
    def __init__(self, max_workers: int = 3, max_queue_size: int = 50):
        self._queue = asyncio.PriorityQueue(max_queue_size)
        self._workers: List[asyncio.Task] = []
        self._max_workers = max_workers
        self._running = False
        self._results: Dict[str, Any] = {}

    async def _worker(self, worker_id: int):
        while self._running:
            try:
                item: WorkItem = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                try:
                    result = await asyncio.wait_for(item.task_func(*item.args, **item.kwargs), timeout=item.timeout)
                    self._results[item.id] = result
                except asyncio.TimeoutError:
                    self._results[item.id] = Exception(f"Task {item.id} timed out")
                except Exception as e:
                    self._results[item.id] = e
                finally:
                    self._queue.task_done()
            except asyncio.TimeoutError:
                if not self._running:
                    break
                continue
            except Exception as e:
                await asyncio.sleep(1)

    async def start(self):
        if not self._running:
            self._running = True
            self._workers = [asyncio.create_task(self._worker(i)) for i in range(self._max_workers)]

    async def stop(self):
        if self._running:
            self._running = False
            if self._workers:
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

    async def get_results(self, specific_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        await self._queue.join()
        if specific_ids:
            return {id: self._results.get(id) for id in specific_ids if id in self._results}
        return self._results.copy()

    async def clear_results(self):
        self._results.clear()

class JSXTemplateAnalyzer:
    """JSX 템플릿 분석기 (CrewAI 기반 로깅 시스템 통합, 복원력 강화)"""

    def __init__(self):
        self.llm = get_azure_llm()
        self.templates_cache = {}
        self.vector_manager = PDFVectorManager()
        self.logger = get_agent_logger()
        self.result_manager = get_complete_data_manager()
        
        # --- Resilience Infrastructure ---
        self.work_queue = AsyncWorkQueue(max_workers=3, max_queue_size=50)
        self.crew_circuit_breaker = CircuitBreaker(failure_threshold=8, recovery_timeout=30.0)  # 수정된 값 적용
        self.vector_db_circuit_breaker = CircuitBreaker(failure_threshold=8, recovery_timeout=30.0)  # 수정된 값 적용
        self.recursion_threshold = 800  # 수정된 값 적용
        self.fallback_to_sync = False
        
        # 실행 통계 추가
        self.execution_stats = {
            "total_attempts": 0,
            "successful_executions": 0,
            "fallback_used": 0,
            "circuit_breaker_triggered": 0,
            "timeout_occurred": 0
        }

        # CrewAI 에이전트들 생성 (기존 방식 유지)
        self.template_analysis_agent = self._create_template_analysis_agent()
        self.vector_enhancement_agent = self._create_vector_enhancement_agent()
        self.agent_result_integrator = self._create_agent_result_integrator()
        self.template_selector_agent = self._create_template_selector_agent()

    def _check_recursion_depth(self):
        """현재 재귀 깊이 확인"""
        frame = sys._getframe()
        depth = 0
        while frame:
            depth += 1
            frame = frame.f_back
        return depth

    def _should_use_sync(self):
        """동기 모드로 전환할지 판단"""
        current_depth = self._check_recursion_depth()
        if current_depth > self.recursion_threshold:
            print(f"⚠️ JSXTemplateAnalyzer 재귀 깊이 {current_depth} 감지 - 동기 모드로 전환")
            self.fallback_to_sync = True
            return True
        return self.fallback_to_sync

    async def _execute_with_resilience(self, task_func: Callable, task_id: str,
                                     circuit_breaker: CircuitBreaker = None,
                                     timeout: float = 120.0, max_retries: int = 2,
                                     fallback_value: Any = None,
                                     *args, **kwargs) -> Any:
        """복원력 있는 작업 실행"""
        current_retry = 0
        last_exception = None

        while current_retry <= max_retries:
            try:
                if circuit_breaker:
                    result = await asyncio.wait_for(
                        circuit_breaker.execute(task_func, *args, **kwargs),
                        timeout=timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        task_func(*args, **kwargs),
                        timeout=timeout
                    )
                return result
            except asyncio.TimeoutError as e:
                last_exception = e
                self.execution_stats["timeout_occurred"] += 1
                print(f"⏰ 작업 {task_id} 타임아웃 (시도 {current_retry + 1})")
            except Exception as e:
                last_exception = e
                print(f"❌ 작업 {task_id} 실패 (시도 {current_retry + 1}): {e}")

            current_retry += 1
            if current_retry <= max_retries:
                backoff_time = min((2 ** (current_retry - 1)), 30)
                await asyncio.sleep(backoff_time)

        print(f"⚠️ 작업 {task_id} 모든 재시도 실패 - 폴백 값 반환")
        self.execution_stats["fallback_used"] += 1
        if fallback_value is not None:
            return fallback_value
        raise last_exception

    def _get_fallback_result(self, task_id: str, context: Optional[Dict] = None) -> Any:
        """폴백 결과 생성"""
        self.execution_stats["fallback_used"] += 1
        
        if "analyze_jsx_templates" in task_id:
            return {}
        if "get_best_template_for_content" in task_id:
            return "Section01.jsx"
        if context and "single_template_analysis" in task_id and "file_name" in context:
            return self._create_default_template_analysis(context["file_name"])
        return None

    def _create_template_analysis_agent(self):
        """템플릿 분석 전문 에이전트"""
        return Agent(
            role="JSX 템플릿 구조 분석 전문가",
            goal="JSX 템플릿 파일들의 구조적 특성과 레이아웃 패턴을 정밀 분석하여 최적화된 분류 및 특성 정보를 제공",
            backstory="""당신은 12년간 React 및 JSX 생태계에서 컴포넌트 아키텍처 분석과 패턴 인식을 담당해온 전문가입니다. 다양한 JSX 템플릿의 구조적 특성을 분석하여 최적의 사용 시나리오를 도출하는 데 특화되어 있습니다.

**전문 영역:**
- JSX 컴포넌트 구조 분석
- Styled-components 패턴 인식
- 레이아웃 시스템 분류
- 템플릿 복잡도 평가

**분석 방법론:**
"모든 JSX 템플릿은 고유한 설계 철학과 사용 목적을 가지고 있으며, 이를 정확히 분석하여 최적의 콘텐츠 매칭을 가능하게 합니다."

**핵심 역량:**
- 컴포넌트명 및 Props 추출
- Styled-components 패턴 분석
- 레이아웃 타입 분류 (simple/hero/grid/gallery)
- 이미지 전략 및 텍스트 전략 평가
- 복잡도 수준 측정""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_vector_enhancement_agent(self):
        """벡터 데이터 강화 전문가"""
        return Agent(
            role="PDF 벡터 데이터 기반 템플릿 강화 전문가",
            goal="PDF 벡터 데이터베이스와 템플릿 특성을 매칭하여 템플릿 분석 결과를 강화하고 최적화된 사용 권장사항을 제공",
            backstory="""당신은 10년간 벡터 데이터베이스와 유사도 검색 시스템을 활용한 템플릿 최적화를 담당해온 전문가입니다. Azure Cognitive Search와 PDF 벡터 데이터를 활용하여 템플릿의 잠재적 활용도를 극대화하는 데 특화되어 있습니다.

**기술 전문성:**
- 벡터 유사도 검색 및 매칭
- PDF 레이아웃 패턴 분석
- 템플릿-콘텐츠 호환성 평가
- 사용 시나리오 최적화

**강화 전략:**
"벡터 데이터의 풍부한 레이아웃 정보를 활용하여 각 템플릿의 최적 활용 시나리오를 식별하고 신뢰도를 향상시킵니다."

**출력 강화 요소:**
- 벡터 매칭 기반 신뢰도 계산
- 유사 레이아웃 기반 사용 권장
- PDF 소스 기반 용도 분류
- 레이아웃 패턴 최적화""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_agent_result_integrator(self):
        """에이전트 결과 통합 전문가"""
        return Agent(
            role="에이전트 결과 통합 및 템플릿 강화 전문가",
            goal="BindingAgent와 OrgAgent의 실행 결과를 분석하여 템플릿 특성을 강화하고 최적화된 인사이트를 제공",
            backstory="""당신은 8년간 다중 에이전트 시스템의 결과 통합과 패턴 분석을 담당해온 전문가입니다. BindingAgent의 이미지 배치 전략과 OrgAgent의 텍스트 구조 분석 결과를 템플릿 특성 강화에 활용하는 데 특화되어 있습니다.

**통합 전문성:**
- BindingAgent 이미지 배치 인사이트 활용
- OrgAgent 텍스트 구조 분석 통합
- 에이전트 간 시너지 효과 극대화
- 템플릿 신뢰도 향상

**분석 방법론:**
"각 에이전트의 전문성을 템플릿 분석에 반영하여 단일 분석으로는 달성할 수 없는 수준의 정확도와 신뢰도를 확보합니다."

**강화 영역:**
- 그리드/갤러리 레이아웃 최적화
- 이미지 배치 전략 반영
- 텍스트 구조 복잡도 조정
- 매거진 스타일 최적화""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_template_selector_agent(self):
        """템플릿 선택 전문가"""
        return Agent(
            role="콘텐츠 기반 최적 템플릿 선택 전문가",
            goal="콘텐츠 특성과 템플릿 분석 결과를 종합하여 가장 적합한 템플릿을 선택하고 선택 근거를 제공",
            backstory="""당신은 15년간 콘텐츠 관리 시스템과 템플릿 매칭 알고리즘을 설계해온 전문가입니다. 복잡한 콘텐츠 특성과 다양한 템플릿 옵션 중에서 최적의 조합을 찾아내는 데 특화되어 있습니다.

**선택 전문성:**
- 콘텐츠-템플릿 호환성 분석
- 다차원 점수 계산 시스템
- 벡터 데이터 기반 매칭
- 에이전트 인사이트 통합

**선택 철학:**
"완벽한 템플릿 선택은 콘텐츠의 본질적 특성과 템플릿의 구조적 강점이 완벽히 조화를 이루는 지점에서 이루어집니다."

**평가 기준:**
- 이미지 개수 및 전략 매칭
- 텍스트 길이 및 복잡도 적합성
- 벡터 데이터 기반 보너스
- 에이전트 인사이트 반영
- 감정 톤 및 용도 일치성""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def analyze_jsx_templates(self, templates_dir: str = "jsx_templates") -> Dict[str, Dict]:
        """jsx_templates 폴더의 모든 템플릿 분석 (CrewAI 기반 벡터 데이터 활용 + 로깅, 복원력 강화)"""
        operation_id = f"analyze_jsx_templates:{templates_dir}"
        self.execution_stats["total_attempts"] += 1
        
        # 재귀 깊이 확인 및 동기 모드 전환
        if self._should_use_sync():
            print("🔄 템플릿 분석 동기 모드로 전환")
            return await self._analyze_jsx_templates_sync_mode(templates_dir)

        try:
            # 개선된 배치 기반 비동기 모드 실행
            return await self._analyze_jsx_templates_batch_mode(templates_dir)
        except RecursionError:
            print("🔄 템플릿 분석 RecursionError 감지 - 동기 모드로 전환")
            self.fallback_to_sync = True
            return await self._analyze_jsx_templates_sync_mode(templates_dir)
        except Exception as e:
            print(f"❌ 템플릿 분석 중 예외 발생: {e} - 동기 모드로 폴백 시도")
            self.fallback_to_sync = True
            return await self._analyze_jsx_templates_sync_mode(templates_dir)

    async def _analyze_jsx_templates_batch_mode(self, templates_dir: str) -> Dict[str, Dict]:
        """개선된 배치 기반 템플릿 분석"""
        print("📦 템플릿 분석 배치 모드 시작")
        
        # 이전 에이전트 결과 수집
        previous_results = self.result_manager.get_all_outputs(exclude_agent="JSXTemplateAnalyzer")
        binding_results = [r for r in previous_results if "BindingAgent" in r.get('agent_name', '')]
        org_results = [r for r in previous_results if "OrgAgent" in r.get('agent_name', '')]
        print(f"📊 이전 에이전트 결과 수집: 전체 {len(previous_results)}개, BindingAgent {len(binding_results)}개, OrgAgent {len(org_results)}개")

        if not os.path.exists(templates_dir):
            print(f"❌ 템플릿 폴더 없음: {templates_dir}")
            raise FileNotFoundError(f"Template directory not found: {templates_dir}")

        jsx_files = [f for f in os.listdir(templates_dir) if f.endswith('.jsx')]
        if not jsx_files:
            print(f"❌ JSX 템플릿 파일 없음: {templates_dir}")
            raise FileNotFoundError(f"No .jsx files found in {templates_dir}")

        # CrewAI 실행
        crew_result = await self._execute_crew_analysis_safe(templates_dir, jsx_files, binding_results, org_results)

        # 개별 템플릿 분석 (배치 처리)
        analyzed_templates = await self._execute_template_analysis_via_queue(
            crew_result, templates_dir, jsx_files, binding_results, org_results
        )

        self.templates_cache.update(analyzed_templates)
        
        # 결과 로깅
        successful_analyses = sum(1 for t in analyzed_templates.values() if isinstance(t, dict) and t.get('analysis_success', False))
        self.result_manager.store_agent_output(
            agent_name="JSXTemplateAnalyzer",
            agent_role="JSX 템플릿 분석기",
            task_description=f"CrewAI 기반 {len(jsx_files)}개 JSX 템플릿 분석 (Resilient)",
            final_answer=f"성공적으로 {successful_analyses}/{len(jsx_files)}개 템플릿 분석 완료",
            reasoning_process=f"CrewAI 분석 ({'성공' if crew_result else '실패/폴백'}). 개별 파일 분석 완료.",
            raw_output=analyzed_templates,
            performance_metrics={
                "total_templates": len(jsx_files),
                "successful_analyses": successful_analyses,
                "crewai_kickoff_successful": bool(crew_result),
                "resilient_execution": True
            }
        )
        
        self.execution_stats["successful_executions"] += 1
        print("✅ PDF 벡터 기반 배치 템플릿 분석 완료")
        return analyzed_templates

    async def _execute_crew_analysis_safe(self, templates_dir: str, jsx_files: List[str],
                                        binding_results: List[Dict], org_results: List[Dict]):
        """안전한 CrewAI 분석 실행"""
        try:
            # 태스크 생성
            template_analysis_task = self._create_template_analysis_task(templates_dir, jsx_files)
            vector_enhancement_task = self._create_vector_enhancement_task()
            agent_integration_task = self._create_agent_integration_task(binding_results, org_results)

            # CrewAI Crew 생성 및 실행
            analysis_crew = Crew(
                agents=[self.template_analysis_agent, self.vector_enhancement_agent, self.agent_result_integrator],
                tasks=[template_analysis_task, vector_enhancement_task, agent_integration_task],
                process=Process.sequential,
                verbose=True
            )

            crew_result = await self._execute_with_resilience(
                task_func=lambda: analysis_crew.kickoff(),
                task_id=f"crew_analysis:{templates_dir}",
                circuit_breaker=self.crew_circuit_breaker,
                timeout=600.0,  # 10분으로 증가
                fallback_value=None
            )

            return crew_result

        except Exception as e:
            print(f"⚠️ CrewAI 분석 실행 실패: {e}")
            return None

    async def _execute_template_analysis_via_queue(self, crew_result, templates_dir: str, jsx_files: List[str],
                                                 binding_results: List[Dict], org_results: List[Dict]) -> Dict[str, Dict]:
        """큐를 통한 템플릿 분석 실행"""
        print(f"📁 {len(jsx_files)}개 템플릿 파일 배치 분석 시작")
        
        await self.work_queue.start()
        await self.work_queue.clear_results()

        # 작업 아이템 생성 및 큐에 추가
        submitted_ids = []
        for i, jsx_file in enumerate(jsx_files):
            work_item_id = f"analyze_{jsx_file}_{time.time_ns()}"
            submitted_ids.append(work_item_id)
            
            work_item = WorkItem(
                id=work_item_id,
                task_func=self._analyze_single_template_with_enhancements,
                args=(jsx_file, templates_dir, crew_result, binding_results, org_results),
                priority=i,
                timeout=180.0
            )
            await self.work_queue.enqueue_work(work_item)

        # 결과 수집
        raw_results = await self.work_queue.get_results(specific_ids=submitted_ids)
        
        processed_templates = {}
        for item_id in submitted_ids:
            jsx_filename = item_id.split('_')[1]  # "analyze_" 제거
            result = raw_results.get(item_id)

            if isinstance(result, dict) and result.get('analysis_success'):
                processed_templates[jsx_filename] = result
            else:
                print(f"⚠️ {jsx_filename} 분석 실패 - 기본 분석 사용")
                processed_templates[jsx_filename] = self._create_default_template_analysis(jsx_filename)

        return processed_templates

    async def _analyze_single_template_with_enhancements(self, jsx_file: str, templates_dir: str,
                                                       crew_result, binding_results: List[Dict],
                                                       org_results: List[Dict]) -> Dict:
        """개별 템플릿 분석 (강화 포함)"""
        file_path = os.path.join(templates_dir, jsx_file)
        
        try:
            # 1. 기본 분석
            template_analysis = await self._analyze_single_template(file_path, jsx_file)
            if not template_analysis.get('analysis_success'):
                return template_analysis

            # 2. 벡터 데이터 강화
            template_analysis = await self._execute_with_resilience(
                task_func=self._enhance_with_vector_data_async,
                task_id=f"vector_enhance:{jsx_file}",
                circuit_breaker=self.vector_db_circuit_breaker,
                timeout=45.0,
                fallback_value=template_analysis.copy(),
                template_analysis=template_analysis.copy(),
                jsx_file=jsx_file
            )

            # 3. 에이전트 결과 강화
            template_analysis = await self._enhance_with_agent_results(template_analysis, binding_results, org_results)

            print(f"✅ {jsx_file} 복원력 있는 분석 완료: {template_analysis.get('layout_type', 'N/A')}")
            return template_analysis

        except Exception as e:
            print(f"❌ {jsx_file} 분석 중 오류: {e}")
            return self._create_default_template_analysis(jsx_file)

    async def _analyze_jsx_templates_sync_mode(self, templates_dir: str) -> Dict[str, Dict]:
        """동기 모드 템플릿 분석"""
        print("🔄 템플릿 분석 동기 모드 실행")
        
        # 이전 에이전트 결과 수집
        previous_results = self.result_manager.get_all_outputs(exclude_agent="JSXTemplateAnalyzer")
        binding_results = [r for r in previous_results if "BindingAgent" in r.get('agent_name', '')]
        org_results = [r for r in previous_results if "OrgAgent" in r.get('agent_name', '')]

        if not os.path.exists(templates_dir):
            return {}

        jsx_files = [f for f in os.listdir(templates_dir) if f.endswith('.jsx')]
        if not jsx_files:
            return {}

        # 간소화된 분석
        analyzed_templates = {}
        for jsx_file in jsx_files:
            file_path = os.path.join(templates_dir, jsx_file)
            template_analysis = await self._analyze_single_template(file_path, jsx_file)
            template_analysis = await self._enhance_with_agent_results(template_analysis, binding_results, org_results)
            analyzed_templates[jsx_file] = template_analysis

        self.templates_cache.update(analyzed_templates)
        print("✅ 동기 모드 템플릿 분석 완료")
        return analyzed_templates

    async def get_best_template_for_content(self, content: Dict, analysis: Dict) -> str:
        """콘텐츠에 가장 적합한 템플릿 선택 (CrewAI 기반 벡터 데이터 + 에이전트 결과 활용 + 로깅, 복원력 강화)"""
        content_title = content.get('title', 'untitled_content')
        operation_id = f"get_best_template_for_content:{content_title}"
        self.execution_stats["total_attempts"] += 1

        # 재귀 깊이 확인 및 동기 모드 전환
        if self._should_use_sync():
            print("🔄 템플릿 선택 동기 모드로 전환")
            return await self._get_best_template_for_content_sync_mode(content, analysis)

        try:
            # 개선된 배치 기반 비동기 모드 실행
            return await self._get_best_template_for_content_batch_mode(content, analysis)
        except RecursionError:
            print("🔄 템플릿 선택 RecursionError 감지 - 동기 모드로 전환")
            self.fallback_to_sync = True
            return await self._get_best_template_for_content_sync_mode(content, analysis)
        except Exception as e:
            print(f"❌ 템플릿 선택 중 예외 발생: {e} - 동기 모드로 폴백 시도")
            self.fallback_to_sync = True
            return await self._get_best_template_for_content_sync_mode(content, analysis)

    async def _get_best_template_for_content_batch_mode(self, content: Dict, analysis: Dict) -> str:
        """배치 모드 템플릿 선택"""
        print("📦 템플릿 선택 배치 모드 시작")
        
        # 이전 에이전트 결과 수집
        previous_results = self.result_manager.get_all_outputs(exclude_agent="JSXTemplateAnalyzer")
        binding_results = [r for r in previous_results if "BindingAgent" in r.get('agent_name', '')]
        org_results = [r for r in previous_results if "OrgAgent" in r.get('agent_name', '')]

        if not self.templates_cache:
            selected_template = "Section01.jsx"
            self.result_manager.store_agent_output(
                agent_name="JSXTemplateAnalyzer_Selector",
                agent_role="템플릿 선택기",
                task_description="콘텐츠 기반 최적 템플릿 선택 (Resilient)",
                final_answer=selected_template,
                reasoning_process="템플릿 캐시 없어 기본 템플릿 선택",
                raw_input={"content": content, "analysis": analysis},
                performance_metrics={"fallback_selection_due_to_empty_cache": True}
            )
            return selected_template

        # CrewAI 실행
        crew_result = await self._execute_crew_selection_safe(content, analysis, previous_results)

        # 실제 선택 수행
        selected_template = await self._execute_template_selection_resilient(
            crew_result, content, analysis, previous_results, binding_results, org_results
        )
        
        self.execution_stats["successful_executions"] += 1
        return selected_template

    async def _execute_crew_selection_safe(self, content: Dict, analysis: Dict, previous_results: List[Dict]):
        """안전한 CrewAI 선택 실행"""
        try:
            template_selection_task = self._create_template_selection_task(content, analysis, previous_results)
            selection_crew = Crew(
                agents=[self.template_selector_agent],
                tasks=[template_selection_task],
                process=Process.sequential,
                verbose=True
            )

            crew_result = await self._execute_with_resilience(
                task_func=lambda: selection_crew.kickoff(),
                task_id=f"crew_selection:{content.get('title', 'untitled')}",
                circuit_breaker=self.crew_circuit_breaker,
                timeout=180.0,
                fallback_value=None
            )

            return crew_result

        except Exception as e:
            print(f"⚠️ CrewAI 선택 실행 실패: {e}")
            return None

    async def _execute_template_selection_resilient(self, crew_result, content: Dict, analysis: Dict,
                                                  previous_results: List[Dict], binding_results: List[Dict],
                                                  org_results: List[Dict]) -> str:
        """복원력 있는 템플릿 선택 실행"""
        image_count = len(content.get('images', []))
        text_length = len(content.get('body', ''))
        content_emotion = analysis.get('emotion_tone', 'neutral')

        # 콘텐츠 기반 벡터 검색
        content_query = f"{content.get('title', '')} {content.get('body', '')[:200]}"
        content_vectors = await self._execute_with_resilience(
            task_func=self.vector_manager.search_similar_layouts,
            task_id=f"vector_search:{content.get('title', 'untitled')}",
            circuit_breaker=self.vector_db_circuit_breaker,
            timeout=45.0,
            fallback_value=[],
            query=content_query,
            index_name="magazine_layout",
            top_k=5
        )

        best_template = None
        best_score = 0
        scoring_details = []

        for template_name, template_info in self.templates_cache.items():
            if not isinstance(template_info, dict):
                continue
                
            score = 0
            score_breakdown = {"template": template_name}

            # 기본 매칭 점수
            template_images = template_info.get('image_strategy', 0)
            if image_count == 0 and template_images == 0:
                score += 30
                score_breakdown["image_match"] = 30
            elif image_count == 1 and template_images == 1:
                score += 30
                score_breakdown["image_match"] = 30
            elif image_count > 1 and template_images > 1:
                score += 20
                score_breakdown["image_match"] = 20

            # 텍스트 길이 매칭
            layout_type = template_info.get('layout_type', 'simple')
            if text_length < 300 and layout_type in ['simple', 'hero']:
                score += 20
                score_breakdown["text_match"] = 20
            elif text_length > 500 and layout_type in ['grid', 'gallery']:
                score += 20
                score_breakdown["text_match"] = 20

            # 벡터 데이터 기반 보너스 점수
            if template_info.get('vector_matched', False):
                vector_bonus = template_info.get('layout_confidence', 0) * 30
                score += vector_bonus
                score_breakdown["vector_bonus"] = vector_bonus

            # 에이전트 결과 기반 보너스 점수
            if template_info.get('agent_enhanced', False):
                agent_bonus = 0
                
                # BindingAgent 인사이트 보너스
                binding_insights = template_info.get('binding_insights', [])
                if binding_insights:
                    if image_count > 1 and 'grid_layout_optimized' in binding_insights:
                        agent_bonus += 15
                    if image_count > 3 and 'gallery_layout_optimized' in binding_insights:
                        agent_bonus += 15
                    if 'professional_image_placement' in binding_insights:
                        agent_bonus += 10

                # OrgAgent 인사이트 보너스
                org_insights = template_info.get('org_insights', [])
                if org_insights:
                    if text_length > 500 and 'structured_text_layout' in org_insights:
                        agent_bonus += 15
                    if 'magazine_style_optimized' in org_insights:
                        agent_bonus += 20
                    if text_length > 800 and 'complex_content_support' in org_insights:
                        agent_bonus += 10

                score += agent_bonus
                score_breakdown["agent_bonus"] = agent_bonus

            # 콘텐츠 벡터와 템플릿 벡터 매칭
            template_vectors = template_info.get('similar_pdf_layouts', [])
            vector_match_bonus = await self._calculate_vector_content_match(content_vectors, template_vectors) * 20
            score += vector_match_bonus
            score_breakdown["content_vector_match"] = vector_match_bonus

            # 감정 톤 매칭
            recommended_usage = template_info.get('recommended_usage', 'general')
            if content_emotion == 'peaceful' and 'culture' in recommended_usage:
                score += 15
                score_breakdown["emotion_match"] = 15
            elif content_emotion == 'exciting' and 'travel' in recommended_usage:
                score += 15
                score_breakdown["emotion_match"] = 15

            score_breakdown["total_score"] = score
            scoring_details.append(score_breakdown)

            if score > best_score:
                best_score = score
                best_template = template_name

        selected_template = best_template or "Section01.jsx"
        selected_info = self.templates_cache.get(selected_template, {})

        # 선택 결과 로깅
        self.result_manager.store_agent_output(
            agent_name="JSXTemplateAnalyzer_Selector",
            agent_role="템플릿 선택기",
            task_description="CrewAI 및 벡터 기반 최적 템플릿 선택 (Resilient)",
            final_answer=selected_template,
            reasoning_process=f"CrewAI 결과 ({'있음' if crew_result else '없음/실패'}), 벡터 검색 ({len(content_vectors)}개 유사 레이아웃) 기반 점수화. 최고 점수: {best_score}",
            raw_input={
                "content_summary": content.get("title"),
                "analysis_emotion": content_emotion,
                "crew_result_available": bool(crew_result)
            },
            raw_output={
                "selected_template": selected_template,
                "best_score": best_score,
                "scoring_details": scoring_details,
                "selected_info_summary": selected_info.get("layout_type")
            },
            performance_metrics={
                "templates_evaluated": len(self.templates_cache),
                "best_score": best_score,
                "vector_matched_for_selection": bool(content_vectors),
                "resilient_execution": True
            }
        )

        print(f"🎯 최종 선택된 템플릿 (Resilient): '{selected_template}' (점수: {best_score})")
        return selected_template

    async def _get_best_template_for_content_sync_mode(self, content: Dict, analysis: Dict) -> str:
        """동기 모드 템플릿 선택"""
        print("🔄 템플릿 선택 동기 모드 실행")
        
        if not self.templates_cache:
            return "Section01.jsx"

        # 간소화된 선택 로직
        image_count = len(content.get('images', []))
        text_length = len(content.get('body', ''))

        best_template = None
        best_score = 0

        for template_name, template_info in self.templates_cache.items():
            if not isinstance(template_info, dict):
                continue
                
            score = 0
            template_images = template_info.get('image_strategy', 0)
            
            if image_count == template_images:
                score += 30
            elif abs(image_count - template_images) <= 1:
                score += 20

            layout_type = template_info.get('layout_type', 'simple')
            if text_length < 300 and layout_type in ['simple', 'hero']:
                score += 20
            elif text_length > 500 and layout_type in ['grid', 'gallery']:
                score += 20

            if score > best_score:
                best_score = score
                best_template = template_name

        selected_template = best_template or "Section01.jsx"
        print(f"🎯 동기 모드 선택된 템플릿: '{selected_template}' (점수: {best_score})")
        return selected_template

    # 기존 메서드들 유지 (변경 없음)
    def _create_template_analysis_task(self, templates_dir: str, jsx_files: List[str]) -> Task:
        """템플릿 분석 태스크"""
        return Task(
            description=f"""
{templates_dir} 폴더의 {len(jsx_files)}개 JSX 템플릿 파일들을 체계적으로 분석하세요.

**분석 대상 파일들:**
{', '.join(jsx_files)}

**분석 요구사항:**
1. 각 JSX 파일의 구조적 특성 분석
2. 컴포넌트명 및 Props 추출
3. Styled-components 패턴 인식
4. 레이아웃 타입 분류 (simple/hero/grid/gallery/overlay)
5. 이미지 전략 및 텍스트 전략 평가
6. 복잡도 수준 측정 (simple/moderate/complex)

**분석 결과 구조:**
각 템플릿별로 다음 정보 포함:
- 기본 정보 (파일명, 컴포넌트명, props)
- 레이아웃 특성 (타입, 특징, 그리드 구조)
- 콘텐츠 전략 (이미지, 텍스트)
- 복잡도 및 사용 권장사항

모든 템플릿의 상세 분석 결과를 제공하세요.
""",
            expected_output="JSX 템플릿별 상세 분석 결과",
            agent=self.template_analysis_agent
        )

    def _create_vector_enhancement_task(self) -> Task:
        """벡터 강화 태스크"""
        return Task(
            description="""
PDF 벡터 데이터베이스를 활용하여 템플릿 분석 결과를 강화하세요.

**강화 요구사항:**
1. 각 템플릿의 레이아웃 특성을 벡터 검색 쿼리로 변환
2. 유사한 매거진 레이아웃 패턴 검색 (top 3)
3. 벡터 매칭 기반 신뢰도 계산
4. PDF 소스 기반 사용 용도 분류

**강화 영역:**
- 레이아웃 신뢰도 향상
- 사용 시나리오 최적화
- 벡터 매칭 상태 표시
- 유사 레이아웃 정보 제공

**출력 요구사항:**
- 벡터 매칭 성공/실패 상태
- 신뢰도 점수 (0.0-1.0)
- 권장 사용 용도
- 유사 레이아웃 목록

이전 태스크의 분석 결과를 벡터 데이터로 강화하세요.
""",
            expected_output="벡터 데이터 기반 강화된 템플릿 분석 결과",
            agent=self.vector_enhancement_agent,
            context=[self._create_template_analysis_task("", [])]
        )

    def _create_agent_integration_task(self, binding_results: List[Dict], org_results: List[Dict]) -> Task:
        """에이전트 통합 태스크"""
        return Task(
            description=f"""
BindingAgent와 OrgAgent의 실행 결과를 분석하여 템플릿 특성을 더욱 강화하세요.

**통합 대상:**
- BindingAgent 결과: {len(binding_results)}개
- OrgAgent 결과: {len(org_results)}개

**BindingAgent 인사이트 활용:**
1. 이미지 배치 전략 분석 (그리드/갤러리)
2. 시각적 일관성 평가 결과 반영
3. 전문적 이미지 배치 인사이트 통합

**OrgAgent 인사이트 활용:**
1. 텍스트 구조 복잡도 분석
2. 매거진 스타일 최적화 정보
3. 구조화된 레이아웃 인사이트

**강화 방법:**
- 템플릿 신뢰도 점수 향상
- 레이아웃 타입별 보너스 적용
- 사용 권장사항 정교화
- 에이전트 인사이트 메타데이터 추가

이전 태스크들의 결과에 에이전트 인사이트를 통합하여 최종 강화된 템플릿 분석을 완성하세요.
""",
            expected_output="에이전트 인사이트가 통합된 최종 템플릿 분석 결과",
            agent=self.agent_result_integrator,
            context=[self._create_template_analysis_task("", []), self._create_vector_enhancement_task()]
        )

    def _create_template_selection_task(self, content: Dict, analysis: Dict, previous_results: List[Dict]) -> Task:
        """템플릿 선택 태스크"""
        return Task(
            description=f"""
콘텐츠 특성과 템플릿 분석 결과를 종합하여 가장 적합한 템플릿을 선택하세요.

**콘텐츠 특성:**
- 이미지 개수: {len(content.get('images', []))}개
- 텍스트 길이: {len(content.get('body', ''))} 문자
- 감정 톤: {analysis.get('emotion_tone', 'neutral')}
- 제목: {content.get('title', 'N/A')}

**이전 에이전트 결과:** {len(previous_results)}개

**선택 기준:**
1. 이미지 개수 및 전략 매칭 (30점)
2. 텍스트 길이 및 복잡도 적합성 (20점)
3. 벡터 데이터 기반 보너스 (최대 30점)
4. 에이전트 인사이트 보너스 (최대 40점)
5. 콘텐츠 벡터 매칭 (20점)
6. 감정 톤 매칭 (15점)

**에이전트 인사이트 활용:**
- BindingAgent: 이미지 배치 전략 최적화
- OrgAgent: 텍스트 구조 및 매거진 스타일

**최종 출력:**
- 선택된 템플릿명
- 총 점수 및 점수 세부사항
- 선택 근거 및 신뢰도

모든 템플릿을 평가하여 최고 점수의 템플릿을 선택하세요.
""",
            expected_output="최적 템플릿 선택 결과 및 상세 점수 분석",
            agent=self.template_selector_agent
        )

    # 기존 메서드들 유지 (변경 없음)
    async def _enhance_with_agent_results(self, template_analysis: Dict, binding_results: List[Dict], org_results: List[Dict]) -> Dict:
        """에이전트 결과 데이터로 템플릿 분석 강화"""
        enhanced_analysis = template_analysis.copy()
        enhanced_analysis['agent_enhanced'] = False
        enhanced_analysis['binding_insights'] = enhanced_analysis.get('binding_insights', [])
        enhanced_analysis['org_insights'] = enhanced_analysis.get('org_insights', [])

        if not binding_results and not org_results:
            return enhanced_analysis

        enhanced_analysis['agent_enhanced'] = True

        # BindingAgent 결과 활용
        if binding_results:
            latest_binding = binding_results[-1]
            binding_answer = latest_binding.get('agent_final_answer', latest_binding.get('final_answer', ''))
            
            if '그리드' in binding_answer or 'grid' in binding_answer.lower():
                enhanced_analysis['binding_insights'].append('grid_layout_optimized')
            if '갤러리' in binding_answer or 'gallery' in binding_answer.lower():
                enhanced_analysis['binding_insights'].append('gallery_layout_optimized')
            if '배치' in binding_answer or 'placement' in binding_answer.lower():
                enhanced_analysis['binding_insights'].append('professional_image_placement')

        # OrgAgent 결과 활용
        if org_results:
            latest_org = org_results[-1]
            org_answer = latest_org.get('agent_final_answer', latest_org.get('final_answer', ''))
            
            if '구조' in org_answer or 'structure' in org_answer.lower():
                enhanced_analysis['org_insights'].append('structured_text_layout')
            if '매거진' in org_answer or 'magazine' in org_answer.lower():
                enhanced_analysis['org_insights'].append('magazine_style_optimized')
            if '복잡' in org_answer or 'complex' in org_answer.lower():
                enhanced_analysis['org_insights'].append('complex_content_support')

        # 신뢰도 조정
        if enhanced_analysis['agent_enhanced']:
            current_confidence = enhanced_analysis.get('layout_confidence', 0.5)
            enhanced_analysis['layout_confidence'] = min(
                current_confidence + 0.05 * (len(enhanced_analysis['binding_insights']) + len(enhanced_analysis['org_insights'])), 
                1.0
            )

        return enhanced_analysis

    async def _enhance_with_vector_data_async(self, template_analysis: Dict, jsx_file: str) -> Dict:
        """벡터 데이터로 템플릿 분석 강화 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._enhance_with_vector_data, template_analysis, jsx_file
        )

    def _enhance_with_vector_data(self, template_analysis: Dict, jsx_file: str) -> Dict:
        """벡터 데이터로 템플릿 분석 강화"""
        try:
            # 템플릿의 레이아웃 특성을 쿼리로 변환
            layout_query = self._create_layout_query_from_template(template_analysis)
            
            # 벡터 검색으로 유사한 매거진 레이아웃 찾기
            similar_layouts = self.vector_manager.search_similar_layouts(
                layout_query,
                "magazine_layout",
                top_k=3
            )

            # 벡터 데이터로 템플릿 특성 보강
            if similar_layouts:
                template_analysis['vector_matched'] = True
                template_analysis['similar_pdf_layouts'] = similar_layouts
                template_analysis['layout_confidence'] = self._calculate_layout_confidence(template_analysis, similar_layouts)
                template_analysis['recommended_usage'] = self._determine_usage_from_vectors(similar_layouts)
            else:
                template_analysis['vector_matched'] = False
                template_analysis['similar_pdf_layouts'] = []
                template_analysis['layout_confidence'] = 0.5
                template_analysis['recommended_usage'] = 'general'

        except Exception as e:
            print(f"⚠️ 벡터 데이터 통합 실패 ({jsx_file}): {e}")
            template_analysis['vector_matched'] = False
            template_analysis['similar_pdf_layouts'] = []
            template_analysis['layout_confidence'] = 0.3

        return template_analysis

    def _create_layout_query_from_template(self, template_analysis: Dict) -> str:
        """템플릿 분석 결과를 벡터 검색 쿼리로 변환"""
        layout_type = template_analysis.get('layout_type', 'general')
        image_count = template_analysis.get('image_strategy', 0)
        complexity = template_analysis.get('complexity_level', 'simple')
        features = template_analysis.get('layout_features', [])
        grid_structure = template_analysis.get('grid_structure', False)

        query_parts = [
            f"{layout_type} layout",
            f"{image_count} images" if image_count > 0 else "text focused",
            f"{complexity} complexity design"
        ]

        if grid_structure:
            query_parts.append("grid system")
        else:
            query_parts.append("flexible layout")

        # 특징 추가
        if 'fixed_height' in features:
            query_parts.append("fixed height sections")
        if 'vertical_layout' in features:
            query_parts.append("vertical column layout")
        if 'gap_spacing' in features:
            query_parts.append("spaced elements design")

        return " ".join(query_parts)

    def _calculate_layout_confidence(self, template_analysis: Dict, similar_layouts: List[Dict]) -> float:
        """벡터 매칭 기반 레이아웃 신뢰도 계산"""
        if not similar_layouts:
            return 0.3

        # 유사도 점수 평균
        avg_similarity = sum(layout.get('score', 0) for layout in similar_layouts) / len(similar_layouts)

        # 템플릿 복잡도와 매칭 정도
        complexity_bonus = 0.2 if template_analysis.get('complexity_level') == 'moderate' else 0.1

        # 이미지 전략 매칭 보너스
        image_bonus = 0.1 if template_analysis.get('image_strategy', 0) > 0 else 0.05

        confidence = min(avg_similarity + complexity_bonus + image_bonus, 1.0)
        return round(confidence, 2)

    def _determine_usage_from_vectors(self, similar_layouts: List[Dict]) -> str:
        """벡터 데이터 기반 사용 용도 결정"""
        if not similar_layouts:
            return 'general'

        # PDF 소스 분석
        pdf_sources = [layout.get('pdf_name', '').lower() for layout in similar_layouts]

        # 매거진 타입 추론
        if any('travel' in source for source in pdf_sources):
            return 'travel_focused'
        elif any('culture' in source for source in pdf_sources):
            return 'culture_focused'
        elif any('lifestyle' in source for source in pdf_sources):
            return 'lifestyle_focused'
        else:
            return 'editorial'

    async def _calculate_vector_content_match(self, content_vectors: List[Dict], template_vectors: List[Dict]) -> float:
        """콘텐츠 벡터와 템플릿 벡터 간 매칭 점수"""
        if not content_vectors or not template_vectors:
            return 0.0

        # PDF 소스 기반 매칭
        content_sources = set(v.get('pdf_name', '') for v in content_vectors)
        template_sources = set(v.get('pdf_name', '') for v in template_vectors)

        # 공통 소스 비율
        common_sources = content_sources.intersection(template_sources)
        if content_sources:
            match_ratio = len(common_sources) / len(content_sources)
            return min(match_ratio, 1.0)

        return 0.0

    async def _analyze_single_template(self, file_path: str, file_name: str) -> Dict:
        """개별 JSX 템플릿 분석 (기존 메서드 유지)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                jsx_content = f.read()

            # 기본 정보 추출
            component_name = self._extract_component_name(jsx_content)
            props = self._extract_props(jsx_content)
            styled_components = self._extract_styled_components(jsx_content)
            layout_structure = self._analyze_layout_structure(jsx_content)

            return {
                'file_name': file_name,
                'component_name': component_name,
                'props': props,
                'styled_components': styled_components,
                'layout_type': layout_structure['type'],
                'layout_features': layout_structure['features'],
                'grid_structure': layout_structure['grid'],
                'image_strategy': layout_structure['images'],
                'text_strategy': layout_structure['text'],
                'complexity_level': layout_structure['complexity'],
                'original_jsx': jsx_content,
                'analysis_success': True
            }

        except Exception as e:
            print(f"⚠️ {file_name} 분석 실패: {e}")
            # 개별 템플릿 분석 실패 로깅
            # 개별 템플릿 분석 실패 로깅
            self.result_manager.store_agent_output(
                agent_name="JSXTemplateAnalyzer_SingleTemplate",
                agent_role="개별 템플릿 분석기",
                task_description=f"JSX 템플릿 {file_name} 분석 실패",
                final_answer=f"템플릿 분석 실패: {str(e)}",
                reasoning_process=f"파일 읽기 또는 구조 분석 중 오류 발생",
                raw_input={"file_path": file_path, "file_name": file_name},
                performance_metrics={"analysis_failed": True, "error": str(e)}
            )
            return self._create_default_template_analysis(file_name)

    def _create_default_template_analysis(self, file_name: str) -> Dict:
        """기본 템플릿 분석 결과 생성"""
        return {
            'file_name': file_name,
            'component_name': file_name.replace('.jsx', ''),
            'props': [],
            'styled_components': [],
            'layout_type': 'simple',
            'layout_features': ['basic'],
            'grid_structure': False,
            'image_strategy': 1,
            'text_strategy': 'basic',
            'complexity_level': 'simple',
            'original_jsx': f'// Default fallback for {file_name}',
            'analysis_success': False,
            'fallback_reason': 'analysis_failed'
        }

    def _extract_component_name(self, jsx_content: str) -> str:
        """JSX에서 컴포넌트명 추출"""
        match = re.search(r'export\s+const\s+(\w+)', jsx_content)
        return match.group(1) if match else 'UnknownComponent'

    def _extract_props(self, jsx_content: str) -> List[str]:
        """JSX에서 props 추출"""
        props = []
        # props 패턴 찾기
        prop_patterns = [
            r'\{(\w+)\}',  # {propName}
            r'props\.(\w+)',  # props.propName
            r'=\s*\{(\w+)\}'  # ={propName}
        ]
        
        for pattern in prop_patterns:
            matches = re.findall(pattern, jsx_content)
            props.extend(matches)
        
        # 중복 제거 및 일반적인 JSX 키워드 제외
        excluded = {'children', 'key', 'ref', 'className', 'style', 'onClick', 'onChange'}
        return list(set(props) - excluded)

    def _extract_styled_components(self, jsx_content: str) -> List[str]:
        """styled-components 추출"""
        styled_pattern = r'const\s+(\w+)\s*=\s*styled\.'
        matches = re.findall(styled_pattern, jsx_content)
        return matches

    def _analyze_layout_structure(self, jsx_content: str) -> Dict:
        """레이아웃 구조 분석"""
        structure = {
            'type': 'simple',
            'features': [],
            'grid': False,
            'images': 0,
            'text': 'basic',
            'complexity': 'simple'
        }

        # 그리드 구조 확인
        if 'grid' in jsx_content.lower() or 'Grid' in jsx_content:
            structure['grid'] = True
            structure['features'].append('grid_layout')

        # 이미지 전략 분석
        img_count = jsx_content.count('<img') + jsx_content.count('Image')
        structure['images'] = min(img_count, 5)  # 최대 5개로 제한

        # 레이아웃 타입 결정
        if 'hero' in jsx_content.lower() or 'Hero' in jsx_content:
            structure['type'] = 'hero'
        elif structure['grid'] or img_count > 2:
            structure['type'] = 'grid' if img_count <= 4 else 'gallery'
        elif 'overlay' in jsx_content.lower():
            structure['type'] = 'overlay'

        # 복잡도 분석
        styled_count = jsx_content.count('styled.')
        component_count = len(re.findall(r'const\s+\w+\s*=', jsx_content))
        
        if styled_count > 5 or component_count > 8:
            structure['complexity'] = 'complex'
        elif styled_count > 2 or component_count > 4:
            structure['complexity'] = 'moderate'

        # 특징 추가
        if 'height:' in jsx_content and 'vh' in jsx_content:
            structure['features'].append('fixed_height')
        if 'flex-direction: column' in jsx_content or 'flexDirection: "column"' in jsx_content:
            structure['features'].append('vertical_layout')
        if 'gap:' in jsx_content or 'margin:' in jsx_content:
            structure['features'].append('gap_spacing')

        return structure

    # 디버깅 및 모니터링 메서드
    def get_execution_statistics(self) -> Dict:
        """실행 통계 조회"""
        return {
            **self.execution_stats,
            "success_rate": (
                self.execution_stats["successful_executions"] / 
                max(self.execution_stats["total_attempts"], 1)
            ) * 100,
            "crew_circuit_breaker_state": self.crew_circuit_breaker.state,
            "vector_circuit_breaker_state": self.vector_db_circuit_breaker.state,
            "templates_cached": len(self.templates_cache)
        }

    def reset_system_state(self) -> None:
        """시스템 상태 리셋"""
        print("🔄 JSXTemplateAnalyzer 시스템 상태 리셋")
        
        # Circuit Breaker 리셋
        self.crew_circuit_breaker._reset_counts()
        self.crew_circuit_breaker._state = CircuitBreakerState.CLOSED
        self.vector_db_circuit_breaker._reset_counts()
        self.vector_db_circuit_breaker._state = CircuitBreakerState.CLOSED
        
        # 폴백 플래그 리셋
        self.fallback_to_sync = False
        
        # 캐시 클리어
        self.templates_cache.clear()
        
        # 통계 초기화
        self.execution_stats = {
            "total_attempts": 0,
            "successful_executions": 0,
            "fallback_used": 0,
            "circuit_breaker_triggered": 0,
            "timeout_occurred": 0
        }
        
        print("✅ 시스템 상태가 리셋되었습니다.")

    def get_performance_metrics(self) -> Dict:
        """성능 메트릭 수집"""
        return {
            "circuit_breakers": {
                "crew": {
                    "state": self.crew_circuit_breaker.state,
                    "failure_count": self.crew_circuit_breaker._failure_count,
                    "failure_threshold": self.crew_circuit_breaker.failure_threshold
                },
                "vector_db": {
                    "state": self.vector_db_circuit_breaker.state,
                    "failure_count": self.vector_db_circuit_breaker._failure_count,
                    "failure_threshold": self.vector_db_circuit_breaker.failure_threshold
                }
            },
            "work_queue": {
                "running": self.work_queue._running,
                "workers": len(self.work_queue._workers),
                "results_count": len(self.work_queue._results)
            },
            "system": {
                "recursion_threshold": self.recursion_threshold,
                "fallback_to_sync": self.fallback_to_sync,
                "templates_cached": len(self.templates_cache)
            },
            "execution_stats": self.execution_stats
        }

    def get_system_info(self) -> Dict:
        """시스템 정보 조회"""
        return {
            "class_name": self.__class__.__name__,
            "version": "2.0_resilient",
            "features": [
                "CrewAI 기반 템플릿 분석",
                "PDF 벡터 데이터 통합",
                "복원력 있는 실행",
                "Circuit Breaker 패턴",
                "재귀 깊이 감지",
                "동기/비동기 폴백",
                "에이전트 결과 통합"
            ],
            "agents": [
                "template_analysis_agent",
                "vector_enhancement_agent",
                "agent_result_integrator",
                "template_selector_agent"
            ],
            "execution_modes": ["batch_resilient", "sync_fallback"],
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
                self.result_manager
            ]
            
            for component in required_components:
                if component is None:
                    return False
            
            # CrewAI 에이전트들 확인
            crewai_agents = [
                self.template_analysis_agent,
                self.vector_enhancement_agent,
                self.agent_result_integrator,
                self.template_selector_agent
            ]
            
            for agent in crewai_agents:
                if agent is None:
                    return False
            
            # 복원력 시스템 확인
            if (self.work_queue is None or 
                self.crew_circuit_breaker is None or 
                self.vector_db_circuit_breaker is None):
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ 시스템 무결성 검증 실패: {e}")
            return False

    def get_template_cache_info(self) -> Dict:
        """템플릿 캐시 정보 조회"""
        cache_info = {
            "total_templates": len(self.templates_cache),
            "templates": [],
            "layout_types": {},
            "complexity_levels": {},
            "vector_matched_count": 0,
            "agent_enhanced_count": 0
        }
        
        for template_name, template_data in self.templates_cache.items():
            if isinstance(template_data, dict):
                template_info = {
                    "name": template_name,
                    "layout_type": template_data.get('layout_type', 'unknown'),
                    "complexity": template_data.get('complexity_level', 'unknown'),
                    "image_strategy": template_data.get('image_strategy', 0),
                    "vector_matched": template_data.get('vector_matched', False),
                    "agent_enhanced": template_data.get('agent_enhanced', False)
                }
                cache_info["templates"].append(template_info)
                
                # 통계 집계
                layout_type = template_data.get('layout_type', 'unknown')
                cache_info["layout_types"][layout_type] = cache_info["layout_types"].get(layout_type, 0) + 1
                
                complexity = template_data.get('complexity_level', 'unknown')
                cache_info["complexity_levels"][complexity] = cache_info["complexity_levels"].get(complexity, 0) + 1
                
                if template_data.get('vector_matched', False):
                    cache_info["vector_matched_count"] += 1
                    
                if template_data.get('agent_enhanced', False):
                    cache_info["agent_enhanced_count"] += 1
        
        return cache_info

    def clear_template_cache(self) -> None:
        """템플릿 캐시 클리어"""
        print(f"🗑️ 템플릿 캐시 클리어: {len(self.templates_cache)}개 템플릿 제거")
        self.templates_cache.clear()

    async def refresh_template_analysis(self, templates_dir: str = "jsx_templates") -> Dict[str, Dict]:
        """템플릿 분석 새로고침"""
        print("🔄 템플릿 분석 새로고침 시작")
        self.clear_template_cache()
        return await self.analyze_jsx_templates(templates_dir)

    # 기존 동기 버전 메서드들 (호환성 유지)
    def analyze_jsx_templates_sync(self, templates_dir: str = "jsx_templates") -> Dict[str, Dict]:
        """동기 버전 템플릿 분석 (호환성 유지)"""
        return asyncio.run(self.analyze_jsx_templates(templates_dir))

    def get_best_template_for_content_sync(self, content: Dict, analysis: Dict) -> str:
        """동기 버전 템플릿 선택 (호환성 유지)"""
        return asyncio.run(self.get_best_template_for_content(content, analysis))

    # 캐시 관리 메서드들 (기존 유지)
    def get_cached_templates(self) -> Dict[str, Dict]:
        """캐시된 템플릿 정보 반환"""
        return self.templates_cache.copy()

    def get_template_info(self, template_name: str) -> Optional[Dict]:
        """특정 템플릿 정보 조회"""
        return self.templates_cache.get(template_name)

    def update_template_cache(self, template_name: str, template_info: Dict) -> None:
        """템플릿 캐시 업데이트"""
        self.templates_cache[template_name] = template_info

    # 벡터 관련 유틸리티 메서드들 (기존 유지)
    def search_templates_by_layout(self, layout_type: str) -> List[str]:
        """레이아웃 타입으로 템플릿 검색"""
        matching_templates = []
        for template_name, template_info in self.templates_cache.items():
            if isinstance(template_info, dict) and template_info.get('layout_type') == layout_type:
                matching_templates.append(template_name)
        return matching_templates

    def search_templates_by_complexity(self, complexity_level: str) -> List[str]:
        """복잡도로 템플릿 검색"""
        matching_templates = []
        for template_name, template_info in self.templates_cache.items():
            if isinstance(template_info, dict) and template_info.get('complexity_level') == complexity_level:
                matching_templates.append(template_name)
        return matching_templates

    def search_templates_by_image_count(self, image_count: int, tolerance: int = 1) -> List[str]:
        """이미지 개수로 템플릿 검색"""
        matching_templates = []
        for template_name, template_info in self.templates_cache.items():
            if isinstance(template_info, dict):
                template_images = template_info.get('image_strategy', 0)
                if abs(template_images - image_count) <= tolerance:
                    matching_templates.append(template_name)
        return matching_templates

    def get_vector_enhanced_templates(self) -> List[str]:
        """벡터 데이터로 강화된 템플릿 목록"""
        enhanced_templates = []
        for template_name, template_info in self.templates_cache.items():
            if isinstance(template_info, dict) and template_info.get('vector_matched', False):
                enhanced_templates.append(template_name)
        return enhanced_templates

    def get_agent_enhanced_templates(self) -> List[str]:
        """에이전트 결과로 강화된 템플릿 목록"""
        enhanced_templates = []
        for template_name, template_info in self.templates_cache.items():
            if isinstance(template_info, dict) and template_info.get('agent_enhanced', False):
                enhanced_templates.append(template_name)
        return enhanced_templates

    # 통계 및 분석 메서드들
    def get_template_statistics(self) -> Dict:
        """템플릿 통계 정보"""
        stats = {
            "total_templates": len(self.templates_cache),
            "layout_distribution": {},
            "complexity_distribution": {},
            "image_strategy_distribution": {},
            "vector_enhancement_rate": 0,
            "agent_enhancement_rate": 0,
            "average_confidence": 0
        }

        if not self.templates_cache:
            return stats

        layout_types = []
        complexities = []
        image_strategies = []
        confidences = []
        vector_enhanced = 0
        agent_enhanced = 0

        for template_info in self.templates_cache.values():
            if isinstance(template_info, dict):
                layout_types.append(template_info.get('layout_type', 'unknown'))
                complexities.append(template_info.get('complexity_level', 'unknown'))
                image_strategies.append(template_info.get('image_strategy', 0))
                
                confidence = template_info.get('layout_confidence', 0)
                if confidence > 0:
                    confidences.append(confidence)
                
                if template_info.get('vector_matched', False):
                    vector_enhanced += 1
                if template_info.get('agent_enhanced', False):
                    agent_enhanced += 1

        # 분포 계산
        for layout_type in set(layout_types):
            stats["layout_distribution"][layout_type] = layout_types.count(layout_type)
        
        for complexity in set(complexities):
            stats["complexity_distribution"][complexity] = complexities.count(complexity)
        
        for strategy in set(image_strategies):
            stats["image_strategy_distribution"][str(strategy)] = image_strategies.count(strategy)

        # 비율 계산
        total = len(self.templates_cache)
        stats["vector_enhancement_rate"] = (vector_enhanced / total) * 100 if total > 0 else 0
        stats["agent_enhancement_rate"] = (agent_enhanced / total) * 100 if total > 0 else 0
        stats["average_confidence"] = sum(confidences) / len(confidences) if confidences else 0

        return stats

    async def cleanup_resources(self) -> None:
        """리소스 정리"""
        print("🧹 JSXTemplateAnalyzer 리소스 정리 시작")
        
        try:
            # 작업 큐 정리
            await self.work_queue.stop()
            
            # 캐시 정리
            self.templates_cache.clear()
            
            print("✅ 리소스 정리 완료")
        except Exception as e:
            print(f"⚠️ 리소스 정리 중 오류: {e}")

    def __del__(self):
        """소멸자 - 리소스 정리"""
        try:
            if hasattr(self, 'work_queue') and self.work_queue._running:
                asyncio.create_task(self.work_queue.stop())
        except Exception:
            pass  # 소멸자에서는 예외를 무시
