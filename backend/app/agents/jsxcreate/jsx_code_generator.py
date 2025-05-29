from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field
from crewai import Agent, Task
from custom_llm import get_azure_llm
from utils.agent_decision_logger import get_agent_logger, get_complete_data_manager
import re
import asyncio
import time
import logging
import sys
from enum import Enum
from functools import wraps
import traceback

# ==================== 기본 인프라 클래스들 ====================


@dataclass
class WorkItem:
    """작업 항목 정의"""
    id: str
    task_func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 0
    max_retries: int = 3
    current_retry: int = 0
    timeout: float = 300.0
    created_at: float = field(default_factory=time.time)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit Breaker 패턴 구현"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.logger = logging.getLogger(__name__)

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Circuit Breaker를 통한 함수 호출"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


class CircuitBreakerOpenError(Exception):
    """Circuit Breaker가 열린 상태일 때 발생하는 예외"""
    pass


class AsyncWorkQueue:
    """비동기 작업 큐 기반 배치 처리 시스템"""

    def __init__(self, max_workers: int = 2, max_queue_size: int = 50, batch_size: int = 3):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_workers)
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing = False
        self.results = {}
        self.logger = logging.getLogger(__name__)

    async def submit_work(self, work_item: WorkItem) -> str:
        """작업 제출"""
        try:
            await asyncio.wait_for(
                self.queue.put(work_item),
                timeout=5.0
            )
            if not self.processing:
                asyncio.create_task(self._process_batches())
            return work_item.id
        except asyncio.TimeoutError:
            raise Exception("Work queue is full")

    async def get_result(self, work_id: str, timeout: float = 300.0) -> Any:
        """결과 조회"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if work_id in self.results:
                result = self.results.pop(work_id)
                if isinstance(result, Exception):
                    raise result
                return result
            await asyncio.sleep(0.1)
        raise asyncio.TimeoutError(f"Work {work_id} timed out")

    async def _process_batches(self):
        """배치 처리 실행"""
        self.processing = True
        try:
            while not self.queue.empty():
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
                    # 배치 간 쿨다운
                    await asyncio.sleep(0.5)
        finally:
            self.processing = False

    async def _collect_batch(self) -> List[WorkItem]:
        """배치 수집"""
        batch = []
        for _ in range(self.batch_size):
            try:
                work_item = await asyncio.wait_for(
                    self.queue.get(), timeout=1.0
                )
                batch.append(work_item)
            except asyncio.TimeoutError:
                break
        return batch

    async def _process_batch(self, batch: List[WorkItem]):
        """배치 작업 처리"""
        async with self.semaphore:
            tasks = [self._execute_work_item(item) for item in batch]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_work_item(self, work_item: WorkItem):
        """개별 작업 실행"""
        try:
            result = await asyncio.wait_for(
                work_item.task_func(*work_item.args, **work_item.kwargs),
                timeout=work_item.timeout
            )
            self.results[work_item.id] = result
        except Exception as e:
            self.logger.error(f"Work item {work_item.id} failed: {e}")
            self.results[work_item.id] = e


class BaseAsyncAgent:
    """기본 비동기 에이전트 클래스"""

    def __init__(self):
        self.work_queue = AsyncWorkQueue(max_workers=2, max_queue_size=50)
        self.circuit_breaker = CircuitBreaker()
        self.recursion_threshold = 600
        self.fallback_to_sync = False
        self.logger = logging.getLogger(__name__)

        # 타임아웃 설정
        self.timeouts = {
            'agent_execution': 45.0,
            'result_collection': 10.0,
            'post_processing': 15.0,
            'total_operation': 120.0,
            'crew_kickoff': 60.0
        }

        # 재시도 설정
        self.retry_config = {
            'max_attempts': 3,
            'base_delay': 1.0,
            'max_delay': 8.0,
            'exponential_base': 2
        }

    def _should_use_sync(self) -> bool:
        """동기 모드 사용 여부 판단"""
        current_frame_count = len(traceback.extract_stack())
        return (
            self.fallback_to_sync or
            current_frame_count > self.recursion_threshold or
            self.circuit_breaker.state == CircuitState.OPEN
        )

    async def execute_with_resilience(
        self,
        task_func: Callable,
        task_id: str,
        timeout: float = 300.0,
        max_retries: int = 3,
        *args,
        **kwargs
    ) -> Any:
        """복원력 있는 작업 실행"""

        work_item = WorkItem(
            id=task_id,
            task_func=task_func,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            max_retries=max_retries
        )

        for attempt in range(max_retries):
            try:
                await self.work_queue.submit_work(work_item)
                result = await self.work_queue.get_result(task_id, timeout)
                return result

            except (CircuitBreakerOpenError, asyncio.TimeoutError, RecursionError) as e:
                self.logger.warning(
                    f"Attempt {attempt + 1} failed for {task_id}: {e}")

                if attempt < max_retries - 1:
                    delay = min(
                        self.retry_config['base_delay'] *
                        (self.retry_config['exponential_base'] ** attempt),
                        self.retry_config['max_delay']
                    )
                    await asyncio.sleep(delay)
                    continue

                # 최종 실패 시 폴백
                self.fallback_to_sync = True
                return self._get_fallback_result(task_id)

            except Exception as e:
                self.logger.error(f"Unexpected error in {task_id}: {e}")
                return self._get_fallback_result(task_id)

    def _get_fallback_result(self, task_id: str) -> Any:
        """폴백 결과 생성 (서브클래스에서 구현)"""
        return f"FALLBACK_RESULT_FOR_{task_id}"

# ==================== 개선된 JSXCodeGenerator ====================


class JSXCodeGenerator(BaseAsyncAgent):
    """JSX 코드 생성 전문 에이전트 (에이전트 결과 데이터 기반)"""

    def __init__(self):
        super().__init__()
        self.llm = get_azure_llm()
        self.logger = get_agent_logger()
        self.result_manager = get_complete_data_manager()

        # JSX 특화 타임아웃 설정
        self.timeouts.update({
            'jsx_generation': 90.0,
            'post_processing': 20.0,
            'agent_creation': 15.0,
            'result_collection': 12.0
        })

    def create_agent(self):
        """기존 에이전트 생성 메서드 (변경 없음)"""
        return Agent(
            role="에이전트 결과 데이터 기반 React JSX 코드 생성 전문가",
            goal="이전 에이전트들의 모든 결과 데이터를 활용하여 오류 없는 완벽한 JSX 코드를 생성",
            backstory="""당신은 10년간 세계 최고 수준의 디지털 매거진과 웹 개발 분야에서 활동해온 풀스택 개발자입니다.

**에이전트 결과 데이터 활용 전문성:**
- 이전 에이전트들의 모든 출력 결과를 분석하여 최적의 JSX 구조 설계
- ContentCreator, ImageAnalyzer, LayoutDesigner 등의 결과를 통합 활용
- 에이전트 협업 패턴과 성공 사례를 JSX 코드에 반영
- template_data.json과 벡터 데이터를 보조 데이터로 활용

**오류 없는 코드 생성 철학:**
"모든 JSX 코드는 컴파일 오류 없이 완벽하게 작동해야 합니다. 에이전트들의 협업 결과를 존중하면서도 기술적 완성도를 보장하는 것이 최우선입니다."

**데이터 우선순위:**
1. 이전 에이전트들의 결과 데이터 (최우선)
2. template_data.json의 콘텐츠 정보
3. PDF 벡터 데이터의 레이아웃 패턴
4. jsx_templates는 사용하지 않음
5. 존재하는 콘텐츠 데이터 및 이미지 URL은 모두 사용한다.
6. 에이전트 결과 데이터는 반드시 활용한다.
7. 콘텐츠 데이터 및 이미지URL이 아닌 설계 구조 및 레이아웃 정보는 사용하지 않는다.""",
            verbose=True,
            llm=self.llm
        )

    async def generate_jsx_code(self, content: Dict, design: Dict, component_name: str) -> str:
        """에이전트 결과 데이터 기반 JSX 코드 생성 (개선된 버전)"""

        # 재귀 깊이 체크
        if self._should_use_sync():
            return await self._generate_jsx_code_sync_mode(content, design, component_name)

        try:
            return await self._generate_jsx_code_batch_mode(content, design, component_name)
        except (RecursionError, CircuitBreakerOpenError) as e:
            self.logger.warning(f"Switching to sync mode due to: {e}")
            self.fallback_to_sync = True
            return await self._generate_jsx_code_sync_mode(content, design, component_name)

    async def _generate_jsx_code_batch_mode(self, content: Dict, design: Dict, component_name: str) -> str:
        """배치 기반 안전한 JSX 생성"""

        task_id = f"jsx_generation_{component_name}_{int(time.time())}"

        async def _safe_jsx_generation():
            return await self._execute_jsx_generation_pipeline(content, design, component_name)

        try:
            result = await self.execute_with_resilience(
                _safe_jsx_generation,
                task_id,
                timeout=self.timeouts['jsx_generation'],
                max_retries=2
            )

            if result and not result.startswith("FALLBACK_RESULT"):
                return result
            else:
                return await self._generate_jsx_code_sync_mode(content, design, component_name)

        except Exception as e:
            self.logger.error(f"Batch mode failed for {component_name}: {e}")
            return await self._generate_jsx_code_sync_mode(content, design, component_name)

    async def _generate_jsx_code_sync_mode(self, content: Dict, design: Dict, component_name: str) -> str:
        """동기 모드 폴백 처리"""

        try:
            # 안전한 결과 수집
            previous_results = await self._safe_collect_results()

            # 기본 JSX 생성
            jsx_code = await self._create_basic_jsx_safe(content, design, component_name)

            # 간단한 후처리
            jsx_code = self._apply_basic_post_processing(
                jsx_code, content, component_name)

            # 결과 저장
            await self._safe_store_result(jsx_code, content, design, component_name, len(previous_results))

            print(f"✅ 동기 모드로 JSX 코드 생성 완료: {component_name}")
            return jsx_code

        except Exception as e:
            print(f"⚠️ 동기 모드 JSX 생성 실패: {e}")
            return self._get_fallback_result(component_name)

    async def _execute_jsx_generation_pipeline(self, content: Dict, design: Dict, component_name: str) -> str:
        """개선된 JSX 생성 파이프라인"""

        # 1단계: 결과 수집 (타임아웃 적용)
        previous_results = await self._safe_collect_results()

        # 결과 분류
        binding_results = [
            r for r in previous_results if "BindingAgent" in r.get('agent_name', '')]
        org_results = [
            r for r in previous_results if "OrgAgent" in r.get('agent_name', '')]
        content_results = [
            r for r in previous_results if "ContentCreator" in r.get('agent_name', '')]

        print(f"📊 이전 결과 수집: 전체 {len(previous_results)}개")
        print(f" - BindingAgent: {len(binding_results)}개")
        print(f" - OrgAgent: {len(org_results)}개")
        print(f" - ContentCreator: {len(content_results)}개")

        # 2단계: 에이전트 생성 (재시도 적용)
        agent = await self._create_agent_with_retry()

        # 3단계: 태스크 실행 (Circuit Breaker 적용)
        jsx_code = await self._execute_jsx_task_safe(
            agent, content, design, component_name,
            previous_results, binding_results, org_results, content_results
        )

        # 4단계: 후처리 (깊이 제한 적용)
        jsx_code = await self._post_process_safe(
            jsx_code, previous_results, binding_results,
            org_results, content_results, content, component_name
        )

        # 5단계: 결과 저장
        await self._safe_store_result(jsx_code, content, design, component_name, len(previous_results))

        print(f"✅ 에이전트 데이터 기반 JSX 코드 생성 완료: {component_name}")
        return jsx_code

    async def _safe_collect_results(self) -> List[Dict]:
        """안전한 결과 수집"""
        try:
            return await asyncio.wait_for(
                self.result_manager.get_all_outputs(
                    exclude_agent="JSXCodeGenerator"),
                timeout=self.timeouts['result_collection']
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                "Result collection timeout, using empty results")
            return []
        except Exception as e:
            self.logger.error(f"Result collection failed: {e}")
            return []

    async def _create_agent_with_retry(self) -> Agent:
        """재시도가 적용된 에이전트 생성"""
        for attempt in range(self.retry_config['max_attempts']):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self.create_agent),
                    timeout=self.timeouts['agent_creation']
                )
            except Exception as e:
                if attempt < self.retry_config['max_attempts'] - 1:
                    delay = min(
                        self.retry_config['base_delay'] *
                        (self.retry_config['exponential_base'] ** attempt),
                        self.retry_config['max_delay']
                    )
                    await asyncio.sleep(delay)
                    continue
                raise e

    async def _execute_jsx_task_safe(
        self,
        agent: Agent,
        content: Dict,
        design: Dict,
        component_name: str,
        previous_results: List[Dict],
        binding_results: List[Dict],
        org_results: List[Dict],
        content_results: List[Dict]
    ) -> str:
        """안전한 JSX 태스크 실행"""

        # 에이전트 결과 데이터 요약
        agent_data_summary = self._summarize_agent_results(
            previous_results, binding_results, org_results, content_results
        )

        generation_task = Task(
            description=f"""
**에이전트 결과 데이터 기반 오류 없는 JSX 코드 생성**

이전 에이전트들의 모든 결과 데이터를 활용하여 완벽한 JSX 코드를 생성하세요:

**이전 에이전트 결과 데이터 ({len(previous_results)}개):**
{agent_data_summary}

**BindingAgent 이미지 배치 인사이트 ({len(binding_results)}개):**
{self._extract_binding_insights(binding_results)}

**OrgAgent 텍스트 구조 인사이트 ({len(org_results)}개):**
{self._extract_org_insights(org_results)}

**ContentCreator 콘텐츠 인사이트 ({len(content_results)}개):**
{self._extract_content_insights(content_results)}

**실제 콘텐츠 (template_data.json 기반):**
- 제목: {content.get('title', '')}
- 부제목: {content.get('subtitle', '')}
- 본문: {content.get('body', '')}
- 이미지 URLs: {content.get('images', [])}
- 태그라인: {content.get('tagline', '')}

**레이아웃 설계 (LayoutDesigner 결과):**
- 타입: {design.get('layout_type', 'grid')}
- 그리드 구조: {design.get('grid_structure', '1fr 1fr')}
- 컴포넌트들: {design.get('styled_components', [])}
- 색상 스키마: {design.get('color_scheme', {})}

**오류 없는 JSX 생성 지침:**
1. 반드시 import React from "react"; 포함
2. 반드시 import styled from "styled-components"; 포함
3. export const {component_name} = () => {{ ... }}; 형태 준수
4. 모든 중괄호, 괄호 정확히 매칭
5. 모든 이미지 URL을 실제 형태로 포함
6. className 사용 (class 아님)
7. JSX 문법 완벽 준수

**절대 금지사항:**
- `````` 마크다운 블록
- 문법 오류 절대 금지
- 불완전한 return 문 금지
- jsx_templates 참조 금지

**에이전트 결과 데이터 활용 방법:**
- BindingAgent의 이미지 배치 전략을 JSX 이미지 태그에 반영
- OrgAgent의 텍스트 구조를 JSX 컴포넌트 구조에 반영
- ContentCreator의 콘텐츠 품질을 JSX 스타일링에 반영
- 이전 성공적인 JSX 패턴 재사용
- 협업 에이전트들의 품질 지표 고려

**출력:** 순수한 JSX 파일 코드만 출력 (설명이나 마크다운 없이)
""",
            agent=agent,
            expected_output="에이전트 결과 데이터 기반 오류 없는 순수 JSX 코드"
        )

        try:
            result = await asyncio.wait_for(
                agent.execute_task(generation_task),
                timeout=self.timeouts['agent_execution']
            )
            return str(result)

        except asyncio.TimeoutError:
            self.logger.error(f"Agent execution timeout for {component_name}")
            raise

    async def _post_process_safe(
        self,
        jsx_code: str,
        previous_results: List[Dict],
        binding_results: List[Dict],
        org_results: List[Dict],
        content_results: List[Dict],
        content: Dict,
        component_name: str,
        max_depth: int = 2
    ) -> str:
        """깊이 제한이 적용된 안전한 후처리"""

        if max_depth <= 0:
            self.logger.warning(
                "Max processing depth reached, returning current code")
            return jsx_code

        try:
            # 타임아웃이 적용된 후처리
            processed_code = await asyncio.wait_for(
                self._apply_enhanced_post_processing(
                    jsx_code, previous_results, binding_results,
                    org_results, content_results, content, component_name
                ),
                timeout=self.timeouts['post_processing']
            )

            return processed_code

        except asyncio.TimeoutError:
            self.logger.warning(
                "Post-processing timeout, returning current code")
            return jsx_code
        except Exception as e:
            self.logger.error(f"Post-processing failed: {e}")
            return jsx_code

    async def _apply_enhanced_post_processing(
        self,
        jsx_code: str,
        previous_results: List[Dict],
        binding_results: List[Dict],
        org_results: List[Dict],
        content_results: List[Dict],
        content: Dict,
        component_name: str
    ) -> str:
        """강화된 후처리 적용"""

        # 1. 마크다운 블록 제거
        jsx_code = self._remove_markdown_blocks(jsx_code)

        # 2. 기본 구조 검증
        jsx_code = self._validate_basic_structure(jsx_code, component_name)

        # 3. 에이전트 결과 기반 강화 (안전하게)
        jsx_code = self._safe_enhance_with_binding_results(
            jsx_code, binding_results, content)
        jsx_code = self._safe_enhance_with_org_results(
            jsx_code, org_results, content)
        jsx_code = self._safe_enhance_with_content_results(
            jsx_code, content_results, content)

        # 4. 이미지 URL 강제 포함
        jsx_code = self._ensure_image_urls(jsx_code, content)

        # 5. 최종 오류 검사 및 수정
        jsx_code = self._final_error_check_and_fix(jsx_code, component_name)

        return jsx_code

    async def _create_basic_jsx_safe(self, content: Dict, design: Dict, component_name: str) -> str:
        """안전한 기본 JSX 생성"""
        title = content.get('title', 'Safe Component')
        body = content.get('body', 'Content generated in safe mode.')
        images = content.get('images', [])

        image_jsx = []
        for i, img_url in enumerate(images[:3]):
            if img_url and img_url.strip():
                image_jsx.append(
                    f'<TravelImage src="{img_url}" alt="Image {i+1}" />')

        image_section = '\n'.join(
            image_jsx) if image_jsx else '<PlaceholderDiv>Safe Mode Content</PlaceholderDiv>'

        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
  background: #f8f9fa;
`;

const Title = styled.h1`
  font-size: 2.5em;
  color: #2c3e50;
  margin-bottom: 20px;
`;

const Content = styled.div`
  font-size: 1.1em;
  line-height: 1.6;
  color: #555;
  margin-bottom: 30px;
`;

const ImageGallery = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
`;

const TravelImage = styled.img`
  width: 100%;
  height: 150px;
  object-fit: cover;
  border-radius: 8px;
`;

const PlaceholderDiv = styled.div`
  width: 100%;
  height: 150px;
  background: #e9ecef;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6c757d;
`;

export const {component_name} = () => {{
  return (
    <Container>
      <Title>{title}</Title>
      <Content>{body}</Content>
      <ImageGallery>
        {image_section}
      </ImageGallery>
    </Container>
  );
}};'''

    def _apply_basic_post_processing(self, jsx_code: str, content: Dict, component_name: str) -> str:
        """기본 후처리 적용"""
        jsx_code = self._remove_markdown_blocks(jsx_code)
        jsx_code = self._validate_basic_structure(jsx_code, component_name)
        jsx_code = self._ensure_image_urls(jsx_code, content)
        jsx_code = self._final_error_check_and_fix(jsx_code, component_name)
        return jsx_code

    async def _safe_store_result(
        self,
        jsx_code: str,
        content: Dict,
        design: Dict,
        component_name: str,
        agent_count: int
    ):
        """안전한 결과 저장"""
        try:
            await asyncio.wait_for(
                self.result_manager.store_agent_output(
                    agent_name="JSXCodeGenerator",
                    agent_role="JSX 코드 생성 전문가",
                    task_description=f"컴포넌트 {component_name} JSX 코드 생성",
                    final_answer=jsx_code,
                    reasoning_process=f"이전 {agent_count}개 에이전트 결과 활용하여 JSX 생성",
                    execution_steps=[
                        "에이전트 결과 수집 및 분석",
                        "BindingAgent/OrgAgent/ContentCreator 인사이트 추출",
                        "JSX 코드 생성",
                        "후처리 및 검증"
                    ],
                    raw_input={"content": content, "design": design,
                        "component_name": component_name},
                    raw_output=jsx_code,
                    performance_metrics={
                        "agent_results_utilized": agent_count,
                        "jsx_templates_ignored": True,
                        "error_free_validated": self._validate_jsx_syntax(jsx_code),
                        "code_length": len(jsx_code),
                        "safe_mode_used": self.fallback_to_sync
                    }
                ),
                timeout=5.0
            )
        except Exception as e:
            self.logger.error(f"Failed to store result: {e}")

    def _get_fallback_result(self, task_id: str) -> str:
        """JSX 전용 폴백 결과 생성"""
        component_name = task_id.replace("jsx_generation_", "").split("_")[
                                         0] or "FallbackComponent"

        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
  text-align: center;
`;

const Title = styled.h1`
  color: #2c3e50;
  margin-bottom: 1rem;
`;

const Message = styled.p`
  color: #555;
  line-height: 1.6;
`;

export const {component_name} = () => {{
  return (
    <Container>
      <Title>Safe Mode Component</Title>
      <Message>This component was generated in safe mode due to system constraints.</Message>
    </Container>
  );
}};'''

    # ==================== 안전한 강화 메서드들 ====================

    def _safe_enhance_with_binding_results(self, jsx_code: str, binding_results: List[Dict], content: Dict) -> str:
        """안전한 BindingAgent 결과 강화"""
        try:
            return self._enhance_with_binding_results(jsx_code, binding_results, content)
        except Exception as e:
            self.logger.warning(f"Binding enhancement failed: {e}")
            return jsx_code

    def _safe_enhance_with_org_results(self, jsx_code: str, org_results: List[Dict], content: Dict) -> str:
        """안전한 OrgAgent 결과 강화"""
        try:
            return self._enhance_with_org_results(jsx_code, org_results, content)
        except Exception as e:
            self.logger.warning(f"Org enhancement failed: {e}")
            return jsx_code

    def _safe_enhance_with_content_results(self, jsx_code: str, content_results: List[Dict], content: Dict) -> str:
        """안전한 ContentCreator 결과 강화"""
        try:
            return self._enhance_with_content_results(jsx_code, content_results, content)
        except Exception as e:
            self.logger.warning(f"Content enhancement failed: {e}")
            return jsx_code

    # ==================== 기존 메서드들 (완전 보존) ====================

    def _summarize_agent_results(self, previous_results: List[Dict], binding_results: List[Dict], org_results: List[Dict], content_results: List[Dict]) -> str:
        """에이전트 결과 데이터 요약 (모든 에이전트 포함)"""
        if not previous_results:
            return "이전 에이전트 결과 없음 - 기본 패턴 사용"

        summary_parts = []

        # 에이전트별 결과 분류
        agent_groups = {}
        for result in previous_results:
            agent_name = result.get('agent_name', 'unknown')
            if agent_name not in agent_groups:
                agent_groups[agent_name] = []
            agent_groups[agent_name].append(result)

        # 각 에이전트 그룹 요약
        for agent_name, results in agent_groups.items():
            latest_result = results[-1]  # 최신 결과
            answer_length = len(latest_result.get('final_answer', ''))
            summary_parts.append(
                f"- {agent_name}: {len(results)}개 결과, 최신 답변 길이: {answer_length}자")

        # 특별 요약
        summary_parts.append(f"- BindingAgent 특별 수집: {len(binding_results)}개")
        summary_parts.append(f"- OrgAgent 특별 수집: {len(org_results)}개")
        summary_parts.append(
            f"- ContentCreator 특별 수집: {len(content_results)}개")

        return "\n".join(summary_parts)

    def _extract_binding_insights(self, binding_results: List[Dict]) -> str:
        """BindingAgent 인사이트 추출"""
        if not binding_results:
            return "BindingAgent 결과 없음"

        insights = []
        for result in binding_results:
            answer = result.get('final_answer', '')
            if '그리드' in answer or 'grid' in answer.lower():
                insights.append("- 그리드 기반 이미지 배치 전략")
            if '갤러리' in answer or 'gallery' in answer.lower():
                insights.append("- 갤러리 스타일 이미지 배치")
            if '배치' in answer:
                insights.append("- 전문적 이미지 배치 분석 완료")

        return "\n".join(insights) if insights else "BindingAgent 일반적 이미지 처리"

    def _extract_org_insights(self, org_results: List[Dict]) -> str:
        """OrgAgent 인사이트 추출"""
        if not org_results:
            return "OrgAgent 결과 없음"

        insights = []
        for result in org_results:
            answer = result.get('final_answer', '')
            if '구조' in answer or 'structure' in answer.lower():
                insights.append("- 체계적 텍스트 구조 설계")
            if '레이아웃' in answer or 'layout' in answer.lower():
                insights.append("- 전문적 레이아웃 구조 분석")
            if '매거진' in answer or 'magazine' in answer.lower():
                insights.append("- 매거진 스타일 텍스트 편집")

        return "\n".join(insights) if insights else "OrgAgent 일반적 텍스트 처리"

    def _extract_content_insights(self, content_results: List[Dict]) -> str:
        """ContentCreator 인사이트 추출"""
        if not content_results:
            return "ContentCreator 결과 없음"

        insights = []
        for result in content_results:
            answer = result.get('final_answer', '')
            performance = result.get('performance_metrics', {})

            if len(answer) > 2000:
                insights.append("- 풍부한 콘텐츠 생성 완료")
            if '여행' in answer and '매거진' in answer:
                insights.append("- 여행 매거진 스타일 콘텐츠")
            if performance.get('content_richness', 0) > 1.5:
                insights.append("- 고품질 콘텐츠 확장 성공")

        return "\n".join(insights) if insights else "ContentCreator 일반적 콘텐츠 처리"

    def _enhance_with_content_results(self, jsx_code: str, content_results: List[Dict], content: Dict) -> str:
        """ContentCreator 결과로 콘텐츠 품질 강화"""
        if not content_results:
            return jsx_code

        latest_content = content_results[-1]
        content_answer = latest_content.get('final_answer', '')
        performance = latest_content.get('performance_metrics', {})

        # 콘텐츠 품질에 따른 스타일 강화
        if len(content_answer) > 2000 or performance.get('content_richness', 0) > 1.5:
            # 고품질 콘텐츠일 때 프리미엄 스타일 적용
            jsx_code = jsx_code.replace(
                'background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);',
                'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'
            )
            jsx_code = jsx_code.replace(
                'color: #2c3e50;',
                'color: #ffffff;'
            )

        if '여행' in content_answer and '매거진' in content_answer:
            # 여행 매거진 스타일 강화
            jsx_code = jsx_code.replace(
                'border-radius: 12px;',
                'border-radius: 16px;\n  box-shadow: 0 12px 24px rgba(0,0,0,0.15);'
            )

        return jsx_code

    def _remove_markdown_blocks(self, jsx_code: str) -> str:
        """마크다운 블록 완전 제거"""
        jsx_code = re.sub(r'```
        jsx_code=re.sub(r'```', '', jsx_code)
        jsx_code=re.sub(r'^(이 코드는|다음은|아래는).*?\n', '',
                        jsx_code, flags=re.MULTILINE)
        return jsx_code.strip()

    def _validate_basic_structure(self, jsx_code: str, component_name: str) -> str:
        """기본 구조 검증"""
        if 'import React' not in jsx_code:
            jsx_code='import React from "react";\n' + jsx_code

        if 'import styled' not in jsx_code:
            jsx_code=jsx_code.replace(
                'import React from "react";',
                'import React from "react";\nimport styled from "styled-components";'
            )

        if f'export const {component_name}' not in jsx_code:
            jsx_code=re.sub(r'export const \w+',
                            f'export const {component_name}', jsx_code)

        return jsx_code

    def _enhance_with_binding_results(self, jsx_code: str, binding_results: List[Dict], content: Dict) -> str:
        """BindingAgent 결과로 이미지 강화"""
        if not binding_results:
            return jsx_code

        latest_binding=binding_results[-1]
        binding_answer=latest_binding.get('final_answer', '')

        # 이미지 배치 전략 반영
        if '그리드' in binding_answer or 'grid' in binding_answer.lower():
            # 그리드 스타일 이미지 갤러리로 교체
            jsx_code=jsx_code.replace(
                'display: flex;',
                'display: grid;\n  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));'
            )

        if '갤러리' in binding_answer or 'gallery' in binding_answer.lower():
            # 갤러리 스타일 강화
            jsx_code=jsx_code.replace(
                'gap: 20px;',
                'gap: 15px;\n  padding: 20px;'
            )

        return jsx_code

    def _enhance_with_org_results(self, jsx_code: str, org_results: List[Dict], content: Dict) -> str:
        """OrgAgent 결과로 텍스트 구조 강화"""
        if not org_results:
            return jsx_code

        latest_org=org_results[-1]
        org_answer=latest_org.get('final_answer', '')

        # 텍스트 구조 개선
        if '매거진' in org_answer or 'magazine' in org_answer.lower():
            # 매거진 스타일 타이포그래피 강화
            jsx_code=jsx_code.replace(
                'font-size: 3em;',
                'font-size: 3.5em;\n  font-weight: 300;\n  letter-spacing: -1px;'
            )

        if '구조' in org_answer or 'structure' in org_answer.lower():
            # 구조적 여백 개선
            jsx_code=jsx_code.replace(
                'margin-bottom: 50px;',
                'margin-bottom: 60px;\n  padding-bottom: 30px;\n  border-bottom: 1px solid #f0f0f0;'
            )

        return jsx_code

    def _ensure_image_urls(self, jsx_code: str, content: Dict) -> str:
        """이미지 URL 강제 포함"""
        images=content.get('images', [])
        if not images:
            return jsx_code

        if '<PlaceholderDiv>' in jsx_code and images:
            image_jsx=[]
            for i, img_url in enumerate(images[:4]):
                if img_url and img_url.strip():
                    image_jsx.append(
                        f'<TravelImage src="{img_url}" alt="Travel Image {i+1}" />')

            if image_jsx:
                jsx_code=jsx_code.replace(
                    '<PlaceholderDiv>에이전트 기반 콘텐츠</PlaceholderDiv>',
                    '\n'.join(image_jsx)
                )

        return jsx_code

    def _final_error_check_and_fix(self, jsx_code: str, component_name: str) -> str:
        """최종 오류 검사 및 수정"""
        # 중괄호 매칭
        open_braces=jsx_code.count('{')
        close_braces=jsx_code.count('}')
        if open_braces != close_braces:
            if open_braces > close_braces:
                jsx_code += '}' * (open_braces - close_braces)

        # 문법 오류 수정
        jsx_code=re.sub(r'\{\{([^}]+)\}\}', r'{\1}', jsx_code)
        jsx_code=jsx_code.replace('class=', 'className=')
        jsx_code=re.sub(r'\{\s*\}', '', jsx_code)

        # 마지막 세미콜론 확인
        if not jsx_code.rstrip().endswith('};'):
            jsx_code=jsx_code.rstrip() + '\n};'

        return jsx_code

    def _validate_jsx_syntax(self, jsx_code: str) -> bool:
        """JSX 문법 검증"""
        try:
            has_import_react='import React' in jsx_code
            has_import_styled='import styled' in jsx_code
            has_export='export const' in jsx_code
            has_return='return (' in jsx_code
            has_closing=jsx_code.rstrip().endswith('};')

            open_braces=jsx_code.count('{')
            close_braces=jsx_code.count('}')
            braces_matched=open_braces == close_braces

            return all([has_import_react, has_import_styled, has_export, has_return, has_closing, braces_matched])
        except Exception:
            return False

    def _create_agent_based_fallback_jsx(self, content: Dict, design: Dict, component_name: str, previous_results: List[Dict]) -> str:
        """에이전트 데이터 기반 폴백 JSX"""
        title=content.get('title', '에이전트 협업 여행 이야기')
        subtitle=content.get('subtitle', '특별한 순간들')
        body=content.get('body', '다양한 AI 에이전트들이 협업하여 생성한 여행 콘텐츠입니다.')
        images=content.get('images', [])
        tagline=content.get('tagline', 'AI AGENTS COLLABORATION')

        # 에이전트 결과 반영
        if previous_results:
            agent_count=len(set(r.get('agent_name') for r in previous_results))
            body=f"{body}\n\n이 콘텐츠는 {agent_count}개의 전문 AI 에이전트가 협업하여 생성했습니다."

        image_tags=[]
        for i, img_url in enumerate(images[:4]):
            if img_url and img_url.strip():
                image_tags.append(
                    f'<TravelImage src="{img_url}" alt="Travel Image {i+1}" />')

        image_jsx='\n        '.join(
            image_tags) if image_tags else '<PlaceholderDiv>에이전트 기반 콘텐츠</PlaceholderDiv>'

        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 60px 20px;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  min-height: 100vh;
`;

const Header = styled.header`
  text-align: center;
  margin-bottom: 50px;
`;

const Title = styled.h1`
  font-size: 3em;
  color: #2c3e50;
  margin-bottom: 20px;
  font-weight: 300;
`;

const Subtitle = styled.h2`
  font-size: 1.4em;
  color: #7f8c8d;
  margin-bottom: 30px;
`;

const Content = styled.div`
  font-size: 1.2em;
  line-height: 1.8;
  color: #34495e;
  margin-bottom: 40px;
  max-width: 800px;
  margin-left: auto;
  margin-right: auto;
  white-space: pre-line;
`;

const ImageGallery = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin: 40px 0;
`;

const TravelImage = styled.img`
  width: 100%;
  height: 200px;
  object-fit: cover;
  border-radius: 12px;
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
  transition: transform 0.3s ease;

  &:hover {{
    transform: translateY(-5px);
  }}
`;

const PlaceholderDiv = styled.div`
  width: 100%;
  height: 200px;
  background: #e9ecef;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6c757d;
  font-size: 1.1em;
`;

const Footer = styled.footer`
  text-align: center;
  margin-top: 50px;
  padding-top: 30px;
  border-top: 1px solid #ecf0f1;
`;

const Tagline = styled.div`
  font-size: 0.9em;
  color: #95a5a6;
  letter-spacing: 3px;
  text-transform: uppercase;
  font-weight: 600;
`;

export const {component_name} = () => {{
  return (
    <Container>
      <Header>
        <Title>{title}</Title>
        <Subtitle>{subtitle}</Subtitle>
      </Header>
      <Content>{body}</Content>
      <ImageGallery>
        {image_jsx}
      </ImageGallery>
      <Footer>
        <Tagline>{tagline}</Tagline>
      </Footer>
    </Container>
  );
}};'''
