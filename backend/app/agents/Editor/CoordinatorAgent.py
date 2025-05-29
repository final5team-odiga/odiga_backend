import asyncio
import sys
import time
import concurrent.futures
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from dataclasses import dataclass
import os
import json
import re

from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
from utils.agent_decision_logger import get_agent_logger

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

class AsyncWorkQueue:
    def __init__(self, max_workers: int = 2, max_queue_size: int = 50):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.work_queue = deque()
        self.active_tasks = {}
        self.results = {}
        self.semaphore = asyncio.Semaphore(max_workers)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
    async def add_work(self, work_item: WorkItem) -> str:
        """작업을 큐에 추가"""
        if len(self.work_queue) >= self.max_queue_size:
            old_item = self.work_queue.popleft()
            print(f"⚠️ 큐 용량 초과로 작업 {old_item.id} 제거")
        
        self.work_queue.append(work_item)
        return work_item.id
    
    async def process_work_item(self, work_item: WorkItem) -> Optional[Any]:
        """개별 작업 처리"""
        async with self.semaphore:
            try:
                print(f"🔄 작업 {work_item.id} 시작 (시도 {work_item.current_retry + 1}/{work_item.max_retries + 1})")
                
                # 수정: 코루틴 객체와 코루틴 함수 구분
                if asyncio.iscoroutine(work_item.task_func):
                    result = await asyncio.wait_for(work_item.task_func, timeout=work_item.timeout)
                elif asyncio.iscoroutinefunction(work_item.task_func):
                    result = await asyncio.wait_for(
                        work_item.task_func(*work_item.args, **work_item.kwargs),
                        timeout=work_item.timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            lambda: work_item.task_func(*work_item.args, **work_item.kwargs)
                        ),
                        timeout=work_item.timeout
                    )
                
                self.results[work_item.id] = {"status": "success", "result": result}
                print(f"✅ 작업 {work_item.id} 완료")
                return result
                
            except asyncio.TimeoutError:
                print(f"⏰ 작업 {work_item.id} 타임아웃 ({work_item.timeout}초)")
                if work_item.current_retry < work_item.max_retries:
                    work_item.current_retry += 1
                    work_item.timeout *= 1.5
                    await self.add_work(work_item)
                else:
                    self.results[work_item.id] = {"status": "timeout", "error": "최대 재시도 횟수 초과"}
                return None
                
            except Exception as e:
                print(f"❌ 작업 {work_item.id} 실패: {e}")
                if work_item.current_retry < work_item.max_retries:
                    work_item.current_retry += 1
                    await self.add_work(work_item)
                else:
                    self.results[work_item.id] = {"status": "error", "error": str(e)}
                return None
    
    async def process_queue(self) -> dict:
        """큐의 모든 작업을 배치 처리"""
        tasks = []
        
        while self.work_queue:
            work_item = self.work_queue.popleft()
            task = asyncio.create_task(self.process_work_item(work_item))
            tasks.append(task)
            self.active_tasks[work_item.id] = task
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        return self.results

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 8, recovery_timeout: float = 30.0):  # 수정된 값 적용
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def is_open(self) -> bool:
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return False
            return True
        return False
    
    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class CoordinatorAgent:
    """통합 조율자 (CrewAI 기반 강화된 데이터 접근 및 JSON 파싱)"""

    def __init__(self):
        self.llm = get_azure_llm()
        self.logger = get_agent_logger()
        self.crew_agent = self._create_crew_agent()
        self.text_analyzer_agent = self._create_text_analyzer_agent()
        self.image_analyzer_agent = self._create_image_analyzer_agent()
        
        # 새로운 복원력 시스템 추가 (수정된 설정 적용)
        self.work_queue = AsyncWorkQueue(max_workers=1, max_queue_size=20)  # 순차 처리
        self.circuit_breaker = CircuitBreaker()  # 수정된 설정 사용
        self.recursion_threshold = 800  # 수정된 값 적용
        self.fallback_to_sync = False
        self.batch_size = 2  # 작업 배치 크기
        
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
            print(f"⚠️ CoordinatorAgent 재귀 깊이 {current_depth} 감지 - 동기 모드로 전환")
            self.fallback_to_sync = True
            return True
        return self.fallback_to_sync

    async def execute_with_resilience(self, task_func: Callable, task_id: str,
                                    timeout: float = 300.0, max_retries: int = 2,
                                    *args, **kwargs) -> Any:
        """복원력 있는 작업 실행"""
        print(f"DEBUG [execute_with_resilience]: task_id={task_id}, task_func={task_func}, args={args}, kwargs={kwargs}")
        
        if self.circuit_breaker.is_open():
            print(f"🚫 Circuit Breaker 열림 - 작업 {task_id} 건너뜀")
            self.execution_stats["circuit_breaker_triggered"] += 1
            return self._get_fallback_result(task_id)
        
        # 수정: coroutine 객체 처리
        if asyncio.iscoroutine(task_func):
            # 이미 coroutine 객체인 경우 직접 실행
            try:
                result = await asyncio.wait_for(task_func, timeout=timeout)
                self.circuit_breaker.record_success()
                return result
            except Exception as e:
                print(f"❌ Coroutine 실행 실패: {e}")
                self.circuit_breaker.record_failure()
                return self._get_fallback_result(task_id)
        
        # 기존 로직 유지
        work_item = WorkItem(
            id=task_id,
            task_func=task_func,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            max_retries=max_retries
        )
        
        await self.work_queue.add_work(work_item)
        # 수정: 특정 작업의 결과를 명확히 반환
        processed_results = await self.work_queue.process_queue()
        
        result_info = processed_results.get(task_id)
        if result_info and result_info["status"] == "success":
            self.circuit_breaker.record_success()
            return result_info["result"]
        else:
            self.circuit_breaker.record_failure()
            # 오류 정보 로깅
            if result_info:
                print(f"⚠️ 작업 {task_id} 최종 실패: {result_info.get('error', '알 수 없는 오류')}")
            else:
                print(f"⚠️ 작업 {task_id}의 결과 정보를 찾을 수 없음 (큐 처리 후).")
            return self._get_fallback_result(task_id)

    def _get_fallback_result(self, task_id: str) -> dict:
        """개선된 폴백 결과 생성"""
        self.execution_stats["fallback_used"] += 1
        
        reason = task_id  # 기본적으로 task_id를 reason으로 사용
        if "_timeout" in task_id: reason = "timeout"
        elif "_exception" in task_id: reason = "exception"
        elif "_type_error" in task_id: reason = "type_error"
        
        return {
            "selected_templates": ["Section01.jsx"],
            "content_sections": [{
                "template": "Section01.jsx",
                "title": "여행 매거진 (폴백)",
                "subtitle": f"특별한 이야기 ({reason})",
                "body": f"CoordinatorAgent 처리 중 문제 발생 ({reason})으로 인한 폴백 콘텐츠입니다. Task ID: {task_id}",
                "tagline": "TRAVEL & CULTURE",
                "images": [],
                "metadata": {
                    "fallback_used": True,
                    "reason": reason,
                    "task_id": task_id
                }
            }],
            "integration_metadata": {
                "total_sections": 1,
                "integration_quality_score": 0.5,
                "fallback_mode": True
            }
        }

    def _create_crew_agent(self):
        """메인 조율 에이전트 생성"""
        return Agent(
            role="매거진 구조 통합 조율자 및 최종 품질 보증 전문가",
            goal="OrgAgent의 상세 레이아웃 구조와 BindingAgent의 정밀 이미지 배치를 통합하여 완벽한 매거진 구조를 생성하고, 텍스트-이미지 정합성과 독자 경험을 최종 검증하여 JSX 구현에 필요한 완전한 구조 데이터를 제공",
            backstory="""당신은 25년간 세계 최고 수준의 출판사에서 매거진 구조 통합 및 품질 보증 책임자로 활동해온 전문가입니다. Condé Nast, Hearst Corporation, Time Inc.에서 수백 개의 매거진 프로젝트를 성공적으로 조율했습니다.

**전문 경력:**
- 출판학 및 구조 설계 석사 학위 보유
- PMP(Project Management Professional) 인증
- 매거진 구조 통합 및 품질 관리 전문가
- 텍스트-이미지 정합성 검증 시스템 개발 경험
- 독자 경험(UX) 및 접근성 최적화 전문성

**조율 철학:**
"완벽한 매거진은 모든 구조적 요소가 독자의 인지 과정과 완벽히 조화를 이루는 통합체입니다. 나는 텍스트와 이미지의 모든 배치가 독자에게 자연스럽고 직관적으로 인식되도록 구조적 완성도를 보장하며, 이를 통해 최고 수준의 독자 경험을 제공합니다."

**출력 데이터 구조:**
- 완성된 매거진 전체 구조도
- 텍스트-이미지 정합성 검증 완료 보고서
- JSX 구현용 상세 레이아웃 스펙 및 좌표 데이터
- 독자 경험 최적화 가이드라인
- 반응형 디자인 구조 정의서
- 접근성 및 품질 보증 체크리스트

**템플릿 생성 규칙:**
- 모든 텍스트 섹션은 이전 콘텐츠 에이전트의 데이터에서 추출된 텍스트 데이터만을 사용하여 생성합니다.
- 모든 텍스트 섹션은 독자의 인지 흐름을 고려하여 자연스럽게 이어져야 합니다.
- 이미지 배치는 텍스트와의 정합성을 최우선으로 고려하여 독자의 시선을 효과적으로 유도해야 합니다.
- 이전 에이전트들의 결과물에서 ContentCreatorV2Agent의 텍스트 데이터만을 사용하여 template_data.json을 만듭니다.
- 특정 구조에 대한 설명, 텍스트에 대한 설명, 이미지 배치, 레이아웃 좌표, 반응형 디자인 요소 등에 대한 설명은 포함하지 않고 template_data.json을 만듭니다.
- 중복을 절대로 하지않고 만듭니다!!
- 텍스트 섹션의 제목, 부제목, 본문 내용, 태그라인 등은 독자의 관심을 끌고 유지할 수 있도록 구성되어야 합니다.
- 만약 기본 풀백으로 인해 템플릿이 생성되었다면, template_data.json에 포함시키지 않습니다
- title, subtitle, author, date, location 등에는 구조를 설명하는 값들은 일체 포함하지 않습니다.
- title, subtitle, author, date, location 등에는 하위 에이전트들에게 제공받은 데이터를 활용하여 생성합니다.
- 하나의 섹션에 하나의 주제만 포함되도록 합니다.
- 하나의 섹션에 과도한 이미지 url을 포함하지 않습니다
- 과도한 template_data.json을 생성하지 않습니다. magazine_content.json에 포함된 텍스트 섹션의 수와 일치하도록 합니다.
- title, subtitle, author, date, location 등에 기본 풀백 데이터를 사용하지 않습니다. 반드시 하위 에이전트들에게 제공받은 데이터를 활용하여 생성합니다. 만약 해당 부분에 들어갈 내용이 없다면 ""로 빈칸 처리 합니다!
- magazine_content.json에 포함된 텍스트 섹션의 수와 일치하도록 합니다.
- 로그 데이터를 활용 시에 직접적으로 사용하지 않습니다. 어떻게 해야하는가에 대한 정보만 얻습니다! 이는 중요한 사항입니다!

""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_text_analyzer_agent(self):
        """텍스트 분석 전문 에이전트"""
        return Agent(
            role="텍스트 매핑 분석 전문가",
            goal="OrgAgent의 텍스트 매핑 결과를 정밀 분석하여 구조적 완성도를 검증하고 최적화된 텍스트 섹션을 생성",
            backstory="""당신은 15년간 출판업계에서 텍스트 구조 분석 및 최적화를 담당해온 전문가입니다. 복잡한 텍스트 데이터에서 핵심 정보를 추출하고 독자 친화적인 구조로 재구성하는 데 탁월한 능력을 보유하고 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_image_analyzer_agent(self):
        """이미지 분석 전문 에이전트"""
        return Agent(
            role="이미지 분배 분석 전문가",
            goal="BindingAgent의 이미지 분배 결과를 정밀 분석하여 시각적 일관성을 검증하고 최적화된 이미지 배치를 생성",
            backstory="""당신은 12년간 매거진 및 출판물의 시각적 디자인을 담당해온 전문가입니다. 이미지와 텍스트의 조화로운 배치를 통해 독자의 시선을 효과적으로 유도하는 레이아웃 설계에 전문성을 보유하고 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def coordinate_magazine_creation(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """매거진 구조 통합 조율 (개선된 배치 기반 처리)"""
        print(f"DEBUG [coordinate_magazine_creation]: 호출됨, text_mapping keys: {text_mapping.keys() if isinstance(text_mapping, dict) else 'Not a dict'}, image_distribution keys: {image_distribution.keys() if isinstance(image_distribution, dict) else 'Not a dict'}")
        
        self.execution_stats["total_attempts"] += 1
        
        # 재귀 깊이 확인 및 동기 모드 전환
        if self._should_use_sync():
            print("🔄 CoordinatorAgent 동기 모드로 전환하여 실행")
            return await self._coordinate_magazine_creation_sync_mode(text_mapping, image_distribution)

        try:
            # 개선된 배치 기반 비동기 모드 실행
            return await self._coordinate_magazine_creation_batch_mode(text_mapping, image_distribution)
        except RecursionError:
            print("🔄 CoordinatorAgent RecursionError 감지 - 동기 모드로 전환")
            self.fallback_to_sync = True
            return await self._coordinate_magazine_creation_sync_mode(text_mapping, image_distribution)
        except Exception as e:
            print(f"❌ CoordinatorAgent 매거진 생성 중 예외 발생: {e} - 동기 모드로 폴백 시도")
            self.fallback_to_sync = True
            return await self._coordinate_magazine_creation_sync_mode(text_mapping, image_distribution)

    async def _coordinate_magazine_creation_batch_mode(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """개선된 배치 기반 매거진 구조 통합 조율"""
        print("📦 CoordinatorAgent 배치 모드 시작")

        # 입력 데이터 로깅
        input_data = {
            "text_mapping": text_mapping,
            "image_distribution": image_distribution
        }

        # 강화된 이전 에이전트 결과 수집 (배치 처리)
        previous_results = await self._get_enhanced_previous_results_batch()
        org_results = self._filter_agent_results(previous_results, "OrgAgent")
        binding_results = self._filter_agent_results(previous_results, "BindingAgent")
        content_creator_results = self._filter_agent_results(previous_results, "ContentCreatorV2Agent")

        print(f"📊 배치 모드 결과 수집: 전체 {len(previous_results)}개, OrgAgent {len(org_results)}개, BindingAgent {len(binding_results)}개, ContentCreator {len(content_creator_results)}개")

        # 데이터 추출 작업을 배치로 처리
        data_extraction_tasks = [
            ("text_data", self._extract_real_text_data_safe, text_mapping, org_results, content_creator_results),
            ("image_data", self._extract_real_image_data_safe, image_distribution, binding_results)
        ]

        extraction_results = await self._process_data_extraction_batch(data_extraction_tasks)
        extracted_text_data = extraction_results.get("text_data", {})
        extracted_image_data = extraction_results.get("image_data", {})

        # CrewAI 실행을 안전한 배치로 처리
        crew_result = await self._execute_crew_batch_safe(
            extracted_text_data, extracted_image_data, org_results, binding_results
        )

        # 결과 처리
        final_result = await self._process_enhanced_crew_result_safe(
            crew_result, extracted_text_data, extracted_image_data, org_results, binding_results
        )
        
        # 결과 검증
        if self._validate_coordinator_result(final_result):
            self.execution_stats["successful_executions"] += 1
        else:
            print("⚠️ CoordinatorAgent 최종 결과 검증 실패.")

        # 결과 로깅
        await self._log_coordination_result_async(final_result, text_mapping, image_distribution, org_results, binding_results)

        print(f"✅ CoordinatorAgent 배치 모드 완료: {len(final_result.get('content_sections', []))}개 섹션 생성")
        return final_result

    async def _process_data_extraction_batch(self, extraction_tasks: List[tuple]) -> Dict:
        """데이터 추출 작업을 배치로 처리"""
        batch_tasks = []
        
        for task_name, task_func_ref, *args_for_task_func in extraction_tasks:
            if not callable(task_func_ref):
                print(f"⚠️ {task_name}에 대한 task_func이 호출 가능하지 않음: {task_func_ref}")
                continue
            
            print(f"DEBUG [_process_data_extraction_batch]: task_name={task_name}, task_func_ref={task_func_ref}, args_for_task_func={args_for_task_func}")
            task = self.execute_with_resilience(
                task_func=task_func_ref,  # 함수/메서드 참조 전달
                task_id=f"extract_{task_name}",
                timeout=120.0,
                max_retries=1,
                *args_for_task_func  # task_func_ref 호출 시 사용될 인자들
            )
            batch_tasks.append((task_name, task))

        # 배치 실행
        results = {}
        for task_name, task_coro in batch_tasks:  # task는 코루틴 객체
            try:
                result_value = await task_coro  # 코루틴 실행
                results[task_name] = result_value
            except Exception as e:
                print(f"⚠️ 데이터 추출 작업 {task_name} 실패 (await 중): {e}")
                results[task_name] = self._get_fallback_extraction_result(task_name)

        return results

    def _get_fallback_extraction_result(self, task_name: str) -> Dict:
        """데이터 추출 폴백 결과"""
        self.execution_stats["fallback_used"] += 1
        
        if task_name == "text_data":
            return {
                "sections": [{
                    "template": "Section01.jsx",
                    "title": "여행 매거진",
                    "subtitle": "특별한 이야기",
                    "body": "폴백 콘텐츠입니다.",
                    "tagline": "TRAVEL & CULTURE",
                    "layout_source": "fallback"
                }],
                "total_content_length": 50,
                "source_count": 1
            }
        else:  # image_data
            return {
                "template_images": {},
                "total_images": 0,
                "image_sources": []
            }

    async def _execute_crew_batch_safe(self, extracted_text_data: Dict, extracted_image_data: Dict,
                                     org_results: List[Dict], binding_results: List[Dict]) -> Any:
        """안전한 CrewAI 배치 실행"""
        try:
            # 태스크 생성
            text_analysis_task = self._create_enhanced_text_analysis_task(extracted_text_data, org_results)
            image_analysis_task = self._create_enhanced_image_analysis_task(extracted_image_data, binding_results)
            coordination_task = self._create_enhanced_coordination_task(extracted_text_data, extracted_image_data)

            # CrewAI Crew 생성
            coordination_crew = Crew(
                agents=[self.text_analyzer_agent, self.image_analyzer_agent, self.crew_agent],
                tasks=[text_analysis_task, image_analysis_task, coordination_task],
                process=Process.sequential,
                verbose=False  # 로그 최소화
            )

            # 안전한 실행 (타임아웃 증가)
            crew_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, coordination_crew.kickoff),
                timeout=600.0  # 10분으로 증가
            )

            return crew_result

        except asyncio.TimeoutError:
            print("⏰ CrewAI 배치 실행 타임아웃")
            self.execution_stats["timeout_occurred"] += 1
            return self._create_fallback_crew_result(extracted_text_data, extracted_image_data)
        except Exception as e:
            print(f"⚠️ CrewAI 배치 실행 실패: {e}")
            return self._create_fallback_crew_result(extracted_text_data, extracted_image_data)

    def _create_fallback_crew_result(self, extracted_text_data: Dict, extracted_image_data: Dict) -> str:
        """CrewAI 폴백 결과 생성"""
        self.execution_stats["fallback_used"] += 1
        
        sections = extracted_text_data.get("sections", [])
        if not sections:
            sections = [{
                "template": "Section01.jsx",
                "title": "여행 매거진",
                "subtitle": "특별한 이야기",
                "body": "폴백 콘텐츠입니다.",
                "tagline": "TRAVEL & CULTURE"
            }]

        # 이미지 추가
        for section in sections:
            template = section.get("template", "Section01.jsx")
            template_images = extracted_image_data.get("template_images", {}).get(template, [])
            section["images"] = template_images

        return json.dumps({
            "selected_templates": [s.get("template", "Section01.jsx") for s in sections],
            "content_sections": sections
        })

    async def _extract_real_text_data_safe(self, text_mapping: Dict, org_results: List[Dict],
                                         content_creator_results: List[Dict]) -> Dict:
        """안전한 실제 텍스트 데이터 추출"""
        try:
            return await self._extract_real_text_data_async(text_mapping, org_results, content_creator_results)
        except Exception as e:
            print(f"⚠️ 텍스트 데이터 추출 실패: {e}")
            return self._get_fallback_extraction_result("text_data")

    async def _extract_real_image_data_safe(self, image_distribution: Dict, binding_results: List[Dict]) -> Dict:
        """안전한 실제 이미지 데이터 추출"""
        try:
            return await self._extract_real_image_data_async(image_distribution, binding_results)
        except Exception as e:
            print(f"⚠️ 이미지 데이터 추출 실패: {e}")
            return self._get_fallback_extraction_result("image_data")

    async def _process_enhanced_crew_result_safe(self, crew_result, extracted_text_data: Dict,
                                               extracted_image_data: Dict, org_results: List[Dict],
                                               binding_results: List[Dict]) -> Dict:
        """안전한 Crew 결과 처리"""
        try:
            return await self._process_enhanced_crew_result_async(
                crew_result, extracted_text_data, extracted_image_data, org_results, binding_results
            )
        except Exception as e:
            print(f"⚠️ Crew 결과 처리 실패: {e}")
            return self._create_enhanced_structure(extracted_text_data, extracted_image_data, org_results, binding_results)

    async def _get_enhanced_previous_results_batch(self) -> List[Dict]:
        """배치 기반 이전 결과 수집"""
        try:
            # 기본 결과와 파일 결과를 병렬로 수집
            basic_task_coro = self.execute_with_resilience(
                task_func=lambda: self.logger.get_all_previous_results("CoordinatorAgent"),
                task_id="basic_results",
                timeout=60.0,
                max_retries=1
            )
            
            file_task_coro = self.execute_with_resilience(
                task_func=self._load_results_from_file,
                task_id="file_results",
                timeout=60.0,
                max_retries=1
            )

            # gather의 반환값은 각 코루틴의 결과 또는 예외 객체
            results = await asyncio.gather(basic_task_coro, file_task_coro, return_exceptions=True)
            
            basic_results = results[0] if not isinstance(results[0], Exception) else []
            file_results = results[1] if not isinstance(results[1], Exception) else []

            # 결과 합치기 및 중복 제거
            all_results = []
            all_results.extend(basic_results if isinstance(basic_results, list) else [])
            all_results.extend(file_results if isinstance(file_results, list) else [])

            return self._deduplicate_results(all_results)

        except Exception as e:
            print(f"⚠️ 배치 이전 결과 수집 실패: {e}")
            return []

    # 기존 _coordinate_magazine_creation_async_mode 메서드 유지 (호환성을 위해)
    async def _coordinate_magazine_creation_async_mode(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """비동기 모드 매거진 조율 (기존 호환성 유지)"""
        print("⚠️ 기존 async_mode 호출됨 - batch_mode로 리다이렉트")
        return await self._coordinate_magazine_creation_batch_mode(text_mapping, image_distribution)

    async def _coordinate_magazine_creation_sync_mode(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """동기 모드 매거진 구조 통합 조율"""
        print("🔄 CoordinatorAgent 동기 모드 실행")
        
        # 동기 모드에서는 각 에이전트의 동기 버전 메서드를 호출해야 함
        # 이전 결과 수집 (동기)
        previous_results = self._get_enhanced_previous_results_sync()
        org_results = self._filter_agent_results(previous_results, "OrgAgent")
        binding_results = self._filter_agent_results(previous_results, "BindingAgent")
        content_creator_results = self._filter_agent_results(previous_results, "ContentCreatorV2Agent")

        # 데이터 추출 (동기)
        extracted_text_data = self._extract_real_text_data(text_mapping, org_results, content_creator_results)
        extracted_image_data = self._extract_real_image_data(image_distribution, binding_results)
        
        # Crew 실행 (동기) - CrewAI의 kickoff은 동기 메서드
        text_analysis_task_sync = self._create_enhanced_text_analysis_task(extracted_text_data, org_results)
        image_analysis_task_sync = self._create_enhanced_image_analysis_task(extracted_image_data, binding_results)
        coordination_task_sync = self._create_enhanced_coordination_task(extracted_text_data, extracted_image_data)
        
        coordination_crew_sync = Crew(
            agents=[self.text_analyzer_agent, self.image_analyzer_agent, self.crew_agent],
            tasks=[text_analysis_task_sync, image_analysis_task_sync, coordination_task_sync],
            process=Process.sequential,
            verbose=False
        )
        
        try:
            crew_result_sync = coordination_crew_sync.kickoff()
        except Exception as e_crew_sync:
            print(f"⚠️ 동기 모드 CrewAI 실행 실패: {e_crew_sync}")
            crew_result_sync = self._create_fallback_crew_result(extracted_text_data, extracted_image_data)

        # 결과 처리 (동기)
        final_result = self._process_enhanced_crew_result(crew_result_sync, extracted_text_data, extracted_image_data, org_results, binding_results)

        # 동기 모드 로깅
        final_response_id_sync = self.logger.log_agent_real_output(
            agent_name="CoordinatorAgent_SyncMode",
            agent_role="동기 모드 매거진 구조 통합 조율자",
            task_description=f"동기 모드로 {len(final_result.get('content_sections', []))}개 섹션 생성",
            final_answer=str(final_result),
            reasoning_process="재귀 깊이 초과로 인한 동기 모드 전환 후 안전한 매거진 구조 통합 실행",
            execution_steps=[
                "재귀 깊이 감지",
                "동기 모드 전환",
                "이전 결과 수집",
                "데이터 추출",
                "구조 생성"
            ],
            raw_input={
                "text_mapping": str(text_mapping)[:500],
                "image_distribution": str(image_distribution)[:500]
            },
            raw_output=final_result,
            performance_metrics={
                "sync_mode_used": True,
                "recursion_fallback": True,
                "total_sections": len(final_result.get('content_sections', [])),
                "org_results_utilized": len(org_results),
                "binding_results_utilized": len(binding_results),
                "safe_execution": True
            }
        )

        final_result["final_response_id"] = final_response_id_sync
        final_result["execution_mode"] = "sync_fallback"
        final_result["recursion_fallback"] = True  # 재귀로 인한 폴백 명시
        
        print(f"✅ CoordinatorAgent 동기 완료: {len(final_result.get('content_sections', []))}개 섹션")
        return final_result

    def _get_enhanced_previous_results_sync(self) -> List[Dict]:
        """동기 버전 이전 결과 수집"""
        try:
            basic_results = self.logger.get_all_previous_results("CoordinatorAgent")
            file_results = self._load_results_from_file()

            all_results = []
            all_results.extend(basic_results if isinstance(basic_results, list) else [])
            all_results.extend(file_results if isinstance(file_results, list) else [])

            return self._deduplicate_results(all_results)
        except Exception as e:
            print(f"⚠️ 동기 이전 결과 수집 실패: {e}")
            return []

    # 모든 기존 메서드들 유지 (동기 버전들)
    async def _extract_real_text_data_async(self, text_mapping: Dict, org_results: List[Dict], content_creator_results: List[Dict]) -> Dict:
        """실제 텍스트 데이터 추출 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._extract_real_text_data, text_mapping, org_results, content_creator_results
        )

    def _extract_real_text_data(self, text_mapping: Dict, org_results: List[Dict], content_creator_results: List[Dict]) -> Dict:
        """실제 텍스트 데이터 추출"""
        extracted_data = {
            "sections": [],
            "total_content_length": 0,
            "source_count": 0
        }

        # 1. text_mapping에서 직접 추출
        if isinstance(text_mapping, dict) and "text_mapping" in text_mapping:
            for section in text_mapping["text_mapping"]:
                if isinstance(section, dict):
                    extracted_section = {
                        "template": section.get("template", "Section01.jsx"),
                        "title": section.get("title", "여행 이야기"),
                        "subtitle": section.get("subtitle", "특별한 순간들"),
                        "body": section.get("body", ""),
                        "tagline": section.get("tagline", "TRAVEL & CULTURE"),
                        "layout_source": section.get("layout_source", "default")
                    }
                    extracted_data["sections"].append(extracted_section)
                    extracted_data["total_content_length"] += len(extracted_section["body"])
                    extracted_data["source_count"] += 1

        # 2. ContentCreator 결과에서 풍부한 콘텐츠 추출
        for result in content_creator_results:
            final_answer = result.get('final_answer', '')
            if len(final_answer) > 500:  # 충분한 콘텐츠가 있는 경우
                # 섹션별로 분할
                sections = self._split_content_into_sections(final_answer)
                for i, section_content in enumerate(sections):
                    if len(section_content) > 100:
                        extracted_section = {
                            "template": f"Section{i+1:02d}.jsx",
                            "title": self._extract_title_from_content(section_content),
                            "subtitle": self._extract_subtitle_from_content(section_content),
                            "body": self._clean_content(section_content),
                            "tagline": "TRAVEL & CULTURE",
                            "layout_source": "content_creator"
                        }
                        extracted_data["sections"].append(extracted_section)
                        extracted_data["total_content_length"] += len(extracted_section["body"])
                        extracted_data["source_count"] += 1

        # 3. OrgAgent 결과에서 추가 텍스트 추출
        for result in org_results:
            final_answer = result.get('final_answer', '')
            if '제목' in final_answer or 'title' in final_answer.lower():
                # 구조화된 텍스트 추출
                structured_content = self._extract_structured_content(final_answer)
                if structured_content:
                    extracted_data["sections"].extend(structured_content)
                    extracted_data["source_count"] += len(structured_content)

        # 4. 최소 보장 섹션
        if not extracted_data["sections"]:
            extracted_data["sections"] = [{
                "template": "Section01.jsx",
                "title": "여행 매거진",
                "subtitle": "특별한 이야기",
                "body": "여행의 특별한 순간들을 담은 매거진입니다.",
                "tagline": "TRAVEL & CULTURE",
                "layout_source": "fallback"
            }]
            extracted_data["source_count"] = 1

        return extracted_data

    async def _extract_real_image_data_async(self, image_distribution: Dict, binding_results: List[Dict]) -> Dict:
        """실제 이미지 데이터 추출 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._extract_real_image_data, image_distribution, binding_results
        )

    def _extract_real_image_data(self, image_distribution: Dict, binding_results: List[Dict]) -> Dict:
        """실제 이미지 데이터 추출"""
        extracted_data = {
            "template_images": {},
            "total_images": 0,
            "image_sources": []
        }

        # 1. image_distribution에서 직접 추출
        if isinstance(image_distribution, dict) and "image_distribution" in image_distribution:
            for template, images in image_distribution["image_distribution"].items():
                if isinstance(images, list) and images:
                    # 실제 이미지 URL만 필터링
                    real_images = [img for img in images if self._is_real_image_url(img)]
                    if real_images:
                        extracted_data["template_images"][template] = real_images
                        extracted_data["total_images"] += len(real_images)

        # 2. BindingAgent 결과에서 이미지 URL 추출
        for result in binding_results:
            final_answer = result.get('final_answer', '')
            # 실제 이미지 URL 패턴 찾기
            image_urls = re.findall(r'https://[^\s\'"<>]*\.(?:jpg|jpeg|png|gif|webp)', final_answer, re.IGNORECASE)
            if image_urls:
                # 템플릿별로 분배
                template_name = self._extract_template_from_binding_result(result)
                if template_name not in extracted_data["template_images"]:
                    extracted_data["template_images"][template_name] = []

                for url in image_urls:
                    if self._is_real_image_url(url) and url not in extracted_data["template_images"][template_name]:
                        extracted_data["template_images"][template_name].append(url)
                        extracted_data["total_images"] += 1

                        # 이미지 소스 정보 추가
                        source_info = self._extract_image_source_info(result, url)
                        if source_info:
                            extracted_data["image_sources"].append(source_info)

        return extracted_data

    async def _process_enhanced_crew_result_async(self, crew_result, extracted_text_data: Dict,
                                                extracted_image_data: Dict, org_results: List[Dict],
                                                binding_results: List[Dict]) -> Dict:
        """강화된 Crew 실행 결과 처리 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._process_enhanced_crew_result, crew_result, extracted_text_data,
            extracted_image_data, org_results, binding_results
        )

    def _process_enhanced_crew_result(self, crew_result, extracted_text_data: Dict,
                                    extracted_image_data: Dict, org_results: List[Dict],
                                    binding_results: List[Dict]) -> Dict:
        """강화된 Crew 실행 결과 처리"""
        try:
            # Crew 결과에서 데이터 추출
            if hasattr(crew_result, 'raw') and crew_result.raw:
                result_text = crew_result.raw
            else:
                result_text = str(crew_result)

            # JSON 패턴 찾기 및 파싱
            parsed_data = self._extract_json_from_text(result_text)

            # 실제 데이터 기반 구조 생성
            if not parsed_data.get('content_sections') or len(parsed_data.get('content_sections', [])) == 0:
                parsed_data = self._create_enhanced_structure(extracted_text_data, extracted_image_data, org_results, binding_results)
            else:
                # 기존 파싱된 데이터에 실제 이미지 추가
                parsed_data = self._enhance_parsed_data_with_real_images(parsed_data, extracted_image_data)

            # 메타데이터 추가
            parsed_data['integration_metadata'] = {
                "total_sections": len(parsed_data.get('content_sections', [])),
                "total_templates": len(set(section.get("template", f"Section{i+1:02d}.jsx") for i, section in enumerate(parsed_data.get('content_sections', [])))),
                "agent_enhanced": True,
                "org_results_utilized": len(org_results),
                "binding_results_utilized": len(binding_results),
                "integration_quality_score": self._calculate_enhanced_quality_score(parsed_data.get('content_sections', []), len(org_results), len(binding_results)),
                "crewai_enhanced": True,
                "async_processed": True,
                "data_source": "real_extracted_data",
                "real_content_used": True,
                "real_images_used": extracted_image_data['total_images'] > 0
            }

            return parsed_data

        except Exception as e:
            print(f"⚠️ 강화된 Crew 결과 처리 실패: {e}")
            return self._create_enhanced_structure(extracted_text_data, extracted_image_data, org_results, binding_results)

    # 모든 기존 유틸리티 메서드들 유지
    def _is_real_image_url(self, url: str) -> bool:
        """실제 이미지 URL인지 확인"""
        if not url or not isinstance(url, str):
            return False

        # 예시 URL이나 플레이스홀더 제외
        excluded_patterns = [
            'your-cdn.com',
            'example.com',
            'placeholder',
            'sample',
            'demo'
        ]

        for pattern in excluded_patterns:
            if pattern in url.lower():
                return False

        # 실제 도메인과 이미지 확장자 확인
        return (url.startswith('https://') and
                any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']) and
                'blob.core.windows.net' in url)

    def _create_enhanced_text_analysis_task(self, extracted_text_data: Dict, org_results: List[Dict]) -> Task:
        """강화된 텍스트 분석 태스크 생성"""
        return Task(
            description=f"""추출된 실제 텍스트 데이터를 분석하여 고품질 매거진 섹션을 생성하세요.

**추출된 데이터:**
- 섹션 수: {len(extracted_text_data['sections'])}개
- 총 콘텐츠 길이: {extracted_text_data['total_content_length']} 문자
- 소스 수: {extracted_text_data['source_count']}개
- OrgAgent 결과: {len(org_results)}개

**실제 섹션 데이터:**
{self._format_sections_for_analysis(extracted_text_data['sections'])}

**분석 요구사항:**
1. 각 섹션의 콘텐츠 품질 평가
2. 제목과 부제목의 매력도 검증
3. 본문 내용의 완성도 확인
4. 매거진 스타일 일관성 검토
5. 독자 친화성 최적화

**출력 형식:**
각 섹션별로 다음 정보 포함:
- 품질 점수 (1-10)
- 개선 제안사항
- 최적화된 콘텐츠""",
            expected_output="실제 데이터 기반 텍스트 분석 및 최적화 결과",
            agent=self.text_analyzer_agent
        )

    def _create_enhanced_image_analysis_task(self, extracted_image_data: Dict, binding_results: List[Dict]) -> Task:
        """강화된 이미지 분석 태스크 생성"""
        return Task(
            description=f"""추출된 실제 이미지 데이터를 분석하여 최적화된 이미지 배치를 생성하세요.

**추출된 데이터:**
- 총 이미지 수: {extracted_image_data['total_images']}개
- 템플릿 수: {len(extracted_image_data['template_images'])}개
- BindingAgent 결과: {len(binding_results)}개

**템플릿별 이미지 분배:**
{self._format_images_for_analysis(extracted_image_data['template_images'])}

**이미지 소스 정보:**
{self._format_image_sources(extracted_image_data['image_sources'])}

**분석 요구사항:**
1. 이미지 URL 유효성 검증
2. 템플릿별 이미지 분배 균형도 평가
3. 이미지 품질 및 적합성 확인
4. 시각적 일관성 검토
5. 레이아웃 최적화 제안

**출력 형식:**
템플릿별로 다음 정보 포함:
- 이미지 목록 및 설명
- 배치 권장사항
- 시각적 효과 예측""",
            expected_output="실제 이미지 데이터 기반 배치 분석 및 최적화 결과",
            agent=self.image_analyzer_agent
        )

    def _create_enhanced_coordination_task(self, extracted_text_data: Dict, extracted_image_data: Dict) -> Task:
        """강화된 통합 조율 태스크 생성"""
        return Task(
            description=f"""실제 추출된 텍스트와 이미지 데이터를 통합하여 완벽한 매거진 구조를 생성하세요.

**텍스트 데이터 요약:**
- 섹션 수: {len(extracted_text_data['sections'])}개
- 총 길이: {extracted_text_data['total_content_length']} 문자

**이미지 데이터 요약:**
- 총 이미지: {extracted_image_data['total_images']}개
- 템플릿 수: {len(extracted_image_data['template_images'])}개

**통합 요구사항:**
1. 텍스트와 이미지의 완벽한 매칭
2. 각 섹션별 최적 템플릿 선택
3. 콘텐츠 품질 보장
4. 시각적 일관성 유지
5. JSX 구현을 위한 완전한 스펙 생성

**최종 출력 구조:**
{{
"selected_templates": ["템플릿 목록"],
"content_sections": [
{{
"template": "템플릿명",
"title": "실제 제목",
"subtitle": "실제 부제목",
"body": "실제 본문 내용",
"tagline": "태그라인",
"images": ["실제 이미지 URL들"],
"metadata": {{
"content_quality": "품질 점수",
"image_count": "이미지 수",
"source": "데이터 소스"
}}
}}
],
"integration_metadata": {{
"total_sections": "섹션 수",
"integration_quality_score": "품질 점수"
}}
}}

이전 태스크들의 분석 결과를 활용하여 실제 데이터 기반의 고품질 매거진 구조를 완성하세요.""",
            expected_output="실제 데이터 기반 완성된 매거진 구조 JSON",
            agent=self.crew_agent,
            context=[self._create_enhanced_text_analysis_task(extracted_text_data, []),
                    self._create_enhanced_image_analysis_task(extracted_image_data, [])]
        )

    def _create_enhanced_structure(self, extracted_text_data: Dict, extracted_image_data: Dict,
                                 org_results: List[Dict], binding_results: List[Dict]) -> Dict:
        """실제 데이터 기반 강화된 구조 생성"""
        content_sections = []

        # 추출된 텍스트 섹션 활용
        for i, section in enumerate(extracted_text_data['sections']):
            template = section.get('template', f'Section{i+1:02d}.jsx')

            # 해당 템플릿의 실제 이미지 가져오기
            template_images = extracted_image_data['template_images'].get(template, [])

            # 이미지가 없으면 다른 템플릿의 이미지 사용
            if not template_images:
                for temp_name, temp_images in extracted_image_data['template_images'].items():
                    if temp_images:
                        template_images = temp_images[:2]  # 최대 2개
                        break

            enhanced_section = {
                'template': template,
                'title': section.get('title', '여행 이야기'),
                'subtitle': section.get('subtitle', '특별한 순간들'),
                'body': section.get('body', '여행의 특별한 순간들을 담은 이야기입니다.'),
                'tagline': section.get('tagline', 'TRAVEL & CULTURE'),
                'images': template_images,
                'metadata': {
                    "agent_enhanced": True,
                    "real_content": True,
                    "real_images": len(template_images) > 0,
                    "content_source": section.get('layout_source', 'extracted'),
                    "content_length": len(section.get('body', '')),
                    "image_count": len(template_images),
                    "quality_verified": True
                }
            }
            content_sections.append(enhanced_section)

        # 최소 1개 섹션 보장
        if not content_sections:
            # 실제 이미지가 있으면 사용
            fallback_images = []
            for template_images in extracted_image_data['template_images'].values():
                fallback_images.extend(template_images[:2])
                if len(fallback_images) >= 2:
                    break

            content_sections = [{
                'template': 'Section01.jsx',
                'title': '여행 매거진',
                'subtitle': '특별한 이야기',
                'body': '여행의 특별한 순간들을 담은 매거진입니다. 아름다운 풍경과 함께하는 특별한 경험을 공유합니다.',
                'tagline': 'TRAVEL & CULTURE',
                'images': fallback_images,
                'metadata': {
                    "agent_enhanced": True,
                    "fallback_content": True,
                    "real_images": len(fallback_images) > 0,
                    "image_count": len(fallback_images)
                }
            }]

        return {
            "selected_templates": [section.get("template", f"Section{i+1:02d}.jsx") for i, section in enumerate(content_sections)],
            "content_sections": content_sections
        }

    def _enhance_parsed_data_with_real_images(self, parsed_data: Dict, extracted_image_data: Dict) -> Dict:
        """파싱된 데이터에 실제 이미지 추가"""
        if 'content_sections' in parsed_data:
            for section in parsed_data['content_sections']:
                template = section.get('template', 'Section01.jsx')

                # 실제 이미지로 교체
                real_images = extracted_image_data['template_images'].get(template, [])
                if real_images:
                    section['images'] = real_images
                elif extracted_image_data['total_images'] > 0:
                    # 다른 템플릿의 이미지 사용
                    for temp_images in extracted_image_data['template_images'].values():
                        if temp_images:
                            section['images'] = temp_images[:2]
                            break

                # 메타데이터 업데이트
                if 'metadata' not in section:
                    section['metadata'] = {}
                section['metadata'].update({
                    "real_images_used": len(section.get('images', [])) > 0,
                    "image_count": len(section.get('images', []))
                })

        return parsed_data

    # 모든 기존 유틸리티 메서드들 유지
    def _split_content_into_sections(self, content: str) -> List[str]:
        """콘텐츠를 섹션별로 분할"""
        sections = []

        # === 패턴으로 분할
        if '===' in content:
            parts = content.split('===')
            for part in parts:
                clean_part = part.strip()
                if len(clean_part) > 100:
                    sections.append(clean_part)
        # 문단 기반 분할
        elif '\n\n' in content:
            paragraphs = content.split('\n\n')
            current_section = ""
            for paragraph in paragraphs:
                if len(current_section + paragraph) > 800:
                    if current_section:
                        sections.append(current_section.strip())
                    current_section = paragraph
                else:
                    current_section += "\n\n" + paragraph
            if current_section:
                sections.append(current_section.strip())
        # 전체를 하나의 섹션으로
        else:
            sections = [content]

        return [s for s in sections if len(s) > 50]

    def _extract_title_from_content(self, content: str) -> str:
        """콘텐츠에서 제목 추출"""
        lines = content.split('\n')
        for line in lines[:3]:  # 처음 3줄에서 찾기
            line = line.strip()
            if line and len(line) < 100:
                # 제목 패턴 정리
                title = re.sub(r'^[#\*\-\s]+', '', line)
                title = re.sub(r'[#\*\-\s]+$', '', title)
                if len(title) > 5:
                    return title[:50]
        return "여행 이야기"

    def _extract_subtitle_from_content(self, content: str) -> str:
        """콘텐츠에서 부제목 추출"""
        lines = content.split('\n')
        for i, line in enumerate(lines[1:4]):  # 2-4번째 줄에서 찾기
            line = line.strip()
            if line and len(line) < 80 and len(line) > 5:
                subtitle = re.sub(r'^[#\*\-\s]+', '', line)
                subtitle = re.sub(r'[#\*\-\s]+$', '', subtitle)
                if len(subtitle) > 3:
                    return subtitle[:40]
        return "특별한 순간들"

    def _clean_content(self, content: str) -> str:
        """콘텐츠 정리"""
        # 불필요한 패턴 제거
        cleaned = re.sub(r'^[#\*\-\s]+', '', content, flags=re.MULTILINE)
        # 연속된 줄바꿈 정리
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        # 빈 줄 제거
        cleaned = re.sub(r'^\s*$\n', '', cleaned, flags=re.MULTILINE)
        return cleaned.strip()

    def _extract_structured_content(self, text: str) -> List[Dict]:
        """구조화된 콘텐츠 추출"""
        sections = []
        
        # 제목 패턴 찾기
        title_patterns = [
            r'제목[:\s]*([^\n]+)',
            r'title[:\s]*([^\n]+)',
            r'## ([^\n]+)',
            r'# ([^\n]+)'
        ]
        
        for pattern in title_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                title = match.group(1).strip()
                if len(title) > 3:
                    section = {
                        "template": f"Section{len(sections)+1:02d}.jsx",
                        "title": title[:50],
                        "subtitle": "여행의 특별한 순간",
                        "body": f"{title}에 대한 자세한 이야기를 담고 있습니다.",
                        "tagline": "TRAVEL & CULTURE",
                        "layout_source": "org_agent"
                    }
                    sections.append(section)
                    if len(sections) >= 3:
                        break
            if sections:
                break
        
        return sections

    def _extract_template_from_binding_result(self, result: Dict) -> str:
        """BindingAgent 결과에서 템플릿명 추출"""
        task_description = result.get('task_description', '')
        template_match = re.search(r'Section\d+\.jsx', task_description)
        return template_match.group() if template_match else "Section01.jsx"

    def _extract_image_source_info(self, result: Dict, url: str) -> Dict:
        """이미지 소스 정보 추출"""
        return {
            "url": url,
            "template": self._extract_template_from_binding_result(result),
            "source": "binding_agent",
            "timestamp": result.get('timestamp', ''),
            "quality_verified": True
        }

    def _format_sections_for_analysis(self, sections: List[Dict]) -> str:
        """분석용 섹션 포맷팅"""
        formatted = []
        for i, section in enumerate(sections[:3]):  # 최대 3개만 표시
            formatted.append(f"""섹션 {i+1}:
- 템플릿: {section.get('template', 'N/A')}
- 제목: {section.get('title', 'N/A')}
- 부제목: {section.get('subtitle', 'N/A')}
- 본문 길이: {len(section.get('body', ''))} 문자
- 소스: {section.get('layout_source', 'N/A')}""")
        return "\n".join(formatted)

    def _format_images_for_analysis(self, template_images: Dict) -> str:
        """분석용 이미지 포맷팅"""
        formatted = []
        for template, images in template_images.items():
            formatted.append(f"""{template}: {len(images)}개 이미지
{chr(10).join([f'  - {img}' for img in images[:2]])}""")
        return "\n".join(formatted)

    def _format_image_sources(self, image_sources: List[Dict]) -> str:
        """이미지 소스 정보 포맷팅"""
        if not image_sources:
            return "이미지 소스 정보 없음"
        
        formatted = []
        for source in image_sources[:3]:  # 최대 3개만 표시
            formatted.append(f"- {source.get('template', 'N/A')}: {source.get('url', 'N/A')}")
        return "\n".join(formatted)

    def _extract_json_from_text(self, text: str) -> Dict:
        """텍스트에서 JSON 추출"""
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        parsed_data = {}
        
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                if len(match) < 10000:  # 너무 큰 JSON 제외
                    data = json.loads(match)
                    if isinstance(data, dict):
                        parsed_data.update(data)
            except json.JSONDecodeError:
                continue
        
        return parsed_data

    async def _log_coordination_result_async(self, final_result: Dict, text_mapping: Dict, 
                                           image_distribution: Dict, org_results: List[Dict], 
                                           binding_results: List[Dict]) -> None:
        """조율 결과 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="CoordinatorAgent",
                agent_role="통합 조율자 (CrewAI 기반 강화된 데이터 접근 및 JSON 파싱)",
                task_description=f"실제 데이터 기반 배치 매거진 구조 통합 조율: {len(final_result.get('content_sections', []))}개 섹션 생성",
                final_answer=str(final_result),
                reasoning_process=f"실제 데이터 기반 배치 처리로 {len(org_results)}개 OrgAgent 결과와 {len(binding_results)}개 BindingAgent 결과 통합",
                execution_steps=[
                    "재귀 깊이 체크",
                    "배치 기반 처리 모드 선택",
                    "이전 결과 수집",
                    "실제 데이터 추출",
                    "CrewAI 안전 실행",
                    "결과 통합 및 검증"
                ],
                raw_input={
                    "text_mapping": str(text_mapping)[:500],
                    "image_distribution": str(image_distribution)[:500]
                },
                raw_output=final_result,
                performance_metrics={
                    "total_sections_generated": len(final_result.get('content_sections', [])),
                    "org_results_utilized": len(org_results),
                    "binding_results_utilized": len(binding_results),
                    "real_data_extraction": True,
                    "crewai_enhanced": True,
                    "batch_processing": True,
                    "safe_execution": True,
                    "integration_quality_score": final_result.get('integration_metadata', {}).get('integration_quality_score', 0.8)
                }
            )
        )

    def _calculate_enhanced_quality_score(self, content_sections: List[Dict], org_count: int, binding_count: int) -> float:
        """강화된 품질 점수 계산"""
        base_score = 0.5
        
        # 섹션 수에 따른 점수
        if len(content_sections) >= 3:
            base_score += 0.2
        elif len(content_sections) >= 1:
            base_score += 0.1
        
        # 실제 콘텐츠 품질
        for section in content_sections:
            if len(section.get('body', '')) > 100:
                base_score += 0.05
            if section.get('images'):
                base_score += 0.05
            if section.get('metadata', {}).get('real_content'):
                base_score += 0.05
        
        # 에이전트 결과 활용도
        if org_count > 0:
            base_score += 0.1
        if binding_count > 0:
            base_score += 0.1
        
        return min(base_score, 1.0)

    def _filter_agent_results(self, results: List[Dict], agent_name: str) -> List[Dict]:
        """특정 에이전트 결과 필터링"""
        filtered = []
        for result in results:
            if isinstance(result, dict):
                result_agent = result.get('agent_name', '')
                if agent_name.lower() in result_agent.lower():
                    filtered.append(result)
        return filtered

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """결과 중복 제거"""
        seen = set()
        unique_results = []
        
        for result in results:
            if isinstance(result, dict):
                # 고유 키 생성
                key = f"{result.get('agent_name', '')}_{result.get('timestamp', '')}_{len(str(result.get('final_answer', '')))}"
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)
        
        return unique_results

    def _load_results_from_file(self) -> List[Dict]:
        """파일에서 결과 로드"""
        try:
            # 로그 파일들에서 결과 수집
            log_files = [
                "logs/agent_responses.json",
                "logs/org_agent_responses.json",
                "logs/binding_agent_responses.json",
                "logs/coordinator_responses.json"
            ]
            
            all_results = []
            for file_path in log_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            file_data = json.load(f)
                            if isinstance(file_data, list):
                                all_results.extend(file_data)
                            elif isinstance(file_data, dict):
                                all_results.append(file_data)
                        except json.JSONDecodeError:
                            continue
            
            return all_results
        except Exception as e:
            print(f"⚠️ 파일에서 결과 로드 실패: {e}")
            return []

    def _validate_coordinator_result(self, result: Dict) -> bool:
        """CoordinatorAgent 결과 검증"""
        if not isinstance(result, dict):
            return False
        
        required_fields = ["content_sections", "selected_templates"]
        for field in required_fields:
            if field not in result:
                return False
        
        content_sections = result.get("content_sections", [])
        if not isinstance(content_sections, list) or len(content_sections) == 0:
            return False
        
        for section in content_sections:
            if not isinstance(section, dict):
                return False
            required_section_fields = ["template", "title", "subtitle", "body", "tagline"]
            if not all(field in section for field in required_section_fields):
                return False
        
        return True

    # 기존 동기 버전 메서드들 유지 (호환성 보장)
    def coordinate_magazine_creation_sync(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """동기 버전 매거진 조율 (호환성 유지)"""
        return asyncio.run(self.coordinate_magazine_creation(text_mapping, image_distribution))

    def get_previous_results(self, agent_filter: str = None) -> List[Dict]:
        """이전 결과 가져오기 (동기 버전)"""
        try:
            return self.logger.get_all_previous_results("CoordinatorAgent")
        except Exception as e:
            print(f"⚠️ 이전 결과 가져오기 실패: {e}")
            return []

    def create_enhanced_magazine_structure(self, text_data: Dict, image_data: Dict) -> Dict:
        """강화된 매거진 구조 생성 (동기 버전)"""
        return self._create_enhanced_structure(text_data, image_data, [], [])

    def validate_magazine_structure(self, structure: Dict) -> bool:
        """매거진 구조 유효성 검증"""
        try:
            # 필수 필드 확인
            if not isinstance(structure, dict):
                return False
            
            if 'content_sections' not in structure:
                return False
            
            content_sections = structure['content_sections']
            if not isinstance(content_sections, list) or len(content_sections) == 0:
                return False
            
            # 각 섹션 유효성 확인
            for section in content_sections:
                if not isinstance(section, dict):
                    return False
                
                required_fields = ['template', 'title', 'subtitle', 'body', 'tagline']
                for field in required_fields:
                    if field not in section:
                        return False
                    if not isinstance(section[field], str):
                        return False
                
                # 이미지 필드 확인 (선택적)
                if 'images' in section:
                    if not isinstance(section['images'], list):
                        return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ 구조 유효성 검증 실패: {e}")
            return False

    def get_structure_statistics(self, structure: Dict) -> Dict:
        """구조 통계 정보 생성"""
        try:
            stats = {
                "total_sections": 0,
                "total_templates": 0,
                "total_images": 0,
                "total_content_length": 0,
                "average_content_length": 0,
                "sections_with_images": 0,
                "unique_templates": set(),
                "quality_indicators": {
                    "has_real_content": False,
                    "has_real_images": False,
                    "structure_complete": False
                }
            }
            
            if not isinstance(structure, dict) or 'content_sections' not in structure:
                return stats
            
            content_sections = structure['content_sections']
            if not isinstance(content_sections, list):
                return stats
            
            stats["total_sections"] = len(content_sections)
            
            total_length = 0
            for section in content_sections:
                if isinstance(section, dict):
                    # 템플릿 수집
                    template = section.get('template', '')
                    if template:
                        stats["unique_templates"].add(template)
                    
                    # 콘텐츠 길이
                    body = section.get('body', '')
                    if isinstance(body, str):
                        total_length += len(body)
                    
                    # 이미지 수
                    images = section.get('images', [])
                    if isinstance(images, list):
                        stats["total_images"] += len(images)
                        if len(images) > 0:
                            stats["sections_with_images"] += 1
                    
                    # 품질 지표
                    metadata = section.get('metadata', {})
                    if isinstance(metadata, dict):
                        if metadata.get('real_content'):
                            stats["quality_indicators"]["has_real_content"] = True
                        if metadata.get('real_images_used'):
                            stats["quality_indicators"]["has_real_images"] = True
            
            stats["total_templates"] = len(stats["unique_templates"])
            stats["unique_templates"] = list(stats["unique_templates"])
            stats["total_content_length"] = total_length
            stats["average_content_length"] = total_length / max(stats["total_sections"], 1)
            
            # 구조 완성도
            stats["quality_indicators"]["structure_complete"] = (
                stats["total_sections"] > 0 and
                stats["total_content_length"] > 100 and
                stats["total_templates"] > 0
            )
            
            return stats
            
        except Exception as e:
            print(f"⚠️ 구조 통계 생성 실패: {e}")
            return {"error": str(e)}

    def export_structure_to_json(self, structure: Dict, file_path: str = None) -> str:
        """구조를 JSON 파일로 내보내기"""
        try:
            if file_path is None:
                file_path = f"magazine_structure_{int(time.time())}.json"
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
            
            # JSON 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(structure, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 매거진 구조가 {file_path}에 저장되었습니다.")
            return file_path
            
        except Exception as e:
            print(f"⚠️ JSON 내보내기 실패: {e}")
            return ""

    def import_structure_from_json(self, file_path: str) -> Dict:
        """JSON 파일에서 구조 가져오기"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                structure = json.load(f)
            
            if self.validate_magazine_structure(structure):
                print(f"✅ 매거진 구조가 {file_path}에서 로드되었습니다.")
                return structure
            else:
                print(f"⚠️ 유효하지 않은 매거진 구조: {file_path}")
                return {}
                
        except Exception as e:
            print(f"⚠️ JSON 가져오기 실패: {e}")
            return {}

    # 디버깅 및 모니터링 메서드
    def debug_agent_results(self, results: List[Dict]) -> None:
        """에이전트 결과 디버깅"""
        print("=== 에이전트 결과 디버깅 ===")
        
        agent_counts = {}
        for result in results:
            agent_name = result.get('agent_name', 'Unknown')
            agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1
        
        print(f"총 결과 수: {len(results)}")
        for agent, count in agent_counts.items():
            print(f"- {agent}: {count}개")
        
        # 최근 결과 샘플
        print("\n=== 최근 결과 샘플 ===")
        for i, result in enumerate(results[-3:]):
            print(f"결과 {i+1}:")
            print(f"  에이전트: {result.get('agent_name', 'N/A')}")
            print(f"  시간: {result.get('timestamp', 'N/A')}")
            print(f"  응답 길이: {len(str(result.get('final_answer', '')))}")
            print(f"  작업: {result.get('task_description', 'N/A')[:100]}...")
            print()

    def monitor_system_health(self) -> Dict:
        """시스템 건강 상태 모니터링"""
        health_status = {
            "circuit_breaker_state": self.circuit_breaker.state,
            "failure_count": self.circuit_breaker.failure_count,
            "work_queue_size": len(self.work_queue.work_queue),
            "active_tasks": len(self.work_queue.active_tasks),
            "recursion_fallback_active": self.fallback_to_sync,
            "last_execution_mode": "unknown",
            "system_status": "healthy"
        }
        
        # 건강 상태 평가
        if self.circuit_breaker.state == "OPEN":
            health_status["system_status"] = "degraded"
        elif self.fallback_to_sync:
            health_status["system_status"] = "fallback_mode"
        elif len(self.work_queue.work_queue) > 10:
            health_status["system_status"] = "high_load"
        
        return health_status

    def reset_system_state(self) -> None:
        """시스템 상태 리셋"""
        print("🔄 CoordinatorAgent 시스템 상태 리셋")
        
        # Circuit Breaker 리셋
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.state = "CLOSED"
        self.circuit_breaker.last_failure_time = None
        
        # 폴백 플래그 리셋
        self.fallback_to_sync = False
        
        # 작업 큐 클리어
        self.work_queue.work_queue.clear()
        self.work_queue.active_tasks.clear()
        self.work_queue.results.clear()
        
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
        metrics = {
            "circuit_breaker": {
                "state": self.circuit_breaker.state,
                "failure_count": self.circuit_breaker.failure_count,
                "failure_threshold": self.circuit_breaker.failure_threshold
            },
            "work_queue": {
                "max_workers": self.work_queue.max_workers,
                "queue_size": len(self.work_queue.work_queue),
                "active_tasks": len(self.work_queue.active_tasks),
                "completed_results": len(self.work_queue.results)
            },
            "system": {
                "recursion_threshold": self.recursion_threshold,
                "fallback_to_sync": self.fallback_to_sync,
                "batch_size": self.batch_size
            }
        }
        
        if hasattr(self, 'execution_stats'):
            metrics["execution_stats"] = self.execution_stats
        
        return metrics

    def get_execution_statistics(self) -> Dict:
        """실행 통계 조회"""
        return {
            **self.execution_stats,
            "success_rate": (
                self.execution_stats["successful_executions"] / 
                max(self.execution_stats["total_attempts"], 1)
            ) * 100,
            "circuit_breaker_state": self.circuit_breaker.state,
            "current_queue_size": len(self.work_queue.work_queue)
        }

    def validate_system_integrity(self) -> bool:
        """시스템 무결성 검증"""
        try:
            # 필수 컴포넌트 확인
            required_components = [
                self.llm,
                self.logger,
                self.crew_agent,
                self.text_analyzer_agent,
                self.image_analyzer_agent
            ]
            
            for component in required_components:
                if component is None:
                    return False
            
            # 복원력 시스템 확인
            if self.work_queue is None or self.circuit_breaker is None:
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ 시스템 무결성 검증 실패: {e}")
            return False

    def get_system_info(self) -> Dict:
        """시스템 정보 조회"""
        return {
            "class_name": self.__class__.__name__,
            "version": "2.0_batch_resilient",
            "features": [
                "CrewAI 기반 강화된 데이터 접근",
                "JSON 파싱 및 구조 생성",
                "비동기 배치 처리",
                "Circuit Breaker 패턴",
                "재귀 깊이 감지 및 폴백",
                "안전한 에이전트 실행",
                "복원력 있는 작업 큐"
            ],
            "agents": {
                "crew_agent": "매거진 구조 통합 조율자",
                "text_analyzer_agent": "텍스트 매핑 분석 전문가",
                "image_analyzer_agent": "이미지 분배 분석 전문가"
            },
            "execution_modes": ["batch_async", "sync_fallback"],
            "safety_features": [
                "재귀 깊이 모니터링",
                "타임아웃 처리",
                "Circuit Breaker",
                "점진적 백오프",
                "폴백 메커니즘"
            ]
        }
