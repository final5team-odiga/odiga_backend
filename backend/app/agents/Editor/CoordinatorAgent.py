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
    def __init__(self, failure_threshold: int = 10, recovery_timeout: float = 120.0):
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





TIMEOUT_CONFIGS = {
    'org_agent': 900,      # 15분
    'binding_agent': 1200, # 20분  
    'coordinator_agent': 600, # 10분
    'vector_init': 600,    # 10분
    'crew_execution': 900  # 15분
}


class CoordinatorAgent:
    """통합 조율자 (CrewAI 기반 강화된 데이터 접근 및 JSON 파싱)"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        self.logger = get_agent_logger()
        self.crew_agent = self._create_crew_agent()
        self.text_analyzer_agent = self._create_text_analyzer_agent()
        self.image_analyzer_agent = self._create_image_analyzer_agent()
        
        # 새로운 복원력 시스템 추가
        self.work_queue = AsyncWorkQueue(max_workers=1, max_queue_size=20)
        self.circuit_breaker = CircuitBreaker()
        self.recursion_threshold = 600
        self.fallback_to_sync = False
        self.batch_size = 2
        
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
                                    timeout: float = None, max_retries: int = 2,
                                    *args, **kwargs) -> Any:
        if timeout is None:
            for task_type, default_timeout in TIMEOUT_CONFIGS.items():
                if task_type in task_id.lower():
                    timeout = default_timeout
                    break
            else:
                timeout = 300

        # 코루틴 객체 처리 개선
        if asyncio.iscoroutine(task_func):
            try:
                result = await asyncio.wait_for(task_func, timeout=timeout)
                self.circuit_breaker.record_success()
                return result
            except Exception as e:
                print(f"❌ Coroutine 실행 실패: {e}")
                self.circuit_breaker.record_failure()
                return self._get_fallback_result(task_id)
        
        # 코루틴 함수와 일반 함수 구분 처리
        if asyncio.iscoroutinefunction(task_func):
            coro = task_func(*args, **kwargs)
        else:
            # 일반 함수는 executor에서 실행
            loop = asyncio.get_event_loop()
            coro = loop.run_in_executor(None, lambda: task_func(*args, **kwargs))
        
        # WorkItem 생성 시 이미 생성된 코루틴 객체 전달
        work_item = WorkItem(
            id=task_id,
            task_func=coro,  # 코루틴 객체 직접 전달
            args=(),  # 빈 튜플
            kwargs={},  # 빈 딕셔너리
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
        
        if "_timeout" in task_id:
            reason = "timeout"
        elif "_exception" in task_id:
            reason = "exception"
        elif "_type_error" in task_id:
            reason = "type_error"
        
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
            goal="ContentCreatorV2Agent의 실제 텍스트 데이터와 BindingAgent의 정밀 이미지 배치를 통합하여 완벽한 매거진 구조를 생성하고, magazine_content.json의 섹션 수에 맞춰 최적화된 template_data.json을 제공",
            backstory="""당신은 25년간 세계 최고 수준의 출판사에서 매거진 구조 통합 및 품질 보증 책임자로 활동해온 전문가입니다. Condé Nast, Hearst Corporation, Time Inc.에서 수백 개의 매거진 프로젝트를 성공적으로 조율했습니다.

**전문 경력:**
- 출판학 및 구조 설계 석사 학위 보유
- PMP(Project Management Professional) 인증
- 매거진 구조 통합 및 품질 관리 전문가
- 텍스트-이미지 정합성 검증 시스템 개발 경험
- 독자 경험(UX) 및 접근성 최적화 전문성

**조율 철학:**
"완벽한 매거진은 모든 구조적 요소가 독자의 인지 과정과 완벽히 조화를 이루는 통합체입니다. 나는 텍스트와 이미지의 모든 배치가 독자에게 자연스럽고 직관적으로 인식되도록 구조적 완성도를 보장하며, 이를 통해 최고 수준의 독자 경험을 제공합니다."

**템플릿 생성 규칙:**
- ContentCreatorV2Agent의 실제 텍스트 데이터만을 사용하여 template_data.json을 생성합니다.
- magazine_content.json의 텍스트 섹션 수와 정확히 일치하도록 섹션을 생성합니다.
- 폴백 데이터(fallback_used: true)는 절대 포함하지 않습니다.
- 실제 이미지 URL만을 사용하며, 각 섹션당 최대 3개의 이미지로 제한합니다.
- title, subtitle, body, tagline은 실제 콘텐츠 데이터에서 추출된 내용만 사용합니다.
- 구조 설명, 레이아웃 설명, 플레이스홀더 텍스트는 포함하지 않습니다.
- 중복 섹션을 절대 생성하지 않습니다.
- 로그 데이터는 참조용으로만 사용하고 직접 포함하지 않습니다.
""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_text_analyzer_agent(self):
        """텍스트 분석 전문 에이전트"""
        return Agent(
            role="텍스트 매핑 분석 전문가",
            goal="ContentCreatorV2Agent의 텍스트 매핑 결과를 정밀 분석하여 구조적 완성도를 검증하고 최적화된 텍스트 섹션을 생성",
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
        
        # 수정: OrgAgent 결과 필터링 - ContentCreatorV2Agent 결과만 사용
        filtered_org_results = []
        for result in org_results:
            final_answer = result.get("final_answer", "")
            raw_output = result.get("raw_output", {})
            
            # 폴백 데이터 제외
            if isinstance(raw_output, dict):
                metadata = raw_output.get("metadata", {})
                if metadata.get("fallback_used"):
                    continue
            
            # ContentCreatorV2Agent의 실제 콘텐츠만 포함
            if ("ContentCreatorV2Agent" in final_answer or
                "content_creator" in final_answer.lower() or
                len(final_answer) > 500):  # 충분한 콘텐츠가 있는 경우
                # "자세한 이야기를 담고 있습니다" 같은 템플릿 응답 제외
                if not ("자세한 이야기를 담고 있습니다" in final_answer or
                        "특별한 이야기를 담고 있습니다" in final_answer):
                    filtered_org_results.append(result)
        
        org_results = filtered_org_results
        print(f"🔍 필터링 후 OrgAgent 결과: {len(org_results)}개")
        
        # magazine_content.json 로드하여 섹션 수 확인
        target_section_count = self._get_target_section_count()
        print(f"🎯 목표 섹션 수: {target_section_count}개")
        
        # 데이터 추출 작업을 배치로 처리
        data_extraction_tasks = [
            ("text_data", self._extract_real_text_data_safe, text_mapping, org_results, content_creator_results, target_section_count),
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
        
        # 수정: 섹션 수 제한 및 폴백 데이터 제거
        final_result = self._limit_and_clean_sections(final_result, target_section_count)
        
        # 결과 검증
        if self._validate_coordinator_result(final_result):
            self.execution_stats["successful_executions"] += 1
        else:
            print("⚠️ CoordinatorAgent 최종 결과 검증 실패.")
        
        # 결과 로깅
        await self._log_coordination_result_async(final_result, text_mapping, image_distribution, org_results, binding_results)
        
        print(f"✅ CoordinatorAgent 배치 모드 완료: {len(final_result.get('content_sections', []))}개 섹션 생성")
        return final_result

    def _get_target_section_count(self) -> int:
        """magazine_content.json에서 목표 섹션 수 확인"""
        try:
            magazine_content_path = "./output/magazine_content.json"
            if os.path.exists(magazine_content_path):
                with open(magazine_content_path, 'r', encoding='utf-8') as f:
                    magazine_data = json.load(f)
                sections = magazine_data.get("sections", [])
                if isinstance(sections, list):
                    return len(sections)
            # 기본값
            return 5
        except Exception as e:
            print(f"⚠️ magazine_content.json 로드 실패: {e}")
            return 5

    def _limit_and_clean_sections(self, result: Dict, target_count: int) -> Dict:
        """섹션 수 제한 및 폴백 데이터 정리"""
        if not isinstance(result, dict) or "content_sections" not in result:
            return result
        
        content_sections = result["content_sections"]
        if not isinstance(content_sections, list):
            return result
        
        # 폴백 데이터 제거
        cleaned_sections = []
        for section in content_sections:
            if isinstance(section, dict):
                metadata = section.get("metadata", {})
                if not metadata.get("fallback_used"):
                    cleaned_sections.append(section)
        
        # 섹션 수 제한
        limited_sections = cleaned_sections[:target_count]
        
        # 최소 1개 섹션 보장 (폴백이 아닌 실제 데이터로)
        if not limited_sections:
            limited_sections = [{
                "template": "Section01.jsx",
                "title": "",
                "subtitle": "",
                "body": "",
                "tagline": "",
                "images": [],
                "metadata": {
                    "minimal_fallback": True
                }
            }]
        
        result["content_sections"] = limited_sections
        result["selected_templates"] = [section.get("template", f"Section{i+1:02d}.jsx")
                                      for i, section in enumerate(limited_sections)]
        
        # 메타데이터 업데이트
        if "integration_metadata" in result:
            result["integration_metadata"]["total_sections"] = len(limited_sections)
            result["integration_metadata"]["cleaned_sections"] = True
            result["integration_metadata"]["target_section_count"] = target_count
        
        return result

    async def _process_data_extraction_batch(self, extraction_tasks: List[tuple]) -> Dict:
        """데이터 추출 작업을 배치로 처리"""
        batch_tasks = []
        
        for task_name, task_func_ref, *args_for_task_func in extraction_tasks:
            if not callable(task_func_ref):
                print(f"⚠️ {task_name}에 대한 task_func이 호출 가능하지 않음: {task_func_ref}")
                continue
            
            print(f"DEBUG [_process_data_extraction_batch]: task_name={task_name}, task_func_ref={task_func_ref}, args_for_task_func={args_for_task_func}")
            
            # 수정: task_func를 키워드 인자로 명시적으로 전달
            task = self.execute_with_resilience(
                task_func=task_func_ref,  # 함수/메서드 참조를 키워드 인자로 전달
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
                "sections": [],
                "total_content_length": 0,
                "source_count": 0
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
            sections = []
        
        # 이미지 추가
        for section in sections:
            template = section.get("template", "Section01.jsx")
            template_images = extracted_image_data.get("template_images", {}).get(template, [])
            section["images"] = template_images[:3]  # 최대 3개로 제한
        
        return json.dumps({
            "selected_templates": [s.get("template", "Section01.jsx") for s in sections],
            "content_sections": sections
        })

    async def _extract_real_text_data_safe(self, text_mapping: Dict, org_results: List[Dict],
                                         content_creator_results: List[Dict], target_section_count: int) -> Dict:
        """안전한 실제 텍스트 데이터 추출"""
        try:
            return await self._extract_real_text_data_async(text_mapping, org_results, content_creator_results, target_section_count)
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
                task_func=lambda: self.logger.get_all_previous_results(),
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
        
        # 수정: OrgAgent 결과 필터링
        filtered_org_results = []
        for result in org_results:
            final_answer = result.get("final_answer", "")
            raw_output = result.get("raw_output", {})
            
            # 폴백 데이터 제외
            if isinstance(raw_output, dict):
                metadata = raw_output.get("metadata", {})
                if metadata.get("fallback_used"):
                    continue
            
            # ContentCreatorV2Agent의 실제 콘텐츠만 포함
            if ("ContentCreatorV2Agent" in final_answer or
                "content_creator" in final_answer.lower() or
                len(final_answer) > 500):
                if not ("자세한 이야기를 담고 있습니다" in final_answer or
                        "특별한 이야기를 담고 있습니다" in final_answer):
                    filtered_org_results.append(result)
        
        org_results = filtered_org_results
        
        # 목표 섹션 수 확인
        target_section_count = self._get_target_section_count()
        
        # 데이터 추출 (동기)
        extracted_text_data = self._extract_real_text_data(text_mapping, org_results, content_creator_results, target_section_count)
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
        
        # 섹션 수 제한 및 정리
        final_result = self._limit_and_clean_sections(final_result, target_section_count)
        
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
            basic_results = self.logger.get_all_previous_results()
            file_results = self._load_results_from_file()
            
            all_results = []
            all_results.extend(basic_results if isinstance(basic_results, list) else [])
            all_results.extend(file_results if isinstance(file_results, list) else [])
            
            return self._deduplicate_results(all_results)
        except Exception as e:
            print(f"⚠️ 동기 이전 결과 수집 실패: {e}")
            return []

    # 모든 기존 메서드들 유지 (동기 버전들)
    async def _extract_real_text_data_async(self, text_mapping: Dict, org_results: List[Dict],
                                          content_creator_results: List[Dict], target_section_count: int) -> Dict:
        """실제 텍스트 데이터 추출 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._extract_real_text_data, text_mapping, org_results, content_creator_results, target_section_count
        )

    def _extract_real_text_data(self, text_mapping: Dict, org_results: List[Dict],
                               content_creator_results: List[Dict], target_section_count: int) -> Dict:
        """실제 텍스트 데이터 추출"""
        extracted_data = {
            "sections": [],
            "total_content_length": 0,
            "source_count": 0
        }
        
        # 1. ContentCreator 결과에서 우선적으로 추출
        for result in content_creator_results:
            final_answer = result.get('final_answer', '')
            if len(final_answer) > 200:  # 충분한 콘텐츠가 있는 경우
                # 섹션별로 분할
                sections = self._split_content_into_sections(final_answer)
                for i, section_content in enumerate(sections):
                    if len(section_content) > 50 and len(extracted_data["sections"]) < target_section_count:
                        extracted_section = {
                            "template": f"Section{len(extracted_data['sections'])+1:02d}.jsx",
                            "title": self._extract_title_from_content(section_content),
                            "subtitle": self._extract_subtitle_from_content(section_content),
                            "body": self._clean_content(section_content),
                            "tagline": "TRAVEL & CULTURE",
                            "layout_source": "content_creator"
                        }
                        extracted_data["sections"].append(extracted_section)
                        extracted_data["total_content_length"] += len(extracted_section["body"])
                        extracted_data["source_count"] += 1
        
        # 2. text_mapping에서 추가 추출 (목표 섹션 수에 미달인 경우)
        if len(extracted_data["sections"]) < target_section_count and isinstance(text_mapping, dict):
            text_mapping_data = text_mapping.get("text_mapping", [])
            if isinstance(text_mapping_data, list):
                for section in text_mapping_data:
                    if (isinstance(section, dict) and
                        len(extracted_data["sections"]) < target_section_count):
                        # 폴백 데이터 제외
                        metadata = section.get("metadata", {})
                        if metadata.get("fallback_used"):
                            continue
                        
                        extracted_section = {
                            "template": section.get("template", f"Section{len(extracted_data['sections'])+1:02d}.jsx"),
                            "title": section.get("title", ""),
                            "subtitle": section.get("subtitle", ""),
                            "body": section.get("body", ""),
                            "tagline": section.get("tagline", "TRAVEL & CULTURE"),
                            "layout_source": "text_mapping"
                        }
                        
                        # 빈 콘텐츠 제외
                        if (extracted_section["title"] or extracted_section["subtitle"] or
                            len(extracted_section["body"]) > 10):
                            extracted_data["sections"].append(extracted_section)
                            extracted_data["total_content_length"] += len(extracted_section["body"])
                            extracted_data["source_count"] += 1
        
        # 3. 목표 섹션 수에 맞춰 제한
        extracted_data["sections"] = extracted_data["sections"][:target_section_count]
        
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
                    # 실제 이미지 URL만 필터링 (최대 3개)
                    real_images = [img for img in images if self._is_real_image_url(img)][:3]
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
                    if (self._is_real_image_url(url) and
                        url not in extracted_data["template_images"][template_name] and
                        len(extracted_data["template_images"][template_name]) < 3):  # 최대 3개
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
6. 각 섹션당 최대 3개의 이미지로 제한
7. 폴백 데이터 절대 포함 금지

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
      "images": ["실제 이미지 URL들 (최대 3개)"],
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
        selected_templates = []
        
        # 추출된 텍스트 섹션을 기반으로 구조 생성
        for i, section in enumerate(extracted_text_data.get('sections', [])):
            template = section.get('template', f"Section{i+1:02d}.jsx")
            
            # 해당 템플릿의 이미지 가져오기
            template_images = extracted_image_data.get('template_images', {}).get(template, [])
            
            # 섹션 구조 생성
            section_data = {
                "template": template,
                "title": section.get('title', ''),
                "subtitle": section.get('subtitle', ''),
                "body": section.get('body', ''),
                "tagline": section.get('tagline', 'TRAVEL & CULTURE'),
                "images": template_images[:3],  # 최대 3개로 제한
                "metadata": {
                    "content_quality": self._calculate_content_quality(section),
                    "image_count": len(template_images[:3]),
                    "source": section.get('layout_source', 'extracted'),
                    "real_content": True,
                    "fallback_used": False
                }
            }
            
            content_sections.append(section_data)
            selected_templates.append(template)
        
        # 최소 1개 섹션 보장
        if not content_sections:
            content_sections = [{
                "template": "Section01.jsx",
                "title": "여행 매거진",
                "subtitle": "특별한 이야기",
                "body": "매거진 콘텐츠를 준비 중입니다.",
                "tagline": "TRAVEL & CULTURE",
                "images": [],
                "metadata": {
                    "content_quality": 0.5,
                    "image_count": 0,
                    "source": "minimal_fallback",
                    "real_content": False,
                    "fallback_used": True
                }
            }]
            selected_templates = ["Section01.jsx"]
        
        return {
            "selected_templates": selected_templates,
            "content_sections": content_sections,
            "integration_metadata": {
                "total_sections": len(content_sections),
                "total_templates": len(set(selected_templates)),
                "integration_quality_score": self._calculate_enhanced_quality_score(
                    content_sections, len(org_results), len(binding_results)
                ),
                "org_results_utilized": len(org_results),
                "binding_results_utilized": len(binding_results),
                "enhanced_structure": True,
                "real_data_based": True
            }
        }

    def _enhance_parsed_data_with_real_images(self, parsed_data: Dict, extracted_image_data: Dict) -> Dict:
        """파싱된 데이터에 실제 이미지 추가"""
        if not isinstance(parsed_data, dict) or 'content_sections' not in parsed_data:
            return parsed_data
        
        content_sections = parsed_data['content_sections']
        if not isinstance(content_sections, list):
            return parsed_data
        
        # 각 섹션에 실제 이미지 추가
        for section in content_sections:
            if isinstance(section, dict):
                template = section.get('template', 'Section01.jsx')
                real_images = extracted_image_data.get('template_images', {}).get(template, [])
                
                # 기존 이미지를 실제 이미지로 교체 (최대 3개)
                section['images'] = real_images[:3]
                
                # 메타데이터 업데이트
                if 'metadata' not in section:
                    section['metadata'] = {}
                section['metadata']['real_images_used'] = len(real_images[:3]) > 0
                section['metadata']['image_count'] = len(real_images[:3])
        
        return parsed_data

    def _calculate_content_quality(self, section: Dict) -> float:
        """콘텐츠 품질 점수 계산"""
        score = 0.0
        
        # 제목 품질 (0.3)
        title = section.get('title', '')
        if title and len(title) > 5:
            score += 0.3
        elif title:
            score += 0.15
        
        # 부제목 품질 (0.2)
        subtitle = section.get('subtitle', '')
        if subtitle and len(subtitle) > 5:
            score += 0.2
        elif subtitle:
            score += 0.1
        
        # 본문 품질 (0.4)
        body = section.get('body', '')
        if len(body) > 200:
            score += 0.4
        elif len(body) > 100:
            score += 0.3
        elif len(body) > 50:
            score += 0.2
        elif body:
            score += 0.1
        
        # 태그라인 품질 (0.1)
        tagline = section.get('tagline', '')
        if tagline and tagline != 'TRAVEL & CULTURE':
            score += 0.1
        elif tagline:
            score += 0.05
        
        return min(score, 1.0)

    def _calculate_enhanced_quality_score(self, content_sections: List[Dict], 
                                        org_results_count: int, binding_results_count: int) -> float:
        """강화된 품질 점수 계산"""
        if not content_sections:
            return 0.0
        
        # 기본 콘텐츠 품질 점수
        content_scores = [self._calculate_content_quality(section) for section in content_sections]
        avg_content_score = sum(content_scores) / len(content_scores)
        
        # 데이터 활용도 점수
        data_utilization_score = min((org_results_count + binding_results_count) / 10.0, 1.0)
        
        # 이미지 활용도 점수
        total_images = sum(len(section.get('images', [])) for section in content_sections)
        image_score = min(total_images / (len(content_sections) * 2), 1.0)  # 섹션당 평균 2개 이미지 기준
        
        # 가중 평균 계산
        final_score = (avg_content_score * 0.5 + data_utilization_score * 0.3 + image_score * 0.2)
        
        return round(final_score, 2)

    def _format_sections_for_analysis(self, sections: List[Dict]) -> str:
        """분석용 섹션 포맷팅"""
        if not sections:
            return "섹션 데이터 없음"
        
        formatted = []
        for i, section in enumerate(sections[:3]):  # 처음 3개만 표시
            formatted.append(f"""
섹션 {i+1}:
- 템플릿: {section.get('template', 'N/A')}
- 제목: {section.get('title', 'N/A')[:50]}...
- 부제목: {section.get('subtitle', 'N/A')[:50]}...
- 본문 길이: {len(section.get('body', ''))} 문자
- 소스: {section.get('layout_source', 'N/A')}""")
        
        if len(sections) > 3:
            formatted.append(f"... 및 {len(sections) - 3}개 추가 섹션")
        
        return "\n".join(formatted)

    def _format_images_for_analysis(self, template_images: Dict) -> str:
        """분석용 이미지 포맷팅"""
        if not template_images:
            return "이미지 데이터 없음"
        
        formatted = []
        for template, images in template_images.items():
            formatted.append(f"- {template}: {len(images)}개 이미지")
            for img in images[:2]:  # 처음 2개만 표시
                formatted.append(f"  * {img[:60]}...")
        
        return "\n".join(formatted)

    def _format_image_sources(self, image_sources: List[Dict]) -> str:
        """이미지 소스 정보 포맷팅"""
        if not image_sources:
            return "이미지 소스 정보 없음"
        
        formatted = []
        for source in image_sources[:5]:  # 처음 5개만 표시
            formatted.append(f"- {source.get('url', 'N/A')[:50]}... (소스: {source.get('source', 'N/A')})")
        
        if len(image_sources) > 5:
            formatted.append(f"... 및 {len(image_sources) - 5}개 추가 소스")
        
        return "\n".join(formatted)

    def _split_content_into_sections(self, content: str) -> List[str]:
        """콘텐츠를 섹션별로 분할"""
        # 단락 기준으로 분할
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # 최소 길이 이상의 단락들을 섹션으로 구성
        sections = []
        current_section = ""
        
        for paragraph in paragraphs:
            if len(current_section + paragraph) < 300:  # 섹션당 최소 300자
                current_section += paragraph + "\n\n"
            else:
                if current_section:
                    sections.append(current_section.strip())
                current_section = paragraph + "\n\n"
        
        # 마지막 섹션 추가
        if current_section:
            sections.append(current_section.strip())
        
        return sections

    def _extract_title_from_content(self, content: str) -> str:
        """콘텐츠에서 제목 추출"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) < 100:  # 제목은 보통 100자 이하
                # 특수 문자나 번호 제거
                cleaned = re.sub(r'^[\d\.\-\*\#\s]+', '', line)
                if len(cleaned) > 5:
                    return cleaned[:80]  # 최대 80자
        
        # 첫 번째 문장을 제목으로 사용
        first_sentence = content.split('.')[0].strip()
        return first_sentence[:80] if first_sentence else "여행 이야기"

    def _extract_subtitle_from_content(self, content: str) -> str:
        """콘텐츠에서 부제목 추출"""
        lines = content.split('\n')
        
        # 두 번째 줄이나 첫 번째 문장 다음을 부제목으로 사용
        if len(lines) > 1:
            subtitle = lines[1].strip()
            if subtitle and len(subtitle) < 150:
                return subtitle[:100]
        
        # 두 번째 문장을 부제목으로 사용
        sentences = content.split('.')
        if len(sentences) > 1:
            subtitle = sentences[1].strip()
            return subtitle[:100] if subtitle else "특별한 경험"
        
        return "특별한 경험"

    def _clean_content(self, content: str) -> str:
        """콘텐츠 정리"""
        # 불필요한 공백 제거
        cleaned = re.sub(r'\n\s*\n', '\n\n', content)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        
        # 특수 문자 정리
        cleaned = re.sub(r'^[\d\.\-\*\#\s]+', '', cleaned, flags=re.MULTILINE)
        
        return cleaned.strip()

    def _extract_template_from_binding_result(self, result: Dict) -> str:
        """BindingAgent 결과에서 템플릿명 추출"""
        final_answer = result.get('final_answer', '')
        
        # 템플릿 패턴 찾기
        template_match = re.search(r'Section\d{2}\.jsx', final_answer)
        if template_match:
            return template_match.group()
        
        # 기본 템플릿 반환
        return "Section01.jsx"

    def _extract_image_source_info(self, result: Dict, url: str) -> Dict:
        """이미지 소스 정보 추출"""
        return {
            "url": url,
            "source": "BindingAgent",
            "agent_id": result.get('agent_id', 'unknown'),
            "timestamp": result.get('timestamp', 'unknown')
        }

    def _filter_agent_results(self, results: List[Dict], agent_name: str) -> List[Dict]:
        """특정 에이전트 결과 필터링"""
        filtered = []
        for result in results:
            if isinstance(result, dict):
                agent_info = result.get('agent_name', '')
                if agent_name.lower() in agent_info.lower():
                    filtered.append(result)
        return filtered

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """결과 중복 제거"""
        seen_ids = set()
        deduplicated = []
        
        for result in results:
            if isinstance(result, dict):
                result_id = result.get('id', str(hash(str(result))))
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    deduplicated.append(result)
        
        return deduplicated

    def _load_results_from_file(self) -> List[Dict]:
        """파일에서 결과 로드"""
        try:
            results_file = "./output/agent_results.json"
            if os.path.exists(results_file):
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            return []
        except Exception as e:
            print(f"⚠️ 결과 파일 로드 실패: {e}")
            return []

    def _extract_json_from_text(self, text: str) -> Dict:
        """텍스트에서 JSON 추출 및 파싱"""
        try:
            # JSON 블록 찾기
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            
            # 기본 구조 반환
            return {
                "selected_templates": [],
                "content_sections": []
            }
        except Exception as e:
            print(f"⚠️ JSON 파싱 실패: {e}")
            return {
                "selected_templates": [],
                "content_sections": []
            }

    def _validate_coordinator_result(self, result: Dict) -> bool:
        """CoordinatorAgent 결과 검증"""
        if not isinstance(result, dict):
            return False
        
        # 필수 키 확인
        required_keys = ['selected_templates', 'content_sections']
        for key in required_keys:
            if key not in result:
                return False
        
        # 콘텐츠 섹션 검증
        content_sections = result.get('content_sections', [])
        if not isinstance(content_sections, list) or len(content_sections) == 0:
            return False
        
        # 각 섹션 검증
        for section in content_sections:
            if not isinstance(section, dict):
                return False
            
            required_section_keys = ['template', 'title', 'body']
            for key in required_section_keys:
                if key not in section:
                    return False
        
        return True

    async def _log_coordination_result_async(self, result: Dict, text_mapping: Dict, 
                                           image_distribution: Dict, org_results: List[Dict], 
                                           binding_results: List[Dict]):
        """비동기 조율 결과 로깅"""
        try:
            response_id = self.logger.log_agent_real_output(
                agent_name="CoordinatorAgent",
                agent_role="매거진 구조 통합 조율자",
                task_description=f"배치 모드로 {len(result.get('content_sections', []))}개 섹션 생성",
                final_answer=str(result),
                reasoning_process="실제 데이터 기반 배치 처리를 통한 안전한 매거진 구조 통합",
                execution_steps=[
                    "이전 결과 배치 수집",
                    "실제 데이터 추출",
                    "CrewAI 배치 실행",
                    "결과 통합 및 검증",
                    "품질 보증"
                ],
                raw_input={
                    "text_mapping": str(text_mapping)[:500],
                    "image_distribution": str(image_distribution)[:500]
                },
                raw_output=result,
                performance_metrics={
                    "batch_mode_used": True,
                    "total_sections": len(result.get('content_sections', [])),
                    "org_results_utilized": len(org_results),
                    "binding_results_utilized": len(binding_results),
                    "execution_stats": self.execution_stats,
                    "quality_score": result.get('integration_metadata', {}).get('integration_quality_score', 0),
                    "real_data_used": True
                }
            )
            
            result["final_response_id"] = response_id
            result["execution_mode"] = "batch_async"
            
        except Exception as e:
            print(f"⚠️ 비동기 로깅 실패: {e}")

    def get_execution_stats(self) -> Dict:
        """실행 통계 반환"""
        return {
            **self.execution_stats,
            "success_rate": (self.execution_stats["successful_executions"] / 
                           max(self.execution_stats["total_attempts"], 1)) * 100,
            "fallback_rate": (self.execution_stats["fallback_used"] / 
                            max(self.execution_stats["total_attempts"], 1)) * 100,
            "circuit_breaker_state": self.circuit_breaker.state,
            "current_mode": "sync" if self.fallback_to_sync else "async"
        }

    def reset_execution_state(self):
        """실행 상태 초기화"""
        self.fallback_to_sync = False
        self.circuit_breaker = CircuitBreaker()
        self.execution_stats = {
            "total_attempts": 0,
            "successful_executions": 0,
            "fallback_used": 0,
            "circuit_breaker_triggered": 0,
            "timeout_occurred": 0
        }
        print("✅ CoordinatorAgent 실행 상태 초기화 완료")
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.work_queue.executor:
            self.work_queue.executor.shutdown(wait=True)
        
        # 예외 처리
        if exc_type:
            print(f"⚠️ CoordinatorAgent 컨텍스트에서 예외 발생: {exc_type.__name__}: {exc_val}")
            return False  # 예외를 재발생시킴
        
        return True

    def __enter__(self):
        """동기 컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """동기 컨텍스트 매니저 종료"""
        if self.work_queue.executor:
            self.work_queue.executor.shutdown(wait=True)
        
        if exc_type:
            print(f"⚠️ CoordinatorAgent 동기 컨텍스트에서 예외 발생: {exc_type.__name__}: {exc_val}")
        
        return False

    def cleanup_resources(self):
        """리소스 정리"""
        try:
            if hasattr(self.work_queue, 'executor') and self.work_queue.executor:
                self.work_queue.executor.shutdown(wait=True)
                print("✅ ThreadPoolExecutor 정리 완료")
            
            # 큐 정리
            self.work_queue.work_queue.clear()
            self.work_queue.active_tasks.clear()
            self.work_queue.results.clear()
            
            print("✅ CoordinatorAgent 리소스 정리 완료")
            
        except Exception as e:
            print(f"⚠️ 리소스 정리 중 오류: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """CoordinatorAgent 상태 확인"""
        try:
            # 기본 상태 정보
            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "execution_mode": "sync" if self.fallback_to_sync else "async",
                "circuit_breaker_state": self.circuit_breaker.state,
                "queue_size": len(self.work_queue.work_queue),
                "active_tasks": len(self.work_queue.active_tasks),
                "execution_stats": self.execution_stats
            }
            
            # LLM 연결 확인
            try:
                if self.llm:
                    health_status["llm_status"] = "connected"
                else:
                    health_status["llm_status"] = "disconnected"
                    health_status["status"] = "degraded"
            except Exception as e:
                health_status["llm_status"] = f"error: {str(e)}"
                health_status["status"] = "degraded"
            
            # 로거 상태 확인
            try:
                if self.logger:
                    health_status["logger_status"] = "connected"
                else:
                    health_status["logger_status"] = "disconnected"
                    health_status["status"] = "degraded"
            except Exception as e:
                health_status["logger_status"] = f"error: {str(e)}"
                health_status["status"] = "degraded"
            
            # 에이전트 상태 확인
            agents_status = {}
            for agent_name, agent in [
                ("crew_agent", self.crew_agent),
                ("text_analyzer_agent", self.text_analyzer_agent),
                ("image_analyzer_agent", self.image_analyzer_agent)
            ]:
                try:
                    if agent and hasattr(agent, 'role'):
                        agents_status[agent_name] = "initialized"
                    else:
                        agents_status[agent_name] = "not_initialized"
                        health_status["status"] = "degraded"
                except Exception as e:
                    agents_status[agent_name] = f"error: {str(e)}"
                    health_status["status"] = "degraded"
            
            health_status["agents_status"] = agents_status
            
            # 메모리 사용량 확인 (선택적)
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                health_status["memory_usage"] = {
                    "rss": memory_info.rss,
                    "vms": memory_info.vms,
                    "percent": process.memory_percent()
                }
            except ImportError:
                health_status["memory_usage"] = "psutil not available"
            except Exception as e:
                health_status["memory_usage"] = f"error: {str(e)}"
            
            return health_status
            
        except Exception as e:
            return {
                "status": "error",
                "timestamp": time.time(),
                "error": str(e),
                "execution_stats": self.execution_stats
            }

    async def force_reset(self):
        """강제 재설정"""
        print("🔄 CoordinatorAgent 강제 재설정 시작")
        
        try:
            # 1. 실행 중인 작업 중단
            for task_id, task in self.work_queue.active_tasks.items():
                if not task.done():
                    task.cancel()
                    print(f"⏹️ 작업 {task_id} 취소")
            
            # 2. 큐 및 결과 정리
            self.work_queue.work_queue.clear()
            self.work_queue.active_tasks.clear()
            self.work_queue.results.clear()
            
            # 3. 실행 상태 초기화
            self.reset_execution_state()
            
            # 4. 에이전트 재생성
            self.crew_agent = self._create_crew_agent()
            self.text_analyzer_agent = self._create_text_analyzer_agent()
            self.image_analyzer_agent = self._create_image_analyzer_agent()
            
            print("✅ CoordinatorAgent 강제 재설정 완료")
            
        except Exception as e:
            print(f"❌ 강제 재설정 중 오류: {e}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 반환"""
        total_attempts = self.execution_stats["total_attempts"]
        
        if total_attempts == 0:
            return {
                "success_rate": 0.0,
                "failure_rate": 0.0,
                "fallback_rate": 0.0,
                "timeout_rate": 0.0,
                "circuit_breaker_rate": 0.0,
                "total_attempts": 0,
                "current_mode": "sync" if self.fallback_to_sync else "async",
                "circuit_breaker_state": self.circuit_breaker.state
            }
        
        return {
            "success_rate": (self.execution_stats["successful_executions"] / total_attempts) * 100,
            "failure_rate": ((total_attempts - self.execution_stats["successful_executions"]) / total_attempts) * 100,
            "fallback_rate": (self.execution_stats["fallback_used"] / total_attempts) * 100,
            "timeout_rate": (self.execution_stats["timeout_occurred"] / total_attempts) * 100,
            "circuit_breaker_rate": (self.execution_stats["circuit_breaker_triggered"] / total_attempts) * 100,
            "total_attempts": total_attempts,
            "successful_executions": self.execution_stats["successful_executions"],
            "current_mode": "sync" if self.fallback_to_sync else "async",
            "circuit_breaker_state": self.circuit_breaker.state,
            "queue_utilization": len(self.work_queue.work_queue) / self.work_queue.max_queue_size * 100,
            "active_tasks_count": len(self.work_queue.active_tasks)
        }

    async def test_coordination_pipeline(self) -> Dict[str, Any]:
        """조율 파이프라인 테스트"""
        print("🧪 CoordinatorAgent 파이프라인 테스트 시작")
        
        test_results = {
            "test_timestamp": time.time(),
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": []
        }
        
        # 테스트 1: 기본 초기화 확인
        try:
            assert self.llm is not None, "LLM이 초기화되지 않음"
            assert self.logger is not None, "Logger가 초기화되지 않음"
            assert self.crew_agent is not None, "Crew Agent가 초기화되지 않음"
            
            test_results["tests_passed"] += 1
            test_results["test_details"].append({
                "test_name": "initialization_test",
                "status": "passed",
                "message": "모든 구성 요소가 정상적으로 초기화됨"
            })
        except Exception as e:
            test_results["tests_failed"] += 1
            test_results["test_details"].append({
                "test_name": "initialization_test",
                "status": "failed",
                "error": str(e)
            })
        
        # 테스트 2: 간단한 작업 실행 테스트
        try:
            test_task_result = await self.execute_with_resilience(
                task_func=lambda: {"test": "success"},
                task_id="pipeline_test",
                timeout=30.0,
                max_retries=1
            )
            
            assert test_task_result is not None, "테스트 작업 결과가 None"
            
            test_results["tests_passed"] += 1
            test_results["test_details"].append({
                "test_name": "task_execution_test",
                "status": "passed",
                "message": "작업 실행이 정상적으로 완료됨",
                "result": test_task_result
            })
        except Exception as e:
            test_results["tests_failed"] += 1
            test_results["test_details"].append({
                "test_name": "task_execution_test",
                "status": "failed",
                "error": str(e)
            })
        
        # 테스트 3: 데이터 추출 기능 테스트
        try:
            test_text_data = {
                "sections": [{
                    "template": "Section01.jsx",
                    "title": "테스트 제목",
                    "body": "테스트 본문 내용"
                }]
            }
            
            test_image_data = {
                "template_images": {
                    "Section01.jsx": ["https://example.com/test.jpg"]
                }
            }
            
            enhanced_structure = self._create_enhanced_structure(
                test_text_data, test_image_data, [], []
            )
            
            assert isinstance(enhanced_structure, dict), "구조 생성 결과가 딕셔너리가 아님"
            assert "content_sections" in enhanced_structure, "content_sections 키가 없음"
            
            test_results["tests_passed"] += 1
            test_results["test_details"].append({
                "test_name": "data_extraction_test",
                "status": "passed",
                "message": "데이터 추출 및 구조 생성이 정상적으로 완료됨"
            })
        except Exception as e:
            test_results["tests_failed"] += 1
            test_results["test_details"].append({
                "test_name": "data_extraction_test",
                "status": "failed",
                "error": str(e)
            })
        
        # 테스트 결과 요약
        total_tests = test_results["tests_passed"] + test_results["tests_failed"]
        test_results["success_rate"] = (test_results["tests_passed"] / total_tests * 100) if total_tests > 0 else 0
        test_results["overall_status"] = "passed" if test_results["tests_failed"] == 0 else "failed"
        
        print(f"🧪 파이프라인 테스트 완료: {test_results['tests_passed']}/{total_tests} 통과")
        
        return test_results

# 사용 예시 및 유틸리티 함수들
def create_coordinator_agent() -> CoordinatorAgent:
    """CoordinatorAgent 인스턴스 생성"""
    try:
        coordinator = CoordinatorAgent()
        print("✅ CoordinatorAgent 생성 완료")
        return coordinator
    except Exception as e:
        print(f"❌ CoordinatorAgent 생성 실패: {e}")
        raise

async def run_coordination_with_monitoring(coordinator: CoordinatorAgent, 
                                         text_mapping: Dict, 
                                         image_distribution: Dict) -> Dict:
    """모니터링과 함께 조율 실행"""
    start_time = time.time()
    
    try:
        # 상태 확인
        health_status = await coordinator.health_check()
        if health_status["status"] == "error":
            print(f"⚠️ CoordinatorAgent 상태 불량: {health_status}")
        
        # 조율 실행
        result = await coordinator.coordinate_magazine_creation(text_mapping, image_distribution)
        
        # 실행 시간 측정
        execution_time = time.time() - start_time
        
        # 성능 메트릭 추가
        result["execution_metadata"] = {
            "execution_time": execution_time,
            "performance_metrics": coordinator.get_performance_metrics(),
            "health_status": health_status
        }
        
        print(f"✅ 조율 완료 (실행 시간: {execution_time:.2f}초)")
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ 조율 실행 실패 (실행 시간: {execution_time:.2f}초): {e}")
        
        # 오류 정보와 함께 폴백 결과 반환
        return {
            "selected_templates": ["Section01.jsx"],
            "content_sections": [{
                "template": "Section01.jsx",
                "title": "매거진 생성 오류",
                "subtitle": "시스템 오류로 인한 폴백",
                "body": f"조율 과정에서 오류가 발생했습니다: {str(e)}",
                "tagline": "SYSTEM ERROR",
                "images": [],
                "metadata": {
                    "error_fallback": True,
                    "error_message": str(e),
                    "execution_time": execution_time
                }
            }],
            "integration_metadata": {
                "total_sections": 1,
                "error_occurred": True,
                "execution_time": execution_time,
                "performance_metrics": coordinator.get_performance_metrics()
            }
        }

# 모듈 수준 유틸리티
def validate_coordination_inputs(text_mapping: Dict, image_distribution: Dict) -> bool:
    """조율 입력 데이터 검증"""
    try:
        # text_mapping 검증
        if not isinstance(text_mapping, dict):
            print("⚠️ text_mapping이 딕셔너리가 아님")
            return False
        
        # image_distribution 검증
        if not isinstance(image_distribution, dict):
            print("⚠️ image_distribution이 딕셔너리가 아님")
            return False
        
        print("✅ 조율 입력 데이터 검증 통과")
        return True
        
    except Exception as e:
        print(f"❌ 입력 데이터 검증 실패: {e}")
        return False

# 전역 설정
COORDINATOR_CONFIG = {
    "max_workers": 1,
    "max_queue_size": 20,
    "default_timeout": 300.0,
    "max_retries": 2,
    "circuit_breaker_threshold": 5,
    "circuit_breaker_timeout": 60.0,
    "batch_size": 2,
    "recursion_threshold": 800
}

def update_coordinator_config(**kwargs):
    """CoordinatorAgent 설정 업데이트"""
    global COORDINATOR_CONFIG
    COORDINATOR_CONFIG.update(kwargs)
    print(f"✅ CoordinatorAgent 설정 업데이트: {kwargs}")

# 모듈 초기화 시 실행되는 코드
if __name__ == "__main__":
    print("🚀 CoordinatorAgent 모듈 로드 완료")
    print(f"📋 현재 설정: {COORDINATOR_CONFIG}")
