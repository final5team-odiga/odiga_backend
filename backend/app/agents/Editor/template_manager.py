import asyncio
import os
from typing import Dict, List
from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
from agents.Editor.OrgAgent import OrgAgent
from agents.Editor.BindingAgent import BindingAgent
from agents.Editor.CoordinatorAgent import CoordinatorAgent
from utils.pdf_vector_manager import PDFVectorManager
from utils.agent_decision_logger import get_agent_logger

class MultiAgentTemplateManager:
    """PDF 벡터 데이터 기반 다중 에이전트 템플릿 관리자 (CrewAI 통합 로깅 시스템 - 비동기 처리)"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        self.org_agent = OrgAgent()
        self.binding_agent = BindingAgent()
        self.coordinator_agent = CoordinatorAgent()
        self.vector_manager = PDFVectorManager()
        self.logger = get_agent_logger()  # 로깅 시스템 추가
        
        # CrewAI 에이전트들 생성
        self.vector_init_agent = self._create_vector_init_agent()
        self.template_loader_agent = self._create_template_loader_agent()
        self.requirement_analyzer_agent = self._create_requirement_analyzer_agent()
        self.data_prep_agent = self._create_data_prep_agent()
        self.coordination_agent = self._create_coordination_agent()

    def _create_vector_init_agent(self):
        """벡터 시스템 초기화 에이전트"""
        return Agent(
            role="PDF 벡터 시스템 초기화 전문가",
            goal="Azure Cognitive Search 기반 PDF 벡터 시스템을 안정적으로 초기화하고 템플릿 데이터를 효율적으로 처리",
            backstory="""당신은 10년간 검색 엔진 및 벡터 데이터베이스 시스템을 설계하고 운영해온 전문가입니다. Azure Cognitive Search, Elasticsearch, 그리고 다양한 벡터 데이터베이스 시스템의 최적화에 특화되어 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_template_loader_agent(self):
        """템플릿 로더 에이전트"""
        return Agent(
            role="JSX 템플릿 관리 및 로딩 전문가",
            goal="템플릿 폴더에서 JSX 파일들을 효율적으로 스캔하고 매거진 생성에 최적화된 템플릿 목록을 제공",
            backstory="""당신은 8년간 React 및 JSX 기반 웹 개발 프로젝트를 관리해온 전문가입니다. 템플릿 시스템 설계와 동적 컴포넌트 로딩에 특화되어 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_requirement_analyzer_agent(self):
        """요구사항 분석 에이전트"""
        return Agent(
            role="템플릿 요구사항 분석 전문가",
            goal="각 JSX 템플릿의 구조적 특성을 분석하여 이미지 요구사항과 레이아웃 스펙을 정확히 도출",
            backstory="""당신은 12년간 UI/UX 설계 및 템플릿 시스템 분석을 담당해온 전문가입니다. 다양한 레이아웃 패턴과 이미지 배치 최적화에 대한 깊은 이해를 보유하고 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_data_prep_agent(self):
        """데이터 준비 에이전트"""
        return Agent(
            role="매거진 데이터 준비 및 전처리 전문가",
            goal="매거진 생성에 필요한 모든 데이터를 수집, 정리, 검증하여 다중 에이전트 시스템이 효율적으로 작동할 수 있도록 준비",
            backstory="""당신은 15년간 출판업계에서 데이터 관리 및 전처리를 담당해온 전문가입니다. 복잡한 멀티미디어 데이터의 구조화와 품질 관리에 특화되어 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_coordination_agent(self):
        """조율 관리 에이전트"""
        return Agent(
            role="다중 에이전트 조율 및 프로세스 관리 전문가",
            goal="OrgAgent, BindingAgent, CoordinatorAgent의 순차적 실행을 관리하고 각 단계의 결과를 최적화하여 최고 품질의 매거진 데이터를 생성",
            backstory="""당신은 20년간 복잡한 소프트웨어 시스템의 프로젝트 관리와 다중 에이전트 조율을 담당해온 전문가입니다. 워크플로우 최적화와 품질 보증에 특화되어 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def initialize_vector_system(self, template_folder: str = "templates"):
        """벡터 시스템 초기화 - PDF 처리 및 인덱싱 (CrewAI 기반 로깅 추가 - 비동기 처리)"""
        print("=== PDF 벡터 시스템 초기화 (비동기 처리) ===")
        
        # CrewAI Task 생성
        init_task = Task(
            description=f"""
            PDF 벡터 시스템을 초기화하고 템플릿 폴더 '{template_folder}'를 처리하세요.
            
            **초기화 단계:**
            1. Azure Cognitive Search 인덱스 상태 확인
            2. 기존 인덱스가 없으면 새로 생성
            3. PDF 템플릿 파일들을 스캔하고 벡터화
            4. 인덱스에 데이터 저장 및 검증
            
            **성공 기준:**
            - 인덱스가 성공적으로 생성되거나 기존 인덱스 확인
            - 모든 PDF 템플릿이 처리되어 검색 가능한 상태
            - 벡터 검색 기능이 정상 작동
            
            초기화 과정에서 발생하는 모든 단계를 상세히 보고하세요.
            """,
            expected_output="벡터 시스템 초기화 완료 보고서 (성공/실패 상태 및 상세 로그 포함)",
            agent=self.vector_init_agent
        )
        
        # CrewAI Crew 생성 및 실행
        init_crew = Crew(
            agents=[self.vector_init_agent],
            tasks=[init_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            # 초기화 시작 로깅 (비동기)
            await self._log_initialization_start_async(template_folder)
            
            # CrewAI 비동기 실행
            crew_result = await asyncio.get_event_loop().run_in_executor(
                None, init_crew.kickoff
            )
            
            # 실제 벡터 시스템 초기화 수행 (비동기)
            await asyncio.gather(
                asyncio.get_event_loop().run_in_executor(None, self.vector_manager.initialize_search_index),
                asyncio.get_event_loop().run_in_executor(None, self.vector_manager.process_pdf_templates, template_folder)
            )
            
            # 초기화 완료 로깅 (비동기)
            await self._log_initialization_complete_async(template_folder, crew_result)
            
            print("✅ PDF 벡터 시스템 초기화 완료 (비동기)")
            
        except Exception as e:
            # 초기화 실패 로깅 (비동기)
            await self._log_initialization_error_async(template_folder, str(e))
            print(f"❌ PDF 벡터 시스템 초기화 실패: {e}")
            raise e

    async def should_initialize_vector_system(self) -> bool:
        """벡터 시스템 초기화 필요 여부 확인 (CrewAI 기반 로깅 추가 - 비동기 처리)"""
        
        # CrewAI Task 생성
        check_task = Task(
            description="""
            벡터 시스템의 현재 상태를 확인하고 초기화가 필요한지 판단하세요.
            
            **확인 항목:**
            1. Azure Cognitive Search 인덱스 존재 여부
            2. 인덱스 내 데이터 존재 여부
            3. 벡터 검색 기능 정상 작동 여부
            
            **판단 기준:**
            - 인덱스가 없으면: 초기화 필요
            - 인덱스는 있지만 데이터가 없으면: 초기화 필요
            - 인덱스와 데이터가 모두 있으면: 초기화 불필요
            
            상태 확인 결과를 상세히 보고하세요.
            """,
            expected_output="벡터 시스템 상태 확인 결과 및 초기화 필요성 판단",
            agent=self.vector_init_agent
        )
        
        # CrewAI Crew 생성 및 실행
        check_crew = Crew(
            agents=[self.vector_init_agent],
            tasks=[check_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            # CrewAI 비동기 실행
            crew_result = await asyncio.get_event_loop().run_in_executor(
                None, check_crew.kickoff
            )
            
            # 기존 인덱스 존재 여부 확인 (비동기)
            index_exists, data_exists = await asyncio.gather(
                self._check_index_exists_async(),
                self._check_data_exists_async(),
                return_exceptions=True
            )
            
            # 결과가 있으면 초기화 불필요
            if not isinstance(data_exists, Exception) and data_exists:
                # 기존 시스템 발견 로깅 (비동기)
                await self._log_existing_system_found_async()
                print("✅ 기존 벡터 인덱스와 데이터 발견 - 초기화 생략 (비동기)")
                return False
            
            # 인덱스는 있지만 데이터 없음
            if not isinstance(index_exists, Exception) and index_exists:
                await self._log_index_exists_no_data_async()
                print("⚠️ 인덱스는 있지만 데이터 없음 - 초기화 필요 (비동기)")
                return True
                
        except Exception as e:
            # 인덱스 없음 로깅 (비동기)
            await self._log_no_index_found_async()
            print(f"📄 벡터 인덱스 없음 - 초기화 필요 (비동기)")
            return True

    async def get_available_templates(self):
        """사용 가능한 템플릿 목록 (CrewAI 기반 로깅 추가 - 비동기 처리)"""
        
        # CrewAI Task 생성
        template_task = Task(
            description="""
            템플릿 폴더에서 사용 가능한 JSX 템플릿 파일들을 스캔하고 목록을 생성하세요.
            
            **스캔 요구사항:**
            1. 'jsx_template' 폴더 존재 여부 확인
            2. .jsx 확장자를 가진 파일들 검색
            3. 파일명 유효성 검증
            4. 기본 템플릿 목록 준비 (폴더가 없는 경우)
            
            **기본 템플릿:**
            - Section01.jsx, Section03.jsx, Section06.jsx, Section08.jsx
            
            스캔 결과를 상세히 보고하세요.
            """,
            expected_output="사용 가능한 JSX 템플릿 파일 목록",
            agent=self.template_loader_agent
        )
        
        # CrewAI Crew 생성 및 실행
        template_crew = Crew(
            agents=[self.template_loader_agent],
            tasks=[template_task],
            process=Process.sequential,
            verbose=True
        )
        
        # CrewAI 비동기 실행
        crew_result = await asyncio.get_event_loop().run_in_executor(
            None, template_crew.kickoff
        )
        
        # 실제 템플릿 로딩 수행 (비동기)
        templates_dir = "jsx_template"
        
        if not os.path.exists(templates_dir):
            default_templates = ["Section01.jsx", "Section03.jsx", "Section06.jsx", "Section08.jsx"]
            
            # 기본 템플릿 사용 로깅 (비동기)
            await self._log_default_templates_async(templates_dir, default_templates)
            return default_templates
        
        # 파일 목록 읽기 (비동기)
        template_files = await asyncio.get_event_loop().run_in_executor(
            None, lambda: [f for f in os.listdir(templates_dir) if f.endswith('.jsx')]
        )
        
        final_templates = template_files if template_files else ["Section01.jsx", "Section03.jsx", "Section06.jsx"]
        
        # 템플릿 로드 완료 로깅 (비동기)
        await self._log_templates_loaded_async(templates_dir, template_files, final_templates)
        
        return final_templates

    async def analyze_template_requirements(self, template_files: List[str]) -> List[Dict]:
        """템플릿 요구사항 분석 (CrewAI 기반 로깅 추가 - 비동기 처리)"""
        
        # CrewAI Task 생성
        analysis_task = Task(
            description=f"""
            제공된 {len(template_files)}개의 JSX 템플릿 파일들을 분석하여 각각의 요구사항을 도출하세요.
            
            **분석 대상 템플릿:** {', '.join(template_files)}
            
            **분석 항목:**
            1. 각 템플릿의 이미지 요구사항 (메인 이미지, 서브 이미지)
            2. 예상 이미지 개수
            3. 레이아웃 특성 및 구조적 요구사항
            
            **출력 형식:**
            각 템플릿별로 다음 정보를 포함:
            - template: 템플릿 파일명
            - image_requirements: 이미지 요구사항 상세
            - total_estimated: 예상 총 이미지 개수
            
            분석 결과를 상세히 보고하세요.
            """,
            expected_output="템플릿별 요구사항 분석 결과 목록",
            agent=self.requirement_analyzer_agent
        )
        
        # CrewAI Crew 생성 및 실행
        analysis_crew = Crew(
            agents=[self.requirement_analyzer_agent],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=True
        )
        
        # CrewAI 비동기 실행
        crew_result = await asyncio.get_event_loop().run_in_executor(
            None, analysis_crew.kickoff
        )
        
        # 실제 요구사항 분석 수행 (비동기)
        requirements = await asyncio.get_event_loop().run_in_executor(
            None, self._analyze_requirements_sync, template_files
        )
        
        # 요구사항 분석 로깅 (비동기)
        await self._log_requirements_analysis_async(template_files, requirements)
        
        return requirements

    async def create_magazine_data(self, magazine_content, image_analysis_results: List[Dict]) -> Dict:
        """PDF 벡터 데이터 기반 매거진 데이터 생성 (CrewAI 통합 로깅 시스템 - 비동기 처리)"""
        print("=== PDF 벡터 기반 다중 에이전트 매거진 생성 시작 (비동기 처리) ===")
        
        # CrewAI Task들 생성
        data_prep_task = self._create_data_prep_task(magazine_content, image_analysis_results)
        coordination_task = self._create_coordination_task()
        
        # CrewAI Crew 생성
        magazine_crew = Crew(
            agents=[self.data_prep_agent, self.coordination_agent],
            tasks=[data_prep_task, coordination_task],
            process=Process.sequential,
            verbose=True
        )
        
        # 전체 프로세스 시작 로깅 (비동기)
        await self._log_process_start_async(magazine_content, image_analysis_results)
        
        # CrewAI 비동기 실행
        crew_result = await asyncio.get_event_loop().run_in_executor(
            None, magazine_crew.kickoff
        )
        
        # 벡터 시스템 확인 및 필요시에만 초기화 (비동기)
        should_init = await self.should_initialize_vector_system()
        if should_init:
            print("\n=== PDF 벡터 시스템 초기화 (필요한 경우만) ===")
            await asyncio.get_event_loop().run_in_executor(
                None, self.vector_manager.process_pdf_templates, "templates"
            )
        else:
            print("\n=== 기존 벡터 데이터 사용 ===")
        
        # 기본 데이터 준비 (비동기 병렬 처리)
        available_templates, template_requirements = await asyncio.gather(
            self.get_available_templates(),
            self.analyze_template_requirements(await self.get_available_templates())
        )
        
        image_urls = [result.get('image_url', '') for result in image_analysis_results if result.get('image_url')]
        image_locations = [result.get('location', '') for result in image_analysis_results if result.get('location')]
        
        # 데이터 준비 완료 로깅 (비동기)
        await self._log_data_prep_complete_async(available_templates, image_urls, template_requirements)
        
        # 다중 에이전트 처리 (비동기 병렬 실행)
        print("\n=== 다중 에이전트 병렬 처리 시작 ===")
        
        # OrgAgent와 BindingAgent 병렬 실행
        org_task = self._run_org_agent_async(magazine_content, available_templates)
        binding_task = self._run_binding_agent_async(image_urls, image_locations, template_requirements)
        
        text_mapping, image_distribution = await asyncio.gather(org_task, binding_task)
        
        # CoordinatorAgent 실행
        final_template_data = await self._run_coordinator_agent_async(text_mapping, image_distribution)
        
        # 벡터 데이터 메타정보 추가
        final_template_data["vector_enhanced"] = True
        final_template_data["crewai_enhanced"] = True
        final_template_data["async_processed"] = True
        final_template_data["pdf_sources"] = await asyncio.get_event_loop().run_in_executor(
            None, self._extract_pdf_sources, text_mapping, image_distribution
        )
        
        # 최종 완료 로깅 (비동기)
        await self._log_final_complete_async(final_template_data)
        
        print("✅ PDF 벡터 기반 매거진 데이터 생성 완료 (비동기 처리)")
        return final_template_data

    async def _run_org_agent_async(self, magazine_content, available_templates):
        """OrgAgent 비동기 실행"""
        print("\n=== OrgAgent: PDF 벡터 기반 텍스트 처리 (비동기) ===")
        
        # OrgAgent가 비동기 메서드를 가지고 있다면 직접 호출, 아니면 executor 사용
        if hasattr(self.org_agent, 'process_content') and asyncio.iscoroutinefunction(self.org_agent.process_content):
            text_mapping = await self.org_agent.process_content(magazine_content, available_templates)
        else:
            text_mapping = await asyncio.get_event_loop().run_in_executor(
                None, self.org_agent.process_content, magazine_content, available_templates
            )
        
        # OrgAgent 완료 로깅 (비동기)
        await self._log_org_agent_complete_async(text_mapping)
        
        return text_mapping

    async def _run_binding_agent_async(self, image_urls, image_locations, template_requirements):
        """BindingAgent 비동기 실행"""
        print("\n=== BindingAgent: PDF 벡터 기반 이미지 처리 (비동기) ===")
        
        # BindingAgent가 비동기 메서드를 가지고 있다면 직접 호출, 아니면 executor 사용
        if hasattr(self.binding_agent, 'process_images') and asyncio.iscoroutinefunction(self.binding_agent.process_images):
            image_distribution = await self.binding_agent.process_images(image_urls, image_locations, template_requirements)
        else:
            image_distribution = await asyncio.get_event_loop().run_in_executor(
                None, self.binding_agent.process_images, image_urls, image_locations, template_requirements
            )
        
        # BindingAgent 완료 로깅 (비동기)
        await self._log_binding_agent_complete_async(image_distribution)
        
        return image_distribution

    async def _run_coordinator_agent_async(self, text_mapping, image_distribution):
        """CoordinatorAgent 비동기 실행"""
        print("\n=== CoordinatorAgent: 벡터 기반 결과 통합 (비동기) ===")
        
        # CoordinatorAgent가 비동기 메서드를 가지고 있다면 직접 호출, 아니면 executor 사용
        if hasattr(self.coordinator_agent, 'coordinate_magazine_creation') and asyncio.iscoroutinefunction(self.coordinator_agent.coordinate_magazine_creation):
            final_template_data = await self.coordinator_agent.coordinate_magazine_creation(text_mapping, image_distribution)
        else:
            final_template_data = await asyncio.get_event_loop().run_in_executor(
                None, self.coordinator_agent.coordinate_magazine_creation, text_mapping, image_distribution
            )
        
        # CoordinatorAgent 완료 로깅 (비동기)
        await self._log_coordinator_agent_complete_async(final_template_data)
        
        return final_template_data

    # 비동기 로깅 메서드들
    async def _log_initialization_start_async(self, template_folder: str):
        """초기화 시작 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_VectorInit",
                agent_role="벡터 시스템 초기화 관리자",
                task_description=f"PDF 벡터 시스템 초기화: {template_folder} 폴더 처리",
                final_answer="벡터 시스템 초기화 시작",
                reasoning_process="PDF 템플릿 처리 및 Azure Cognitive Search 인덱스 초기화",
                execution_steps=[
                    "CrewAI 초기화 에이전트 생성",
                    "Azure Cognitive Search 인덱스 초기화",
                    "PDF 템플릿 처리 및 벡터화 시작"
                ],
                raw_input={"template_folder": template_folder},
                performance_metrics={
                    "initialization_started": True,
                    "template_folder": template_folder,
                    "crewai_enabled": True,
                    "async_processing": True
                }
            )
        )

    async def _log_initialization_complete_async(self, template_folder: str, crew_result):
        """초기화 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_VectorInit",
                agent_role="벡터 시스템 초기화 관리자",
                task_description="PDF 벡터 시스템 초기화 완료",
                final_answer="PDF 벡터 시스템 초기화 성공적으로 완료",
                reasoning_process="CrewAI 기반 비동기 Azure Cognitive Search 인덱스 생성 및 PDF 템플릿 벡터화 완료",
                execution_steps=[
                    "CrewAI 초기화 태스크 비동기 실행 완료",
                    "Azure Cognitive Search 인덱스 생성 완료",
                    "PDF 템플릿 처리 및 벡터화 완료",
                    "벡터 시스템 활성화"
                ],
                raw_output={
                    "initialization_success": True,
                    "crew_result": str(crew_result)[:500]
                },
                performance_metrics={
                    "initialization_completed": True,
                    "vector_system_active": True,
                    "template_folder_processed": template_folder,
                    "crewai_execution_success": True,
                    "async_processing": True
                }
            )
        )

    async def _log_initialization_error_async(self, template_folder: str, error: str):
        """초기화 실패 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_VectorInit",
                agent_role="벡터 시스템 초기화 관리자",
                task_description="PDF 벡터 시스템 초기화 실패",
                final_answer=f"ERROR: 벡터 시스템 초기화 실패 - {error}",
                reasoning_process="CrewAI 기반 비동기 벡터 시스템 초기화 중 예외 발생",
                error_logs=[{"error": error, "template_folder": template_folder}],
                performance_metrics={
                    "initialization_failed": True,
                    "error_occurred": True,
                    "crewai_execution_failed": True,
                    "async_processing": True
                }
            )
        )

    async def _check_index_exists_async(self) -> bool:
        """인덱스 존재 여부 확인 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._check_index_exists_sync()
        )

    async def _check_data_exists_async(self) -> bool:
        """데이터 존재 여부 확인 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._check_data_exists_sync()
        )

    def _check_index_exists_sync(self) -> bool:
        """인덱스 존재 여부 확인 (동기)"""
        try:
            index_client = self.vector_manager.search_index_client
            index_client.get_index(self.vector_manager.search_index_name)
            return True
        except:
            return False

    def _check_data_exists_sync(self) -> bool:
        """데이터 존재 여부 확인 (동기)"""
        try:
            search_client = self.vector_manager.search_client
            results = search_client.search("*", top=1)
            for _ in results:
                return True
            return False
        except:
            return False

    async def _log_existing_system_found_async(self):
        """기존 시스템 발견 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_VectorCheck",
                agent_role="벡터 시스템 상태 확인자",
                task_description="벡터 시스템 초기화 필요성 검사",
                final_answer="기존 벡터 인덱스와 데이터 발견 - 초기화 불필요",
                reasoning_process="CrewAI 기반 비동기 기존 Azure Cognitive Search 인덱스에 데이터가 존재함",
                performance_metrics={
                    "existing_system_found": True,
                    "initialization_required": False,
                    "data_available": True,
                    "crewai_check_completed": True,
                    "async_processing": True
                }
            )
        )

    async def _log_index_exists_no_data_async(self):
        """인덱스 있지만 데이터 없음 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_VectorCheck",
                agent_role="벡터 시스템 상태 확인자",
                task_description="벡터 시스템 상태 확인",
                final_answer="인덱스는 있지만 데이터 없음 - 초기화 필요",
                reasoning_process="CrewAI 기반 비동기 Azure Cognitive Search 인덱스는 존재하지만 데이터가 없음",
                performance_metrics={
                    "index_exists": True,
                    "data_available": False,
                    "initialization_required": True,
                    "crewai_check_completed": True,
                    "async_processing": True
                }
            )
        )

    async def _log_no_index_found_async(self):
        """인덱스 없음 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_VectorCheck",
                agent_role="벡터 시스템 상태 확인자",
                task_description="벡터 시스템 상태 확인",
                final_answer="벡터 인덱스 없음 - 초기화 필요",
                reasoning_process="CrewAI 기반 비동기 Azure Cognitive Search 인덱스가 존재하지 않음",
                performance_metrics={
                    "index_exists": False,
                    "initialization_required": True,
                    "first_time_setup": True,
                    "crewai_check_completed": True,
                    "async_processing": True
                }
            )
        )

    async def _log_default_templates_async(self, templates_dir: str, default_templates: List[str]):
        """기본 템플릿 사용 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_TemplateLoader",
                agent_role="템플릿 로더",
                task_description="사용 가능한 템플릿 목록 조회",
                final_answer=f"템플릿 폴더 없음 - 기본 템플릿 {len(default_templates)}개 사용",
                reasoning_process=f"CrewAI 기반 비동기 템플릿 폴더 {templates_dir}가 존재하지 않아 기본 템플릿 목록 반환",
                raw_output=default_templates,
                performance_metrics={
                    "templates_dir_exists": False,
                    "default_templates_used": True,
                    "template_count": len(default_templates),
                    "crewai_execution_completed": True,
                    "async_processing": True
                }
            )
        )

    async def _log_templates_loaded_async(self, templates_dir: str, template_files: List[str], final_templates: List[str]):
        """템플릿 로드 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_TemplateLoader",
                agent_role="템플릿 로더",
                task_description="템플릿 폴더에서 JSX 템플릿 목록 조회",
                final_answer=f"템플릿 {len(final_templates)}개 로드 완료",
                reasoning_process=f"CrewAI 기반 비동기 템플릿 폴더 {templates_dir}에서 JSX 파일 검색 및 목록 생성",
                raw_input={"templates_dir": templates_dir},
                raw_output=final_templates,
                performance_metrics={
                    "templates_dir_exists": True,
                    "jsx_files_found": len(template_files),
                    "final_template_count": len(final_templates),
                    "fallback_used": len(template_files) == 0,
                    "crewai_execution_completed": True,
                    "async_processing": True
                }
            )
        )

    async def _log_requirements_analysis_async(self, template_files: List[str], requirements: List[Dict]):
        """요구사항 분석 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_RequirementAnalyzer",
                agent_role="템플릿 요구사항 분석자",
                task_description=f"{len(template_files)}개 템플릿 요구사항 분석",
                final_answer=f"템플릿 요구사항 분석 완료: {len(requirements)}개 템플릿",
                reasoning_process="CrewAI 기반 비동기 각 템플릿별 이미지 요구사항 및 구조 분석",
                execution_steps=[
                    "CrewAI 분석 태스크 비동기 실행",
                    "템플릿 파일 목록 분석",
                    "이미지 요구사항 계산",
                    "구조적 요구사항 정의"
                ],
                raw_input=template_files,
                raw_output=requirements,
                performance_metrics={
                    "templates_analyzed": len(template_files),
                    "requirements_generated": len(requirements),
                    "avg_images_per_template": 2,
                    "crewai_execution_completed": True,
                    "async_processing": True
                }
            )
        )

    async def _log_process_start_async(self, magazine_content, image_analysis_results: List[Dict]):
        """프로세스 시작 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager",
                agent_role="다중 에이전트 템플릿 관리자",
                task_description="PDF 벡터 기반 매거진 데이터 생성 프로세스 시작",
                final_answer="CrewAI 기반 비동기 다중 에이전트 매거진 생성 프로세스 시작",
                reasoning_process=f"CrewAI 통합 비동기 매거진 콘텐츠와 {len(image_analysis_results)}개 이미지 분석 결과를 활용한 매거진 데이터 생성",
                execution_steps=[
                    "CrewAI 에이전트 및 태스크 생성",
                    "비동기 벡터 시스템 확인",
                    "비동기 템플릿 준비",
                    "다중 에이전트 병렬 실행 준비"
                ],
                raw_input={
                    "magazine_content": str(magazine_content)[:500],
                    "image_analysis_count": len(image_analysis_results)
                },
                performance_metrics={
                    "process_started": True,
                    "image_analysis_count": len(image_analysis_results),
                    "crewai_enabled": True,
                    "async_processing": True
                }
            )
        )

    async def _log_data_prep_complete_async(self, available_templates: List[str], image_urls: List[str], template_requirements: List[Dict]):
        """데이터 준비 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_DataPrep",
                agent_role="데이터 준비 관리자",
                task_description="매거진 생성을 위한 기본 데이터 준비",
                final_answer=f"CrewAI 기반 비동기 데이터 준비 완료: 템플릿 {len(available_templates)}개, 이미지 {len(image_urls)}개",
                reasoning_process="CrewAI 통합 비동기 템플릿, 이미지, 요구사항 데이터 수집 및 정리",
                raw_output={
                    "available_templates": available_templates,
                    "template_requirements": template_requirements,
                    "image_urls": image_urls
                },
                performance_metrics={
                    "templates_prepared": len(available_templates),
                    "images_prepared": len(image_urls),
                    "requirements_prepared": len(template_requirements),
                    "pdf_vector_active": True,
                    "crewai_data_prep_completed": True,
                    "async_processing": True
                }
            )
        )

    async def _log_org_agent_complete_async(self, text_mapping: Dict):
        """OrgAgent 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_OrgCoordination",
                agent_role="OrgAgent 조율자",
                task_description="OrgAgent PDF 벡터 기반 텍스트 처리 완료",
                final_answer=f"OrgAgent 비동기 텍스트 처리 완료: {text_mapping.get('total_sections', 0)}개 섹션",
                reasoning_process="OrgAgent가 PDF 벡터 데이터를 활용하여 비동기 텍스트 레이아웃 및 구조 처리 완료",
                raw_output=text_mapping,
                performance_metrics={
                    "org_agent_completed": True,
                    "sections_processed": text_mapping.get('total_sections', 0),
                    "vector_enhanced": text_mapping.get('vector_enhanced', False),
                    "crewai_coordinated": True,
                    "async_processing": True
                }
            )
        )

    async def _log_binding_agent_complete_async(self, image_distribution: Dict):
        """BindingAgent 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_BindingCoordination",
                agent_role="BindingAgent 조율자",
                task_description="BindingAgent PDF 벡터 기반 이미지 처리 완료",
                final_answer=f"BindingAgent 비동기 이미지 처리 완료: {len(image_distribution.get('image_distribution', {}))}개 템플릿",
                reasoning_process="BindingAgent가 PDF 벡터 데이터를 활용하여 비동기 이미지 배치 및 분배 처리 완료",
                raw_output=image_distribution,
                performance_metrics={
                    "binding_agent_completed": True,
                    "templates_processed": len(image_distribution.get('image_distribution', {})),
                    "vector_enhanced": image_distribution.get('vector_enhanced', False),
                    "crewai_coordinated": True,
                    "async_processing": True
                }
            )
        )

    async def _log_coordinator_agent_complete_async(self, final_template_data: Dict):
        """CoordinatorAgent 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_CoordinatorCoordination",
                agent_role="CoordinatorAgent 조율자",
                task_description="CoordinatorAgent 벡터 기반 결과 통합 완료",
                final_answer=f"CoordinatorAgent 비동기 통합 완료: {len(final_template_data.get('content_sections', []))}개 최종 섹션",
                reasoning_process="CoordinatorAgent가 OrgAgent와 BindingAgent 결과를 비동기 통합하여 최종 매거진 구조 생성",
                raw_output=final_template_data,
                performance_metrics={
                    "coordinator_agent_completed": True,
                    "final_sections": len(final_template_data.get('content_sections', [])),
                    "integration_quality": final_template_data.get('integration_metadata', {}).get('integration_quality_score', 0),
                    "crewai_coordinated": True,
                    "async_processing": True
                }
            )
        )

    async def _log_final_complete_async(self, final_template_data: Dict):
        """최종 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager",
                agent_role="다중 에이전트 템플릿 관리자",
                task_description="PDF 벡터 기반 매거진 데이터 생성 완료",
                final_answer=f"CrewAI 기반 비동기 매거진 데이터 생성 완료: {len(final_template_data.get('content_sections', []))}개 섹션, 벡터 강화됨",
                reasoning_process="OrgAgent, BindingAgent, CoordinatorAgent 비동기 병렬 실행으로 PDF 벡터 데이터 기반 매거진 생성 완료",
                execution_steps=[
                    "CrewAI 기반 비동기 벡터 시스템 확인 및 초기화",
                    "비동기 기본 데이터 준비",
                    "OrgAgent 비동기 텍스트 처리",
                    "BindingAgent 비동기 이미지 처리",
                    "CoordinatorAgent 비동기 결과 통합",
                    "벡터 메타정보 추가"
                ],
                raw_output=final_template_data,
                performance_metrics={
                    "total_process_completed": True,
                    "final_sections_count": len(final_template_data.get('content_sections', [])),
                    "vector_enhanced": True,
                    "crewai_enhanced": True,
                    "async_enhanced": True,
                    "pdf_sources_extracted": len(final_template_data.get('pdf_sources', {}).get('text_sources', [])) + len(final_template_data.get('pdf_sources', {}).get('image_sources', [])),
                    "all_agents_completed": True,
                    "integration_quality_score": final_template_data.get('integration_metadata', {}).get('integration_quality_score', 0)
                }
            )
        )

    # 동기 버전 메서드들 (호환성 유지)
    def _analyze_requirements_sync(self, template_files: List[str]) -> List[Dict]:
        """템플릿 요구사항 분석 (동기 버전)"""
        requirements = []
        for template_file in template_files:
            requirement = {
                "template": template_file,
                "image_requirements": {
                    "main_images": 1,
                    "sub_images": True,
                    "total_estimated": 2
                }
            }
            requirements.append(requirement)
        return requirements

    def _create_data_prep_task(self, magazine_content, image_analysis_results: List[Dict]) -> Task:
        """데이터 준비 태스크 생성"""
        return Task(
            description=f"""
            매거진 생성에 필요한 모든 데이터를 수집, 정리, 검증하여 다중 에이전트 시스템이 효율적으로 작동할 수 있도록 준비하세요.
            
            **처리 대상:**
            - 매거진 콘텐츠: {len(str(magazine_content))} 문자
            - 이미지 분석 결과: {len(image_analysis_results)}개
            
            **데이터 준비 요구사항:**
            1. 매거진 콘텐츠 구조 분석 및 정리
            2. 이미지 분석 결과에서 유효한 URL 추출
            3. 템플릿 요구사항과 이미지 데이터 매칭
            4. 데이터 품질 검증 및 정리
            
            **출력 형식:**
            - 정리된 매거진 콘텐츠 구조
            - 검증된 이미지 URL 목록
            - 템플릿별 데이터 매핑 정보
            
            데이터 준비 과정을 상세히 보고하세요.
            """,
            expected_output="매거진 생성을 위한 완전히 준비된 데이터 패키지",
            agent=self.data_prep_agent
        )

    def _create_coordination_task(self) -> Task:
        """조율 태스크 생성"""
        return Task(
            description="""
            OrgAgent, BindingAgent, CoordinatorAgent의 순차적 실행을 관리하고 각 단계의 결과를 최적화하여 최고 품질의 매거진 데이터를 생성하세요.
            
            **조율 요구사항:**
            1. 에이전트 간 데이터 흐름 최적화
            2. 각 단계별 품질 검증
            3. 오류 발생 시 복구 전략 실행
            4. 최종 결과 품질 보증
            
            **관리 대상:**
            - OrgAgent: 텍스트 레이아웃 처리
            - BindingAgent: 이미지 배치 처리
            - CoordinatorAgent: 결과 통합
            
            **최종 목표:**
            - 고품질 매거진 구조 데이터 생성
            - 모든 에이전트 결과의 완벽한 통합
            - JSX 구현을 위한 완전한 스펙 제공
            
            조율 과정과 결과를 상세히 보고하세요.
            """,
            expected_output="다중 에이전트 조율 완료 보고서 및 최종 매거진 데이터",
            agent=self.coordination_agent
        )

    # 기존 메서드들 유지 (변경 없음)
    def _extract_pdf_sources(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """사용된 PDF 소스 정보 추출 (로깅 추가)"""
        sources = {
            "text_sources": [],
            "image_sources": []
        }
        
        # 텍스트 소스 추출
        if isinstance(text_mapping, dict) and "text_mapping" in text_mapping:
            for section in text_mapping["text_mapping"]:
                if isinstance(section, dict) and "layout_source" in section:
                    source = section["layout_source"]
                    if source and source != "default" and source not in sources["text_sources"]:
                        sources["text_sources"].append(source)
        
        # 이미지 소스 추출
        if isinstance(image_distribution, dict) and "template_distributions" in image_distribution:
            for dist in image_distribution["template_distributions"]:
                if isinstance(dist, dict) and "layout_source" in dist:
                    source = dist["layout_source"]
                    if source and source != "default" and source not in sources["image_sources"]:
                        sources["image_sources"].append(source)
        
        return sources

    # 동기 버전 메서드들 (호환성 보장)
    def initialize_vector_system_sync(self, template_folder: str = "templates"):
        """벡터 시스템 초기화 (동기 버전 - 호환성 유지)"""
        return asyncio.run(self.initialize_vector_system(template_folder))

    def should_initialize_vector_system_sync(self) -> bool:
        """벡터 시스템 초기화 필요 여부 확인 (동기 버전 - 호환성 유지)"""
        return asyncio.run(self.should_initialize_vector_system())

    def get_available_templates_sync(self):
        """사용 가능한 템플릿 목록 (동기 버전 - 호환성 유지)"""
        return asyncio.run(self.get_available_templates())

    def analyze_template_requirements_sync(self, template_files: List[str]) -> List[Dict]:
        """템플릿 요구사항 분석 (동기 버전 - 호환성 유지)"""
        return asyncio.run(self.analyze_template_requirements(template_files))

    def create_magazine_data_sync(self, magazine_content, image_analysis_results: List[Dict]) -> Dict:
        """PDF 벡터 데이터 기반 매거진 데이터 생성 (동기 버전 - 호환성 유지)"""
        return asyncio.run(self.create_magazine_data(magazine_content, image_analysis_results))
