import os
from typing import Dict, List
from crewai import Crew
from custom_llm import get_azure_llm
from agents.Editor.OrgAgent import OrgAgent
from agents.Editor.BindingAgent import BindingAgent
from agents.Editor.CoordinatorAgent import CoordinatorAgent
from utils.pdf_vector_manager import PDFVectorManager

class MultiAgentTemplateManager:
    """PDF 벡터 데이터 기반 다중 에이전트 템플릿 관리자"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        self.org_agent = OrgAgent()
        self.binding_agent = BindingAgent()
        self.coordinator_agent = CoordinatorAgent()
        self.vector_manager = PDFVectorManager()
        
    def initialize_vector_system(self, template_folder: str = "templates"):
        """벡터 시스템 초기화 - PDF 처리 및 인덱싱"""
        print("=== PDF 벡터 시스템 초기화 ===")
        
        # Azure Cognitive Search 인덱스 초기화
        self.vector_manager.initialize_search_index()
        
        # PDF 템플릿 처리 및 벡터화
        self.vector_manager.process_pdf_templates(template_folder)
        
        print("✅ PDF 벡터 시스템 초기화 완료")

    def should_initialize_vector_system(self) -> bool:
        """벡터 시스템 초기화 필요 여부 확인"""
        try:
            # 기존 인덱스 존재 여부 확인
            index_client = self.vector_manager.search_index_client
            index_client.get_index(self.vector_manager.search_index_name)
            
            # 인덱스에 데이터가 있는지 확인
            search_client = self.vector_manager.search_client
            results = search_client.search("*", top=1)
            
            # 결과가 있으면 초기화 불필요
            for _ in results:
                print("✅ 기존 벡터 인덱스와 데이터 발견 - 초기화 생략")
                return False
            
            print("⚠️ 인덱스는 있지만 데이터 없음 - 초기화 필요")
            return True
            
        except Exception as e:
            print(f"📄 벡터 인덱스 없음 - 초기화 필요")
            return True
        
    def get_available_templates(self):
        """사용 가능한 템플릿 목록"""
        templates_dir = "templates"
        if not os.path.exists(templates_dir):
            return ["Section01.jsx", "Section03.jsx", "Section06.jsx", "Section08.jsx"]
        
        template_files = [f for f in os.listdir(templates_dir) if f.endswith('.jsx')]
        return template_files if template_files else ["Section01.jsx", "Section03.jsx", "Section06.jsx"]
    
    def analyze_template_requirements(self, template_files: List[str]) -> List[Dict]:
        """템플릿 요구사항 분석"""
        requirements = []
        for template_file in template_files:
            requirements.append({
                "template": template_file,
                "image_requirements": {
                    "main_images": 1,
                    "sub_images": True,
                    "total_estimated": 2
                }
            })
        return requirements
    
    def create_magazine_data(self, magazine_content, image_analysis_results: List[Dict]) -> Dict:
        """PDF 벡터 데이터 기반 매거진 데이터 생성"""
        
        print("=== PDF 벡터 기반 다중 에이전트 매거진 생성 시작 ===")
        
        # 벡터 시스템 확인 및 필요시에만 초기화
        if self.should_initialize_vector_system():
            print("\n=== PDF 벡터 시스템 초기화 (필요한 경우만) ===")
            self.vector_manager.process_pdf_templates("templates")
        else:
            print("\n=== 기존 벡터 데이터 사용 ===")
        
        # 기본 데이터 준비
        available_templates = self.get_available_templates()
        template_requirements = self.analyze_template_requirements(available_templates)
        
        image_urls = [result.get('image_url', '') for result in image_analysis_results if result.get('image_url')]
        image_locations = [result.get('location', '') for result in image_analysis_results if result.get('location')]
        
        print(f"- 템플릿: {len(available_templates)}개")
        print(f"- 이미지: {len(image_urls)}개")
        print(f"- PDF 벡터 데이터 활용: 활성화")
        
        # 1. PDF 벡터 기반 텍스트 처리
        print("\n=== OrgAgent: PDF 벡터 기반 텍스트 처리 ===")
        text_mapping = self.org_agent.process_content(magazine_content, available_templates)
        
        # 2. PDF 벡터 기반 이미지 처리
        print("\n=== BindingAgent: PDF 벡터 기반 이미지 처리 ===")
        image_distribution = self.binding_agent.process_images(image_urls, image_locations, template_requirements)
        
        # 3. 결과 통합
        print("\n=== CoordinatorAgent: 벡터 기반 결과 통합 ===")
        final_template_data = self.coordinator_agent.coordinate_magazine_creation(text_mapping, image_distribution)
        
        # 벡터 데이터 메타정보 추가
        final_template_data["vector_enhanced"] = True
        final_template_data["pdf_sources"] = self._extract_pdf_sources(text_mapping, image_distribution)
        
        print("✅ PDF 벡터 기반 매거진 데이터 생성 완료")
        return final_template_data
    
    def _extract_pdf_sources(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """사용된 PDF 소스 정보 추출"""
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
    
    def generate_react_app(self, template_data: Dict, file_manager, project_name: str):
        """React 앱 생성"""
        project_folder = file_manager.create_project_folder(project_name)
        src_folder, components_folder = file_manager.create_react_app(project_folder)
        
        # template_data.json 저장
        template_data_path = os.path.join(project_folder, "template_data.json")
        file_manager.save_json(template_data, template_data_path)
        
        # 벡터 데이터 메타정보 저장
        if template_data.get("vector_enhanced"):
            vector_info_path = os.path.join(project_folder, "vector_sources.json")
            vector_info = {
                "enhanced_by_pdf_vectors": True,
                "pdf_sources": template_data.get("pdf_sources", {}),
                "generation_timestamp": file_manager._get_current_timestamp() if hasattr(file_manager, '_get_current_timestamp') else "2025-01-26"
            }
            file_manager.save_json(vector_info, vector_info_path)
        
        print(f"✅ PDF 벡터 기반 React 앱 생성: {project_folder}")
        return project_folder, template_data_path
