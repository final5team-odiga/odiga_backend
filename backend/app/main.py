import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from agents.image_analyzer import ImageAnalyzerAgent
from agents.contents.content_creator import ContentCreatorV2Crew
from agents.Editor.template_manager import MultiAgentTemplateManager
from agents.jsxcreate.jsx_generator import JSXCreatorAgent
from utils.blob_storage import BlobStorageManager
from utils.file_manager import FileManager
from utils.agent_decision_logger import get_agent_logger
from crewai import Crew
from pathlib import Path

# nest_asyncio 적용 (이벤트 루프 중첩 허용)
nest_asyncio.apply()

dotenv_path = Path(r'C:\Users\EL0021\Desktop\odiga_agent\.env')

# 환경 변수 로드
load_dotenv(dotenv_path=dotenv_path, override=True)

async def main_async():
    print("=== CrewAI 여행 매거진 생성 시스템 (에이전트 학습 기반) ===")
    
    # 전역 로거 초기화
    logger = get_agent_logger()
    print(f"📝 에이전트 의사결정 로깅 시스템 초기화 완료")
    
    try:
        # 초기화
        blob_manager = BlobStorageManager()
        file_manager = FileManager(output_folder=os.getenv("OUTPUT_FOLDER", "./output"))
        
        # **중요: 출력 폴더 강제 생성**
        os.makedirs(file_manager.output_folder, exist_ok=True)
        print(f"✅ 출력 폴더 생성: {file_manager.output_folder}")
        
        # 데이터 수집
        images = blob_manager.get_images()
        text_blobs = blob_manager.get_texts()
        texts = [blob_manager.read_text_file(text_blob) for text_blob in text_blobs]
        
        print(f"이미지: {len(images)}개, 텍스트: {len(texts)}개")
        
        # 에이전트 초기화
        image_analyzer = ImageAnalyzerAgent()
        content_creator_v2 = ContentCreatorV2Crew()
        template_manager = MultiAgentTemplateManager()
        jsx_creator = JSXCreatorAgent()
        
        # 전체 프로세스 시작 로깅
        logger.log_agent_decision(
            agent_name="MainProcess",
            agent_role="전체 시스템 조율자",
            input_data={
                "images_count": len(images),
                "texts_count": len(texts),
                "output_folder": file_manager.output_folder
            },
            decision_process={
                "step": "system_initialization",
                "agents_initialized": ["ImageAnalyzer", "ContentCreator", "TemplateManager", "JSXCreator"]
            },
            output_result={
                "initialization_success": True,
                "data_loaded": True
            },
            reasoning="시스템 초기화 및 데이터 로드 완료, 에이전트 학습 시스템 활성화",
            confidence_score=1.0
        )
        
        # PDF 벡터 시스템 초기화
        print("\n=== PDF 벡터 시스템 초기화 ===")
        if template_manager.should_initialize_vector_system():
            print("\n=== PDF 벡터 시스템 초기화 (최초 실행) ===")
            template_manager.initialize_vector_system("templates")
        else:
            print("\n=== 기존 벡터 인덱스 사용 ===")
        
        # 1. 이미지 분석 (비동기 호환)
        print("\n=== 이미지 분석 ===")
        crew = Crew(agents=[image_analyzer.create_agent()], verbose=True)
        
        if hasattr(image_analyzer, 'analyze_images_async'):
            image_analysis_results = await image_analyzer.analyze_images_async(images, crew)
        else:
            loop = asyncio.get_event_loop()
            image_analysis_results = await loop.run_in_executor(
                None, image_analyzer.analyze_images, images, crew
            )
        
        # 이미지 분석 완료 로깅
        logger.log_agent_interaction(
            source_agent="ImageAnalyzerAgent",
            target_agent="MainProcess",
            interaction_type="analysis_completion",
            data_transferred={
                "analysis_results_count": len(image_analysis_results),
                "images_processed": len(images)
            }
        )
        
        # 분석 결과 저장
        analysis_path = os.path.join(file_manager.output_folder, "image_analysis_results.json")
        file_manager.save_json(image_analysis_results, analysis_path)
        print(f"이미지 분석 결과 저장: {analysis_path}")
        
        # 2. 콘텐츠 생성 (인터뷰 + 에세이 통합) - 학습 기반
        print("\n=== 콘텐츠 생성 (인터뷰 + 에세이) - 학습 기반 ===")
        magazine_content = content_creator_v2.execute_content_creation(texts, image_analysis_results)
        
        # 콘텐츠 생성 완료 로깅
        logger.log_agent_interaction(
            source_agent="ContentCreatorV2Crew",
            target_agent="MainProcess",
            interaction_type="content_generation",
            data_transferred={
                "content_length": len(magazine_content),
                "source_texts": len(texts),
                "learning_applied": True
            }
        )
        
        # 콘텐츠 저장 (JSON 형식)
        content_path = os.path.join(file_manager.output_folder, "magazine_content.json")
        file_manager.save_magazine_content_json(magazine_content, content_path)
        print(f"매거진 콘텐츠 저장: {content_path}")
        
        # 3. PDF 벡터 기반 템플릿 데이터 생성 - 학습 기반
        print("\n=== PDF 벡터 기반 템플릿 데이터 생성 - 학습 기반 ===")
        template_data = template_manager.create_magazine_data(magazine_content, image_analysis_results)
        
        # 템플릿 데이터 저장
        template_path = os.path.join(file_manager.output_folder, "template_data.json")
        file_manager.save_json(template_data, template_path)
        print(f"템플릿 데이터 저장: {template_path}")
        
        # 4. React 앱 기본 구조 생성
        print("\n=== React 앱 기본 구조 생성 ===")
        project_name = "travel-magazine"
        
        # **중요: 프로젝트 폴더 강제 생성**
        project_folder = file_manager.create_project_folder(project_name)
        print(f"프로젝트 폴더 생성: {project_folder}")
        
        # **중요: React 앱 구조 생성**
        file_manager.create_react_app(project_folder)
        print(f"React 앱 구조 생성: {project_folder}")
        
        # **중요: template_data.json을 프로젝트 폴더에 복사**
        template_data_path = os.path.join(project_folder, "template_data.json")
        file_manager.save_json(template_data, template_data_path)
        print(f"프로젝트 템플릿 데이터 저장: {template_data_path}")
        
        # 5. JSX 생성 (학습 기반 비동기 코드 리뷰 포함)
        print("\n=== JSX 생성 - 학습 기반 ===")
        generated_components_data = await jsx_creator.generate_jsx_components_async(template_data_path)
        print(f"JSX 컴포넌트 생성: {len(generated_components_data)}개 (학습 기반)")
        
        # 6. 컴포넌트 저장
        components_folder = os.path.join(project_folder, "src", "components")
        
        # **중요: 컴포넌트 폴더 강제 생성**
        os.makedirs(components_folder, exist_ok=True)
        print(f"✅ 컴포넌트 폴더 생성: {components_folder}")
        
        # **중요: 실제 파일 저장 전 디버깅**
        print(f"📁 저장할 컴포넌트 수: {len(generated_components_data)}")
        for i, component in enumerate(generated_components_data):
            learning_applied = component.get('learning_insights_applied', False)
            print(f"  {i+1}. {component.get('name', 'Unknown')} -> {component.get('file', 'Unknown')} (학습적용: {learning_applied})")
        
        saved_components = jsx_creator.save_jsx_components(generated_components_data, components_folder)
        
        # **중요: 저장 결과 확인**
        print(f"📁 실제 저장된 컴포넌트 수: {len(saved_components)}")
        if os.path.exists(components_folder):
            actual_files = os.listdir(components_folder)
            print(f"📁 실제 생성된 파일들: {actual_files}")
        
        # 7. App.js 생성
        if saved_components:
            app_js = generate_vector_enhanced_app_js(saved_components, template_data)
            app_js_path = os.path.join(project_folder, "src", "App.js")
            file_manager.save_content(app_js, app_js_path)
            print(f"✅ App.js 생성 완료: {app_js_path}")
        
        # 8. 실행 방법 안내
        print("\n=== 매거진 실행 방법 ===")
        print(f"1. 터미널에서 다음 명령어를 실행하세요:")
        print(f"   cd {project_folder}")
        print(f"   npm install")
        print(f"   npm start")
        print(f"2. 웹 브라우저에서 http://localhost:3000 으로 접속하세요.")
        
        # 9. 최종 프로세스 완료 로깅
        final_learning_insights = logger.get_learning_insights("MainProcess")
        
        logger.log_agent_decision(
            agent_name="MainProcess",
            agent_role="전체 시스템 조율자",
            input_data={
                "total_components_generated": len(generated_components_data),
                "total_components_saved": len(saved_components)
            },
            decision_process={
                "step": "system_completion",
                "learning_insights_generated": len(final_learning_insights.get('recommendations', []))
            },
            output_result={
                "magazine_generated": True,
                "react_app_created": True,
                "learning_system_active": True,
                "final_success": True
            },
            reasoning="전체 매거진 생성 프로세스 완료, 에이전트 학습 데이터 축적",
            confidence_score=0.95,
            performance_metrics={
                "overall_success_rate": len(saved_components) / max(len(generated_components_data), 1),
                "learning_insights_count": len(final_learning_insights.get('recommendations', [])),
                "system_efficiency": 1.0
            }
        )
      
        
        # PDF 소스 정보 출력
        if template_data.get("vector_enhanced") and template_data.get("pdf_sources"):
            pdf_sources = template_data["pdf_sources"]
            if pdf_sources.get("text_sources"):
                print(f"📄 텍스트 레이아웃 소스: {', '.join(pdf_sources['text_sources'])}")
            if pdf_sources.get("image_sources"):
                print(f"🖼️ 이미지 레이아웃 소스: {', '.join(pdf_sources['image_sources'])}")
        
        # 학습 인사이트 요약
        total_insights = len(final_learning_insights.get('recommendations', []))
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # 오류 로깅
        logger.log_agent_decision(
            agent_name="MainProcess",
            agent_role="전체 시스템 조율자",
            input_data={"error_occurred": True},
            decision_process={"step": "error_handling"},
            output_result={"system_failure": True, "error_message": str(e)},
            reasoning=f"시스템 실행 중 오류 발생: {str(e)}",
            confidence_score=0.0
        )

def main():
    asyncio.run(main_async())

def generate_vector_enhanced_app_js(saved_components, template_data):
    """PDF 벡터 기반 App.js 생성 (학습 시스템 포함)"""
    imports = []
    components = []
    
    for component in saved_components:
        component_name = component['name']
        file_name = component['file']
        imports.append(f"import {{ {component_name} }} from './components/{file_name}';")
        
        # 학습 적용 여부 표시
        learning_applied = component.get('learning_insights_applied', False)
        learning_indicator = " /* 학습 적용 */" if learning_applied else ""
        components.append(f"      <{component_name} />{learning_indicator}")
    
    # PDF 소스 정보
    pdf_sources = template_data.get("pdf_sources", {})
    text_sources = pdf_sources.get("text_sources", [])
    image_sources = pdf_sources.get("image_sources", [])
    
    source_info = ""
    if text_sources or image_sources:
        source_info = f"""
        <div className="pdf-sources-info">
          <h3>📄 PDF 벡터 데이터 기반 생성</h3>
          {f'<p>텍스트 레이아웃: {", ".join(text_sources)}</p>' if text_sources else ''}
          {f'<p>이미지 레이아웃: {", ".join(image_sources)}</p>' if image_sources else ''}
        </div>"""
    
    return f'''import React from 'react';
import './App.css';
{chr(10).join(imports)}

function App() {{
  return (
    <div className="App">
      <header className="magazine-header">
        <h1>✈️ 여행 매거진</h1>
        <p>PDF 벡터 데이터 기반 AI 매거진 (에이전트 학습 시스템)</p>
        {source_info}
      </header>
      
      <main className="magazine-content">
{chr(10).join(components)}
      </main>
      
      <footer className="magazine-footer">
        <div className="generation-info">
          <h3>생성 시스템 정보</h3>
          <div className="agent-info">
            <div className="agent-group">
              <h4>콘텐츠 생성 (학습 기반)</h4>
              <p>🎤 InterviewAgent | ✍️ EssayAgent | 📝 ContentCreatorV2</p>
            </div>
            <div className="agent-group">
              <h4>PDF 벡터 기반 편집 (학습 기반)</h4>
              <p>📄 OrgAgent | 🖼️ BindingAgent | 🎯 CoordinatorAgent</p>
            </div>
            <div className="agent-group">
              <h4>기술 스택 (학습 시스템)</h4>
              <p>Azure Form Recognizer | Azure Cognitive Search | OpenAI Embeddings | Agent Decision Logger</p>
            </div>
          </div>
        </div>
        <p className="copyright">이 매거진은 실제 매거진 PDF의 벡터 데이터를 학습한 AI 시스템으로 생성되었습니다.</p>
        <p className="learning-info">🧠 에이전트 학습 시스템이 적용되어 지속적으로 품질이 향상됩니다.</p>
      </footer>
    </div>
  );
}}

export default App;'''

if __name__ == "__main__":
    main()
