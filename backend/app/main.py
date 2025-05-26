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
from crewai import Crew
from pathlib import Path

# nest_asyncio 적용 (이벤트 루프 중첩 허용)
nest_asyncio.apply()

dotenv_path = Path(r'C:\Users\EL0021\Desktop\odiga_agent\.env')

# 환경 변수 로드
load_dotenv(dotenv_path=dotenv_path, override=True)

async def main_async():
    print("=== CrewAI 여행 매거진 생성 시스템 ===")
    
    try:
        # 초기화
        blob_manager = BlobStorageManager()
        file_manager = FileManager(output_folder=os.getenv("OUTPUT_FOLDER", "./output"))
        
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
        
        # 비동기 메서드가 있으면 사용, 없으면 동기 메서드 사용
        if hasattr(image_analyzer, 'analyze_images_async'):
            image_analysis_results = await image_analyzer.analyze_images_async(images, crew)
        else:
            # 동기 메서드를 비동기 컨텍스트에서 안전하게 실행
            loop = asyncio.get_event_loop()
            image_analysis_results = await loop.run_in_executor(
                None, image_analyzer.analyze_images, images, crew
            )
        
        # 분석 결과 저장
        analysis_path = os.path.join(file_manager.output_folder, "image_analysis_results.json")
        file_manager.save_json(image_analysis_results, analysis_path)
        print(f"이미지 분석 결과 저장: {analysis_path}")
        
        # 2. 콘텐츠 생성 (인터뷰 + 에세이 통합)
        print("\n=== 콘텐츠 생성 (인터뷰 + 에세이) ===")
        magazine_content = content_creator_v2.execute_content_creation(texts, image_analysis_results)
        
        # 콘텐츠 저장 (JSON 형식)
        content_path = os.path.join(file_manager.output_folder, "magazine_content.json")
        file_manager.save_magazine_content_json(magazine_content, content_path)
        print(f"매거진 콘텐츠 저장: {content_path}")
        
        # 3. PDF 벡터 기반 템플릿 데이터 생성
        print("\n=== PDF 벡터 기반 템플릿 데이터 생성 ===")
        template_data = template_manager.create_magazine_data(magazine_content, image_analysis_results)
        
        # 템플릿 데이터 저장
        template_path = os.path.join(file_manager.output_folder, "template_data.json")
        file_manager.save_json(template_data, template_path)
        print(f"템플릿 데이터 저장: {template_path}")
        
        # 4. React 앱 기본 구조 생성
        print("\n=== React 앱 기본 구조 생성 ===")
        project_name = "travel-magazine"
        project_folder, template_data_path = template_manager.generate_react_app(template_data, file_manager, project_name)
        print(f"React 앱 구조 생성: {project_folder}")
        
        # 5. JSX 생성 (비동기 코드 리뷰 포함)
        print("\n=== JSX 생성 ===")
        if hasattr(jsx_creator, 'generate_jsx_components_async'):
            generated_components_data = await jsx_creator.generate_jsx_components_async(template_data_path)
        else:
            generated_components_data = jsx_creator.generate_jsx_components(template_data_path)
        print(f"JSX 컴포넌트 생성: {len(generated_components_data)}개")
        
        # 6. 컴포넌트 저장
        components_folder = os.path.join(project_folder, "src", "components")
        saved_components = jsx_creator.save_jsx_components(generated_components_data, components_folder)
        
        # 7. App.js 생성
        if saved_components:
            app_js = generate_vector_enhanced_app_js(saved_components, template_data)
            app_js_path = os.path.join(project_folder, "src", "App.js")
            file_manager.save_content(app_js, app_js_path)
            print(f"✅ App.js 생성 완료")
        
        # 8. 실행 방법 안내
        print("\n=== 매거진 실행 방법 ===")
        print(f"1. 터미널에서 다음 명령어를 실행하세요:")
        print(f"   cd {project_folder}")
        print(f"   npm install")
        print(f"   npm start")
        print(f"2. 웹 브라우저에서 http://localhost:3000 으로 접속하세요.")
        
        # 9. 생성 결과 요약
        print("\n=== PDF 벡터 기반 매거진 생성 완료 ===")
        print(f"✅ PDF 벡터 시스템: 활성화")
        print(f"✅ InterviewAgent: 인터뷰 형식 콘텐츠 생성")
        print(f"✅ EssayAgent: 에세이 형식 콘텐츠 생성")
        print(f"✅ ContentCreatorV2: 통합 매거진 콘텐츠 생성")
        print(f"✅ OrgAgent: PDF 벡터 기반 텍스트 배치")
        print(f"✅ BindingAgent: PDF 벡터 기반 이미지 배치")
        print(f"✅ CoordinatorAgent: 벡터 기반 결과 통합")
        print(f"✅ JSXCreatorAgent: JSX 파일 생성")
        
        # PDF 소스 정보 출력
        if template_data.get("vector_enhanced") and template_data.get("pdf_sources"):
            pdf_sources = template_data["pdf_sources"]
            if pdf_sources.get("text_sources"):
                print(f"📄 텍스트 레이아웃 소스: {', '.join(pdf_sources['text_sources'])}")
            if pdf_sources.get("image_sources"):
                print(f"🖼️ 이미지 레이아웃 소스: {', '.join(pdf_sources['image_sources'])}")
        
        print(f"📁 최종 결과: {project_folder}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())

def main():
    # 이벤트 루프 상태 확인 및 적절한 실행 방법 선택
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 이미 실행 중인 루프가 있으면 태스크로 실행
            task = loop.create_task(main_async())
            return task
        else:
            # 실행 중인 루프가 없으면 새로 실행
            return asyncio.run(main_async())
    except RuntimeError:
        # 루프가 없으면 새로 생성
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(main_async())

def generate_vector_enhanced_app_js(saved_components, template_data):
    """PDF 벡터 기반 App.js 생성"""
    imports = []
    components = []
    
    for component in saved_components:
        component_name = component['name']
        file_name = component['file']
        imports.append(f"import {{ {component_name} }} from './components/{file_name}';")
        components.append(f"      <{component_name} />")
    
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
        <p>PDF 벡터 데이터 기반 AI 매거진</p>
        <div className="tech-stack">
          <span>🤖 CrewAI Multi-Agent</span>
          <span>📄 PDF Vector Analysis</span>
          <span>🔍 Azure Cognitive Search</span>
          <span>🧠 OpenAI Embeddings</span>
        </div>
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
              <h4>콘텐츠 생성</h4>
              <p>🎤 InterviewAgent | ✍️ EssayAgent | 📝 ContentCreatorV2</p>
            </div>
            <div className="agent-group">
              <h4>PDF 벡터 기반 편집</h4>
              <p>📄 OrgAgent | 🖼️ BindingAgent | 🎯 CoordinatorAgent</p>
            </div>
            <div className="agent-group">
              <h4>기술 스택</h4>
              <p>Azure Form Recognizer | Azure Cognitive Search | OpenAI Embeddings</p>
            </div>
          </div>
        </div>
        <p className="copyright">이 매거진은 실제 매거진 PDF의 벡터 데이터를 학습한 AI 시스템으로 생성되었습니다.</p>
      </footer>
    </div>
  );
}}

export default App;'''

if __name__ == "__main__":
    main()
