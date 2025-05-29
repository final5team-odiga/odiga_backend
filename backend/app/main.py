import asyncio
import nest_asyncio
from dotenv import load_dotenv
from utils.system_coordinator import SystemCoordinator
from pathlib import Path

# nest_asyncio 적용
nest_asyncio.apply()

dotenv_path = Path(r'C:\Users\EL0021\Desktop\odiga_agent\.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

async def main_async():
    print("=== CrewAI 여행 매거진 생성 시스템 ===")

    # 시스템 코디네이터 초기화
    coordinator = SystemCoordinator()

    try:
        # 1. 시스템 초기화 (비동기)
        data = await coordinator.initialize_system()

        # 2. 이미지 분석 (비동기)
        image_results = await coordinator.process_images(data['images'])

        # 3. 콘텐츠 생성 (비동기)
        magazine_content = await coordinator.create_content(data['texts'], image_results)

        # 4. 템플릿 데이터 생성 (비동기)
        template_data = await coordinator.generate_template_data(magazine_content, image_results)

        # 5. JSX 컴포넌트 생성 (비동기)
        components = await coordinator.generate_jsx_components(template_data)

        # 6. React 앱 생성 (비동기)
        project_path = await coordinator.create_react_app(components, template_data)

        # 7. 결과 출력
        coordinator.display_results(project_path, components)

    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        await coordinator.handle_error(e)

def main():
    print("🚀 CrewAI 여행 매거진 생성 시스템 시작 ")
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
