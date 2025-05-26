from crewai import Agent, Task
from custom_llm import get_azure_llm
import asyncio
import aiohttp
import os
from typing import List, Dict, Any

class ImageAnalyzerAgent:
    def __init__(self):
        self.llm = get_azure_llm()

    def create_agent(self):
        return Agent(
            role="이미지 분석가",
            goal="이미지에서 위치 정보를 정확하게 식별하고 분석",
            backstory="당신은 이미지에서 랜드마크, 지형, 건축물 등을 식별하여 위치를 파악하는 전문가입니다.",
            verbose=True,
            llm=self.llm,
            multimodal=True  # 멀티모달 기능 활성화
        )

    async def analyze_single_image_async(self, session: aiohttp.ClientSession, image, semaphore: asyncio.Semaphore, image_index: int) -> Dict[str, Any]:
        """단일 이미지를 비동기로 분석"""
        async with semaphore:  # 동시 처리 수 제한
            try:
                print(f"\n=== 이미지 {image_index}: '{image.name}' 분석 중 ===")
                
                # 이미지 URL 생성
                from utils.blob_storage import BlobStorageManager
                blob_manager = BlobStorageManager()
                image_url = blob_manager.get_image_url(image)
                print(f"이미지 URL: {image_url}")

                # 비동기 Azure OpenAI API 호출
                headers = {
                    'Content-Type': 'application/json',
                    'api-key': os.getenv("AZURE_API_KEY")
                }
                
                payload = {
                    "model": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                    "messages": [
                        {
                            "role": "system",
                            "content": "당신은 이미지에서 위치를 식별하는 전문가입니다. 이미지에 나타난 랜드마크, 건물, 자연 환경 등을 분석하여 가능한 정확한 위치(국가, 도시, 특정 장소)를 파악하세요."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "이 이미지에 나타난 위치를 식별해주세요. 가능하다면 국가, 도시, 특정 장소명을 포함해주세요."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                }

                api_url = f"{os.getenv('AZURE_API_BASE')}/openai/deployments/{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}/chat/completions?api-version={os.getenv('AZURE_API_VERSION')}"
                
                async with session.post(api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        result = result_data['choices'][0]['message']['content']
                        
                        analysis_result = {
                            "image_name": image.name,
                            "image_url": image_url,
                            "location": result
                        }
                        
                        print(f"이미지 '{image.name}' 분석 완료: {result}")
                        return analysis_result
                    else:
                        error_text = await response.text()
                        raise Exception(f"API 호출 실패: {response.status} - {error_text}")
                        
            except Exception as e:
                print(f"이미지 '{image.name}' 분석 중 오류 발생: {str(e)}")
                import traceback
                print(traceback.format_exc())
                
                return {
                    "image_name": image.name,
                    "image_url": image_url if 'image_url' in locals() else "URL 생성 실패",
                    "location": f"분석 오류: {str(e)}"
                }

    async def analyze_images_batch_async(self, images: List, max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """여러 이미지를 비동기로 배치 분석"""
        semaphore = asyncio.Semaphore(max_concurrent)  # 동시 처리 수 제한
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),  # 5분 타임아웃
            connector=aiohttp.TCPConnector(limit=max_concurrent)
        ) as session:
            tasks = []
            
            for i, image in enumerate(images, 1):
                task = self.analyze_single_image_async(session, image, semaphore, i)
                tasks.append(task)
            
            print(f"총 {len(tasks)}개의 이미지를 동시에 처리합니다 (최대 동시 처리: {max_concurrent}개)")
            
            # 모든 작업을 동시에 실행하고 결과 수집
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 예외 처리된 결과들을 정리
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "image_name": images[i].name if i < len(images) else f"unknown_{i}",
                        "image_url": "처리 실패",
                        "location": f"처리 중 예외 발생: {str(result)}"
                    })
                else:
                    processed_results.append(result)
            
            return processed_results

    def analyze_images(self, images, crew):
        """기존 인터페이스 유지 - 비동기 처리로 내부 구현 변경"""
        print(f"\n=== 비동기 이미지 분석 시작 - 총 {len(images)}개 이미지 ===")
        
        # 비동기 함수 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                self.analyze_images_batch_async(images, max_concurrent=3)  # 동시 처리 수를 3개로 제한
            )
            print(f"\n=== 비동기 이미지 분석 완료 - {len(results)}개 결과 ===")
            return results
        finally:
            loop.close()
