from crewai import Agent
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
            role="이미지 위치 식별 전문가",
            goal="이미지에서 국가, 배경 정보, 촬영 위치를 정확하고 간결하게 식별",
            backstory="""당신은 10년간 세계 각국의 여행 이미지를 분석해온 지리학 및 문화 전문가입니다.

    **전문 분야:**
    - 지리학 박사 학위 보유
    - 100개국 이상의 랜드마크 및 지역 특성 데이터베이스 구축
    - 건축 양식, 자연 환경, 문화적 특징을 통한 위치 식별 전문성
    - 이미지 기반 지리적 위치 추정 시스템 개발 경험

    **분석 방법론:**
    1. **건축 양식 분석**: 건물 스타일, 지붕 형태, 창문 구조를 통한 지역 특성 파악
    2. **자연 환경 분석**: 식생, 지형, 기후 특성을 통한 지리적 위치 추정
    3. **문화적 요소 분석**: 간판, 교통수단, 복장 등을 통한 국가/지역 식별
    4. **랜드마크 식별**: 유명 건물, 다리, 광장 등 특정 위치 마커 인식

    **응답 원칙:**
    - 정확하고 간결한 정보만 제공
    - 추측보다는 확실한 정보 우선
    - 구체적인 위치명 제공 (가능한 경우)
    - 불확실한 경우 "추정" 또는 "유사" 표현 사용

    **출력 형식:**
    반드시 다음 3가지 정보만 간결하게 제공:
    1. 국가: 정확한 국가명
    2. 배경 정보: 도시/자연/해안 등 환경 유형
    3. 촬영 위치: 구체적 장소명 또는 지역명""",
            verbose=True,
            llm=self.llm,
            multimodal=True
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
                            "content": "당신은 이미지에서 위치를 정확하게 식별하는 전문가입니다. 반드시 다음 형식으로만 응답하세요:\n\n국가: [국가명]\n배경 정보: [도시/자연/해안/산악 등]\n촬영 위치: [구체적 장소명 또는 지역명]\n\n추가 설명이나 서술은 하지 마세요."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "이 이미지의 위치를 다음 형식으로 정확히 식별해주세요:\n\n국가: \n배경 정보: \n촬영 위치: "
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
                    "temperature": 0.1,  # 더 정확한 응답을 위해 낮춤
                    "max_tokens": 100    # 간결한 응답을 위해 줄임
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
