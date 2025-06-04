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
            role="지리적 위치 분석 전문가",
            goal="이미지의 지리적 위치를 건축적, 환경적, 문화적 특징을 통해 정확히 식별",
            backstory="""당신은 지리학 및 건축사학 전문가로서 다음 역량을 보유하고 있습니다:

    **전문 분야:**
    - 지리학 박사 학위 및 건축사학 연구
    - 전 세계 도시 계획 및 건축 양식 데이터베이스 구축
    - 위성 이미지 및 항공 사진 분석 전문성
    - 문화적 랜드마크 및 지역 특성 식별 경험

    **분석 방법론:**
    1. 건축 양식 분석: 건물 구조, 재료, 색상, 디자인 패턴
    2. 환경 요소 분석: 지형, 식생, 기후 지표
    3. 인프라 분석: 도로, 교통 시설, 공공 구조물
    4. 문화 요소 분석: 간판, 언어, 전통적 요소
    5. 사진의 구도와 색감

    **출력 원칙:**
    - 객관적이고 구체적인 지리적 정보만 제공
    - 확실한 증거에 기반한 분석
    - 위치 특정에 필요한 핵심 정보 집중

    **출력 형식:**
    국가: [정확한 국가명]
    도시: [구체적 도시/지역명]  
    촬영 위치: [상세 장소명]
    자세한 설명: [사진 고유 특징 키워드]""",
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
                            "content": """당신은 지리적 위치 식별 전문가입니다. 이미지의 지리적, 건축적, 문화적 특징을 분석하여 위치를 특정합니다.

                분석 기준:
                - 건축물의 양식과 구조적 특징
                - 자연환경과 지형적 요소
                - 도시 인프라와 교통 시설
                - 문화적 표식과 언어적 단서
                - 사진의 구도와 색감

                출력 형식:
                국가: [국가명]
                도시: [도시/지역명]
                촬영 위치: [구체적 장소명]
                자세한 설명: [사진 고유 특징]

                분석 시 지리적 정보만 제공하고 다른 내용은 언급하지 마세요."""
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "이 이미지의 지리적 위치를 다음 형식으로 분석해주세요:\n\n국가:\n도시:\n촬영 위치:\n자세한 설명:\n\n건축물, 자연환경, 문화적 요소를 기반으로 위치를 특정해주세요."
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
                    "temperature": 0.1,
                    "max_tokens": 150
                }
 
                api_url = f"{os.getenv('AZURE_API_BASE')}/openai/deployments/{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}/chat/completions?api-version={os.getenv('AZURE_API_VERSION')}"
                
                async with session.post(api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        result = result_data['choices'][0]['message']['content']
                        
                        # 결과 형식 검증 및 정제
                        lines = result.strip().split('\n')
                        parsed_result = {}
                        
                        for line in lines:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                key = key.strip()
                                value = value.strip()
                                
                                if key == "국가":
                                    parsed_result["country"] = value
                                elif key == "도시":
                                    parsed_result["city"] = value
                                elif key == "촬영 위치":
                                    parsed_result["location"] = value
                                elif key == "자세한 설명":
                                    parsed_result["description"] = value
                        
                        analysis_result = {
                            "image_name": image.name,
                            "image_url": image_url,
                            "country": parsed_result.get("country", "미상"),
                            "city": parsed_result.get("city", "미상"),
                            "location": parsed_result.get("location", "미상"),
                            "description": parsed_result.get("description", "특징없음"),
                            "raw_location": result,
                            "confidence_score": 0.9 if all(k in parsed_result for k in ["country", "city", "location"]) else 0.5
                        }
                        
                        print(f"이미지 '{image.name}' 정밀 분석 완료:")
                        print(f"  국가: {parsed_result.get('country', '미상')}")
                        print(f"  도시: {parsed_result.get('city', '미상')}")
                        print(f"  위치: {parsed_result.get('location', '미상')}")
                        print(f"  특징: {parsed_result.get('description', '특징없음')}")
                        
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
