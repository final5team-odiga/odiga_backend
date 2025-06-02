import os
import json
import asyncio
import random
import time
from dotenv import load_dotenv
from crewai import BaseLLM
from openai import AzureOpenAI
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

dotenv_path = Path(r'C:\Users\EL0021\Desktop\odiga_multimodal_agent\.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

class AzureOpenAILLM(BaseLLM):
    """Azure OpenAI API를 직접 사용하는 사용자 정의 LLM 클래스 (개선된 버전)"""

    def __init__(self):
        # 환경 변수 확인 및 가져오기
        self.api_key = os.getenv("AZURE_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_API_BASE")
        self.api_version = os.getenv("AZURE_API_VERSION")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        if not self.api_key:
            raise ValueError("AZURE_API_KEY 환경 변수가 설정되지 않았습니다.")
        if not self.azure_endpoint:
            raise ValueError("AZURE_API_BASE 환경 변수가 설정되지 않았습니다.")
        if not self.api_version:
            raise ValueError("AZURE_API_VERSION 환경 변수가 설정되지 않았습니다.")
        if not self.deployment_name:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME 환경 변수가 설정되지 않았습니다.")

        # 부모 클래스 초기화
        super().__init__(model=f"azure/{self.deployment_name}")

        # ✅ 개선된 Azure OpenAI 클라이언트 초기화 (타임아웃 및 재시도 설정)
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
            timeout=120.0,  # 120초 타임아웃 설정
            max_retries=3   # 최대 3회 재시도
        )
        
        # ✅ Rate limiting을 위한 설정
        self.last_call_time = 0
        self.min_call_interval = 0.5  # 호출 간 최소 간격 (초)
        self.semaphore = asyncio.Semaphore(3)  # 최대 3개 동시 호출

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        """LLM에 메시지를 전송하고 응답을 받습니다 (개선된 버전)."""
        
        # ✅ Rate limiting 적용
        self._apply_rate_limiting()
        
        try:
            # 문자열 메시지를 적절한 형식으로 변환
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]

            # ✅ Exponential backoff를 적용한 재시도 로직
            return self._call_with_backoff(messages, tools, available_functions)

        except Exception as e:
            print(f"LLM 호출 오류: {str(e)}")
            raise RuntimeError(f"LLM 요청 실패: {str(e)}")

    def _apply_rate_limiting(self):
        """Rate limiting 적용"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.min_call_interval:
            wait_time = self.min_call_interval - time_since_last
            time.sleep(wait_time)
        
        self.last_call_time = time.time()

    def _call_with_backoff(self, messages, tools=None, available_functions=None, max_retries=3):
        """Exponential backoff를 적용한 API 호출"""
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # ✅ 재시도 시 지연 적용
                if attempt > 0:
                    delay = min(2 ** attempt + random.uniform(0, 1), 30)  # 최대 30초
                    print(f"재시도 {attempt + 1}/{max_retries} - {delay:.2f}초 대기 중...")
                    time.sleep(delay)

                # API 호출
                if tools and self.supports_function_calling():
                    response = self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=messages,
                        tools=tools,
                        temperature=0.7,
                        max_tokens=4000
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=4000
                    )

                # ✅ 응답 검증
                if not response or not response.choices:
                    raise ValueError("빈 응답 수신")

                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("빈 콘텐츠 수신")

                # 함수 호출 처리 (기존 로직 유지)
                if (tools and self.supports_function_calling() 
                    and response.choices[0].message.tool_calls and available_functions):
                    return self._handle_function_call(response, messages, available_functions)

                return content

            except Exception as e:
                last_exception = e
                print(f"시도 {attempt + 1} 실패: {e}")
                
                if attempt == max_retries - 1:
                    break

        # 모든 재시도 실패
        raise RuntimeError(f"최대 재시도 {max_retries} 후 실패: {last_exception}")

    def _handle_function_call(self, response, messages, available_functions):
        """함수 호출 처리 (기존 로직 유지)"""
        tool_call = response.choices[0].message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        if function_name in available_functions:
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)
            
            messages.append(response.choices[0].message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": str(function_response)
            })
            
            second_response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )
            return second_response.choices[0].message.content

    async def ainvoke(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> str:
        """비동기 LLM 호출 (개선된 버전)"""
        
        async with self.semaphore:  # ✅ 동시 호출 수 제한
            try:
                # ✅ 비동기 Rate limiting 적용
                await self._apply_rate_limiting_async()
                
                # 메시지 형식 변환
                if isinstance(messages, str):
                    formatted_messages = [{"role": "user", "content": messages}]
                elif isinstance(messages, list):
                    if messages and isinstance(messages[0], str):
                        formatted_messages = [{"role": "user", "content": messages[0]}]
                    else:
                        formatted_messages = messages
                else:
                    formatted_messages = [{"role": "user", "content": str(messages)}]

                # ✅ 비동기 Exponential backoff 적용
                return await self._ainvoke_with_backoff(
                    formatted_messages, tools, available_functions
                )

            except Exception as e:
                print(f"비동기 LLM 호출 오류: {str(e)}")
                raise RuntimeError(f"비동기 LLM 요청 실패: {str(e)}")

    async def _apply_rate_limiting_async(self):
        """비동기 Rate limiting 적용"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.min_call_interval:
            wait_time = self.min_call_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_call_time = time.time()

    async def _ainvoke_with_backoff(self, formatted_messages, tools=None, available_functions=None, max_retries=3):
        """비동기 Exponential backoff를 적용한 API 호출"""
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # ✅ 재시도 시 비동기 지연 적용
                if attempt > 0:
                    delay = min(2 ** attempt + random.uniform(0, 1), 30)  # 최대 30초
                    print(f"비동기 재시도 {attempt + 1}/{max_retries} - {delay:.2f}초 대기 중...")
                    await asyncio.sleep(delay)

                # ✅ 비동기 API 호출
                if tools and self.supports_function_calling():
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.chat.completions.create(
                            model=self.deployment_name,
                            messages=formatted_messages,
                            tools=tools,
                            temperature=0.7,
                            max_tokens=4000
                        )
                    )
                else:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.chat.completions.create(
                            model=self.deployment_name,
                            messages=formatted_messages,
                            temperature=0.7,
                            max_tokens=4000
                        )
                    )

                # ✅ 응답 검증 강화
                if not response or not response.choices:
                    raise ValueError("빈 응답 수신")

                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("빈 콘텐츠 수신")

                # 함수 호출 처리
                if (tools and self.supports_function_calling() 
                    and response.choices[0].message.tool_calls and available_functions):
                    return await self._handle_function_call_async(
                        response, formatted_messages, available_functions
                    )

                return content

            except Exception as e:
                last_exception = e
                print(f"비동기 시도 {attempt + 1} 실패: {e}")
                
                if attempt == max_retries - 1:
                    break

        # 모든 재시도 실패
        raise RuntimeError(f"비동기 최대 재시도 {max_retries} 후 실패: {last_exception}")

    async def _handle_function_call_async(self, response, formatted_messages, available_functions):
        """비동기 함수 호출 처리"""
        tool_call = response.choices[0].message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        if function_name in available_functions:
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)
            
            formatted_messages.append({
                "role": "assistant", 
                "content": response.choices[0].message.content or ""
            })
            formatted_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": str(function_response)
            })
            
            second_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=formatted_messages,
                    temperature=0.7,
                    max_tokens=4000
                )
            )
            return second_response.choices[0].message.content or ""

    def supports_function_calling(self) -> bool:
        """함수 호출 지원 여부 확인"""
        return True

    def supports_stop_words(self) -> bool:
        """중지 단어 지원 여부 확인"""
        return True

    def get_context_window_size(self) -> int:
        """LLM의 컨텍스트 윈도우 크기 반환"""
        return 8192

def get_azure_llm():
    """Azure OpenAI LLM 인스턴스 생성"""
    return AzureOpenAILLM()
