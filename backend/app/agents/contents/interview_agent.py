from typing import Dict, List
from crewai import Agent, Task
from ...custom_llm import get_azure_llm

class InterviewAgentBase:
    def __init__(self, name: str, instruction: Dict):
        self.name = name
        self.instruction = instruction
        self.llm = get_azure_llm()

    def get_question(self) -> str:
        return self.instruction.get("page_instruction", {}).get("source", "질문이 없습니다.")

    def create_agent(self):
        return Agent(
            role=f"여행 인터뷰 전문가 - {self.name}",
            goal=self.instruction.get("purpose", "여행 경험에 대한 인터뷰 답변 정제"),
            backstory=f"""당신은 여행 인터뷰 전문가입니다. {self.instruction.get('purpose', '')}
            자연스럽고 진솔한 대화체로 여행자의 경험을 깊이 있게 탐구하며,
            감정과 경험의 디테일을 생생하게 끌어내는 전문성을 가지고 있습니다.""",
            verbose=True,
            llm=self.llm
        )

    def process_interview(self, user_response: str) -> str:
        """사용자 응답을 바탕으로 인터뷰 형식으로 정제"""
        agent = self.create_agent()
        
        question = self.get_question()
        style = self.instruction.get("style", {})
        
        interview_task = Task(
            description=f"""
            다음 질문에 대한 사용자의 응답을 자연스럽고 진솔한 인터뷰 형식으로 정제하세요.
            
            **질문**: {question}
            
            **사용자 응답**: 
            {user_response}
            
            **정제 지침**:
            - 형식: {style.get('format', 'Q&A')}
            - 톤: {style.get('tone', '자연스럽고 진솔한 대화체')}
            - 언어: {style.get('language', '구어체 기반, 존댓말 사용')}
            
            **편집 원칙**:
            {chr(10).join(['- ' + principle for principle in style.get('editing_principle', [])])}
            
            **목표**: {self.instruction.get('page_instruction', {}).get('goal', '진솔한 경험 표현')}
            
            **출력 형식**:
            Q: {question}
            A: [정제된 답변]
            """,
            expected_output="Q&A 형식의 인터뷰 텍스트"
        )
        
        result = agent.execute_task(interview_task)
        return str(result)

class InterviewAgentManager:
    def __init__(self):
        self.agents = self._get_interview_agents()
        
    def _get_interview_agents(self) -> List[InterviewAgentBase]:
        style_common = {
            "format": "Q&A",
            "tone": "자연스럽고 진솔한 대화체",
            "language": "구어체 기반, 존댓말 사용",
            "editing_principle": [
                "말투는 부드럽고 담백하게 정제",
                "너무 긴 문장은 나누되 감정의 흐름은 유지",
                "반복되거나 모호한 표현은 자연스럽게 정리",
                "질문에 맞는 답변이 되도록 포커스를 유지"
            ]
        }

        agents_data = [
            {
                "name": "InterviewAgent1",
                "purpose": "여행 중 인상 깊었던 인물이나 장면에 대한 인터뷰 답변을 정제",
                "page_instruction": {
                    "page": "1page",
                    "source": "여행 중 인상 깊었던 인물이나 장면이 있었나요?",
                    "goal": "인물 혹은 장면에 대한 구체적이고 진솔한 묘사"
                }
            },
            {
                "name": "InterviewAgent2", 
                "purpose": "날씨, 도시 느낌과 여행과 함께한 음악에 대한 인터뷰 답변을 정제",
                "page_instruction": {
                    "page": "2page",
                    "source": "{날씨}의 {도시}은 어떤 느낌이었나요? 여행과 함께한 음악이 있다면요?",
                    "goal": "날씨와 도시의 분위기, 음악과의 연관성 표현"
                }
            },
            {
                "name": "InterviewAgent3",
                "purpose": "가장 만족스러웠던 음식에 대한 인터뷰 답변을 정제", 
                "page_instruction": {
                    "page": "3page",
                    "source": "이 {도시}에서 가장 만족스러웠던 음식은 무엇이었나요?",
                    "goal": "음식에 대한 생생한 묘사와 만족감 표현"
                }
            },
            {
                "name": "InterviewAgent4",
                "purpose": '여행 중 "이건 꼭 해보자"라고 생각한 것이 있었다면에 대한 인터뷰 답변을 정제',
                "page_instruction": {
                    "page": "4page", 
                    "source": '여행 중 "이건 꼭 해보자"라고 생각한 것이 있었다면',
                    "goal": "계획이나 다짐에 대한 구체적이고 솔직한 표현"
                }
            },
            {
                "name": "InterviewAgent5",
                "purpose": "가장 좋았던 공간에 대한 인터뷰 답변을 정제",
                "page_instruction": {
                    "page": "5page",
                    "source": "여행을 돌아보았을 때 가장 좋았던 공간은?", 
                    "goal": "공간에 대한 감정과 기억을 생생히 표현"
                }
            }
        ]

        return [InterviewAgentBase(data["name"], {
            "purpose": data["purpose"],
            "style": style_common,
            "page_instruction": data["page_instruction"],
            "output_format": {
                "type": "interview_text",
                "output": "인터뷰 Q&A 형식의 문자열"
            }
        }) for data in agents_data]

    def process_all_interviews(self, qa_map: Dict[str, str]) -> Dict[str, str]:
        """질문-답변 맵을 기반으로 모든 텍스트 응답을 인터뷰 형식으로 처리"""
        import re

        def normalize_question(q_text: str) -> str:
            """질문 텍스트를 정규화하여 매칭 가능성을 높입니다."""
            # 1. 중괄호 제거
            q_text = q_text.replace('{', '').replace('}', '')
            # 2. 물음표 제거
            q_text = q_text.replace('?', '')
            # 3. '질문: ' 접두어 제거
            q_text = re.sub(r'^질문\s*:\s*', '', q_text)
            # 4. 따옴표 제거 (일반, 스마트 모두)
            q_text = q_text.replace("'", "").replace('"', '').replace('"', '').replace('"', '')
            # 5. 공백 정규화
            q_text = ' '.join(q_text.split())
            # 6. 모든 공백 제거 (비교용)
            q_text_no_space = q_text.replace(' ', '')
            return q_text.lower(), q_text_no_space.lower()

        results = {}
        print("\n=== 인터뷰 에이전트 처리 시작 ===")
        print(f"- 파싱된 질문-답변 맵에는 {len(qa_map)}개의 항목이 있습니다.")
        print("- 사용 가능한 질문 키:")
        for q_key in qa_map.keys():
            print(f"  * '{q_key}'")
        
        for agent in self.agents:
            agent_question = agent.get_question()
            print(f"\n{agent.name}이(가) 찾는 질문: '{agent_question}'")
            
            # 질문 정규화
            norm_agent_q, norm_agent_q_no_space = normalize_question(agent_question)
            print(f"- 정규화된 질문: '{norm_agent_q}'")
            
            # 가장 유사한 질문 찾기
            best_match = None
            best_score = 0
            
            for q_key in qa_map.keys():
                norm_key, norm_key_no_space = normalize_question(q_key)
                
                # 1. 완전 일치 검사
                if norm_agent_q == norm_key:
                    best_match = q_key
                    best_score = 100
                    print(f"- 완전 일치: '{q_key}'")
                    break
                
                # 2. 공백 없는 일치 검사
                elif norm_agent_q_no_space == norm_key_no_space:
                    best_match = q_key
                    best_score = 90
                    print(f"- 공백 무시 일치: '{q_key}'")
                    break
                
                # 3. 부분 문자열 검사
                elif norm_agent_q in norm_key or norm_key in norm_agent_q:
                    # 길이가 비슷한 경우에만 높은 점수 부여
                    similarity = min(len(norm_agent_q), len(norm_key)) / max(len(norm_agent_q), len(norm_key)) * 100
                    if similarity > best_score:
                        best_match = q_key
                        best_score = similarity
                        print(f"- 부분 일치 ({similarity:.1f}%): '{q_key}'")
                
                # 4. 단어 일치율 검사
                else:
                    agent_words = set(norm_agent_q.split())
                    key_words = set(norm_key.split())
                    if agent_words and key_words:  # 빈 집합이 아닌 경우에만
                        common_words = agent_words.intersection(key_words)
                        similarity = len(common_words) / max(len(agent_words), len(key_words)) * 70  # 최대 80% 유사도
                        if similarity > best_score:
                            best_match = q_key
                            best_score = similarity
                            print(f"- 단어 일치 ({similarity:.1f}%): '{q_key}'")
            
            # 최소 80% 이상 유사한 경우에만 매칭으로 간주
            if best_match and best_score >= 70:
                user_response = qa_map[best_match]
                print(f"✓ 매칭된 질문: '{best_match}' (유사도: {best_score:.1f}%)")
                
                interview_result = agent.process_interview(user_response)
                results[agent.name] = interview_result
                print(f"✓ {agent.name} 인터뷰 처리 완료")
            else:
                print(f"⚠️ {agent.name}에 대한 답변을 찾지 못해 인터뷰를 생성하지 않았습니다.")
        
        return results