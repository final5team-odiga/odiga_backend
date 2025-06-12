from typing import Dict, List
from crewai import Agent, Task
from ...custom_llm import get_azure_llm

class EssayAgentBase:
    def __init__(self, name: str, instruction: Dict):
        self.name = name
        self.instruction = instruction
        self.llm = get_azure_llm()

    def create_agent(self):
        return Agent(
            role=f"여행 에세이 작가 - {self.name}",
            goal=self.instruction.get("purpose", "여행 경험을 에세이 형식으로 정제"),
            backstory=f"""당신은 여행 에세이 전문 작가입니다. {self.instruction.get('purpose', '')}
            담백하고 조용한 감정선을 가진 일기체로 여행자의 내면과 경험을 깊이 있게 표현하며,
            '-하다' 말투의 문어체로 자연스럽고 감성적인 에세이를 작성하는 전문성을 가지고 있습니다.""",
            verbose=True,
            llm=self.llm
        )

    def get_question(self) -> str:
        """에이전트가 담당하는 질문 텍스트를 반환합니다."""
        page_instruction = self.instruction.get("page_instruction", {})
        # EssayAgent4와 같은 연속적인 섹션을 위해 'continuation'도 확인
        return page_instruction.get("source", page_instruction.get("continuation", ""))

    def rewrite_text(self, user_input: str) -> str:
        """사용자 입력을 에세이 형식으로 변환"""
        agent = self.create_agent()
        
        style = self.instruction["style"]
        hints = self.instruction.get("page_instruction", {})
        
        essay_task = Task(
            description=f"""
            다음 내용을 기반으로 에세이를 작성하되, 반드시 '-하다' 말투의 문어체로 작성하세요.
            에세이 제목은 넣지 않습니다.
            
            **원본 내용**:
            {user_input}
            
            **에세이 목적**: {self.instruction['purpose']}
            
            **스타일 지침**:
            - 형식: {style['format']}
            - 톤: {style['tone']}
            - 언어: {style['language']}
            - 문단 구성: {style['paragraphing']}
            
            **편집 원칙**:
            {chr(10).join(['- ' + principle for principle in style['editing_principle']])}
            
            **페이지 정보**: {hints.get('page', '')}
            **목표**: {hints.get('goal', '')}
            
            **구조 힌트**:
            {chr(10).join(['- ' + hint for hint in hints.get('structure_hint', [])])}
            
            **출력**: 에세이 형식의 문어체 텍스트 ('-하다' 말투)
            """,
            expected_output="'-하다' 말투의 문어체 에세이"
        )
        
        result = agent.execute_task(essay_task)
        return str(result)

class EssayAgentManager:
    def __init__(self):
        self.agents = self._get_essay_agents()

    def _get_essay_agents(self) -> List[EssayAgentBase]:
        agents = []

        agent1_instruction = {
            "purpose": "사용자가 작성한 여행 에세이 응답 중, '여행을 떠난 이유, 나의 감정은?'에 대한 내용을 에세이 형식으로 정제",
            "style": {
                "format": "에세이",
                "tone": "담백하고 조용한 감정선을 가진 일기체",
                "language": "문어체 기반, 존댓말 아닌 '-하다' 말투",
                "paragraphing": "3~5줄 단위 짧은 단락 구분",
                "editing_principle": [
                    "인터뷰이의 의도와 감정 흐름을 절대 왜곡하지 말 것",
                    "의미를 바꾸지 않는 선에서 문장 표현을 다듬을 것",
                    "불필요한 반복, 추임새, 군더더기는 제거",
                    "단어 선택은 감정을 드러내되 절대 과하지 않게 표현"
                ]
            },
            "page_instruction": {
                "page": "1page",
                "source": "질문: '여행을 떠난 이유, 나의 감정은?'",
                "goal": "출발 동기와 감정 상태를 자연스럽게 서술",
                "structure_hint": [
                    "하루의 시작이나 공항/이동 장면으로 시작",
                    "이유를 명확하게 설명하지 않아도 분위기로 감정 전달",
                    "비워두고 싶거나 명확하지 않은 감정도 솔직하게 표현 가능"
                ]
            },
            "output_format": {
                "type": "essay_text",
                "output": "에세이 형식의 문자열"
            }
        }

        agent2_instruction = {
            "purpose": "사용자의 여행 에세이 응답 중, '{계절}에 {도시}를 찾은 이유가 있다면?'에 대한 내용을 에세이 형식으로 정제",
            "style": {
                "format": "에세이",
                "tone": "담백하고 조용한 감정선을 가진 일기체",
                "language": "문어체 기반, 존댓말 아닌 '-하다' 말투",
                "paragraphing": "3~5줄 단위 짧은 단락 구분",
                "editing_principle": [
                    "인터뷰이의 의도와 감정 흐름을 절대 왜곡하지 말 것",
                    "의미를 바꾸지 않는 선에서 문장 표현을 다듬을 것",
                    "불필요한 반복, 추임새, 군더더기는 제거",
                    "단어 선택은 감정을 드러내되 절대 과하지 않게 표현"
                ]
            },
            "page_instruction": {
                "page": "2page",
                "source": "질문: '{계절}에 {도시}를 찾은 이유가 있다면?'",
                "goal": "도시와 계절이 연결된 개인적인 이유 또는 인상 묘사",
                "structure_hint": [
                    "도시의 풍경이나 계절적 느낌으로 서두를 시작",
                    "기억, 기대, 이미지 같은 주관적 인상이 드러나도록 유도"
                ]
            },
            "output_format": {
                "type": "essay_text",
                "output": "에세이 형식의 문자열"
            }
        }

        agent3_instruction = {
            "purpose": "사용자의 여행 에세이 응답 중, '{이름}님이 그곳에서의 오늘 하루는 어떻게 흘러갔는지 궁금해진다. 어땠는지'에 대한 내용을 에세이 형식으로 정제",
            "style": {
                "format": "에세이",
                "tone": "담백하고 조용한 감정선을 가진 일기체",
                "language": "문어체 기반, 존댓말 아닌 '-하다' 말투/ 일부 구어체 허용",
                "paragraphing": "3~5줄 단위 짧은 단락 구분",
                "editing_principle": [
                    "인터뷰이의 의도와 감정 흐름을 절대 왜곡하지 말 것",
                    "의미를 바꾸지 않는 선에서 문장 표현을 다듬을 것",
                    "불필요한 반복, 추임새, 군더더기는 제거",
                    "단어 선택은 감정을 드러내되 절대 과하지 않게 표현"
                ]
            },
            "page_instruction": {
                "page": "3page",
                "source": "질문: '{이름}님이 그곳에서의 오늘 하루는 어떻게 흘러갔는지 궁금해진다. 어땠는지'",
                "goal": "하루의 흐름과 감정을 따라가는 묘사",
                "structure_hint": [
                    "산책, 카페, 만난 사람, 작은 사건 등 구체적 묘사 포함",
                    "감정 변화가 있으면 드러내되 과장 없이 표현"
                ]
            },
            "output_format": {
                "type": "essay_text",
                "output": "에세이 형식의 문자열"
            }
        }

        agent4_instruction = {
            "purpose": "사용자의 여행 에세이 응답 중, 3페이지 내용의 연장선인 하루 마무리를 에세이 형식으로 정제",
            "style": {
                "format": "에세이",
                "tone": "담백하고 조용한 감정선을 가진 일기체",
                "language": "문어체 기반, 존댓말 아닌 '-하다'말투/ 일부 구어체 허용",
                "paragraphing": "3~5줄 단위 짧은 단락 구분",
                "editing_principle": [
                    "인터뷰이의 의도와 감정 흐름을 절대 왜곡하지 말 것",
                    "의미를 바꾸지 않는 선에서 문장 표현을 다듬을 것",
                    "불필요한 반복, 추임새, 군더더기는 제거",
                    "단어 선택은 감정을 드러내되 절대 과하지 않게 표현"
                ]
            },
            "page_instruction": {
                "page": "4page",
                "continuation": "3page 내용의 연장선",
                "goal": "하루 마무리나 인상 깊은 순간으로 연결",
                "structure_hint": [
                    "해질 무렵의 장면, 노을, 글을 쓰는 장면 등으로 마무리",
                    "짧지만 의미 있는 인용이나 내면 독백으로 끝맺기"
                ]
            },
            "output_format": {
                "type": "essay_text",
                "output": "에세이 형식의 문자열"
            }
        }

        agents.append(EssayAgentBase("EssayFormatAgent1", agent1_instruction))
        agents.append(EssayAgentBase("EssayFormatAgent2", agent2_instruction))
        agents.append(EssayAgentBase("EssayFormatAgent3", agent3_instruction))
        agents.append(EssayAgentBase("EssayFormatAgent4", agent4_instruction))

        return agents

    def run_all(self, qa_map: Dict[str, str]) -> Dict[str, str]:
        """질문-답변 맵을 기반으로 모든 텍스트 응답을 에세이 형식으로 처리"""
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
        print("\n=== 에세이 에이전트 처리 시작 ===")
        print(f"- 파싱된 질문-답변 맵에는 {len(qa_map)}개의 항목이 있습니다.")
        print("- 사용 가능한 질문 키:")
        for q_key in qa_map.keys():
            print(f"  * '{q_key}'")
        
        agent3_answer = None
        agent3_key = None

        # 모든 에이전트의 질문을 먼저 분석하여 매칭 정보 수집
        agent_matches = {}
        for agent in self.agents:
            agent_question = agent.get_question()
            
            # 특수 케이스: EssayAgent4 (3page 내용의 연장선)
            if agent.name == "EssayFormatAgent4" and "3page 내용의 연장선" in agent_question:
                agent_matches[agent.name] = {"special_case": True}
                continue
                
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
                        similarity = len(common_words) / max(len(agent_words), len(key_words)) * 50  # 최대 80% 유사도
                        if similarity > best_score:
                            best_match = q_key
                            best_score = similarity
                            print(f"- 단어 일치 ({similarity:.1f}%): '{q_key}'")
            
            # 최소 80% 이상 유사한 경우에만 매칭으로 간주
            if best_match and best_score >= 50:
                agent_matches[agent.name] = {
                    "match_key": best_match,
                    "score": best_score
                }
                print(f"✓ 매칭된 질문: '{best_match}' (유사도: {best_score:.1f}%)")
                
                # EssayFormatAgent3의 답변 저장 (EssayFormatAgent4를 위해)
                if agent.name == "EssayFormatAgent3":
                    agent3_key = best_match
                    agent3_answer = qa_map[best_match]
            else:
                agent_matches[agent.name] = {"match_key": None, "score": 0}
                print(f"⚠️ {agent.name}에 대한 매칭 질문을 찾지 못했습니다.")
        
        # 매칭 정보를 기반으로 에세이 생성
        for agent in self.agents:
            user_response = None
            
            # EssayFormatAgent4 특별 처리
            if agent.name == "EssayFormatAgent4" and agent_matches[agent.name].get("special_case"):
                if agent3_answer:
                    user_response = agent3_answer
                    print(f"\n{agent.name}: EssayFormatAgent3의 질문 '{agent3_key}'의 답변을 사용합니다.")
            else:
                match_info = agent_matches[agent.name]
                if match_info.get("match_key"):
                    user_response = qa_map[match_info["match_key"]]
                    print(f"\n{agent.name}: 매칭된 질문 '{match_info['match_key']}' 처리 중...")
            
            if user_response:
                result = agent.rewrite_text(user_response)
                results[agent.name] = result
                print(f"✓ {agent.name} 에세이 처리 완료")
            else:
                print(f"⚠️ {agent.name}에 대한 답변을 찾지 못해 에세이를 생성하지 않았습니다.")

        return results