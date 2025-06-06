import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# 추가: Cosmos DB 관련 임포트
from db.cosmos_connection import logging_container
from db.db_utils import update_agent_logs_in_cosmos, get_agent_logs_from_cosmos

@dataclass
class AgentOutput:
    """에이전트 응답 데이터 """
    agent_name: str
    agent_role: str
    output_id: str
    timestamp: str
    # 에이전트 응답 (핵심 데이터만)
    task_description: str
    final_answer: str
    reasoning_process: str
    execution_steps: List[str]
    # 입출력 데이터
    raw_input: Any
    raw_output: Any
    # 성능 메트릭 (선택적)
    performance_metrics: Dict
    error_logs: List[Dict]
    # info 관련 필드 추가
    info_data: Dict

    
    def get_info(self, key: str = None):
        """안전한 info 데이터 접근"""
        if key:
            return self.info_data.get(key)
        return self.info_data
    
    def set_info(self, key: str, value: Any):
        """안전한 info 데이터 설정"""
        if not hasattr(self, 'info_data') or self.info_data is None:
            self.info_data = {}
        self.info_data[key] = value


@dataclass
class AgentInfo:
    """에이전트 정보 데이터"""
    agent_name: str
    info_id: str
    timestamp: str
    info_type: str
    info_content: Dict
    metadata: Dict
    info_data: Dict


class AgentOutputManager:
    """에이전트 응답 전용 관리 시스템"""
    
    def __init__(self, storage_dir: str = "./agent_outputs"):
        # storage_dir 파라미터는 호환성을 위해 유지하지만 실제로 사용하지 않음
        self.current_session_id = self._generate_session_id()
        self.outputs = []  # 에이전트 응답만 메모리에 저장 (로컬 캐싱용)

    def _generate_session_id(self) -> str:
        """세션 ID 생성"""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def store_agent_output(self,
                          agent_name: str,
                          agent_role: str,
                          task_description: str,
                          final_answer: str,
                          reasoning_process: str = "",
                          execution_steps: List[str] = None,
                          raw_input: Any = None,
                          raw_output: Any = None,
                          performance_metrics: Dict = None,
                          error_logs: List[Dict] = None,
                          info_data: Dict = None) -> str:
        """에이전트 응답 저장 (Cosmos DB + 메모리 캐싱)"""
        
        output_id = f"{agent_name}_{int(time.time() * 1000000)}"
        
        agent_output = AgentOutput(
            agent_name=agent_name,
            agent_role=agent_role,
            output_id=output_id,
            timestamp=datetime.now().isoformat(),
            task_description=task_description,
            final_answer=final_answer,
            reasoning_process=reasoning_process,
            execution_steps=execution_steps or [],
            raw_input=self._safe_copy(raw_input),
            raw_output=self._safe_copy(raw_output),
            performance_metrics=performance_metrics or {},
            error_logs=error_logs or [],
            info_data=info_data or {}
        )
        
        # 메모리에 저장 (로컬 캐싱)
        self.outputs.append(agent_output)
        
        # Cosmos DB에도 저장
        try:
            output_data = asdict(agent_output)
            # Cosmos DB에 저장
            update_agent_logs_in_cosmos(
                logging_container, 
                self.current_session_id, 
                agent_name, 
                output_data
            )
            print(f"📦 {agent_name} 응답을 Cosmos DB에 저장: {output_id}")
        except Exception as e:
            print(f"❌ Cosmos DB 저장 실패, 로컬에만 저장됨: {e}")
            # 로컬에는 저장 (폴백)
            self._save_latest_outputs_local()
        
        return output_id

    def _safe_copy(self, data: Any) -> Any:
        """안전한 데이터 복사"""
        try:
            if data is None:
                return None
            if isinstance(data, (str, int, float, bool)):
                return data
            if isinstance(data, (list, tuple)):
                return [self._safe_copy(item) for item in data]
            if isinstance(data, dict):
                return {key: self._safe_copy(value) for key, value in data.items()}
            return str(data)  # 복잡한 객체는 문자열로 변환
        except:
            return str(data)

    def get_all_outputs(self, exclude_agent: str = None) -> List[Dict]:
        """모든 에이전트 응답 조회"""
        try:
            # Cosmos DB에서 먼저 조회
            cosmos_logs = get_agent_logs_from_cosmos(logging_container, self.current_session_id)
            if cosmos_logs:
                all_outputs = []
                for agent, outputs in cosmos_logs.get("agent_outputs", {}).items():
                    if exclude_agent is None or agent != exclude_agent:
                        all_outputs.extend(outputs)
                
                # 타임스탬프로 정렬
                return sorted(all_outputs, key=lambda x: x.get('timestamp', ''))
        except Exception as e:
            print(f"⚠️ Cosmos DB 로그 조회 실패, 로컬 캐시 사용: {e}")
        
        # 로컬 캐시에서 조회 (폴백)
        all_outputs = []
        for output in self.outputs:
            if exclude_agent is None or output.agent_name != exclude_agent:
                all_outputs.append(asdict(output))
        
        return sorted(all_outputs, key=lambda x: x.get('timestamp', ''))

    def get_agent_output(self, agent_name: str, latest: bool = True) -> Optional[Dict]:
        """특정 에이전트의 응답 조회"""
        try:
            # Cosmos DB에서 먼저 조회
            agent_outputs = get_agent_logs_from_cosmos(
                logging_container, 
                self.current_session_id, 
                agent_name
            )
            
            if agent_outputs:
                if latest:
                    return sorted(agent_outputs, key=lambda x: x.get('timestamp', ''), reverse=True)[0]
                else:
                    return agent_outputs
        except Exception as e:
            print(f"⚠️ Cosmos DB 에이전트 로그 조회 실패, 로컬 캐시 사용: {e}")
        
        # 로컬 캐시에서 조회 (폴백)
        agent_outputs = [
            asdict(output) for output in self.outputs
            if output.agent_name == agent_name
        ]
        
        if not agent_outputs:
            return None
        
        if latest:
            return sorted(agent_outputs, key=lambda x: x.get('timestamp', ''), reverse=True)[0]
        else:
            return agent_outputs

    def _save_latest_outputs_local(self):
        """출력을 메모리에만 캐싱 (로컬 파일 시스템 사용 안함)"""
        try:
            outputs_count = len(self.outputs)
            print(f"📄 메모리에 {outputs_count}개 출력 항목 캐싱됨")
            # 메모리 캐시 크기 제한 (선택적)
            if outputs_count > 100:
                # 가장 오래된 항목 일부 제거
                self.outputs = self.outputs[-50:]
                print(f"🔄 메모리 캐시 크기 조정: {len(self.outputs)}개 항목 유지")
        except Exception as e:
            print(f"❌ 메모리 캐싱 실패: {e}")
            
    def store_agent_info(self, agent_name: str, info_type: str, info_content: Dict, metadata: Dict = None) -> str:
        """에이전트 정보 저장"""
        info_id = f"{agent_name}_info_{int(time.time() * 1000000)}"
        
        agent_info = {
            "agent_name": agent_name,
            "info_id": info_id,
            "timestamp": datetime.now().isoformat(),
            "info_type": info_type,
            "info_content": info_content,
            "metadata": metadata or {},
            "info_data": {}
        }
        
        # Cosmos DB에 저장
        update_agent_logs_in_cosmos(
            logging_container,
            self.current_session_id,
            f"{agent_name}_info",
            agent_info
        )
        
        return info_id
        
    def get_agent_info(self, agent_name: str = None, info_type: str = None, latest: bool = True) -> List[Dict]:
        """에이전트 정보 조회"""
        try:
            # Cosmos DB에서 조회
            cosmos_logs = get_agent_logs_from_cosmos(logging_container, self.current_session_id)
            if not cosmos_logs:
                return []
                
            agent_info = []
            
            # 정보 필터링
            for agent, outputs in cosmos_logs.get("agent_outputs", {}).items():
                if agent.endswith("_info"):  # 정보 항목 식별자
                    agent_base_name = agent.replace("_info", "")
                    if agent_name and agent_base_name != agent_name:
                        continue
                        
                    for info in outputs:
                        if info_type and info.get("info_type") != info_type:
                            continue
                        agent_info.append(info)
            
            if latest and agent_info:
                agent_info = sorted(agent_info, key=lambda x: x.get("timestamp", ""), reverse=True)
                if agent_name and info_type:
                    # 특정 에이전트의 특정 타입 정보 중 최신
                    return [agent_info[0]]
                    
            return agent_info
            
        except Exception as e:
            print(f"❌ 에이전트 정보 조회 실패: {e}")
            return []
            
    def get_all_info(self) -> List[Dict]:
        """모든 정보 조회"""
        return self.get_agent_info(agent_name=None, info_type=None, latest=False)

class AgentDecisionLogger:
    """간소화된 에이전트 로거 (명확한 저장 구조)"""
    
    def __init__(self):
        self.current_session_id = self._generate_session_id()
        # 응답 관리자 (Cosmos DB 사용)
        self.output_manager = AgentOutputManager()

    def _generate_session_id(self) -> str:
        """세션 ID 생성"""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def log_agent_real_output(self,
                             agent_name: str,
                             agent_role: str,
                             task_description: str,
                             final_answer: str,
                             reasoning_process: str = "",
                             execution_steps: List[str] = None,
                             raw_input: Any = None,
                             raw_output: Any = None,
                             performance_metrics: Dict = None,
                             error_logs: List[Dict] = None,
                             info_data: Dict = None) -> str:
        """에이전트 응답 로깅"""
        
        return self.output_manager.store_agent_output(
            agent_name=agent_name,
            agent_role=agent_role,
            task_description=task_description,
            final_answer=final_answer,
            reasoning_process=reasoning_process,
            execution_steps=execution_steps,
            raw_input=raw_input,
            raw_output=raw_output,
            performance_metrics=performance_metrics,
            error_logs=error_logs,
            info_data=info_data
        )

    def log_agent_info(self,
                      agent_name: str,
                      info_type: str,
                      info_content: Dict,
                      metadata: Dict = None) -> str:
        """에이전트 정보 로깅 (새로운 기능)"""
        
        return self.output_manager.store_agent_info(
            agent_name=agent_name,
            info_type=info_type,
            info_content=info_content,
            metadata=metadata
        )

    def get_agent_info(self, agent_name: str = None, info_type: str = None, latest: bool = True) -> List[Dict]:
        """에이전트 정보 조회 (새로운 기능)"""
        return self.output_manager.get_agent_info(agent_name, info_type, latest)

    def get_all_info(self) -> List[Dict]:
        """모든 정보 조회 (새로운 기능)"""
        return self.output_manager.get_all_info()

    def get_all_previous_results(self, current_agent: str) -> List[Dict]:
        """모든 이전 응답 조회"""
        return self.output_manager.get_all_outputs(exclude_agent=current_agent)

    def get_previous_agent_result(self, agent_name: str, latest: bool = True) -> Optional[Dict]:
        """이전 에이전트 응답 조회"""
        return self.output_manager.get_agent_output(agent_name, latest)

    def get_learning_insights(self, target_agent: str) -> Dict:
        """학습 인사이트 생성 (간소화)"""
        
        all_outputs = self.output_manager.get_all_outputs()
        
        if not all_outputs:
            return {
                "insights": "이전 에이전트 응답이 없습니다.",
                "patterns": [],
                "recommendations": []
            }
        
        # 간단한 패턴 분석
        patterns = self._analyze_output_patterns(all_outputs)
        recommendations = self._generate_recommendations(patterns, target_agent)
        
        return {
            "target_agent": target_agent,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_outputs_analyzed": len(all_outputs),
            "patterns": patterns,
            "recommendations": recommendations,
            "insights": self._extract_insights(all_outputs, target_agent)
        }

    def _analyze_output_patterns(self, outputs: List[Dict]) -> List[Dict]:
        """응답 패턴 분석"""
        
        # 에이전트별 응답 그룹화
        agent_groups = {}
        
        for output in outputs:
            agent_name = output.get("agent_name", "unknown")
            if agent_name not in agent_groups:
                agent_groups[agent_name] = []
            agent_groups[agent_name].append(output)
        
        patterns = []
        
        # 각 에이전트별 패턴 분석
        for agent_name, agent_outputs in agent_groups.items():
            # 단순 패턴: 응답 길이 평균 및 표준편차
            if not agent_outputs:
                continue
                
            final_answers = [output.get("final_answer", "") for output in agent_outputs]
            avg_length = sum(len(ans) for ans in final_answers) / len(final_answers)
            
            patterns.append({
                "agent": agent_name,
                "response_count": len(agent_outputs),
                "avg_response_length": avg_length,
                "response_pattern": "text" if avg_length > 0 else "structured"
            })
        
        return patterns

    def _generate_recommendations(self, patterns: List[Dict], target_agent: str) -> List[str]:
        """추천 생성"""
        
        if not patterns:
            return ["분석할 패턴이 없습니다."]
            
        recommendations = []
        
        # 타겟 에이전트 패턴
        target_pattern = None
        for pattern in patterns:
            if pattern["agent"] == target_agent:
                target_pattern = pattern
                
        # 응답 길이 관련 추천
        if target_pattern:
            if target_pattern["avg_response_length"] > 500:
                recommendations.append(f"{target_agent}의 응답이 긴 편입니다. 보다 간결한 응답을 고려하세요.")
            elif target_pattern["avg_response_length"] < 20:
                recommendations.append(f"{target_agent}의 응답이 매우 짧습니다. 보다 상세한 응답이 필요할 수 있습니다.")
                
        return recommendations

    def _extract_insights(self, outputs: List[Dict], target_agent: str) -> List[str]:
        """인사이트 추출"""
        
        insights = []
        
        # 타겟 에이전트 출력만 필터링
        target_outputs = [output for output in outputs if output.get("agent_name") == target_agent]
        
        if not target_outputs:
            insights.append(f"{target_agent}의 이전 응답이 없습니다.")
            return insights
            
        # 시간순 정렬
        target_outputs.sort(key=lambda x: x.get("timestamp", ""))
        
        # 최신 응답
        latest_output = target_outputs[-1]
        latest_task = latest_output.get("task_description", "")
        
        insights.append(f"{target_agent}의 최근 작업: {latest_task}")
        
        # 성능 메트릭스 분석
        metrics = [output.get("performance_metrics", {}) for output in target_outputs]
        if metrics and all("response_length" in m for m in metrics):
            avg_length = sum(m.get("response_length", 0) for m in metrics) / len(metrics)
            insights.append(f"평균 응답 길이: {avg_length:.1f}")
            
        return insights

    def log_agent_decision(self, agent_name: str, agent_role: str, input_data: Dict,
                          decision_process: Dict, output_result: Dict, reasoning: str,
                          confidence_score: float = 0.8, context: Dict = None,
                          performance_metrics: Dict = None) -> str:
        """에이전트 결정 로깅 (이전 버전 호환성 유지)"""
        
        metrics = performance_metrics or {}
        metrics["confidence_score"] = confidence_score
        
        return self.log_agent_real_output(
            agent_name=agent_name,
            agent_role=agent_role,
            task_description=f"결정: {list(decision_process.keys())[0] if decision_process else ''}",
            final_answer=output_result.get("answer", str(output_result)),
            reasoning_process=reasoning,
            raw_input=input_data,
            raw_output=output_result,
            performance_metrics=metrics,
            info_data=context
        )

    def log_agent_interaction(self,
                             source_agent: str,
                             target_agent: str,
                             interaction_type: str,
                             data_transferred: Dict,
                             success: bool = True) -> str:
        """에이전트 간 상호작용 로깅"""
        
        return self.log_agent_real_output(
            agent_name=f"{source_agent}_to_{target_agent}",
            agent_role="상호작용",
            task_description=f"{interaction_type} 상호작용",
            final_answer=f"성공: {success}",
            raw_input={
                "source": source_agent,
                "target": target_agent,
                "type": interaction_type
            },
            raw_output=data_transferred,
            performance_metrics={
                "success": success,
                "interaction_type": interaction_type
            }
        )


def get_agent_logger() -> AgentDecisionLogger:
    """전역 에이전트 로거 인스턴스 반환"""
    # 싱글톤 패턴
    if not hasattr(get_agent_logger, "instance"):
        get_agent_logger.instance = AgentDecisionLogger()
    return get_agent_logger.instance


def get_real_output_manager() -> AgentOutputManager:
    """전역 에이전트 출력 관리자 인스턴스 반환"""
    # 싱글톤 패턴
    if not hasattr(get_real_output_manager, "instance"):
        get_real_output_manager.instance = AgentOutputManager()
    return get_real_output_manager.instance


# 간소화된 함수 인터페이스를 제공하기 위한 클래스 메서드 래퍼
def log_agent_decision(self, agent_name: str, agent_role: str = None, input_data: Dict = None,
                      decision_process: Dict = None, output_result: Dict = None, reasoning: str = "",
                      confidence_score: float = 0.8, context: Dict = None,
                      performance_metrics: Dict = None) -> str:
    """에이전트 결정 로깅 편의 함수"""
    logger = get_agent_logger()
    return logger.log_agent_decision(
        agent_name=agent_name,
        agent_role=agent_role or "에이전트",
        input_data=input_data or {},
        decision_process=decision_process or {"default": []},
        output_result=output_result or {},
        reasoning=reasoning,
        confidence_score=confidence_score,
        context=context,
        performance_metrics=performance_metrics
    )


def get_complete_data_manager() -> AgentOutputManager:
    """Cosmos DB 연결된 에이전트 출력 관리자 인스턴스 반환"""
    # 싱글톤 패턴
    if not hasattr(get_complete_data_manager, "instance"):
        # 세션 ID 제공하여 인스턴스 생성
        get_complete_data_manager.instance = AgentOutputManager()
    return get_complete_data_manager.instance



