import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

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

class AgentOutputManager:
    """에이전트 응답 전용 관리 시스템 (수정된 저장 구조)"""
    
    def __init__(self, storage_dir: str = "./agent_outputs"):
        self.storage_dir = storage_dir
        self.current_session_id = self._generate_session_id()
        self.outputs = []  # 에이전트 응답만 저장
        
        # 저장 디렉토리 생성 (수정: 이중 저장 구조)
        os.makedirs(storage_dir, exist_ok=True)
        
        # agent_outputs 폴더에 저장
        self.outputs_dir = os.path.join(storage_dir, "outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)
        
        # 세션별 저장
        self.session_path = os.path.join(self.outputs_dir, f"session_{self.current_session_id}")
        os.makedirs(self.session_path, exist_ok=True)
        
        # 저장 파일 경로들
        self.outputs_path = os.path.join(self.session_path, "agent_outputs.json")
        self.summary_path = os.path.join(self.outputs_dir, "outputs_summary.json")
        self.latest_path = os.path.join(storage_dir, "latest_outputs.json")
        
    
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
                          error_logs: List[Dict] = None) -> str:
        """에이전트 응답 저장 (다중 위치 저장)"""
        
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
            error_logs=error_logs or []
        )
        
        self.outputs.append(agent_output)
        
        # 다중 위치 저장
        self._save_outputs()
        self._save_latest_outputs()
        self._update_summary()
        
        print(f"📦 {agent_name} 응답 저장: {output_id}")
        print(f"  - 세션 저장: {self.outputs_path}")
        print(f"  - 최신 저장: {self.latest_path}")
        
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
        all_outputs = []
        
        for output in self.outputs:
            if exclude_agent is None or output.agent_name != exclude_agent:
                all_outputs.append(asdict(output))
        
        # 이전 세션 출력도 로드
        previous_outputs = self._load_previous_outputs()
        for output in previous_outputs:
            if exclude_agent is None or output.get('agent_name') != exclude_agent:
                if not any(o.get('output_id') == output.get('output_id') for o in all_outputs):
                    all_outputs.append(output)
        
        return sorted(all_outputs, key=lambda x: x.get('timestamp', ''))
    
    def get_agent_output(self, agent_name: str, latest: bool = True) -> Optional[Dict]:
        """특정 에이전트의 응답 조회"""
        agent_outputs = [
            asdict(output) for output in self.outputs
            if output.agent_name == agent_name
        ]
        
        if not agent_outputs:
            # 이전 세션에서 조회
            previous_outputs = self._load_previous_outputs()
            agent_outputs = [o for o in previous_outputs if o.get('agent_name') == agent_name]
        
        if not agent_outputs:
            return None
        
        if latest:
            return sorted(agent_outputs, key=lambda x: x.get('timestamp', ''), reverse=True)[0]
        else:
            return agent_outputs
    
    def _save_outputs(self):
        """세션별 응답 저장"""
        outputs_data = {
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "agent_outputs": [asdict(output) for output in self.outputs],
            "total_outputs": len(self.outputs),
            "storage_info": {
                "session_path": self.session_path,
                "outputs_path": self.outputs_path
            }
        }
        
        with open(self.outputs_path, 'w', encoding='utf-8') as f:
            json.dump(outputs_data, f, ensure_ascii=False, indent=2)
    
    def _save_latest_outputs(self):
        """최신 출력을 agent_outputs 폴더 루트에 저장"""
        latest_data = {
            "last_updated": datetime.now().isoformat(),
            "current_session_id": self.current_session_id,
            "total_outputs_in_session": len(self.outputs),
            "latest_outputs": [asdict(output) for output in self.outputs[-10:]],  # 최신 10개만
            "storage_locations": {
                "full_session_data": self.outputs_path,
                "summary_data": self.summary_path,
                "outputs_directory": self.outputs_dir
            }
        }
        
        with open(self.latest_path, 'w', encoding='utf-8') as f:
            json.dump(latest_data, f, ensure_ascii=False, indent=2)
    
    def _update_summary(self):
        """출력 요약 정보 업데이트"""
        # 기존 요약 로드
        existing_summary = self._load_summary()
        
        # 에이전트별 통계 계산
        agent_stats = {}
        for output in self.outputs:
            agent_name = output.agent_name
            if agent_name not in agent_stats:
                agent_stats[agent_name] = {
                    "count": 0,
                    "total_answer_length": 0,
                    "latest_timestamp": "",
                    "latest_task": ""
                }
            
            agent_stats[agent_name]["count"] += 1
            agent_stats[agent_name]["total_answer_length"] += len(output.final_answer)
            agent_stats[agent_name]["latest_timestamp"] = output.timestamp
            agent_stats[agent_name]["latest_task"] = output.task_description[:100]
        
        # 요약 데이터 생성
        summary_data = {
            "last_updated": datetime.now().isoformat(),
            "current_session": {
                "session_id": self.current_session_id,
                "total_outputs": len(self.outputs),
                "agent_stats": agent_stats
            },
            "all_sessions": existing_summary.get("all_sessions", []) + [{
                "session_id": self.current_session_id,
                "output_count": len(self.outputs),
                "timestamp": datetime.now().isoformat()
            }],
            "storage_info": {
                "outputs_directory": self.outputs_dir,
                "current_session_path": self.session_path,
                "total_sessions": len(existing_summary.get("all_sessions", [])) + 1
            }
        }
        
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    def _load_summary(self) -> Dict:
        """기존 요약 데이터 로드"""
        try:
            with open(self.summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"all_sessions": []}
    
    def _load_previous_outputs(self) -> List[Dict]:
        """이전 세션 출력 로드"""
        try:
            with open(self.outputs_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('agent_outputs', [])
        except:
            return []

class AgentDecisionLogger:
    """간소화된 에이전트 로거 (명확한 저장 구조)"""
    
    def __init__(self):
        self.current_session_id = self._generate_session_id()
        
        # 응답 관리자 (agent_outputs 폴더 사용)
        self.output_manager = AgentOutputManager("./agent_outputs") 
        
        
    
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
                             error_logs: List[Dict] = None) -> str:
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
            error_logs=error_logs
        )
    
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
            "key_insights": self._extract_insights(all_outputs, target_agent)
        }
    
    def _analyze_output_patterns(self, outputs: List[Dict]) -> List[Dict]:
        """응답 패턴 분석 (간소화)"""
        patterns = []
        
        # 에이전트별 응답 길이 패턴
        agent_answer_lengths = {}
        for output in outputs:
            agent_name = output.get('agent_name', 'unknown')
            answer_length = len(output.get('final_answer', ''))
            
            if agent_name not in agent_answer_lengths:
                agent_answer_lengths[agent_name] = []
            agent_answer_lengths[agent_name].append(answer_length)
        
        patterns.append({
            "type": "answer_length_patterns",
            "description": "에이전트별 응답 길이 패턴",
            "data": {
                agent: {
                    "avg_length": sum(lengths) / len(lengths),
                    "max_length": max(lengths),
                    "min_length": min(lengths)
                }
                for agent, lengths in agent_answer_lengths.items()
            }
        })
        
        return patterns
    
    def _generate_recommendations(self, patterns: List[Dict], target_agent: str) -> List[str]:
        """추천사항 생성 (간소화)"""
        recommendations = []
        
        for pattern in patterns:
            if pattern["type"] == "answer_length_patterns":
                data = pattern["data"]
                if data:
                    # 평균 응답 길이가 긴 에이전트 찾기
                    best_agent = max(data.items(), key=lambda x: x[1]["avg_length"])
                    recommendations.append(
                        f"{target_agent}는 {best_agent[0]} 에이전트의 상세한 응답 스타일"
                        f"(평균 {best_agent[1]['avg_length']:.0f}자)을 참고하세요."
                    )
        
        return recommendations
    
    def _extract_insights(self, outputs: List[Dict], target_agent: str) -> List[str]:
        """인사이트 추출 (간소화)"""
        insights = []
        
        if not outputs:
            return ["이전 에이전트 응답이 없어 인사이트를 제공할 수 없습니다."]
        
        # 최신 응답 분석
        recent_outputs = sorted(outputs, key=lambda x: x.get('timestamp', ''), reverse=True)[:3]
        
        if recent_outputs:
            latest_agent = recent_outputs[0].get('agent_name')
            latest_answer = recent_outputs[0].get('final_answer', '')
            insights.append(
                f"가장 최근에 활동한 {latest_agent} 에이전트의 응답"
                f"({len(latest_answer)}자)을 {target_agent}가 참고하세요."
            )
        
        # 에러 없는 고품질 응답 식별
        error_free_outputs = [o for o in outputs if not o.get('error_logs')]
        if error_free_outputs:
            insights.append(
                f"에러 없는 고품질 응답 {len(error_free_outputs)}개를 발견했습니다. "
                f"{target_agent}는 이들의 응답 패턴을 참고하세요."
            )
        
        return insights
    
    # 호환성을 위한 기존 메서드들 (간소화)
    def log_agent_decision(self, agent_name: str, agent_role: str, input_data: Dict,
                          decision_process: Dict, output_result: Dict, reasoning: str,
                          confidence_score: float = 0.8, context: Dict = None,
                          performance_metrics: Dict = None) -> str:
        """기존 호환성 메서드 (응답만 저장)"""
        
        return self.log_agent_real_output(
            agent_name=agent_name,
            agent_role=agent_role,
            task_description=str(input_data),
            final_answer=str(output_result),
            reasoning_process=reasoning,
            raw_input=input_data,
            raw_output=output_result,
            performance_metrics=performance_metrics,
            decision_process=decision_process.get('steps', []),
        )
    
    def log_agent_interaction(self,
                             source_agent: str,
                             target_agent: str,
                             interaction_type: str,
                             data_transferred: Dict,
                             success: bool = True) -> str:
        """에이전트 간 상호작용 로깅 (간소화)"""
        
        # 상호작용도 응답으로 저장
        return self.log_agent_real_output(
            agent_name=f"{source_agent}_to_{target_agent}",
            agent_role="에이전트 상호작용",
            task_description=f"{interaction_type} 상호작용",
            final_answer=f"{source_agent}에서 {target_agent}로 데이터 전달",
            reasoning_process=f"상호작용 타입: {interaction_type}",
            raw_input={"source": source_agent, "target": target_agent},
            raw_output=data_transferred,
            performance_metrics={"success": success}
        )

# 전역 인스턴스
_global_logger = None
_global_output_manager = None

def get_agent_logger() -> AgentDecisionLogger:
    """전역 에이전트 로거 인스턴스 반환"""
    global _global_logger
    if _global_logger is None:
        _global_logger = AgentDecisionLogger()
    return _global_logger

def get_real_output_manager() -> AgentOutputManager:
    """전역 응답 관리자 인스턴스 반환"""
    global _global_output_manager
    if _global_output_manager is None:
        _global_output_manager = AgentOutputManager()
    return _global_output_manager

def get_complete_data_manager() -> AgentOutputManager:
    """호환성을 위한 별칭"""
    return get_real_output_manager()
