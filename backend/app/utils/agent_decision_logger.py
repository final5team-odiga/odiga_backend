import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class AgentDecision:
    """에이전트 의사결정 로그"""
    agent_name: str
    agent_role: str
    decision_id: str
    timestamp: str
    input_data: Dict
    decision_process: Dict
    output_result: Dict
    reasoning: str
    confidence_score: float
    context: Dict
    performance_metrics: Dict

@dataclass
class AgentInteraction:
    """에이전트 간 상호작용 로그"""
    interaction_id: str
    source_agent: str
    target_agent: str
    interaction_type: str  # "handoff", "collaboration", "feedback"
    data_transferred: Dict
    success: bool
    timestamp: str

class AgentDecisionLogger:
    """에이전트 의사결정 로깅 및 학습 시스템"""
    
    def __init__(self, log_dir: str = "./agent_logs"):
        self.log_dir = log_dir
        self.current_session_id = self._generate_session_id()
        self.decision_logs = []
        self.interaction_logs = []
        
        # 로그 디렉토리 생성
        os.makedirs(log_dir, exist_ok=True)
        
        # 세션별 로그 파일 경로
        self.session_log_path = os.path.join(log_dir, f"session_{self.current_session_id}.json")
        self.cumulative_log_path = os.path.join(log_dir, "cumulative_decisions.json")
        self.learning_insights_path = os.path.join(log_dir, "learning_insights.json")
    
    def _generate_session_id(self) -> str:
        """세션 ID 생성"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_agent_decision(self, 
                          agent_name: str,
                          agent_role: str,
                          input_data: Dict,
                          decision_process: Dict,
                          output_result: Dict,
                          reasoning: str,
                          confidence_score: float = 0.8,
                          context: Dict = None,
                          performance_metrics: Dict = None) -> str:
        """에이전트 의사결정 로깅"""
        
        decision_id = f"{agent_name}_{int(time.time() * 1000)}"
        
        decision = AgentDecision(
            agent_name=agent_name,
            agent_role=agent_role,
            decision_id=decision_id,
            timestamp=datetime.now().isoformat(),
            input_data=input_data,
            decision_process=decision_process,
            output_result=output_result,
            reasoning=reasoning,
            confidence_score=confidence_score,
            context=context or {},
            performance_metrics=performance_metrics or {}
        )
        
        self.decision_logs.append(decision)
        
        # 실시간 로그 저장
        self._save_session_logs()
        
        print(f"📝 {agent_name} 의사결정 로그 기록: {decision_id}")
        return decision_id
    
    def log_agent_interaction(self,
                             source_agent: str,
                             target_agent: str,
                             interaction_type: str,
                             data_transferred: Dict,
                             success: bool = True) -> str:
        """에이전트 간 상호작용 로깅"""
        
        interaction_id = f"{source_agent}_to_{target_agent}_{int(time.time() * 1000)}"
        
        interaction = AgentInteraction(
            interaction_id=interaction_id,
            source_agent=source_agent,
            target_agent=target_agent,
            interaction_type=interaction_type,
            data_transferred=data_transferred,
            success=success,
            timestamp=datetime.now().isoformat()
        )
        
        self.interaction_logs.append(interaction)
        
        # 실시간 로그 저장
        self._save_session_logs()
        
        print(f"🔄 에이전트 상호작용 로그: {source_agent} → {target_agent} ({interaction_type})")
        return interaction_id
    
    def get_previous_decisions(self, agent_name: str = None, limit: int = 10) -> List[Dict]:
        """이전 의사결정 로그 조회"""
        
        # 누적 로그에서 이전 의사결정 로드
        previous_decisions = self._load_cumulative_logs()
        
        if agent_name:
            previous_decisions = [
                decision for decision in previous_decisions 
                if decision.get('agent_name') == agent_name
            ]
        
        # 최신 순으로 정렬하여 제한된 수만 반환
        return sorted(previous_decisions, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]
    
    def get_learning_insights(self, target_agent: str) -> Dict:
        """특정 에이전트를 위한 학습 인사이트 생성"""
        
        previous_decisions = self.get_previous_decisions(limit=50)
        
        if not previous_decisions:
            return {
                "insights": "이전 의사결정 로그가 없습니다.",
                "patterns": [],
                "recommendations": []
            }
        
        # 패턴 분석
        patterns = self._analyze_decision_patterns(previous_decisions)
        
        # 성공/실패 분석
        performance_analysis = self._analyze_performance_patterns(previous_decisions)
        
        # 추천사항 생성
        recommendations = self._generate_recommendations(patterns, performance_analysis, target_agent)
        
        insights = {
            "target_agent": target_agent,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_decisions_analyzed": len(previous_decisions),
            "patterns": patterns,
            "performance_analysis": performance_analysis,
            "recommendations": recommendations,
            "key_insights": self._extract_key_insights(previous_decisions, target_agent)
        }
        
        # 인사이트 저장
        self._save_learning_insights(insights)
        
        return insights
    
    def _analyze_decision_patterns(self, decisions: List[Dict]) -> List[Dict]:
        """의사결정 패턴 분석"""
        
        patterns = []
        
        # 에이전트별 의사결정 빈도
        agent_frequency = {}
        for decision in decisions:
            agent_name = decision.get('agent_name', 'unknown')
            agent_frequency[agent_name] = agent_frequency.get(agent_name, 0) + 1
        
        patterns.append({
            "type": "agent_activity",
            "description": "에이전트별 활동 빈도",
            "data": agent_frequency
        })
        
        # 신뢰도 점수 분포
        confidence_scores = [d.get('confidence_score', 0) for d in decisions if d.get('confidence_score')]
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            patterns.append({
                "type": "confidence_distribution",
                "description": "평균 신뢰도 점수",
                "data": {
                    "average": round(avg_confidence, 3),
                    "min": min(confidence_scores),
                    "max": max(confidence_scores),
                    "count": len(confidence_scores)
                }
            })
        
        # 의사결정 시간 패턴
        timestamps = [d.get('timestamp') for d in decisions if d.get('timestamp')]
        if len(timestamps) > 1:
            patterns.append({
                "type": "temporal_pattern",
                "description": "의사결정 시간 분포",
                "data": {
                    "first_decision": timestamps[-1],  # 가장 오래된 것
                    "last_decision": timestamps[0],    # 가장 최근 것
                    "total_span": len(timestamps)
                }
            })
        
        return patterns
    
    def _analyze_performance_patterns(self, decisions: List[Dict]) -> Dict:
        """성능 패턴 분석"""
        
        performance_data = []
        for decision in decisions:
            metrics = decision.get('performance_metrics', {})
            if metrics:
                performance_data.append(metrics)
        
        if not performance_data:
            return {"message": "성능 메트릭 데이터 없음"}
        
        # 성능 지표 평균 계산
        avg_metrics = {}
        for metrics in performance_data:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in avg_metrics:
                        avg_metrics[key] = []
                    avg_metrics[key].append(value)
        
        # 평균값 계산
        for key, values in avg_metrics.items():
            avg_metrics[key] = {
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        
        return {
            "performance_metrics": avg_metrics,
            "total_samples": len(performance_data)
        }
    
    def _generate_recommendations(self, patterns: List[Dict], performance: Dict, target_agent: str) -> List[str]:
        """추천사항 생성"""
        
        recommendations = []
        
        # 신뢰도 기반 추천
        for pattern in patterns:
            if pattern["type"] == "confidence_distribution":
                avg_confidence = pattern["data"].get("average", 0)
                if avg_confidence < 0.7:
                    recommendations.append(
                        f"{target_agent}는 이전 에이전트들의 낮은 신뢰도({avg_confidence:.2f})를 고려하여 "
                        "더 신중한 검증 과정을 거쳐야 합니다."
                    )
                elif avg_confidence > 0.9:
                    recommendations.append(
                        f"이전 에이전트들의 높은 신뢰도({avg_confidence:.2f})를 바탕으로 "
                        f"{target_agent}는 기존 결과를 적극 활용할 수 있습니다."
                    )
        
        # 활동 패턴 기반 추천
        for pattern in patterns:
            if pattern["type"] == "agent_activity":
                most_active_agent = max(pattern["data"].items(), key=lambda x: x[1])
                recommendations.append(
                    f"가장 활발했던 {most_active_agent[0]} 에이전트의 의사결정 패턴을 "
                    f"{target_agent}가 참고하면 도움이 될 것입니다."
                )
        
        # 성능 기반 추천
        if performance.get("performance_metrics"):
            recommendations.append(
                f"{target_agent}는 이전 에이전트들의 성능 메트릭을 참고하여 "
                "유사한 품질 수준을 유지하거나 개선해야 합니다."
            )
        
        return recommendations
    
    def _extract_key_insights(self, decisions: List[Dict], target_agent: str) -> List[str]:
        """핵심 인사이트 추출"""
        
        insights = []
        
        if not decisions:
            return ["이전 의사결정 데이터가 없어 인사이트를 제공할 수 없습니다."]
        
        # 최근 의사결정 트렌드
        recent_decisions = decisions[:5]  # 최근 5개
        recent_agents = [d.get('agent_name') for d in recent_decisions]
        
        if recent_agents:
            most_recent_agent = recent_agents[0]
            insights.append(
                f"가장 최근에 활동한 {most_recent_agent} 에이전트의 결과를 "
                f"{target_agent}가 우선적으로 검토해야 합니다."
            )
        
        # 성공 패턴 식별
        high_confidence_decisions = [
            d for d in decisions 
            if d.get('confidence_score', 0) > 0.8
        ]
        
        if high_confidence_decisions:
            insights.append(
                f"신뢰도가 높은 의사결정 {len(high_confidence_decisions)}개를 발견했습니다. "
                f"{target_agent}는 이들의 접근 방식을 참고할 수 있습니다."
            )
        
        # 일관성 분석
        reasoning_patterns = [d.get('reasoning', '') for d in decisions if d.get('reasoning')]
        if len(reasoning_patterns) > 3:
            insights.append(
                f"이전 에이전트들의 추론 패턴을 분석한 결과, "
                f"{target_agent}는 일관된 논리적 접근을 유지해야 합니다."
            )
        
        return insights
    
    def _save_session_logs(self):
        """현재 세션 로그 저장"""
        
        session_data = {
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "decisions": [asdict(decision) for decision in self.decision_logs],
            "interactions": [asdict(interaction) for interaction in self.interaction_logs]
        }
        
        with open(self.session_log_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        # 누적 로그에도 추가
        self._update_cumulative_logs()
    
    def _update_cumulative_logs(self):
        """누적 로그 업데이트"""
        
        # 기존 누적 로그 로드
        cumulative_data = self._load_cumulative_logs()
        
        # 현재 세션의 의사결정 추가
        for decision in self.decision_logs:
            decision_dict = asdict(decision)
            # 중복 방지
            if not any(d.get('decision_id') == decision_dict['decision_id'] for d in cumulative_data):
                cumulative_data.append(decision_dict)
        
        # 최신 100개만 유지 (메모리 관리)
        cumulative_data = sorted(
            cumulative_data, 
            key=lambda x: x.get('timestamp', ''), 
            reverse=True
        )[:100]
        
        # 저장
        with open(self.cumulative_log_path, 'w', encoding='utf-8') as f:
            json.dump(cumulative_data, f, ensure_ascii=False, indent=2)
    
    def _load_cumulative_logs(self) -> List[Dict]:
        """누적 로그 로드"""
        
        if not os.path.exists(self.cumulative_log_path):
            return []
        
        try:
            with open(self.cumulative_log_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    
    def _save_learning_insights(self, insights: Dict):
        """학습 인사이트 저장"""
        
        # 기존 인사이트 로드
        existing_insights = []
        if os.path.exists(self.learning_insights_path):
            try:
                with open(self.learning_insights_path, 'r', encoding='utf-8') as f:
                    existing_insights = json.load(f)
            except:
                existing_insights = []
        
        # 새 인사이트 추가
        existing_insights.append(insights)
        
        # 최신 10개만 유지
        existing_insights = existing_insights[-10:]
        
        # 저장
        with open(self.learning_insights_path, 'w', encoding='utf-8') as f:
            json.dump(existing_insights, f, ensure_ascii=False, indent=2)

# 전역 로거 인스턴스
_global_logger = None

def get_agent_logger() -> AgentDecisionLogger:
    """전역 에이전트 로거 인스턴스 반환"""
    global _global_logger
    if _global_logger is None:
        _global_logger = AgentDecisionLogger()
    return _global_logger
