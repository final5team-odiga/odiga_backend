import logging
import time
import os
from typing import List, Dict, Any
import sys
import io
from utils.agent_decision_logger import get_agent_logger


if sys.platform.startswith('win'):
    # Windows에서 UTF-8 인코딩 강제 설정
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

class HybridLogger:
    def __init__(self, name: str, agent_logger=None):  # ✅ 파라미터 추가
        self.class_name = name
        self.standard_logger = logging.getLogger(name)
        
        self.agent_logger = agent_logger or self._create_safe_agent_logger()
        
        if not self.standard_logger.handlers:
            # 콘솔 핸들러
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # ✅ UTF-8 인코딩 지원 포매터
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            console_handler.setFormatter(formatter)
            self.standard_logger.addHandler(console_handler)
            self.standard_logger.setLevel(logging.INFO)

        # 하이브리드 로깅 상태
        self.hybrid_enabled = True
        self.fallback_mode = False
        self.standard_logger.info(f"{self.class_name} 하이브리드 로깅 시스템 초기화 완료")


    def _setup_standard_logger(self):
        """표준 로거 설정"""
        try:
            # 핸들러가 없는 경우에만 추가
            if not self.standard_logger.handlers:
                # 콘솔 핸들러
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                
                # 포맷터
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(formatter)
                
                self.standard_logger.addHandler(console_handler)
                self.standard_logger.setLevel(logging.INFO)
                
                # 파일 핸들러 (선택적)
                log_dir = "./logs"
                os.makedirs(log_dir, exist_ok=True)
                file_handler = logging.FileHandler(
                    os.path.join(log_dir, f"{self.class_name}.log")
                )
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                self.standard_logger.addHandler(file_handler)
                
        except Exception as e:
            print(f"표준 로거 설정 실패: {e}")

    def _create_safe_agent_logger(self):
        """안전한 에이전트 로거 생성"""
        try:
            return get_agent_logger()
        except Exception as e:
            self.standard_logger.warning(f"에이전트 로거 생성 실패, 안전 모드 사용: {e}")
            self.fallback_mode = True
            return SafeAgentLogger(self.standard_logger)

    # ==================== 표준 로깅 메서드들 ====================
    
    def info(self, message, *args, **kwargs):
        try:
            # ✅ 이모지 안전 처리
            safe_message = self._make_emoji_safe(message)
            return self.standard_logger.info(safe_message, *args, **kwargs)
        except UnicodeEncodeError:
            # 이모지 제거 후 재시도
            emoji_free_message = self._remove_emojis(message)
            return self.standard_logger.info(emoji_free_message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        try:
            safe_message = self._make_emoji_safe(message)
            return self.standard_logger.error(safe_message, *args, **kwargs)
        except UnicodeEncodeError:
            emoji_free_message = self._remove_emojis(message)
            return self.standard_logger.error(emoji_free_message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        try:
            safe_message = self._make_emoji_safe(message)
            return self.standard_logger.warning(safe_message, *args, **kwargs)
        except UnicodeEncodeError:
            emoji_free_message = self._remove_emojis(message)
            return self.standard_logger.warning(emoji_free_message, *args, **kwargs)
    
    def debug(self, message, *args, **kwargs):
        try:
            safe_message = self._make_emoji_safe(message)
            return self.standard_logger.debug(safe_message, *args, **kwargs)
        except UnicodeEncodeError:
            emoji_free_message = self._remove_emojis(message)
            return self.standard_logger.debug(emoji_free_message, *args, **kwargs)
    
    def _make_emoji_safe(self, message: str) -> str:
        """이모지를 안전하게 처리"""
        if sys.platform.startswith('win'):
            try:
                # UTF-8로 인코딩 가능한지 테스트
                message.encode('utf-8')
                return message
            except UnicodeEncodeError:
                return self._remove_emojis(message)
        return message
    
    def _remove_emojis(self, message: str) -> str:
        """이모지 제거 또는 대체"""
        import re
        
        # 이모지 매핑
        emoji_map = {
            '📦': '[PACKAGE]',
            '✅': '[SUCCESS]',
            '❌': '[ERROR]',
            '⚠️': '[WARNING]',
            '📱': '[MOBILE]',
            '🎨': '[ART]',
            '🚀': '[ROCKET]',
            '📊': '[CHART]',
            '🛡️': '[SHIELD]',
            '📝': '[NOTE]',
            '📁': '[FOLDER]'
        }
        
        # 이모지 대체
        for emoji, replacement in emoji_map.items():
            message = message.replace(emoji, replacement)
        
        # 남은 이모지 제거 (유니코드 범위 기반)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # 감정 표현
            "\U0001F300-\U0001F5FF"  # 기호 및 픽토그램
            "\U0001F680-\U0001F6FF"  # 교통 및 지도
            "\U0001F1E0-\U0001F1FF"  # 국기
            "\U00002702-\U000027B0"  # 기타 기호
            "\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
        
        return emoji_pattern.sub('[EMOJI]', message)



    # ==================== 에이전트 로깅 메서드들 ====================
    
    def log_agent_decision(self, agent_name: str, agent_role: str = None, 
                          input_data: Dict = None, decision_process: Dict = None, 
                          output_result: Dict = None, reasoning: str = "", 
                          confidence_score: float = 0.8, context: Dict = None,
                          performance_metrics: Dict = None) -> str:
        """에이전트 결정 로깅 - 호환성 보장"""
        try:
            # 기본값 설정으로 누락된 인수 문제 해결
            agent_role = agent_role or f"{agent_name} 에이전트"
            input_data = input_data or {}
            decision_process = decision_process or {"steps": ["결정 과정 기록"]}
            output_result = output_result or {"result": "처리 완료"}
            reasoning = reasoning or "에이전트 결정 처리"
            
            # 에이전트 로거가 있는 경우
            if self.agent_logger and hasattr(self.agent_logger, 'log_agent_decision'):
                return self.agent_logger.log_agent_decision(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    input_data=input_data,
                    decision_process=decision_process,
                    output_result=output_result,
                    reasoning=reasoning,
                    confidence_score=confidence_score,
                    context=context,
                    performance_metrics=performance_metrics
                )
            else:
                # 폴백: log_agent_real_output 사용
                return self.log_agent_real_output(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    task_description=str(input_data),
                    final_answer=str(output_result),
                    reasoning_process=reasoning,
                    raw_input=input_data,
                    raw_output=output_result,
                    performance_metrics=performance_metrics
                )
                
        except Exception as e:
            self.error(f"에이전트 결정 로깅 실패: {e}")
            # 최종 폴백: 표준 로깅
            self.info(f"Agent Decision (Fallback) - {agent_name}: {reasoning}")
            return f"fallback_{agent_name}_{int(time.time())}"

    def log_agent_real_output(self, agent_name: str, agent_role: str = None,
                             task_description: str = "", final_answer: str = "",
                             reasoning_process: str = "", execution_steps: List[str] = None,
                             raw_input: Any = None, raw_output: Any = None,
                             performance_metrics: Dict = None, error_logs: List[Dict] = None,
                             info_data: Dict = None) -> str:
        """에이전트 실제 출력 로깅"""
        try:
            # 기본값 설정
            agent_role = agent_role or f"{agent_name} 에이전트"
            task_description = task_description or "작업 수행"
            final_answer = final_answer or "처리 완료"
            
            if self.agent_logger and hasattr(self.agent_logger, 'log_agent_real_output'):
                return self.agent_logger.log_agent_real_output(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    task_description=task_description,
                    final_answer=final_answer,
                    reasoning_process=reasoning_process,
                    execution_steps=execution_steps or [],
                    raw_input=raw_input,
                    raw_output=raw_output,
                    performance_metrics=performance_metrics or {},
                    error_logs=error_logs or [],
                    info_data=info_data or {}
                )
            else:
                # 표준 로거로 폴백
                self.info(f"Agent Output - {agent_name} ({agent_role}): {task_description}")
                return f"standard_{agent_name}_{int(time.time())}"
                
        except Exception as e:
            self.error(f"에이전트 출력 로깅 실패: {e}")
            self.info(f"Agent Output (Error Fallback) - {agent_name}: {task_description}")
            return f"error_fallback_{agent_name}_{int(time.time())}"

    def log_agent_interaction(self, source_agent: str, target_agent: str,
                             interaction_type: str, data_transferred: Dict,
                             success: bool = True) -> str:
        """에이전트 간 상호작용 로깅"""
        try:
            if self.agent_logger and hasattr(self.agent_logger, 'log_agent_interaction'):
                return self.agent_logger.log_agent_interaction(
                    source_agent=source_agent,
                    target_agent=target_agent,
                    interaction_type=interaction_type,
                    data_transferred=data_transferred,
                    success=success
                )
            else:
                # 폴백: 상호작용을 출력으로 로깅
                return self.log_agent_real_output(
                    agent_name=f"{source_agent}_to_{target_agent}",
                    agent_role="에이전트 상호작용",
                    task_description=f"{interaction_type} 상호작용",
                    final_answer=f"데이터 전달 {'성공' if success else '실패'}",
                    raw_input={"source": source_agent, "target": target_agent},
                    raw_output=data_transferred,
                    performance_metrics={"success": success}
                )
                
        except Exception as e:
            self.error(f"에이전트 상호작용 로깅 실패: {e}")
            self.info(f"Agent Interaction (Fallback) - {source_agent} -> {target_agent}: {interaction_type}")
            return f"interaction_fallback_{int(time.time())}"

    def get_learning_insights(self, target_agent: str = None) -> Dict:
        """에이전트 학습 인사이트 추출"""
        try:
            if self.agent_logger and hasattr(self.agent_logger, 'get_learning_insights'):
                return self.agent_logger.get_learning_insights(target_agent)
            else:
                return self._create_fallback_insights(target_agent)
                
        except Exception as e:
            self.error(f"학습 인사이트 추출 실패: {e}")
            return self._create_fallback_insights(target_agent)

    def _create_fallback_insights(self, target_agent: str = None) -> Dict:
        """폴백 인사이트 생성"""
        return {
            "target_agent": target_agent or "unknown",
            "analysis_timestamp": time.time(),
            "total_outputs_analyzed": 0,
            "patterns": ["hybrid_fallback_mode"],
            "recommendations": ["하이브리드 로거 폴백 모드에서 실행"],
            "key_insights": f"하이브리드 로거 폴백 모드 - {self.class_name}",
            "fallback_mode": True,
            "logger_status": {
                "standard_logger_available": True,
                "agent_logger_available": not self.fallback_mode,
                "hybrid_enabled": self.hybrid_enabled
            }
        }

    # ==================== 하이브리드 전용 메서드들 ====================
    
    def log_hybrid_processing_start(self, approach_type: str, metadata: Dict = None):
        """하이브리드 처리 시작 로깅"""
        try:
            log_data = {
                "event": "hybrid_processing_start",
                "approach_type": approach_type,
                "timestamp": time.time(),
                "data_isolation_enabled": True,
                "metadata": metadata or {}
            }
            
            self.info(f"하이브리드 처리 시작: {approach_type}")
            
            # 에이전트 로거에도 기록
            return self.log_agent_real_output(
                agent_name="HybridSystem",
                agent_role="하이브리드 처리 시스템",
                task_description=f"하이브리드 처리 시작: {approach_type}",
                final_answer="처리 시작됨",
                raw_input={"approach_type": approach_type},
                raw_output=log_data,
                performance_metrics={"processing_started": True}
            )
            
        except Exception as e:
            self.error(f"하이브리드 처리 시작 로깅 실패: {e}")
            return None

    def log_hybrid_quality_report(self, quality_report: Dict, agent_name: str = "QualitySystem"):
        """하이브리드 품질 보고서 로깅"""
        try:
            overall_score = quality_report.get("overall_score", 0)
            
            self.info(f"품질 보고서 생성 완료 - 전체 점수: {overall_score:.2f}")
            
            # 에이전트 로거에 상세 기록
            return self.log_agent_real_output(
                agent_name=agent_name,
                agent_role="품질 평가 시스템",
                task_description="하이브리드 품질 보고서 생성",
                final_answer=f"품질 평가 완료 (점수: {overall_score:.2f})",
                raw_input={"report_request": True},
                raw_output=quality_report,
                performance_metrics={
                    "overall_score": overall_score,
                    "data_integrity": quality_report.get("data_integrity", {}),
                    "report_generated": True
                }
            )
            
        except Exception as e:
            self.error(f"품질 보고서 로깅 실패: {e}")
            return None

    def log_data_integrity_check(self, check_result: Dict, agent_name: str = "IntegrityChecker"):
        """데이터 무결성 검사 로깅"""
        try:
            integrity_score = check_result.get("overall_integrity_score", 0)
            
            self.info(f"데이터 무결성 검사 완료 - 무결성 점수: {integrity_score:.2f}")
            
            return self.log_agent_real_output(
                agent_name=agent_name,
                agent_role="데이터 무결성 검사기",
                task_description="원본 데이터 무결성 검증",
                final_answer=f"무결성 검사 완료 (점수: {integrity_score:.2f})",
                raw_output=check_result,
                performance_metrics={
                    "integrity_score": integrity_score,
                    "check_passed": integrity_score > 0.9
                }
            )
            
        except Exception as e:
            self.error(f"데이터 무결성 검사 로깅 실패: {e}")
            return None

    def log_guideline_application(self, guidelines_data: Dict, agent_name: str = "GuidelineApplicator"):
        """가이드라인 적용 로깅"""
        try:
            confidence_score = guidelines_data.get("quality_metrics", {}).get("confidence_level", 0)
            
            self.info(f"가이드라인 적용 완료 - 신뢰도: {confidence_score:.2f}")
            
            return self.log_agent_real_output(
                agent_name=agent_name,
                agent_role="가이드라인 적용기",
                task_description="AI Search 가이드라인 적용",
                final_answer=f"가이드라인 적용 완료 (신뢰도: {confidence_score:.2f})",
                raw_output=guidelines_data,
                performance_metrics={
                    "confidence_score": confidence_score,
                    "guidelines_applied": True,
                    "ai_search_isolated": True
                }
            )
            
        except Exception as e:
            self.error(f"가이드라인 적용 로깅 실패: {e}")
            return None

    # ==================== 유틸리티 메서드들 ====================
    
    def get_logger_status(self) -> Dict:
        """로거 상태 정보 반환"""
        return {
            "class_name": self.class_name,
            "standard_logger_name": self.standard_logger.name,
            "agent_logger_available": self.agent_logger is not None and not self.fallback_mode,
            "agent_logger_type": type(self.agent_logger).__name__,
            "hybrid_enabled": self.hybrid_enabled,
            "fallback_mode": self.fallback_mode,
            "handlers_count": len(self.standard_logger.handlers),
            "log_level": self.standard_logger.level
        }

    def enable_hybrid_mode(self):
        """하이브리드 모드 활성화"""
        self.hybrid_enabled = True
        self.info("하이브리드 로깅 모드 활성화")

    def disable_hybrid_mode(self):
        """하이브리드 모드 비활성화 (표준 로깅만 사용)"""
        self.hybrid_enabled = False
        self.info("하이브리드 로깅 모드 비활성화 - 표준 로깅만 사용")

    def test_logging_system(self):
        """로깅 시스템 테스트"""
        try:
            self.info("=== 하이브리드 로깅 시스템 테스트 시작 ===")
            
            # 표준 로깅 테스트
            self.debug("DEBUG 레벨 테스트")
            self.info("INFO 레벨 테스트")
            self.warning("WARNING 레벨 테스트")
            self.error("ERROR 레벨 테스트")
            
            # 에이전트 로깅 테스트
            test_output_id = self.log_agent_real_output(
                agent_name="TestAgent",
                agent_role="테스트 에이전트",
                task_description="로깅 시스템 테스트",
                final_answer="테스트 성공",
                performance_metrics={"test_passed": True}
            )
            
            # 에이전트 결정 로깅 테스트
            test_decision_id = self.log_agent_decision(
                agent_name="TestDecisionAgent",
                agent_role="결정 테스트 에이전트",
                input_data={"test": True},
                decision_process={"steps": ["테스트 단계"]},
                output_result={"result": "성공"},
                reasoning="테스트 목적"
            )
            
            # 상태 정보 출력
            status = self.get_logger_status()
            self.info(f"로거 상태: {status}")
            
            self.info("=== 하이브리드 로깅 시스템 테스트 완료 ===")
            
            return {
                "test_passed": True,
                "output_id": test_output_id,
                "decision_id": test_decision_id,
                "status": status
            }
            
        except Exception as e:
            self.error(f"로깅 시스템 테스트 실패: {e}")
            return {"test_passed": False, "error": str(e)}
        

    def get_all_previous_results(self, agent_name: str = None) -> List[Dict]:
        """모든 이전 결과 조회 (CoordinatorAgent 호환성)"""
        try:
            # 에이전트 로거가 있는 경우
            if self.agent_logger and hasattr(self.agent_logger, 'get_all_outputs'):
                try:
                    all_outputs = self.agent_logger.get_all_outputs()
                    if isinstance(all_outputs, list):
                        # 특정 에이전트 필터링
                        if agent_name:
                            filtered_outputs = []
                            for output in all_outputs:
                                if isinstance(output, dict) and agent_name.lower() in output.get('agent_name', '').lower():
                                    filtered_outputs.append(output)
                            return filtered_outputs
                        return all_outputs
                except Exception as e:
                    self.warning(f"에이전트 로거에서 결과 조회 실패: {e}")
            
            # SafeAgentLogger인 경우
            if hasattr(self.agent_logger, 'outputs'):
                outputs = self.agent_logger.outputs
                if isinstance(outputs, list):
                    if agent_name:
                        filtered_outputs = []
                        for output in outputs:
                            if isinstance(output, dict) and agent_name.lower() in output.get('agent_name', '').lower():
                                filtered_outputs.append(output)
                        return filtered_outputs
                    return outputs
            
            # 폴백: learning insights 사용
            insights = self.get_learning_insights(agent_name)
            if isinstance(insights, dict):
                return [{"insight_data": insights, "source": "learning_insights_fallback"}]
            
            # 최종 폴백: 빈 리스트
            self.info(f"이전 결과를 찾을 수 없음 (에이전트: {agent_name})")
            return []
            
        except Exception as e:
            self.error(f"이전 결과 조회 실패: {e}")
            return []

    def get_recent_outputs(self, agent_name: str = None, limit: int = 10) -> List[Dict]:
        """최근 출력 조회"""
        try:
            all_results = self.get_all_previous_results(agent_name)
            
            # 타임스탬프 기준으로 정렬
            sorted_results = sorted(
                all_results,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )
            
            return sorted_results[:limit]
            
        except Exception as e:
            self.error(f"최근 출력 조회 실패: {e}")
            return []

    def get_agent_statistics(self, agent_name: str = None) -> Dict:
        """에이전트 통계 조회"""
        try:
            all_results = self.get_all_previous_results(agent_name)
            
            stats = {
                "total_outputs": len(all_results),
                "agents": {},
                "recent_activity": None,
                "error_count": 0
            }
            
            # 에이전트별 통계
            for result in all_results:
                if isinstance(result, dict):
                    agent = result.get('agent_name', 'unknown')
                    if agent not in stats["agents"]:
                        stats["agents"][agent] = 0
                    stats["agents"][agent] += 1
                    
                    # 에러 카운트
                    if 'error' in result or result.get('final_answer', '').lower().find('error') != -1:
                        stats["error_count"] += 1
                    
                    # 최근 활동
                    timestamp = result.get('timestamp')
                    if timestamp and (not stats["recent_activity"] or timestamp > stats["recent_activity"]):
                        stats["recent_activity"] = timestamp
            
            return stats
            
        except Exception as e:
            self.error(f"에이전트 통계 조회 실패: {e}")
            return {"total_outputs": 0, "agents": {}, "error_count": 0}
        

class SafeAgentLogger:
    """안전한 에이전트 로거 (AgentLogger 없을 때 사용) - 개선됨"""
    
    def __init__(self, standard_logger):
        self.standard_logger = standard_logger
        self.outputs = []
        self.max_outputs = 1000  # 메모리 사용량 제한
    
    def get_all_outputs(self) -> List[Dict]:
        """모든 출력 반환"""
        return self.outputs.copy()
    
    def get_outputs_by_agent(self, agent_name: str) -> List[Dict]:
        """특정 에이전트의 출력만 반환"""
        filtered_outputs = []
        for output in self.outputs:
            if isinstance(output, dict) and agent_name.lower() in output.get('agent_name', '').lower():
                filtered_outputs.append(output)
        return filtered_outputs
    
    def log_agent_decision(self, agent_name: str, agent_role: str = None,
                          input_data: Dict = None, decision_process: Dict = None,
                          output_result: Dict = None, reasoning: str = "",
                          confidence_score: float = 0.8, context: Dict = None,
                          performance_metrics: Dict = None) -> str:
        """안전한 폴백 에이전트 결정 로깅 (개선됨)"""
        try:
            output_id = f"safe_{agent_name}_{int(__import__('time').time() * 1000)}"
            log_entry = {
                "output_id": output_id,
                "agent_name": agent_name,
                "agent_role": agent_role or f"{agent_name} 에이전트",
                "timestamp": __import__('datetime').datetime.now().isoformat(),
                "input_data": input_data or {},
                "decision_process": decision_process or {},
                "output_result": output_result or {},
                "reasoning": reasoning,
                "confidence_score": confidence_score,
                "context": context or {},
                "performance_metrics": performance_metrics or {},
                "safe_mode": True
            }
            
            # 메모리 관리
            if len(self.outputs) >= self.max_outputs:
                self.outputs = self.outputs[-int(self.max_outputs * 0.8):]  # 20% 정리
            
            self.outputs.append(log_entry)
            self.standard_logger.info(f"Agent Decision (Safe Mode) - {agent_name}: {reasoning}")
            return output_id
            
        except Exception as e:
            self.standard_logger.error(f"안전 모드 에이전트 결정 로깅 실패: {e}")
            return f"safe_error_{agent_name}_{int(__import__('time').time())}"
    
    def log_agent_real_output(self, agent_name: str, agent_role: str = None,
                             task_description: str = "", final_answer: str = "",
                             reasoning_process: str = "", execution_steps: List[str] = None,
                             raw_input: Any = None, raw_output: Any = None,
                             performance_metrics: Dict = None, error_logs: List[Dict] = None,
                             info_data: Dict = None) -> str:
        """안전한 폴백 에이전트 출력 로깅 (개선됨)"""
        try:
            output_id = f"safe_output_{agent_name}_{int(__import__('time').time() * 1000)}"
            log_entry = {
                "output_id": output_id,
                "agent_name": agent_name,
                "agent_role": agent_role or f"{agent_name} 에이전트",
                "timestamp": __import__('datetime').datetime.now().isoformat(),
                "task_description": task_description,
                "final_answer": final_answer,
                "reasoning_process": reasoning_process,
                "execution_steps": execution_steps or [],
                "raw_input": raw_input,
                "raw_output": raw_output,
                "performance_metrics": performance_metrics or {},
                "error_logs": error_logs or [],
                "info_data": info_data or {},
                "safe_mode": True
            }
            
            # 메모리 관리
            if len(self.outputs) >= self.max_outputs:
                self.outputs = self.outputs[-int(self.max_outputs * 0.8):]
            
            self.outputs.append(log_entry)
            self.standard_logger.info(f"Agent Output (Safe Mode) - {agent_name}: {task_description}")
            return output_id
            
        except Exception as e:
            self.standard_logger.error(f"안전 모드 에이전트 출력 로깅 실패: {e}")
            return f"safe_output_error_{agent_name}_{int(__import__('time').time())}"


    def log_agent_interaction(self, source_agent: str, target_agent: str,
                             interaction_type: str, data_transferred: Dict,
                             success: bool = True) -> str:
        """안전한 폴백 에이전트 상호작용 로깅"""
        try:
            interaction_id = f"safe_interaction_{int(time.time() * 1000)}"
            
            self.standard_logger.info(
                f"Agent Interaction (Safe Mode) - {source_agent} -> {target_agent}: {interaction_type}"
            )
            
            return interaction_id
            
        except Exception as e:
            self.standard_logger.error(f"안전 모드 상호작용 로깅 실패: {e}")
            return f"safe_interaction_error_{int(time.time())}"

    def get_learning_insights(self, target_agent: str = None) -> Dict:
        """안전한 폴백 인사이트"""
        return {
            "target_agent": target_agent or "unknown",
            "analysis_timestamp": time.time(),
            "total_outputs_analyzed": len(self.outputs),
            "patterns": ["safe_mode"],
            "recommendations": ["안전 모드에서 실행 중"],
            "key_insights": "SafeAgentLogger 모드",
            "safe_mode": True
        }


# ==================== 팩토리 함수들 ====================

def create_hybrid_logger(class_name: str, agent_logger_factory=None) -> HybridLogger:
    """하이브리드 로거 생성 팩토리 함수"""
    try:
        # 에이전트 로거 생성
        if agent_logger_factory:
            agent_logger = agent_logger_factory()
        else:
            try:
                from utils.agent_decision_logger import get_agent_logger
                agent_logger = get_agent_logger()
            except ImportError:
                agent_logger = None

        # ✅ 하이브리드 로거 생성 (올바른 파라미터 전달)
        hybrid_logger = HybridLogger(class_name, agent_logger)
        return hybrid_logger

    except Exception as e:
        # ✅ 최종 폴백: 표준 로거만 사용 (올바른 파라미터 전달)
        print(f"하이브리드 로거 생성 실패, 표준 로거 사용: {e}")
        return HybridLogger(class_name, None)  # ✅ None을 명시적으로 전달


def get_hybrid_logger(class_name: str = None) -> HybridLogger:
    """하이브리드 로거 싱글톤 인스턴스 반환"""
    global _hybrid_logger_instances
    
    if '_hybrid_logger_instances' not in globals():
        _hybrid_logger_instances = {}
    
    if class_name not in _hybrid_logger_instances:
        _hybrid_logger_instances[class_name] = create_hybrid_logger(class_name or "DefaultHybridLogger")
    
    return _hybrid_logger_instances[class_name]

def setup_hybrid_logging_for_class(cls):
    """클래스 데코레이터: 하이브리드 로깅 자동 설정"""
    class_name = cls.__name__
    
    # 클래스에 logger 속성 추가
    cls.logger = get_hybrid_logger(class_name)
    
    # 기존 메서드들에 로깅 래퍼 추가 (선택적)
    original_init = cls.__init__
    
    def wrapped_init(self, *args, **kwargs):
        self.logger = get_hybrid_logger(class_name)
        self.logger.info(f"{class_name} 인스턴스 초기화")
        return original_init(self, *args, **kwargs)
    
    cls.__init__ = wrapped_init
    
    return cls

# ==================== 전역 인스턴스 관리 ====================

_hybrid_logger_instances = {}

def reset_hybrid_loggers():
    """모든 하이브리드 로거 인스턴스 리셋"""
    global _hybrid_logger_instances
    _hybrid_logger_instances = {}

def get_all_hybrid_loggers() -> Dict[str, HybridLogger]:
    """모든 하이브리드 로거 인스턴스 반환"""
    return _hybrid_logger_instances.copy()

# ==================== 테스트 함수 ====================

def test_hybrid_logging_system():
    """하이브리드 로깅 시스템 전체 테스트"""
    print("=== 하이브리드 로깅 시스템 전체 테스트 시작 ===")
    
    try:
        # 하이브리드 로거 생성 테스트
        logger = get_hybrid_logger("TestSystem")
        
        # 로깅 시스템 테스트
        test_result = logger.test_logging_system()
        
        print(f"테스트 결과: {test_result}")
        print("=== 하이브리드 로깅 시스템 전체 테스트 완료 ===")
        
        return test_result
        
    except Exception as e:
        print(f"하이브리드 로깅 시스템 테스트 실패: {e}")
        return {"test_passed": False, "error": str(e)}

if __name__ == "__main__":
    # 테스트 실행
    test_hybrid_logging_system()
