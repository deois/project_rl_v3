"""
Dash 앱 상태 관리자
포트폴리오, 학습 상태, 백테스트 데이터 관리
"""

import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import weakref

from src.utils.logger import get_logger
from src.dash_training_integration import DashRealTrainingManager, DashBacktestManager

# 로거 설정
logger = get_logger("dash_manager")


class CompleteDashManager:
    """완전한 Dash 전용 관리자"""

    def __init__(self):
        # 기본 상태
        self.training_status: Dict[str, Any] = {
            "is_training": False,
            "can_stop": False,
            "current_episode": 0,
            "total_episodes": 0,
            "current_reward": 0.0,
            "average_reward": 0.0,
            "portfolio_value": 0.0,
            "task_id": None,
            "start_time": None,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "mode": "simulation",  # "simulation" 또는 "real"
        }

        # 백테스트 상태 추가
        self.backtest_status: Dict[str, Any] = {
            "is_running": False,
            "progress": 0,
            "current_step": 0,
            "total_steps": 0,
            "task_id": None,
            "start_time": None,
            "results": None,
            "error": None,
        }

        # 로그 및 차트 데이터
        self.logs: List[str] = []
        self.chart_data: Dict[str, List[Any]] = {
            "episodes": [],
            "rewards": [],
            "portfolio_values": [],
            "timestamps": [],
            "actor_losses": [],
            "critic_losses": [],
        }

        # 백테스트 결과 데이터 추가
        self.backtest_data: Dict[str, Any] = {
            "portfolio_values": [],
            "rewards": [],
            "dates": [],
            "allocations": [],  # actions 대신 allocations 사용
            "metrics": {},
        }

        # 백테스트 설정 저장
        self.backtest_config: Dict[str, Any] = {
            "model_path": "./model/rl_ddpg",
            "episode": 0,
            "assets": ["SPY", "DGRO", "SCHD", "EWY"],
            "start_date": None,
            "end_date": None,
        }

        # 실제 학습 관리자들
        self.real_training_manager = DashRealTrainingManager(
            log_callback=self.add_log, status_callback=self.update_real_status
        )

        self.backtest_manager = DashBacktestManager(
            log_callback=self.add_log,
            result_callback=self.update_backtest_data,
            progress_callback=self.update_backtest_progress,
        )

        # 시뮬레이션용
        self.simulation_stop_event: Optional[threading.Event] = None

    def add_log(self, message: str) -> None:
        """로그 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        # 최대 150개 로그만 유지
        if len(self.logs) > 150:
            self.logs = self.logs[-150:]

    def update_real_status(self, **kwargs) -> None:
        """실제 학습 상태 업데이트"""
        # 명시적으로 전달된 상태 업데이트 (학습 종료 시 사용)
        if "is_training" in kwargs:
            self.training_status["is_training"] = kwargs["is_training"]
            logger.info(
                f"📊 학습 상태 강제 업데이트: is_training = {kwargs['is_training']}"
            )

        if "can_stop" in kwargs:
            self.training_status["can_stop"] = kwargs["can_stop"]
            logger.info(
                f"📊 중지 가능 상태 강제 업데이트: can_stop = {kwargs['can_stop']}"
            )

        if hasattr(self.real_training_manager, "current_episode"):
            self.training_status.update(
                {
                    "current_episode": self.real_training_manager.current_episode,
                    "current_reward": self.real_training_manager.current_reward,
                    "portfolio_value": self.real_training_manager.portfolio_value,
                    "average_reward": self.real_training_manager.average_reward,
                    "actor_loss": self.real_training_manager.actor_loss,
                    "critic_loss": self.real_training_manager.critic_loss,
                }
            )

            # 차트 데이터 업데이트
            if self.real_training_manager.current_episode > 0:
                self.update_chart_data(
                    self.real_training_manager.current_episode,
                    self.real_training_manager.current_reward,
                    self.real_training_manager.portfolio_value,
                    self.real_training_manager.actor_loss,
                    self.real_training_manager.critic_loss,
                )

    def update_backtest_status(self, **kwargs) -> None:
        """백테스트 상태 업데이트 (기존 메서드 - 일반적인 상태 변경용)"""
        # old_status = self.backtest_status.copy()
        self.backtest_status.update(kwargs)

        # # 상태 변화 로그
        # if "is_running" in kwargs:
        #     if kwargs["is_running"] and not old_status.get("is_running", False):
        #         logger.info("🚀 백테스트 시작 - 상태를 실행 중으로 변경")
        #     elif not kwargs["is_running"] and old_status.get("is_running", False):
        #         logger.info("✅ 백테스트 완료 - 상태를 대기 중으로 변경")

    def update_backtest_progress(
        self, current_step: int, total_steps: int, progress_percent: float, status: str
    ) -> None:
        """백테스트 진행률 업데이트"""
        # 상태 업데이트 전 현재 상태 로그
        old_progress = self.backtest_status.get("progress", 0)
        old_is_running = self.backtest_status.get("is_running", False)

        # 새로운 상태 데이터
        new_status = {
            "current_step": current_step,
            "total_steps": total_steps,
            "progress": progress_percent,
            "status": status,
            "is_running": True,
            "last_update": time.time(),  # 마지막 업데이트 시간 추가
        }

        # 상태 업데이트
        self.backtest_status.update(new_status)

        # 10% 단위로만 로그 출력
        if int(progress_percent) % 10 == 0 and int(progress_percent) != int(
            old_progress
        ):
            logger.info(
                f"📊 백테스트 진행률: {progress_percent:.0f}% ({current_step}/{total_steps}) - {status}"
            )

        # 특별 메시지 (주요 체크포인트에서)
        if progress_percent > 0 and progress_percent % 25 == 0:
            logger.info(f"🎯 백테스트 주요 체크포인트: {progress_percent:.0f}% 달성!")

        # 완료 임박 시 알림
        if progress_percent >= 95:
            logger.info("🏁 백테스트가 거의 완료되었습니다!")

    def update_backtest_data(
        self,
        portfolio_values: List[float],
        rewards: List[float],
        dates: List[str],
        allocations: Optional[List[Dict[str, float]]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """백테스트 데이터 업데이트 - 강화학습 vs 균등투자 비교 데이터 포함"""
        self.backtest_data = {
            "portfolio_values": portfolio_values,
            "rewards": rewards,
            "dates": dates,
            "allocations": allocations or [],  # actions 대신 allocations 사용
            "metrics": metrics or {},
        }

        # 비교 데이터 처리 (강화학습 vs 균등투자)
        if additional_data:
            # 강화학습 전략 데이터
            if "rl_strategy" in additional_data:
                rl_data = additional_data["rl_strategy"]
                returns_data = {
                    "annualized_returns": rl_data.get("annualized_returns", []),
                    "cumulative_returns": rl_data.get("cumulative_returns", []),
                }
                self.backtest_data.update({"returns_data": returns_data})

            # 균등투자 전략 데이터 추가
            if "equal_strategy" in additional_data:
                equal_data = additional_data["equal_strategy"]
                self.backtest_data.update(
                    {
                        "equal_strategy": {
                            "portfolio_values": equal_data.get("portfolio_values", []),
                            "rewards": equal_data.get("rewards", []),
                            "dates": equal_data.get("dates", []),
                            "allocations": equal_data.get("allocations", []),
                            "allocation_dates": equal_data.get("allocation_dates", []),
                            "annualized_returns": equal_data.get(
                                "annualized_returns", []
                            ),
                            "cumulative_returns": equal_data.get(
                                "cumulative_returns", []
                            ),
                        }
                    }
                )

            # 강화학습 전략의 배분 날짜도 메인 데이터에 추가
            if "rl_strategy" in additional_data:
                rl_data = additional_data["rl_strategy"]
                allocation_dates = rl_data.get("allocation_dates", [])
                if allocation_dates:
                    self.backtest_data["allocation_dates"] = allocation_dates

            # 기존 형식 지원 (하위 호환성)
            if (
                "annualized_returns" in additional_data
                and "rl_strategy" not in additional_data
            ):
                returns_data = {
                    "annualized_returns": additional_data.get("annualized_returns", []),
                    "cumulative_returns": additional_data.get("cumulative_returns", []),
                }
                self.backtest_data.update({"returns_data": returns_data})

        # 디버깅을 위한 로그
        logger.info(
            f"📊 DashManager 백테스트 데이터 업데이트: 포트폴리오={len(portfolio_values)}, 배분={len(allocations or [])}"
        )
        logger.info(
            f"📊 날짜 데이터: {len(dates)}개, 첫 날짜: {dates[0] if dates else 'None'}, 마지막 날짜: {dates[-1] if dates else 'None'}"
        )

        # if allocations and len(allocations) > 0:
        #     logger.info(f"📊 강화학습 첫 번째 배분 데이터: {allocations[0]}")

        # 균등투자 데이터 로그
        if additional_data and "equal_strategy" in additional_data:
            equal_data = additional_data["equal_strategy"]
            # equal_dates = equal_data.get("dates", [])
            # equal_values = equal_data.get("portfolio_values", [])
            # equal_allocations = equal_data.get("allocations", [])

            # logger.info(
            #     f"📊 균등투자 날짜: {len(equal_dates)}개, 포트폴리오: {len(equal_values)}개, 배분: {len(equal_allocations)}개")
            # if equal_dates:
            #     logger.info(f"📊 균등투자 날짜 범위: {equal_dates[0]} ~ {equal_dates[-1]}")
            # if equal_allocations:
            #     logger.info(f"📊 균등투자 첫 번째 배분 데이터: {equal_allocations[0]}")
            # logger.info(f"📊 비교 데이터 포함: 강화학습 vs 균등투자")

        # 최종 저장된 데이터 구조 요약
        logger.info(
            f"📊 최종 저장 데이터 - 메인 날짜: {len(self.backtest_data.get('dates', []))}개"
        )
        if "equal_strategy" in self.backtest_data:
            equal_stored = self.backtest_data["equal_strategy"]
            logger.info(
                f"📊 최종 저장 데이터 - 균등투자 날짜: {len(equal_stored.get('dates', []))}개"
            )

    def reset_backtest_data(self) -> None:
        """백테스트 데이터 초기화"""
        self.backtest_data = {
            "portfolio_values": [],
            "rewards": [],
            "dates": [],
            "allocations": [],  # actions 대신 allocations 사용
            "metrics": {},
        }
        self.backtest_status.update(
            {
                "is_running": False,
                "progress": 0,
                "current_step": 0,
                "total_steps": 0,
                "task_id": None,
                "results": None,
                "error": None,
            }
        )

    def update_chart_data(
        self,
        episode: int,
        reward: float,
        portfolio_value: float,
        actor_loss: float = 0.0,
        critic_loss: float = 0.0,
    ) -> None:
        """차트 데이터 업데이트"""
        self.chart_data["episodes"].append(episode)
        self.chart_data["rewards"].append(reward)
        self.chart_data["portfolio_values"].append(portfolio_value)
        self.chart_data["timestamps"].append(datetime.now())
        self.chart_data["actor_losses"].append(actor_loss)
        self.chart_data["critic_losses"].append(critic_loss)

        # 최대 1000개 데이터 포인트만 유지
        if len(self.chart_data["episodes"]) > 1000:
            for key in self.chart_data:
                self.chart_data[key] = self.chart_data[key][-1000:]

    def reset_chart_data(self) -> None:
        """차트 데이터 초기화"""
        for key in self.chart_data:
            self.chart_data[key] = []
