"""
백테스트 관리자
학습된 모델을 사용한 백테스트 실행 및 결과 분석
"""

import threading
import time
from typing import Callable, Optional, Dict, Any, List, Tuple
import numpy as np
import torch

from src.utils.logger import get_logger
from src.ddpg_algorithm import DDPGAgent
from src.data.merge import load_merged_data_v1
from src.environment.trading_env import TradingEnvironment


class DashBacktestManager:
    """Dash용 백테스트 관리자"""

    def __init__(
        self,
        log_callback: Optional[Callable[[str], None]] = None,
        result_callback: Optional[Callable[..., None]] = None,
        progress_callback: Optional[Callable[[int, int, float, str], None]] = None,
    ):
        self.log_callback = log_callback
        self.result_callback = result_callback  # 결과 콜백 추가
        self.progress_callback = progress_callback  # 진행률 콜백 추가
        self.logger = get_logger("dash_backtest")
        self.is_running = False

    def add_log(self, message: str):
        """로그 메시지 추가"""
        if self.log_callback:
            self.log_callback(message)
        self.logger.info(message)

    def update_progress(
        self, current_step: int, total_steps: int, status: str = "진행 중"
    ):
        """진행률 업데이트"""
        progress_percent = (
            min((current_step / total_steps) * 100, 100) if total_steps > 0 else 0
        )
        if self.progress_callback:
            self.progress_callback(current_step, total_steps, progress_percent, status)

    def update_results(
        self,
        portfolio_values: List[float],
        rewards: List[float],
        dates: List[str],
        allocations: List[Dict[str, float]],
        metrics: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """결과 업데이트"""
        if self.result_callback:
            # 데이터가 제대로 전달되는지 확인하는 로그
            self.add_log(
                f"📤 백테스트 결과 콜백 전송: 포트폴리오={len(portfolio_values)}, 배분={len(allocations)}"
            )
            # 추가 데이터도 콜백으로 전달
            self.result_callback(
                portfolio_values, rewards, dates, allocations, metrics, additional_data
            )

    def start_backtest(self, config: Dict[str, Any]) -> bool:
        """백테스트 시작"""

        if self.is_running:
            self.add_log("⚠️ 이미 백테스트가 진행 중입니다")
            return False

        self.is_running = True

        # 새 스레드에서 백테스트 실행
        backtest_thread = threading.Thread(
            target=self._backtest_worker, args=(config,), daemon=True
        )
        backtest_thread.start()

        return True

    def _backtest_worker(self, config: Dict[str, Any]):
        """백테스트 워커"""

        try:
            self.add_log(
                f"📈 백테스트 시작: {config['model_path']}, 에피소드 {config['episode']}"
            )

            # 장치 설정
            device = self._get_device()
            self.add_log(f"🔧 연산 장치: {device}")

            # 모델 설정 로드 및 환경 구성
            env, agent = self._setup_backtest_environment(config, device)

            # 백테스트 실행
            self.add_log("🚀 백테스트 실행 중...")
            results = self._run_evaluation(env, agent, config["assets"])

            # 결과 처리 및 콜백 전송
            self._process_backtest_results(results)

            # 백테스트 성공 완료 시 진행률 완료 상태로 업데이트
            self.update_progress(100, 100, "백테스트 완료")

        except Exception as e:
            error_message = f"❌ 백테스트 중 오류 발생: {str(e)}"
            self.add_log(error_message)
            self.logger.error(error_message, exc_info=True)
            # 오류 발생 시에도 진행률 상태 업데이트
            self.update_progress(0, 1, "오류 발생")
        finally:
            self.is_running = False

    def _get_device(self):
        """최적 장치 선택"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = "CUDA GPU"
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = "Apple Silicon GPU (MPS)"
        else:
            device = torch.device("cpu")
            device_name = "CPU"

        return device

    def _setup_backtest_environment(
        self, config: Dict[str, Any], device
    ) -> Tuple[TradingEnvironment, DDPGAgent]:
        """백테스트 환경 설정"""

        # 🔧 메타데이터 먼저 읽어서 모델 설정 파악
        self.add_log("📋 모델 메타데이터 읽는 중...")
        from src.dash_utils import (
            get_model_metadata,
            load_model_training_config,
            get_latest_episode_from_model,
        )

        model_metadata = get_model_metadata(config["model_path"])
        training_config = load_model_training_config(config["model_path"])

        # 에피소드가 0인 경우 최신 에피소드 자동 감지
        if config["episode"] == 0 or config["episode"] is None:
            latest_episode = get_latest_episode_from_model(config["model_path"])
            if latest_episode > 0:
                config["episode"] = latest_episode
                self.add_log(f"🔍 최신 에피소드 자동 감지: {latest_episode}")
            else:
                self.add_log(
                    "⚠️ 유효한 에피소드를 찾을 수 없어 최신 체크포인트를 사용합니다"
                )
                config["episode"] = None

        if not training_config:
            # 메타데이터가 없는 경우 기본값 사용
            self.add_log("⚠️ 메타데이터를 찾을 수 없어 기본 설정을 사용합니다")
            model_settings = {
                "hidden_dim": 256,
                "actor_lr": 0.0003,
                "critic_lr": 0.0003,
                "critic_loss_type": "mse",
                "window_size": 60,
            }
        else:
            # 메타데이터에서 모델 설정 추출
            model_settings = {
                "hidden_dim": training_config.get("hidden_dim", 256),
                "actor_lr": training_config.get("actor_lr", 0.0003),
                "critic_lr": training_config.get("critic_lr", 0.0003),
                "critic_loss_type": training_config.get("critic_loss_type", "mse"),
                "window_size": training_config.get("window_size", 60),
            }

        self.add_log(
            f"📊 모델 설정 확인: 히든차원={model_settings['hidden_dim']}, "
            f"윈도우크기={model_settings['window_size']}, "
            f"Actor LR={model_settings['actor_lr']}, Critic LR={model_settings['critic_lr']}"
        )

        # 모델의 ETF 정보 확인 및 사용
        model_etfs = training_config.get("assets", []) if training_config else []
        if model_etfs:
            self.add_log(f"📊 모델 학습 ETF: {', '.join(model_etfs)}")
            # 모델이 학습된 ETF 사용
            config["assets"] = model_etfs
        else:
            self.add_log(f"📊 설정된 ETF 사용: {', '.join(config['assets'])}")

        # 데이터 로드 (ETF 조합에 따른 파일명 생성)
        self.add_log(f"📂 데이터 로딩 중... (ETF: {', '.join(config['assets'])})")
        etf_combination = "_".join(sorted(config["assets"]))
        filename = f"rl_ddpg_{etf_combination}"
        merged_data = load_merged_data_v1(config["assets"], filename, refresh=False)
        self.add_log(f"✅ 데이터 로드 완료 ({len(merged_data)} 행)")

        # 환경 설정 (메타데이터의 윈도우 크기 사용)
        self.add_log("🏗️ 트레이딩 환경 설정 중...")
        env = TradingEnvironment(
            merged_data,
            self.logger,
            window_size=int(model_settings["window_size"]),
            n_assets=len(config["assets"]),
        )

        # 에이전트 설정 (메타데이터의 모델 설정 사용)
        self.add_log("🤖 DDPG 에이전트 생성 중 (메타데이터 기반 설정)...")
        state_dim = (
            env.observation_space.shape[0] if env.observation_space is not None else 0
        )
        action_dim = env.action_space.shape[0] if env.action_space is not None else 0

        agent_ddpg = DDPGAgent(
            self.logger,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=int(model_settings["hidden_dim"]),
            actor_lr=model_settings["actor_lr"],
            critic_lr=model_settings["critic_lr"],
            device=str(device),
            critic_loss_type=model_settings.get("critic_loss_type", "mse"),
        )

        # 모델 로드
        episode_to_load = config["episode"]
        if episode_to_load is None:
            self.add_log(f"📥 최신 체크포인트 로드 중: {config['model_path']}")
        else:
            self.add_log(
                f"📥 에피소드 {episode_to_load} 체크포인트 로드 중: {config['model_path']}"
            )

        ret = agent_ddpg.load_checkpoint(
            config["model_path"], episode_to_load, evaluate=True
        )

        if ret:
            # 로드 성공 시 실제 에피소드 번호 확인 및 로그
            metadata_info = get_model_metadata(config["model_path"])
            if metadata_info:
                actual_episode = metadata_info["episode"]
                self.add_log(f"✅ 체크포인트 로드 완료: 에피소드 {actual_episode}")
            else:
                self.add_log(f"✅ 체크포인트 로드 완료")
        else:
            self.add_log("❌ 체크포인트 로드 실패")
            raise Exception("저장된 모델을 찾을 수 없습니다")

        # 평가 모드 추가 설정
        agent_ddpg.reset_for_evaluation()
        self.add_log("✅ 모델 로드 및 평가 모드 설정 완료")

        return env, agent_ddpg

    def _process_backtest_results(self, results: Dict[str, Any]):
        """백테스트 결과 처리 및 콜백 전송"""

        # 결과를 콜백으로 전달 - 강화학습과 균등투자 비교 데이터
        if results and results.get("RL_Agent") and results.get("Equal_Weight"):
            rl_results = results["RL_Agent"]
            equal_results = results["Equal_Weight"]

            # 디버깅을 위한 로그 출력
            rl_allocations = rl_results.get("allocations", [])
            equal_allocations = equal_results.get("allocations", [])

            # self.add_log(f"📊 강화학습 배분 데이터: {len(rl_allocations)}개 항목")
            # self.add_log(f"📊 균등투자 배분 데이터: {len(equal_allocations)}개 항목")

            # if rl_allocations and len(rl_allocations) > 0:
            #     self.add_log(f"📊 강화학습 첫 배분: {rl_allocations[0]}")
            #     self.add_log(f"📊 강화학습 마지막 배분: {rl_allocations[-1]}")

            # if equal_allocations and len(equal_allocations) > 0:
            #     self.add_log(f"📊 균등투자 첫 배분: {equal_allocations[0]}")

            # 비교 데이터 구성 (두 전략 모두 포함)
            comparison_data = {
                "rl_strategy": {
                    "portfolio_values": rl_results.get("portfolio_values", []),
                    "rewards": rl_results.get("rewards", []),
                    "dates": rl_results.get("dates", []),
                    "allocations": rl_allocations,
                    "allocation_dates": rl_results.get("allocation_dates", []),
                    "annualized_returns": rl_results.get("annualized_returns", []),
                    "cumulative_returns": rl_results.get("cumulative_returns", []),
                },
                "equal_strategy": {
                    "portfolio_values": equal_results.get("portfolio_values", []),
                    "rewards": equal_results.get("rewards", []),
                    "dates": equal_results.get("dates", []),
                    "allocations": equal_allocations,
                    "allocation_dates": equal_results.get("allocation_dates", []),
                    "annualized_returns": equal_results.get("annualized_returns", []),
                    "cumulative_returns": equal_results.get("cumulative_returns", []),
                },
            }

            # 기본적으로는 강화학습 데이터를 메인으로 전달하되, 추가 데이터에 비교 정보 포함
            self.update_results(
                rl_results.get("portfolio_values", []),
                rl_results.get("rewards", []),
                rl_results.get("dates", []),
                rl_allocations,
                results.get("final_metrics", {}),
                comparison_data,
            )

        # # 결과 요약
        # final_metrics = results.get("final_metrics", {})
        # summary = (
        #     f"📊 백테스트 완료! "
        #     f"최종 수익률: {final_metrics.get('total_return', 0):.2f}%, "
        #     f"연환산 수익률: {final_metrics.get('annualized_return', 0):.2f}%, "
        #     f"최종 포트폴리오: ${final_metrics.get('final_portfolio_value', 0):.2f}"
        # )
        # self.add_log(summary)

    def _run_evaluation(self, env, agent, assets: List[str]) -> Dict[str, Any]:
        """실제 평가 로직 - 강화학습 모델과 균등투자 전략 비교"""

        results = {
            "RL_Agent": {  # 강화학습 에이전트 결과
                "portfolio_values": [],
                "rewards": [],
                "dates": [],
                "allocations": [],
                "allocation_dates": [],  # 배분 날짜 별도 저장
                "annualized_returns": [],
                "cumulative_returns": [],
            },
            "Equal_Weight": {  # 균등투자 결과
                "portfolio_values": [],
                "rewards": [],
                "dates": [],
                "allocations": [],
                "allocation_dates": [],  # 배분 날짜 별도 저장
                "annualized_returns": [],
                "cumulative_returns": [],
            },
            "final_metrics": {},
        }

        # 환경 초기화
        state = env.reset()
        step_count = 0

        try:
            total_steps = len(env.data) - env.window_size
        except:
            total_steps = 1000  # 기본값

        self.add_log(f"📈 평가 시작 (예상 스텝: {total_steps})")
        self.add_log(f"🆚 강화학습 vs 균등투자 전략 비교 분석")

        # 백테스트 시작 시 진행률 초기화
        self.update_progress(0, total_steps, "백테스트 시작")

        # 에피소드 보상 추적
        rl_episode_reward = 0
        equal_episode_reward = 0

        while True:
            # 진행률 로그 및 UI 업데이트 (더 자주 업데이트)
            if step_count % 50 == 0 and step_count > 0:  # 100에서 50으로 변경
                progress = (step_count / total_steps) * 100
                self.add_log(
                    f"⏳ 백테스트 진행: {step_count}/{total_steps} ({progress:.1f}%)"
                )
                self.update_progress(step_count, total_steps, "백테스트 진행 중")
                env.render()

            # 특별 진행률 체크포인트 (10% 단위)
            progress_checkpoint = (step_count / total_steps) * 100
            if (
                step_count > 0
                and int(progress_checkpoint) % 10 == 0
                and int(progress_checkpoint)
                != int(((step_count - 1) / total_steps) * 100)
            ):
                self.add_log(
                    f"🎯 백테스트 체크포인트: {int(progress_checkpoint)}% 완료"
                )
                self.update_progress(
                    step_count,
                    total_steps,
                    f"백테스트 {int(progress_checkpoint)}% 완료",
                )

            # 강화학습 에이전트 액션 선택
            # select_action에서 이미 최소 비중 10% 제약을 보장하므로 추가 클리핑 불필요
            rl_action, raw_action = agent.select_action(state, add_noise=False)

            # 액션 유효성 검사 (최소 비중 제약 확인)
            # select_action에서 이미 최소 비중 10%를 보장하지만, 안전을 위해 검증
            min_weight = 0.075
            if np.min(rl_action) < min_weight:
                # 최소 비중 미만인 경우 재조정 (이론적으로 발생하지 않아야 함)
                rl_action = np.maximum(rl_action, min_weight)
                rl_action = rl_action / np.sum(rl_action)

            # 합이 1인지 확인 (이론적으로 이미 보장됨)
            action_sum = np.sum(rl_action)
            if abs(action_sum - 1.0) > 1e-6:
                rl_action = rl_action / action_sum

            # 환경 스텝 실행
            (
                next_state,
                reward_agent,
                reward_monthly_agent,
                reward_monthly_equal,
                done,
                _,
                verification,
            ) = env.step(rl_action)

            # 데이터 수집 및 처리
            self._collect_evaluation_data(
                env,
                rl_action,
                assets,
                verification,
                reward_monthly_agent,
                reward_monthly_equal,
                results,
                rl_episode_reward,
                equal_episode_reward,
            )

            state = next_state
            step_count += 1

            if done:
                break

        # 최종 메트릭 계산 및 성능 비교 로그
        self._calculate_final_metrics(env, results, step_count, total_steps)

        return results

    def _collect_evaluation_data(
        self,
        env,
        rl_action,
        assets,
        verification,
        reward_monthly_agent,
        reward_monthly_equal,
        results,
        rl_episode_reward,
        equal_episode_reward,
    ):
        """평가 데이터 수집"""

        # 강화학습 에이전트 데이터 수집
        current_rl_value = env._calculate_value()
        current_date_str, current_date_obj = env._current_date()  # 튜플 언패킹

        # 균등투자 전략 데이터 수집
        current_equal_value = (
            np.sum(
                env.shares_equal
                * env.original_data.iloc[env.current_step][
                    env.original_data.columns[: env.n_assets]
                ].values
            )
            + env.balance_equal
        )

        # 연환산 수익률과 총 수익률 계산 (환경 객체의 메서드 사용)
        rl_annualized_return = env._calculate_annualized_return()
        rl_total_return = env._calculate_total_return()

        # 균등투자 전략을 위한 별도 수익률 계산
        # 임시로 환경의 상태를 균등투자 값으로 변경하여 계산
        original_balance = env.balance
        original_shares = env.shares.copy()
        original_total_invested = env.total_invested

        # 균등투자 전략의 값으로 임시 변경
        env.balance = env.balance_equal
        env.shares = env.shares_equal.copy()
        env.total_invested = env.total_invested  # 총 투자금은 동일
        equal_annualized_return = env._calculate_annualized_return()
        equal_total_return = env._calculate_total_return()

        # 원래 상태로 복원
        env.balance = original_balance
        env.shares = original_shares
        env.total_invested = original_total_invested

        # 데이터 저장 (날짜는 문자열 형태로 저장)
        results["RL_Agent"]["portfolio_values"].append(current_rl_value)
        results["RL_Agent"]["dates"].append(current_date_str)
        results["RL_Agent"]["rewards"].append(
            reward_monthly_agent if verification else 0
        )
        results["RL_Agent"]["annualized_returns"].append(rl_annualized_return)
        results["RL_Agent"]["cumulative_returns"].append(rl_total_return)

        results["Equal_Weight"]["portfolio_values"].append(current_equal_value)
        results["Equal_Weight"]["dates"].append(current_date_str)
        results["Equal_Weight"]["rewards"].append(
            reward_monthly_equal if verification else 0
        )
        results["Equal_Weight"]["annualized_returns"].append(equal_annualized_return)
        results["Equal_Weight"]["cumulative_returns"].append(equal_total_return)

        # 포트폴리오 배분 저장 (리밸런싱 시점만)
        if verification:
            # 강화학습 에이전트 배분
            rl_allocation = {}
            for i, asset in enumerate(assets):
                rl_allocation[asset] = float(rl_action[i])
            rl_allocation["Cash"] = float(rl_action[-1])
            results["RL_Agent"]["allocations"].append(rl_allocation)
            # 배분 날짜도 별도 저장
            results["RL_Agent"]["allocation_dates"].append(current_date_str)

            # 균등투자 전략 배분 (25%씩 균등분배, 현금 0%)
            equal_allocation = {}
            for asset in assets:
                equal_allocation[asset] = 0.25  # 4개 자산에 25%씩
            equal_allocation["Cash"] = 0.0
            results["Equal_Weight"]["allocations"].append(equal_allocation)
            # 배분 날짜도 별도 저장
            results["Equal_Weight"]["allocation_dates"].append(current_date_str)

            # 보상 누적
            rl_episode_reward += reward_monthly_agent
            equal_episode_reward += reward_monthly_equal

    def _calculate_final_metrics(self, env, results, step_count, total_steps):
        """최종 메트릭 계산 및 성능 비교"""

        # 최종 메트릭 계산
        rl_final_value = (
            results["RL_Agent"]["portfolio_values"][-1]
            if results["RL_Agent"]["portfolio_values"]
            else env.initial_balance
        )
        equal_final_value = (
            results["Equal_Weight"]["portfolio_values"][-1]
            if results["Equal_Weight"]["portfolio_values"]
            else env.initial_balance
        )

        # 총 투자 금액
        total_invested = env.total_invested

        # 수익률 계산
        rl_total_return = (
            ((rl_final_value - total_invested) / total_invested) * 100
            if total_invested > 0
            else 0
        )
        equal_total_return = (
            ((equal_final_value - total_invested) / total_invested) * 100
            if total_invested > 0
            else 0
        )

        # 연환산 수익률 계산 (단순화)
        days_elapsed = len(results["RL_Agent"]["portfolio_values"])
        years_elapsed = days_elapsed / 365.0 if days_elapsed > 0 else 1

        rl_annualized_return = (
            (((rl_final_value / total_invested) ** (1 / years_elapsed)) - 1) * 100
            if total_invested > 0 and years_elapsed > 0
            else 0
        )
        equal_annualized_return = (
            (((equal_final_value / total_invested) ** (1 / years_elapsed)) - 1) * 100
            if total_invested > 0 and years_elapsed > 0
            else 0
        )

        # 수익률 디버깅 로그 출력
        for strategy in ["RL_Agent", "Equal_Weight"]:
            portfolio_values = results[strategy]["portfolio_values"]
            cumulative_returns = results[strategy]["cumulative_returns"]
            annualized_returns = results[strategy]["annualized_returns"]

            if portfolio_values and len(portfolio_values) > 0:
                # 첫 번째 포트폴리오 값을 기준점으로 사용
                initial_value = portfolio_values[0]
                final_value = portfolio_values[-1]

                # 디버깅을 위한 로그 출력
                self.add_log(f"🔍 {strategy} 디버깅:")
                self.add_log(f"   - 첫 번째 값: ${initial_value:.2f}")
                self.add_log(f"   - 최종 값: ${final_value:.2f}")
                self.add_log(f"   - 총 투자금: ${total_invested:.2f}")
                self.add_log(
                    f"   - 첫 번째 값 기준 수익률: {((final_value - initial_value) / initial_value) * 100:.2f}%"
                )
                self.add_log(
                    f"   - 총 투자금 기준 수익률: {((final_value - total_invested) / total_invested) * 100:.2f}%"
                )

                # 마지막 몇 개 값 로그 출력 (이미 환경에서 실시간으로 계산되어 저장됨)
                if len(cumulative_returns) > 5:
                    self.add_log(
                        f"   - 마지막 5개 누적 수익률: {cumulative_returns[-5:]}"
                    )

                # 연환산 수익률 로그 출력 (이미 실시간으로 계산되어 저장됨)
                if len(annualized_returns) > 5:
                    self.add_log(
                        f"   - 마지막 5개 연환산 수익률: {annualized_returns[-5:]}"
                    )

        # 최종 메트릭 요약
        results["final_metrics"] = {
            # 강화학습 메트릭
            "rl_total_return": rl_total_return,
            "rl_annualized_return": rl_annualized_return,
            "rl_final_portfolio_value": rl_final_value,
            # 균등투자 메트릭
            "equal_total_return": equal_total_return,
            "equal_annualized_return": equal_annualized_return,
            "equal_final_portfolio_value": equal_final_value,
            # 공통 메트릭
            "total_invested": total_invested,
            "total_steps": step_count,
            "evaluation_days": days_elapsed,
            # 성능 비교
            "return_difference": rl_total_return - equal_total_return,
            "annualized_return_difference": rl_annualized_return
            - equal_annualized_return,
            "value_difference": rl_final_value - equal_final_value,
        }

        # 백테스트 완료 시 진행률 100% 업데이트
        self.update_progress(step_count, total_steps, "백테스트 완료")

        # 성능 비교 로그
        self.add_log("📊 === 성능 비교 결과 ===")
        self.add_log(
            f"🤖 강화학습: 최종 ${rl_final_value:.2f}, 수익률 {rl_total_return:.2f}%, 연환산 {rl_annualized_return:.2f}%"
        )
        self.add_log(
            f"⚖️ 균등투자: 최종 ${equal_final_value:.2f}, 수익률 {equal_total_return:.2f}%, 연환산 {equal_annualized_return:.2f}%"
        )
        self.add_log(
            f"🏆 성과 차이: ${rl_final_value - equal_final_value:.2f} ({'강화학습 우세' if rl_final_value > equal_final_value else '균등투자 우세'})"
        )

        # 데이터 수집 결과 로그
        rl_dates = results["RL_Agent"]["dates"]
        equal_dates = results["Equal_Weight"]["dates"]
        self.add_log(
            f"📅 수집된 날짜 데이터: 강화학습 {len(rl_dates)}개, 균등투자 {len(equal_dates)}개"
        )
        if rl_dates:
            self.add_log(f"📅 날짜 범위: {rl_dates[0]} ~ {rl_dates[-1]}")
            # 중간 날짜들도 확인
            if len(rl_dates) > 10:
                sample_dates = [
                    rl_dates[i]
                    for i in [
                        0,
                        len(rl_dates) // 4,
                        len(rl_dates) // 2,
                        len(rl_dates) * 3 // 4,
                        -1,
                    ]
                ]
                self.add_log(f"📅 샘플 날짜들: {sample_dates}")
