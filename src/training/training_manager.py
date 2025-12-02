"""
강화학습 훈련 관리자
실제 DDPG 학습 로직을 Dash 콜백에서 실행
"""

import os
import threading
import time
import random
from collections import deque
from typing import Callable, Optional, Dict, Any, Tuple
import numpy as np
import torch

from src.utils.logger import get_logger
from src.ddpg_algorithm import DDPGAgent
from src.data.merge import load_merged_data_v1
from src.environment.trading_env import TradingEnvironment
from .model_utils import calculate_model_hash


class DashRealTrainingManager:
    """Dash용 실제 강화학습 관리자"""

    def __init__(
        self,
        log_callback: Optional[Callable[[str], None]] = None,
        status_callback: Optional[Callable[..., None]] = None,
    ):
        self.log_callback = log_callback
        self.status_callback = status_callback
        self.logger = get_logger("dash_real_training")

        # 학습 상태
        self.is_training = False
        self.should_stop = False
        self.current_episode = 0
        self.total_episodes = 0
        self.current_reward = 0.0
        self.average_reward = 0.0
        self.portfolio_value = 0.0
        self.task_id: Optional[str] = None

        # 학습 컨트롤
        self.stop_event: Optional[threading.Event] = None
        self.training_thread: Optional[threading.Thread] = None

        # 성능 메트릭
        self.actor_loss = 0.0
        self.critic_loss = 0.0
        self.update_count = 0

        # 학습 시간 추적
        self.training_start_time: Optional[float] = None
        self.total_training_time = 0.0
        self.episode_times = []

        # 수동 저장을 위한 현재 학습 상태 저장
        self.current_agent = None
        self.current_config: Optional[Dict[str, Any]] = None
        self.model_dir: Optional[str] = None

    def add_log(self, message: str):
        """로그 메시지 추가"""
        if self.log_callback:
            self.log_callback(message)
        self.logger.info(message)

    def update_status(self, **kwargs):
        """상태 업데이트"""
        if self.status_callback:
            self.status_callback(**kwargs)

    def start_real_training(self, task_id: str, config: Dict[str, Any]) -> bool:
        """실제 강화학습 시작"""

        if self.is_training:
            self.add_log("⚠️ 이미 학습이 진행 중입니다")
            return False

        self.task_id = task_id
        self.is_training = True
        self.stop_event = threading.Event()

        # 학습 시간 추적 시작
        self.training_start_time = time.time()
        self.episode_times = []

        # 새 스레드에서 학습 실행
        self.training_thread = threading.Thread(
            target=self._training_thread_worker, args=(task_id, config), daemon=True
        )
        self.training_thread.start()

        return True

    def stop_training(self):
        """학습 중지"""
        if self.stop_event:
            self.stop_event.set()
            self.add_log(f"🛑 학습 중지 신호 전송 (ID: {self.task_id})")

    def _training_thread_worker(self, task_id: str, config: Dict[str, Any]):
        """학습 워커 스레드"""

        try:
            self.add_log(f"🚀 실제 DDPG 학습 시작 (ID: {task_id})")

            # 장치 설정
            device = self._get_device()
            self.add_log(f"🔧 연산 장치: {device}")

            # 환경 및 에이전트 설정
            env, agent = self._setup_training_environment(config, device)

            # 실제 학습 실행
            self._run_ddpg_training(env, agent, config, task_id)

        except Exception as e:
            error_msg = f"❌ 학습 중 오류 발생 (ID: {task_id}): {str(e)}"
            self.add_log(error_msg)
            self.logger.error(error_msg, exc_info=True)
        finally:
            # 학습 완료 처리
            self.is_training = False
            self.add_log(f"✅ 학습 완료 (ID: {task_id})")

            # 상태 콜백을 통해 DashManager에 종료 상태 전달
            if self.status_callback:
                self.status_callback(is_training=False, can_stop=False)

            # 학습 완료 시 현재 상태 초기화
            self.current_agent = None
            self.current_config = None
            self.current_episode = 0
            self.model_dir = None

    def _get_device(self):
        """최적 장치 선택"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _setup_training_environment(
        self, config: Dict[str, Any], device
    ) -> Tuple[TradingEnvironment, DDPGAgent]:
        """학습 환경 및 에이전트 설정"""

        # 데이터 로드
        self.add_log(f"📂 데이터 로딩 중... (자산: {', '.join(config['assets'])})")

        # ETF 조합에 따른 고유한 파일명 생성
        etf_combination = "_".join(sorted(config["assets"]))
        filename = f"rl_ddpg_{etf_combination}"

        merged_data = load_merged_data_v1(config["assets"], filename, refresh=False)
        self.add_log(f"✅ 데이터 로드 완료 ({len(merged_data)} 행)")

        # 환경 생성
        self.add_log("🏗️ 트레이딩 환경 설정 중...")
        env = TradingEnvironment(
            merged_data,
            self.logger,
            window_size=config["window_size"],
            n_assets=len(config["assets"]),
        )

        # 에이전트 생성
        self.add_log("🤖 DDPG 에이전트 초기화 중...")
        state_dim = (
            env.observation_space.shape[0] if env.observation_space is not None else 0
        )
        action_dim = env.action_space.shape[0] if env.action_space is not None else 0

        agent = DDPGAgent(
            self.logger,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config["hidden_dim"],
            actor_lr=config["actor_lr"],
            critic_lr=config["critic_lr"],
            device=str(device),
            critic_loss_type=config.get("critic_loss_type", "mse"),
        )

        # 체크포인트 로드
        model_dir = f"./model/rl_ddpg_{self.task_id}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            self.add_log(f"📁 모델 디렉토리 생성: {model_dir}")

        self.add_log(
            f"📂 체크포인트 로드 중... (에피소드: {config['episodes_resume']})"
        )
        agent.load_checkpoint(model_dir, config["episodes_resume"])

        return env, agent

    def _run_ddpg_training(self, env, agent, config: Dict[str, Any], task_id: str):
        """실제 DDPG 학습 실행"""

        # 현재 학습 상태 저장 (수동 저장용)
        self.current_agent = agent
        self.current_config = config
        self.model_dir = f"./model/rl_ddpg_{self.task_id}"

        self.total_episodes = config["episodes"]
        episodes_resume = config["episodes_resume"]
        batch_size = config["batch_size"]
        episodes_save = config["episodes_save"]

        # 리플레이 버퍼 초기화
        replay_buffer = deque(maxlen=100000)
        self.add_log(f"💾 리플레이 버퍼 초기화 완료 (최대 크기: 100,000)")

        # 기본 액션들
        action_equal = np.ones(env.action_space.shape) / env.action_space.shape[0]
        action_balance = np.zeros(env.action_space.shape)
        action_balance[env.action_space.shape[0] - 1] = 1

        # 평균 보상 계산용
        reward_window = deque(maxlen=100)

        total_start_time = time.time()
        update = 0

        self.add_log(
            f"🎯 학습 시작! 목표: {self.total_episodes} 에피소드, 배치 크기: {batch_size}"
        )

        # 에피소드 루프
        for episode in range(episodes_resume, self.total_episodes):
            # 🛑 중지 신호 확인
            if self.stop_event and self.stop_event.is_set():
                self.add_log(
                    f"🛑 중지 신호 감지 - 에피소드 {episode}에서 안전하게 중단"
                )
                break

            episode_start_time = time.time()
            self.current_episode = episode

            # 매 에피소드 상태 업데이트 (차트 실시간 갱신용)
            progress_percent = (episode / self.total_episodes) * 100
            self.update_status(
                episode=episode,
                total_episodes=self.total_episodes,
                progress=progress_percent,
                current_reward=getattr(self, "current_reward", 0),
                average_reward=getattr(self, "average_reward", 0),
                portfolio_value=getattr(self, "portfolio_value", 0),
                actor_loss=getattr(self, "actor_loss", 0),
                critic_loss=getattr(self, "critic_loss", 0),
                update_count=getattr(self, "update_count", 0),
            )

            # 에피소드 시작 로그
            if episode % 5 == 0 or episode < 3:
                progress_percent = (episode / self.total_episodes) * 100
                self.add_log(
                    f"📈 에피소드 {episode}/{self.total_episodes} 시작 (진행률: {progress_percent:.1f}%)"
                )

            state = env.reset(episode)
            episode_reward = 0
            step_count = 0

            # 검증 상태 추적
            action_verification = None
            state_verification = None
            state_next_verification = None

            count_monthly_agent = 0
            count_monthly_equal = 0
            count_monthly_balance = 0

            # 에피소드 내 스텝 루프 실행
            episode_reward, step_count, update = self._run_episode_steps(
                env,
                agent,
                replay_buffer,
                batch_size,
                episode,
                episode_reward,
                step_count,
                update,
                reward_window,
            )

            # 🛑 에피소드 완료 후에도 중지 신호 확인
            if self.stop_event and self.stop_event.is_set():
                self.add_log(f"🛑 에피소드 {episode} 완료 후 중지 신호 감지")
                break

            # 에피소드 완료 처리
            self._process_episode_completion(
                env, episode, episode_reward, episode_start_time, step_count
            )

            # 정기 모델 저장 처리
            if episode % config.get("episodes_save", 10) == 0:
                self._save_model_checkpoint(
                    episode, config, total_start_time, reward_window
                )

        total_elapsed_time = time.time() - total_start_time

        # 학습 완료/중단 메시지
        if self.stop_event and self.stop_event.is_set():
            completion_message = f"⏹️ 학습이 중단되었습니다 (에피소드 {self.current_episode}/{self.total_episodes}). 총 시간: {total_elapsed_time:.2f}초"
        else:
            completion_message = f"🎉 학습 완료! 총 시간: {total_elapsed_time:.2f}초"

        self.add_log(completion_message)

    def _run_episode_steps(
        self,
        env,
        agent,
        replay_buffer,
        batch_size,
        episode,
        episode_reward,
        step_count,
        update,
        reward_window,
    ):
        """에피소드 내 스텝들을 실행"""

        # 검증 상태 추적
        action_verification = None
        state_verification = None
        state_next_verification = None

        count_monthly_agent = 0
        count_monthly_equal = 0

        state = env.reset(episode)

        while True:
            # 🛑 매 스텝마다 중지 신호 확인
            if self.stop_event and self.stop_event.is_set():
                self.add_log(f"🛑 에피소드 {episode} 중간에 중지 신호 감지")
                break

            step_count += 1

            # 액션 선택
            # select_action에서 이미 최소 비중 7.5% 제약을 보장하므로 추가 클리핑 불필요
            action, raw_action = agent.select_action(state)

            # 액션 유효성 검사 (최소 비중 제약 확인)
            # select_action에서 이미 최소 비중 7.5%를 보장하지만, 안전을 위해 검증
            min_weight = 0.075
            if np.min(action) < min_weight:
                # 최소 비중 미만인 경우 재조정 (이론적으로 발생하지 않아야 함)
                action = np.maximum(action, min_weight)
                action = action / np.sum(action)

            # 합이 1인지 확인 (이론적으로 이미 보장됨)
            action_sum = np.sum(action)
            if abs(action_sum - 1.0) > 1e-6:
                action = action / action_sum

            # 환경에서 스텝 실행
            (
                next_state,
                reward_agent,
                reward_monthly_agent,
                reward_monthly_equal,
                done,
                _,
                verification,
            ) = env.step(action)

            # 검증된 스텝인 경우 처리
            if verification:
                episode_reward += reward_monthly_agent

                # 리플레이 버퍼에 경험 저장 (에이전트가 실제로 선택한 액션만)
                if (
                    action_verification is not None
                    and state_verification is not None
                    and state_next_verification is not None
                ):
                    replay_buffer.append(
                        (
                            state_verification,
                            action_verification,
                            reward_monthly_agent,
                            state_next_verification,
                            float(done),
                        )
                    )

                    # 최고 성과 추적 (비교 목적)
                    max_reward = max(reward_monthly_agent, reward_monthly_equal)
                    if max_reward == reward_monthly_agent:
                        count_monthly_agent += 1
                    elif max_reward == reward_monthly_equal:
                        count_monthly_equal += 1

                action_verification = action
                state_verification = state
                state_next_verification = next_state

            # 학습 업데이트
            if len(replay_buffer) >= batch_size:
                # 🛑 학습 전에도 중지 신호 확인
                if self.stop_event and self.stop_event.is_set():
                    self.add_log("🛑 학습 업데이트 중 중지 신호 감지")
                    break

                batch = random.sample(replay_buffer, batch_size)
                self.actor_loss, self.critic_loss = agent.update(batch)
                update += 1
                self.update_count = update

                # 주기적으로 손실 값 로그 및 차트 업데이트
                if update % 100 == 0:  # 더 자주 업데이트 (1000 → 100)
                    env.render()

                    # 중간 업데이트 시에도 상태 전송
                    self.update_status(
                        episode=episode,
                        total_episodes=self.total_episodes,
                        progress=(episode / self.total_episodes) * 100,
                        current_reward=episode_reward,
                        average_reward=np.mean(reward_window) if reward_window else 0,
                        portfolio_value=env._calculate_value(),
                        actor_loss=self.actor_loss,
                        critic_loss=self.critic_loss,
                        update_count=update,
                        is_training=True,
                        mid_episode_update=True,  # 중간 업데이트 플래그
                    )

                    self.add_log(
                        f"🔄 학습 진행 - 업데이트: {update}, Actor Loss: {self.actor_loss:.4f}, Critic Loss: {self.critic_loss:.4f}"
                    )

            state = next_state

            if done:
                break

            # 비동기 처리를 위한 대기
            if step_count % 100 == 0:
                time.sleep(0.001)

        return episode_reward, step_count, update

    def _process_episode_completion(
        self, env, episode, episode_reward, episode_start_time, step_count
    ):
        """에피소드 완료 후 처리"""

        # 에피소드 완료 처리
        final_portfolio_value = env._calculate_value()
        self.portfolio_value = final_portfolio_value

        reward_window = deque(
            maxlen=100
        )  # 임시로 생성, 실제로는 메서드 파라미터로 받아야 함
        reward_window.append(episode_reward)
        self.current_reward = episode_reward
        self.average_reward = np.mean(reward_window)

        elapsed_time = time.time() - episode_start_time

        # 에피소드 시간 추적
        self.episode_times.append(elapsed_time)

        # 총 학습 시간 계산
        if self.training_start_time:
            self.total_training_time = time.time() - self.training_start_time

        # 에피소드 완료 후 실시간 상태 업데이트 (풍부한 데이터)
        self.update_status(
            episode=episode,
            total_episodes=self.total_episodes,
            progress=(episode / self.total_episodes) * 100,
            current_reward=episode_reward,
            average_reward=self.average_reward,
            portfolio_value=final_portfolio_value,
            actor_loss=getattr(self, "actor_loss", 0),
            critic_loss=getattr(self, "critic_loss", 0),
            update_count=getattr(self, "update_count", 0),
            episode_time=elapsed_time,
            step_count=step_count,
            is_training=True,
        )

        # 로그 메시지 전송
        if episode % 3 == 0 or episode < 5:
            log_message = (
                f"✅ 에피소드 {episode} 완료: 보상 {episode_reward:.2f}, "
                f"포트폴리오 ${final_portfolio_value:.2f}, "
                f"시간 {elapsed_time:.1f}초, 스텝 {step_count}개"
            )
            self.add_log(log_message)

        # 상세 성과 로그
        if episode % 10 == 0 and episode > 0:
            performance_ratio = (
                final_portfolio_value / env.total_invested
                if env.total_invested > 0
                else 1.0
            )
            detailed_log = (
                f"📊 상세 성과 (에피소드 {episode}) - "
                f"투자금: ${env.total_invested:.2f}, "
                f"수익률: {((performance_ratio - 1) * 100):.2f}%"
            )
            self.add_log(detailed_log)

    def _save_model_checkpoint(self, episode, config, total_start_time, reward_window):
        """모델 체크포인트 저장"""

        self.add_log(f"💾 모델 저장 중 - 에피소드 {episode}")

        # 현재 학습 설정을 메타데이터로 전달
        current_config = config.copy()

        # 학습 시간 통계 계산
        avg_episode_time = np.mean(self.episode_times) if self.episode_times else 0.0
        remaining_episodes = self.total_episodes - episode
        estimated_time_remaining = avg_episode_time * remaining_episodes

        current_config.update(
            {
                "current_episode": episode,
                "total_episodes": self.total_episodes,
                "task_id": self.task_id,
                "average_reward": np.mean(reward_window) if reward_window else 0.0,
                "training_start_time": total_start_time,
                # ETF 정보 추가
                "selected_etfs": config.get("assets", []),
                "etf_count": len(config.get("assets", [])),
                # DDPG 알고리즘 설정 추가
                "max_grad_norm": config.get("max_grad_norm", 0.5),
                "critic_loss_type": config.get("critic_loss_type", "mse"),
                # 새로운 학습 시간 메타데이터
                "total_training_time_hours": self.total_training_time / 3600.0,
                "average_episode_time_seconds": float(avg_episode_time),
                "estimated_remaining_time_hours": estimated_time_remaining / 3600.0,
                "completed_episodes_count": len(self.episode_times),
                "training_efficiency_episodes_per_hour": (
                    len(self.episode_times) / (self.total_training_time / 3600.0)
                    if self.total_training_time > 0
                    else 0.0
                ),
            }
        )

        start_save_time = time.time()

        # 메인 모델 디렉토리에 저장
        model_dir = f"./model/rl_ddpg_{self.task_id}"
        os.makedirs(model_dir, exist_ok=True)
        if self.current_agent:
            self.current_agent.save_checkpoint(model_dir, episode, current_config)

        # 최신 체크포인트 디렉토리에도 저장
        latest_model_dir = "./model/rl_ddpg_latest"
        os.makedirs(latest_model_dir, exist_ok=True)
        if self.current_agent:
            self.current_agent.save_checkpoint(
                latest_model_dir, episode, current_config
            )

        save_time = time.time() - start_save_time
        self.add_log(f"✅ 모델 저장 완료 (소요시간: {save_time:.2f}초)")
        self.add_log(f"📁 저장 위치: {model_dir} 및 {latest_model_dir}")

        # 모델 저장 후 상태 업데이트
        self.update_status(
            episode=episode,
            total_episodes=self.total_episodes,
            progress=(episode / self.total_episodes) * 100,
            current_reward=self.current_reward,
            average_reward=self.average_reward,
            portfolio_value=self.portfolio_value,
            actor_loss=self.actor_loss,
            critic_loss=self.critic_loss,
            update_count=self.update_count,
            is_training=True,
        )

    def manual_save_model(self) -> bool:
        """수동으로 현재 모델 저장"""
        if not self.is_training or not self.current_agent or not self.model_dir:
            self.add_log("❌ 학습 중이 아니거나 저장할 모델이 없습니다")
            return False

        try:
            # 현재 학습 설정을 메타데이터로 구성
            current_config = self.current_config.copy() if self.current_config else {}

            # 학습 시간 통계 계산
            avg_episode_time = (
                np.mean(self.episode_times) if self.episode_times else 0.0
            )
            remaining_episodes = self.total_episodes - self.current_episode
            estimated_time_remaining = avg_episode_time * remaining_episodes

            current_config.update(
                {
                    "current_episode": self.current_episode,
                    "total_episodes": self.total_episodes,
                    "task_id": self.task_id,
                    "manual_save": True,  # 수동 저장 표시
                    "save_time": time.time(),
                    # DDPG 알고리즘 설정 추가
                    "max_grad_norm": (
                        self.current_config.get("max_grad_norm", 0.5)
                        if self.current_config
                        else 0.5
                    ),
                    "critic_loss_type": (
                        self.current_config.get("critic_loss_type", "mse")
                        if self.current_config
                        else "mse"
                    ),
                    # 학습 시간 메타데이터
                    "total_training_time_hours": self.total_training_time / 3600.0,
                    "average_episode_time_seconds": float(avg_episode_time),
                    "estimated_remaining_time_hours": estimated_time_remaining / 3600.0,
                    "completed_episodes_count": len(self.episode_times),
                    "training_efficiency_episodes_per_hour": (
                        len(self.episode_times) / (self.total_training_time / 3600.0)
                        if self.total_training_time > 0
                        else 0.0
                    ),
                }
            )

            # 메인 모델 디렉토리에 저장
            self.current_agent.save_checkpoint(
                self.model_dir, self.current_episode, current_config
            )

            # 최신 체크포인트 디렉토리에도 저장
            latest_model_dir = "./model/rl_ddpg_latest"
            os.makedirs(latest_model_dir, exist_ok=True)
            self.current_agent.save_checkpoint(
                latest_model_dir, self.current_episode, current_config
            )

            # 모델 해시 계산 및 메타데이터에 추가
            model_hash = calculate_model_hash(self.model_dir)
            current_config["model_hash"] = model_hash
            current_config["model_integrity_check"] = (
                True
                if model_hash != "no_model_files"
                and not model_hash.startswith("hash_error")
                else False
            )

            # 해시가 포함된 메타데이터로 다시 저장 (두 곳 모두)
            self.current_agent.save_checkpoint(
                self.model_dir, self.current_episode, current_config
            )
            self.current_agent.save_checkpoint(
                latest_model_dir, self.current_episode, current_config
            )

            # 상세 저장 로그
            self.add_log(f"💾 수동 모델 저장 완료: 에피소드 {self.current_episode}")
            self.add_log(f"📁 저장 위치: {self.model_dir} 및 {latest_model_dir}")
            self.add_log(
                f"📊 현재 학습 시간: {self.total_training_time/3600.0:.2f}시간"
            )
            self.add_log(
                f"🔐 모델 해시: {model_hash[:12]}... (무결성: {'✅' if current_config['model_integrity_check'] else '❌'})"
            )
            return True

        except Exception as e:
            error_msg = f"❌ 수동 모델 저장 실패: {str(e)}"
            self.add_log(error_msg)
            self.logger.error(error_msg, exc_info=True)
            return False
