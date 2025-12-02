"""
Dash 시뮬레이션 모듈
시뮬레이션 학습 및 테스트 기능
"""

import threading
import time
import uuid
from typing import Dict, Any
import numpy as np

from src.utils.logger import get_logger

# 로거 설정
logger = get_logger("dash_simulation")


def start_simulation_training(dash_manager, task_id: str, config: Dict[str, Any]) -> None:
    """시뮬레이션 학습 시작"""

    def simulation_thread():
        try:
            dash_manager.simulation_stop_event = threading.Event()

            dash_manager.add_log(f"🎮 시뮬레이션 학습 시작! 목표: {config['episodes']} 에피소드")

            for episode in range(1, config['episodes'] + 1):
                if dash_manager.simulation_stop_event.is_set():
                    dash_manager.add_log(f"🛑 시뮬레이션 에피소드 {episode}에서 중지됨")
                    break

                # 에피소드 시작 로그
                dash_manager.add_log(f"🎯 에피소드 {episode}/{config['episodes']} 시작")

                # 에피소드 내에서의 스텝 시뮬레이션 (포트폴리오 관리 환경에서는 대략 250일 정도)
                total_steps = 250

                # 에피소드 시작 시 상태 업데이트
                dash_manager.training_status.update({
                    "current_episode": episode,
                    "total_steps_per_episode": total_steps,
                    "current_step": 0,
                    "learning_phase": "에피소드 초기화"
                })

                # 잠깐 대기하여 UI 업데이트 확인
                time.sleep(0.1)

                for step in range(1, total_steps + 1):
                    if dash_manager.simulation_stop_event.is_set():
                        break

                    # 학습 단계별 상태 업데이트
                    if step <= 5:
                        learning_phase = "환경 초기화"
                    elif step <= 20:
                        learning_phase = "데이터 로딩"
                    elif step <= 50:
                        learning_phase = "모델 준비"
                    elif step < total_steps * 0.9:
                        learning_phase = "DDPG 학습"
                    else:
                        learning_phase = "에피소드 마무리"

                    # 실시간 상태 업데이트
                    dash_manager.training_status.update({
                        "current_step": step,
                        "learning_phase": learning_phase
                    })

                    # 빠른 진행을 위해 0.02초마다 업데이트 (총 5초 에피소드)
                    time.sleep(0.02)

                # 에피소드 완료 후 최종 데이터 계산
                base_reward = 80 + episode * 0.5
                reward = base_reward + np.random.normal(0, 15)

                base_portfolio = 10000 + episode * 80
                portfolio_value = base_portfolio + np.random.normal(0, 200)

                # 손실값 시뮬레이션 (감소 추세)
                actor_loss = max(0.001, 0.1 - episode * 0.001 + np.random.normal(0, 0.01))
                critic_loss = max(0.001, 0.15 - episode * 0.0015 + np.random.normal(0, 0.015))

                # 최종 상태 업데이트
                dash_manager.training_status.update({
                    "current_episode": episode,
                    "current_reward": reward,
                    "portfolio_value": portfolio_value,
                    "average_reward": np.mean([reward] * min(episode, 20)),
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "current_step": total_steps,
                    "learning_phase": "에피소드 완료"
                })

                # 차트 데이터 업데이트
                dash_manager.update_chart_data(
                    episode, reward, portfolio_value, actor_loss, critic_loss)

                # 에피소드 완료 로그
                dash_manager.add_log(
                    f"✅ 에피소드 {episode}/{config['episodes']} 완료: "
                    f"보상 {reward:.2f}, 포트폴리오 ${portfolio_value:.2f}"
                )

                # 에피소드 간 잠깐 휴식
                time.sleep(0.5)

        except Exception as e:
            dash_manager.add_log(f"❌ 시뮬레이션 중 오류: {str(e)}")
            logger.error(f"시뮬레이션 오류: {e}")
        finally:
            dash_manager.training_status.update({
                "is_training": False,
                "can_stop": False
            })
            dash_manager.add_log(f"✅ 시뮬레이션 완료 (ID: {task_id})")

    thread = threading.Thread(target=simulation_thread, daemon=True)
    thread.start()
