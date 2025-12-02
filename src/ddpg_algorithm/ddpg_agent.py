"""
DDPG (Deep Deterministic Policy Gradient) 에이전트
강화학습 기반 연속적 포트폴리오 최적화 에이전트
"""

import os
import time
import json
from typing import Tuple, Optional, Dict, Any, List, Union
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .ddpg_models import Actor, Critic
from .ddpg_noise import OUNoise


class DDPGAgent:
    """
    DDPG 에이전트 클래스
    Deep Deterministic Policy Gradient 알고리즘을 사용한 포트폴리오 최적화
    """

    def __init__(
        self,
        logger: logging.Logger,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        device: str = "cpu",
        max_grad_norm: float = 0.5,
        critic_loss_type: str = "mse",
    ):
        """
        Args:
            logger: 로거 인스턴스
            state_dim: 상태 공간 차원
            action_dim: 행동 공간 차원
            hidden_dim: 은닉층 차원
            actor_lr: Actor 네트워크 학습률
            critic_lr: Critic 네트워크 학습률
            device: 연산 장치 ('cpu', 'cuda', 'mps')
            max_grad_norm: Gradient clipping 최대 norm 값 (기본값: 0.5)
            critic_loss_type: Critic loss 함수 타입 ('mse' 또는 'smooth_l1', 기본값: 'mse')
        """
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_lr, weight_decay=1e-4
        )

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=1e-4
        )

        self.gamma = 0.99  # 할인 인수
        self.tau = 0.001  # 타겟 네트워크 업데이트 비율 조정
        self.device = device
        self.logger = logger
        self.max_grad_norm = max_grad_norm  # Gradient clipping 최대 norm
        self.critic_loss_type = critic_loss_type  # Critic loss 함수 타입

        self.noise = OUNoise(action_dim, theta=0.15, sigma=0.2)  # 노이즈 파라미터 조정
        self.update_counter = 0
        self.update_freq = 2  # 타겟 네트워크 업데이트 주기

    def select_action(
        self, state: np.ndarray, add_noise: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        상태에 따른 행동 선택
        최소 비중 10% 제약을 유지하면서 노이즈를 추가

        Args:
            state: 현재 상태
            add_noise: 탐험 노이즈 추가 여부 (학습 시 True, 평가 시 False)

        Returns:
            (클리핑된 행동, 원본 행동)
            - 클리핑된 행동: 최소 비중 10% 제약을 만족하는 액션
            - 원본 행동: Actor 네트워크의 원본 출력 (Affine Scaling 적용됨)
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).squeeze(0)
        action = action.detach().cpu().numpy()
        raw_action = (
            action.copy()
        )  # 원본 액션 저장 (Affine Scaling 적용됨, 이미 최소 10% 보장)

        if add_noise:
            # DDPG 방식: OUNoise를 사용한 탐험 (연속 행동 공간에 적합)
            # 입실론 그리디는 이산 행동 공간용이므로 DDPG에는 부적합
            action += self.noise.sample()  # 탐험을 위한 노이즈 추가

            # 노이즈 추가 후 최소 비중 제약을 유지하는 클리핑
            min_weight = 0.075  # 최소 비중 7.5%

            # 각 요소를 최소 비중 이상으로 클리핑
            action = np.maximum(action, min_weight)

            # 합이 1이 되도록 정규화
            action_sum = np.sum(action)
            if action_sum > 0:
                action = action / action_sum
            else:
                # 모든 값이 0인 경우 균등 분배 (각 요소는 최소 7.5% 이상)
                action = np.ones_like(action) / len(action)

            # 정규화 후에도 최소 비중 제약이 유지되는지 검증
            # (정규화로 인해 일부 요소가 0.075 미만으로 떨어질 수 있으므로 재조정)
            min_val = np.min(action)
            if min_val < min_weight:
                # 최소값이 0.075 미만인 경우, 모든 요소를 최소 0.075 이상으로 조정
                action = np.maximum(action, min_weight)
                # 다시 정규화
                action = action / np.sum(action)
        else:
            # 노이즈 없는 경우: Actor 출력 그대로 사용 (이미 Affine Scaling으로 최소 10% 보장됨)
            # 추가 클리핑 불필요
            pass

        return action, raw_action  # 최종 액션과 원본 액션 반환

    def update(self, batch: List[Tuple]) -> Tuple[float, float]:
        """
        배치 데이터로 네트워크 업데이트

        Args:
            batch: (상태, 행동, 보상, 다음상태, 완료) 튜플들의 리스트

        Returns:
            (actor 손실, critic 손실)
        """
        batch = list(zip(*batch))
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # Critic update
        next_actions = self.actor_target(next_states)
        target_q = self.critic_target(next_states, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * target_q
        current_q = self.critic(states, actions)

        # Critic loss 계산 (loss 타입에 따라 선택)
        if self.critic_loss_type == "smooth_l1":
            critic_loss = F.smooth_l1_loss(current_q, target_q.detach())
        else:  # 기본값: mse
            critic_loss = F.mse_loss(current_q, target_q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping 적용
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Gradient clipping 적용
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # 타겟 네트워크 소프트 업데이트
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        """
        타겟 네트워크 소프트 업데이트

        Args:
            target: 타겟 네트워크
            source: 소스 네트워크
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save_checkpoint(
        self,
        model_dir: str,
        episode: int,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        모델 체크포인트 저장

        Args:
            model_dir: 모델 저장 디렉토리
            episode: 현재 에피소드
            training_config: 학습 설정 메타데이터
        """
        checkpoint_name = os.path.join(model_dir, f"checkpoint_{episode:04d}.pth")
        checkpoint_last = os.path.join(model_dir, f"checkpoint_last.pth")

        # 학습 설정 메타데이터 추가
        checkpoint_data = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "episode": episode,
            "save_time": time.time(),
            "training_metadata": training_config or {},
        }

        # 체크포인트 저장
        torch.save(checkpoint_data, checkpoint_name)
        torch.save(checkpoint_data, checkpoint_last)

        # 메타데이터 JSON 파일로도 별도 저장 (가독성을 위해)
        if training_config:
            metadata_file = os.path.join(model_dir, f"metadata_{episode:04d}.json")
            metadata_last = os.path.join(model_dir, "metadata_last.json")

            metadata = {
                "episode": episode,
                "save_time": checkpoint_data["save_time"],
                "save_datetime": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(checkpoint_data["save_time"])
                ),
                "training_config": training_config,
                "model_info": {
                    "hidden_dim": training_config.get("hidden_dim", 256),
                    "learning_rates": {
                        "actor_lr": training_config.get("actor_lr", 0.0003),
                        "critic_lr": training_config.get("critic_lr", 0.0003),
                    },
                    "critic_loss_type": training_config.get("critic_loss_type", "mse"),
                    "max_grad_norm": training_config.get("max_grad_norm", 0.5),
                    "batch_size": training_config.get("batch_size", 128),
                    "window_size": training_config.get("window_size", 60),
                    "assets": training_config.get("assets", []),
                },
            }

            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            with open(metadata_last, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 모델 및 메타데이터 저장 완료: {checkpoint_name}")

    def load_checkpoint(
        self, model_dir: str, episode: Optional[int] = None, evaluate: bool = False
    ) -> bool:
        """
        모델 체크포인트 로드

        Args:
            model_dir: 모델 저장 디렉토리
            episode: 로드할 에피소드 (None시 최신)
            evaluate: 평가 모드 여부

        Returns:
            로드 성공 여부
        """
        if episode is None:
            checkpoint_name = os.path.join(model_dir, "checkpoint_last.pth")
        else:
            checkpoint_name = os.path.join(model_dir, f"checkpoint_{episode:04d}.pth")
            if not os.path.exists(checkpoint_name):
                self.logger.info(f"Checkpoint {checkpoint_name} does not exist")
                checkpoint_name = os.path.join(model_dir, "checkpoint_last.pth")
                if not os.path.exists(checkpoint_name):
                    self.logger.info(f"Checkpoint {checkpoint_name} does not exist")
                    return False  # 명시적으로 False 반환

        if checkpoint_name is not None and os.path.exists(checkpoint_name):
            try:
                self.logger.info("Loading models from {}".format(checkpoint_name))
                # PyTorch 2.6 호환성을 위해 weights_only=False 명시적 설정
                checkpoint = torch.load(
                    checkpoint_name, map_location=self.device, weights_only=False
                )
                self.logger.info(f"checkpoint: {checkpoint_name}")

                # 메인 네트워크 로드
                self.actor.load_state_dict(checkpoint["actor_state_dict"])
                self.critic.load_state_dict(checkpoint["critic_state_dict"])

                # 타겟 네트워크 로드
                self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
                self.critic_target.load_state_dict(
                    checkpoint["critic_target_state_dict"]
                )

                # 옵티마이저 로드 (학습 재개시에만 필요)
                if not evaluate:
                    self.actor_optimizer.load_state_dict(
                        checkpoint["actor_optimizer_state_dict"]
                    )
                    self.critic_optimizer.load_state_dict(
                        checkpoint["critic_optimizer_state_dict"]
                    )

                # 평가 모드 설정
                if evaluate:
                    self.actor.eval()
                    self.critic.eval()
                    self.actor_target.eval()
                    self.critic_target.eval()

                    # 평가 시 노이즈 상태 리셋
                    self.noise.reset()
                    self.logger.info("🔕 평가 모드: 노이즈 상태 리셋 완료")

                    # 타겟 네트워크를 메인 네트워크와 완전 동기화 (평가용)
                    self.actor_target.load_state_dict(self.actor.state_dict())
                    self.critic_target.load_state_dict(self.critic.state_dict())
                    self.logger.info("🔄 평가 모드: 타겟 네트워크 동기화 완료")
                else:
                    self.actor.train()
                    self.critic.train()
                    self.actor_target.train()
                    self.critic_target.train()

                # 로딩된 모델의 액션 분포 검증 (평가 모드에서만)
                if evaluate:
                    self._verify_model_diversity()

                self.logger.info(f"✅ 모델 로드 성공: {checkpoint_name}")
                return True  # 성공 시 True 반환
            except Exception as e:
                self.logger.error(f"❌ 모델 로드 실패: {e}")
                return False  # 예외 발생 시 False 반환
        else:
            self.logger.error(
                f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_name}"
            )
            return False  # 파일이 없으면 False 반환

    def _verify_model_diversity(self) -> None:
        """모델의 액션 다양성 검증 (평가 모드에서만 실행)"""
        try:
            # 다양한 테스트 상태에 대해 액션 분포 확인
            test_states = []
            for i in range(5):
                # 랜덤 상태 생성 (실제 환경 상태 차원에 맞춰)
                test_state = np.random.randn(self.actor.fc1.in_features)  # 상태 차원
                test_states.append(test_state)

            actions = []
            for state in test_states:
                action, _ = self.select_action(state, add_noise=False)
                actions.append(action)

            actions = np.array(actions)

            # 액션 분산성 계산
            action_std = np.std(actions, axis=0)
            action_mean = np.mean(actions, axis=0)

            self.logger.info(f"🎯 액션 다양성 검증:")
            self.logger.info(f"   - 평균 액션: {action_mean}")
            self.logger.info(f"   - 액션 표준편차: {action_std}")
            self.logger.info(f"   - 분산성 점수: {np.mean(action_std):.4f}")

            # 낮은 분산성 경고
            if np.mean(action_std) < 0.01:
                self.logger.warning(
                    "⚠️ 낮은 액션 분산성 감지 - 모델이 한쪽으로 치우쳐있을 수 있음"
                )

        except Exception as e:
            self.logger.warning(f"⚠️ 액션 다양성 검증 실패: {e}")

    def reset_for_evaluation(self) -> None:
        """평가를 위한 에이전트 상태 리셋"""
        self.noise.reset()
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()
        self.logger.info("🔄 평가를 위한 에이전트 상태 리셋 완료")
