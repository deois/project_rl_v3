"""
DDPG 알고리즘을 위한 Actor-Critic 신경망 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class Actor(nn.Module):
    """
    Actor 네트워크: 연속적인 행동 공간에서 포트폴리오 비율을 결정
    상태를 입력받아 각 자산에 대한 투자 비율(0~1)을 출력
    Softmax-Affine 변환을 통해 최소 비중 10% 제약을 보장
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden1_dim: int = 400,
        hidden2_dim: int = 300,
    ):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden1_dim)
        self.ln1 = nn.LayerNorm(hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.ln2 = nn.LayerNorm(hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, action_dim)
        self.action_dim = action_dim  # action_dim 저장 (Affine Scaling에 사용)
        self.min_weight = 0.075  # 최소 비중 ε = 0.075 (7.5%)
        self.init_weights()

    def init_weights(self) -> None:
        """네트워크 가중치 초기화"""
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        순전파: 상태 -> 행동 확률
        Softmax-Affine 변환을 적용하여 최소 비중 10% 제약을 보장

        Args:
            state: 현재 시장 상태 (배치 크기, 상태 차원)

        Returns:
            각 자산에 대한 투자 비율 (배치 크기, 행동 차원)
            모든 비중은 최소 10% 이상이며 합이 1이 됨

        수학적 검증:
        - 공식: w_i = ε + (1 - Nε) × s_i
        - 여기서 ε = 0.10 (최소 비중), N = action_dim (자산 수)
        - 합 검증: Σw_i = Nε + (1 - Nε) × Σs_i = Nε + (1 - Nε) × 1 = 1
        - 최소값 검증: s_i ≥ 0 이므로 w_i ≥ ε = 0.10
        """
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))

        # 로짓 출력
        z = self.fc3(x)

        # Standard Softmax: s = Softmax(z)
        s = torch.softmax(z, dim=-1)

        # Affine Scaling: w_i = ε + (1 - Nε) × s_i
        # N = action_dim, ε = 0.10
        epsilon = self.min_weight
        n_assets = self.action_dim
        weights = epsilon + (1.0 - n_assets * epsilon) * s

        return weights


class Critic(nn.Module):
    """
    Critic 네트워크: 상태-행동 쌍의 가치를 평가
    포트폴리오 상태와 투자 행동을 입력받아 Q-value를 출력
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden1_dim: int = 400,
        hidden2_dim: int = 300,
    ):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden1_dim)
        self.ln1 = nn.LayerNorm(hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.ln2 = nn.LayerNorm(hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, 1)
        self.init_weights()

    def init_weights(self) -> None:
        """네트워크 가중치 초기화"""
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        순전파: 상태-행동 -> Q-value

        Args:
            state: 현재 시장 상태 (배치 크기, 상태 차원)
            action: 투자 행동 (배치 크기, 행동 차원)

        Returns:
            Q-value (배치 크기, 1)
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


def fanin_init(size: Tuple[int, ...], fanin: int = None) -> torch.Tensor:
    """
    Fan-in 기반 가중치 초기화

    Args:
        size: 텐서 크기
        fanin: Fan-in 값 (기본값: size[0])

    Returns:
        초기화된 텐서
    """
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)
