"""
DDPG 알고리즘을 위한 노이즈 생성기
Ornstein-Uhlenbeck 프로세스를 사용한 탐험 노이즈
"""

import numpy as np
from typing import Union


class OUNoise:
    """
    Ornstein-Uhlenbeck 노이즈 생성기
    연속적인 행동 공간에서 탐험을 위한 상관된 노이즈 생성
    """

    def __init__(self, action_dimension: int, mu: float = 0, theta: float = 0.15, sigma: float = 0.2):
        """
        Args:
            action_dimension: 행동 공간의 차원 수
            mu: 평균 회귀값 (기본값: 0)
            theta: 평균 회귀 속도 (기본값: 0.15)
            sigma: 노이즈 강도 (기본값: 0.2)
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self) -> None:
        """노이즈 상태를 초기값으로 리셋"""
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self) -> np.ndarray:
        """
        Ornstein-Uhlenbeck 프로세스에 따른 노이즈 샘플 생성

        Returns:
            생성된 노이즈 벡터 (action_dimension,)
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
