"""
DDPG 강화학습용 거래 환경
MinMaxScaler 정규화 및 월별 리밸런싱 지원
"""

import numpy as np
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler

from src.environment.common import BaseTradingEnvironment


class TradingEnvironment(BaseTradingEnvironment):
    """
    DDPG 강화학습용 포트폴리오 거래 환경
    MinMaxScaler를 사용한 데이터 정규화 및 월별 리밸런싱 지원
    """

    def __init__(
        self,
        data,
        logger,
        initial_balance=300,
        monthly_investment=300,
        n_assets=4,
        window_size=30,
    ):
        """
        Args:
            data: 시장 데이터 DataFrame
            logger: 로거 인스턴스
            initial_balance: 초기 잔액 (기본값: 300)
            monthly_investment: 월 투자금 (기본값: 300)
            n_assets: 투자 대상 자산 수 (기본값: 4)
            window_size: 관찰 윈도우 크기 (기본값: 30)
        """
        # 원본 데이터 저장
        self.original_data = data.copy()

        # 정규화를 위한 스케일러 초기화 및 데이터 정규화
        self.scaler = MinMaxScaler()
        self.normalized_data = self.normalize_data(data)

        # BaseTradingEnvironment 초기화
        super(TradingEnvironment, self).__init__(
            self.normalized_data,
            logger,
            initial_balance,
            monthly_investment,
            n_assets,
            window_size,
        )

        # 비교 전략용 변수들
        self.shares_equal = None  # 균등 투자시 주식수
        self.balance_equal = None  # 균등 투자시 잔액

        self.value_verification = None  # 리벨런싱 시의 가치
        self.value_verification_equal = None  # 균등 투자 전략의 리벨런싱 시 가치

        # 성과 추적 변수들
        self.balance_max = None  # 가장 현금이 많을 때
        self.balance_max_date = None  # 가장 현금이 많을 때 일시

        self.value_max = None  # 가장 가치가 높을 때
        self.value_max_date = None  # 가장 가치가 높을 때 일시

        self.drop_max = None  # 가장 많이 떨어졌을 때
        self.drop_max_date = None  # 가장 많이 떨어졌을 때 일시

        self.trig_max = None  # 가장 많이 올랐을 때,
        self.trig_max_date = None  # 가장 많이 떨어졌을 때 일시

        # 행동 공간: 자산 + 현금 비율 (0~1)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(n_assets + 1,), dtype=np.float32
        )

        # 관찰 공간: 정규화된 데이터 + 포트폴리오 상태
        # merged_data의 30일치 데이터 + 현재 포트폴리오 비율(주식+현금)(5) + 현재 잔액(1) + 주식수(4) + 총가치(1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(data.shape[1] * window_size + n_assets + 1 + 1 + n_assets + 1,),
            dtype=np.float32,
        )

        self.logger.info(f"Observation space shape: {self.observation_space.shape}")

        self.reset()

    def normalize_data(self, data):
        """데이터 정규화"""
        # 'date' 열이 있는지 확인
        if "date" in data.columns:
            columns_to_normalize = data.columns.drop("date")
        else:
            columns_to_normalize = data.columns

        normalized = data.copy()
        normalized[columns_to_normalize] = self.scaler.fit_transform(
            data[columns_to_normalize]
        )
        return normalized

    def reset(self, step=0):
        """환경 리셋"""
        self.reset_variables()

        # 비교 전략 초기화
        self.shares_equal = np.zeros(self.n_assets)
        self.balance_equal = self.initial_balance

        self.value_verification = 0  # 리벨런싱 시의 가치
        self.value_verification_equal = 0  # 균등 투자 전략의 리벨런싱 시 가치

        # 성과 추적 변수 초기화
        self.balance_max = 0  # 가장 현금이 많을 때
        self.value_max = 0  # 가장 가치가 높을 때
        self.drop_max = 0  # 가장 많이 떨어졌을 때
        self.trig_max = 0  # 가장 많이 올랐을 때,

        self.current_step += step % 500

        self.balance_max_date = self._current_date()  # 가장 현금이 많을 때 일시
        self.value_max_date = self._current_date()  # 가장 가치가 높을 때 일시
        self.drop_max_date = self._current_date()  # 가장 많이 떨어졌을 때 일시

        self.trig_max_date = self._current_date()  # 가장 많이 떨어졌을 때 일시

        return self._next_observation()

    def _next_observation(self):
        """다음 관찰값 생성"""
        # 정규화된 데이터 사용
        merged_data_history = self.normalized_data.iloc[
            self.current_step - self.window_size : self.current_step
        ].values.flatten()

        # 현재 포트폴리오 비율
        portfolio_ratio = self.portfolio

        # 현재 잔액 (정규화하지 않음)
        balance = np.array([self.balance])

        # 주식수 (정규화하지 않음)
        shares = self.shares

        # 가치 (정규화하지 않음)
        total_value = np.array([self._calculate_value()])

        return np.concatenate(
            [merged_data_history, portfolio_ratio, balance, shares, total_value]
        )

    def step(self, action):
        """환경 스텝 실행"""
        self.current_step += 1
        verification = False
        reward_agent = 0
        reward_monthly_agent = 0
        reward_monthly_equal = 0

        # 한 달(30일)이 지났는지 확인
        if (self.current_step - self.last_rebalance_step) >= 30:

            if self.value_verification != 0 and self.value_verification_equal != 0:
                # 매일의 가치 계산 및 보상
                current_value_agent = self._calculate_value()
                current_value_equal = self._calculate_value_equal()

                reward_monthly_agent = (
                    (current_value_agent - self.value_verification)
                    / self.value_verification
                ) * 100
                reward_monthly_equal = (
                    (current_value_equal - self.value_verification_equal)
                    / self.value_verification_equal
                ) * 100

            self.last_rebalance_step = self.current_step

            # Agent와 균등 투자 전략 모두 독립적으로 월별 투자금 추가
            self.balance += self.monthly_investment  # Agent 투자금 추가
            self.balance_equal += (
                self.monthly_investment
            )  # 균등 투자 전략도 독립적으로 투자금 추가
            self.total_invested += self.monthly_investment

            # 포트폴리오 리밸런싱
            self.portfolio = action / np.sum(action)

            # 리벨런싱에 대한 평가를 위해선 리벨런싱 전 가치를 저장해야함
            self.value_verification = self._calculate_value()
            self.value_verification_equal = self._calculate_value_equal()

            max_balance = max(self.balance_max, self.balance)
            if max_balance == self.balance:
                self.balance_max = self.balance
                self.balance_max_date = self._current_date()

            # 거래 실행 - Agent와 균등 투자 전략 각각 독립적으로 실행
            self._rebalance_portfolio_agent(self.original_data)
            self._rebalance_portfolio_equal(self.original_data)

            verification = True

        max_value = max(self.value_max, self._calculate_value())
        if max_value == self._calculate_value():
            self.value_max = self._calculate_value()
            self.value_max_date = self._current_date()

        done = self.current_step >= len(self.data) - 1

        return (
            self._next_observation(),
            reward_agent,
            reward_monthly_agent,
            reward_monthly_equal,
            done,
            {},
            verification,
        )

    def _rebalance_portfolio_agent(self, data):
        """에이전트 포트폴리오 리밸런싱"""
        # 원본 데이터 사용
        current_values = data.iloc[self.current_step][
            data.columns[: self.n_assets]
        ].values

        # self.render_rebalance()

        # 현재 주가 및 목표 가치 계산
        target_values = self.portfolio[:-1] * (
            self.balance + np.sum(self.shares * current_values)
        )

        target_shares = np.floor(target_values / current_values).astype(int)  # 정수
        shares_diff = target_shares - self.shares

        fee = 0

        # 먼저 매도
        for i in range(self.n_assets):
            if shares_diff[i] < 0:  # 매도
                revenue = (
                    -shares_diff[i] * current_values[i] * 0.99735
                )  # 수수료 0.015% + 세금 0.25% 적용
                self.fee += abs(
                    -shares_diff[i] * current_values[i] * 0.00265
                )  # 수수료 0.015% + 세금 0.25% 적용
                fee += abs(
                    -shares_diff[i] * current_values[i] * 0.00265
                )  # 수수료 0.015% + 세금 0.25% 적용
                self.shares[i] += shares_diff[i]
                self.balance += revenue

        # 그 다음 매수
        for i in range(self.n_assets):
            if shares_diff[i] > 0:  # 매수
                cost = (
                    shares_diff[i] * current_values[i] * 1.00015
                )  # 수수료 0.015% 적용
                if self.balance >= cost:
                    self.fee += abs(
                        shares_diff[i] * current_values[i] * 0.00015
                    )  # 수수료 0.015% 적용
                    fee += abs(
                        shares_diff[i] * current_values[i] * 0.00015
                    )  # 수수료 0.015% 적용
                    self.shares[i] += shares_diff[i]
                    self.balance -= cost

    def _rebalance_portfolio_equal(self, data):
        """균등 분배 포트폴리오 리밸런싱"""
        # 원본 데이터 사용
        current_values = data.iloc[self.current_step][
            data.columns[: self.n_assets]
        ].values

        # 균등 분배: 각 자산에 25%씩, 현금 0%
        target_portfolio = np.zeros(self.n_assets + 1)
        target_portfolio[:-1] = 1.0 / self.n_assets  # 각 자산에 균등 분배
        target_portfolio[-1] = 0  # 현금 0%

        # 현재 주가 및 목표 가치 계산
        target_values = target_portfolio[:-1] * (
            self.balance_equal + np.sum(self.shares_equal * current_values)
        )

        target_shares = np.floor(target_values / current_values).astype(int)
        shares_diff = target_shares - self.shares_equal

        # 매도 먼저
        for i in range(self.n_assets):
            if shares_diff[i] < 0:  # 매도
                revenue = -shares_diff[i] * current_values[i] * 0.99735
                self.shares_equal[i] += shares_diff[i]
                self.balance_equal += revenue

        # 매수
        for i in range(self.n_assets):
            if shares_diff[i] > 0:  # 매수
                cost = shares_diff[i] * current_values[i] * 1.00015
                if self.balance_equal >= cost:
                    self.shares_equal[i] += shares_diff[i]
                    self.balance_equal -= cost

    def _calculate_value(self):
        """현재 포트폴리오 가치 계산 (원본 데이터 사용)"""
        # 원본 데이터 사용하여 가치 계산
        current_prices = self.original_data.iloc[self.current_step][
            self.original_data.columns[: self.n_assets]
        ].values
        return np.sum(self.shares * current_prices) + self.balance

    def _calculate_value_equal(self):
        """균등 투자 전략의 현재 포트폴리오 가치 계산 (원본 데이터 사용)"""
        # 원본 데이터 사용하여 가치 계산
        current_prices = self.original_data.iloc[self.current_step][
            self.original_data.columns[: self.n_assets]
        ].values
        return np.sum(self.shares_equal * current_prices) + self.balance_equal
