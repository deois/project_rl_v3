"""
공통 거래 환경 기본 클래스
포트폴리오 관리를 위한 기본 환경 구조
"""

from typing import Tuple, Optional, Any
import gymnasium
import numpy as np
import pandas as pd
import logging


class BaseTradingEnvironment(gymnasium.Env[np.ndarray, np.ndarray]):
    """
    기본 거래 환경 클래스
    OpenAI Gym 환경을 상속받아 포트폴리오 거래 환경의 기본 구조를 제공
    """

    def __init__(self, data: pd.DataFrame, logger: logging.Logger, initial_balance: float = 300, monthly_investment: float = 300, n_assets: int = 4, window_size: int = 30):
        """
        Args:
            data: 시장 데이터 DataFrame
            logger: 로거 인스턴스
            initial_balance: 초기 잔액 (기본값: 300)
            monthly_investment: 월 투자금 (기본값: 300)
            n_assets: 투자 대상 자산 수 (기본값: 4)
            window_size: 관찰 윈도우 크기 (기본값: 30)
        """
        super(BaseTradingEnvironment, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.monthly_investment = monthly_investment
        self.n_assets = n_assets
        self.window_size = window_size
        self.logger = logger

        self.reset()

    def reset_variables(self):
        """환경 변수 초기화"""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.total_invested = self.initial_balance
        self.portfolio = np.zeros(self.n_assets + 1)
        self.portfolio[-1] = 1  # 초기에는 모든 자금이 현금
        self.shares = np.zeros(self.n_assets)
        self.last_rebalance_step = self.current_step
        self.last_investment_step = self.current_step
        self.fee = 0  # 수수료
        # 초기 포트폴리오 값을 저장 (수익률 계산 기준점)
        self.initial_portfolio_value = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """환경 리셋"""
        super().reset(seed=seed)
        self.reset_variables()
        # 첫 번째 관찰값 생성 후 초기 포트폴리오 값 설정
        observation = self._next_observation()
        if self.initial_portfolio_value is None:
            self.initial_portfolio_value = self._calculate_value()
        return observation, {}

    def _next_observation(self):
        """다음 관찰값 생성"""
        merged_data_history = self.data.iloc[self.current_step -
                                             self.window_size:self.current_step].values.flatten()
        portfolio_ratio = self.portfolio
        balance = np.array([self.balance])
        shares = self.shares
        total_value = np.array([self._calculate_value()])
        fee = np.array([self.fee])

        return np.concatenate([merged_data_history, portfolio_ratio, balance, shares, total_value, fee])

    def _calculate_value(self):
        """현재 포트폴리오 가치 계산"""
        current_prices = self.data.iloc[self.current_step][self.data.columns[:self.n_assets]].values
        return np.sum(self.shares * current_prices) + self.balance

    def _calculate_annualized_return(self):
        """연환산 수익률 계산 - total_invested 기준"""
        days_invested = (self.data.index[self.current_step] - self.data.index[0]).days

        # 30일 미만에서는 연환산 수익률이 의미가 없으므로 0 반환
        if days_invested < 30:
            return 0.0

        if days_invested > 0 and self.total_invested > 0:
            years_invested = days_invested / 365.25
            current_value = self._calculate_value()
            return ((current_value / self.total_invested) ** (1 / years_invested) - 1) * 100
        else:
            return 0.0

    def _calculate_total_return(self):
        """총 수익률 계산 - total_invested 기준"""
        if self.total_invested > 0:
            current_value = self._calculate_value()
            return ((current_value - self.total_invested) / self.total_invested) * 100
        else:
            return 0.0

    def _current_date(self):
        """현재 날짜 반환"""
        current_date = self.data.index[self.current_step].strftime('%Y-%m-%d')
        # self.logger.info(f'current date is {current_date}')
        return current_date, self.data.index[self.current_step]

    def render(self, mode: str = 'human') -> str:
        """환경 상태 렌더링"""
        self.logger.info(
            f'Step: {self.current_step}, 가치: {self._calculate_value():.2f}, 현금: {self.balance:.2f} 연수익율 {self._calculate_annualized_return():.2f}%, 수익률 {self._calculate_total_return():.2f}, 수수료 {self.fee:.2f}')
        # portfolio_name = ["QQQ", "SCHD", "EWY", "HYG", "현금"]
        # portfolio_name = self.data.columns.tolist() + ['현금']
        portfolio_name = self.data.columns[:self.n_assets].tolist() + ['현금']
        portfolio_percentages = [f'{ticker}: {p * 100:.1f}%' for ticker,
                                 p in zip(portfolio_name, self.portfolio)]
        self.logger.info(f'Portfolio: {portfolio_percentages}')
        ticker_name = self.data.columns.tolist()
        shares_volume = [f'{ticker}: {p}' for ticker, p in zip(ticker_name, self.shares)]
        self.logger.info(f'Shares: {shares_volume}')

        # Calculate actual ratios
        current_prices = self.data.iloc[self.current_step][self.data.columns[:self.n_assets]].values
        total_value = self._calculate_value()
        actual_ratios = [(self.shares[i] * current_prices[i] / total_value)
                         * 100 for i in range(self.n_assets)]
        cash_ratio = (self.balance / total_value) * 100
        actual_ratios.append(cash_ratio)

        message = (f'가치: {self._calculate_value():.2f}, 현금: {self.balance:.2f}, 연수익율 {self._calculate_annualized_return():.2f}%, 수익률 {self._calculate_total_return():.2f}, 수수료 {self.fee:.2f}\n'
                   f'Portfolio: {portfolio_percentages}\n'
                   f'Shares: {shares_volume}\n')
        return message

    def render_rebalance(self):
        """리밸런싱 상태 렌더링"""
        # self.logger.info the current date
        current_date = self.data.index[self.current_step]
        self.logger.info("--------------------------------------------------")
        self.logger.info(f"Rebalancing on date: {current_date}")

        # Only select the columns corresponding to the assets being traded
        current_price_assets = self.data.iloc[self.current_step][self.data.columns[:self.n_assets]].values
        target_values = self.portfolio[:-1] * \
            (self.balance + np.sum(self.shares * current_price_assets))

        # Calculate current ratios
        total_value = self._calculate_value()
        current_ratios = [(self.shares[i] * current_price_assets[i] / total_value)
                          for i in range(self.n_assets)]
        current_ratios.append(self.balance / total_value)  # Add cash ratio

        # self.logger.info current values with stock names
        stock_names = self.data.columns[:self.n_assets].tolist() + ['현금']

        # self.logger.info current ratios with stock names and without decimal points
        current_ratios_with_names = {
            name: f"{int(ratio * 100)}%" for name, ratio in zip(stock_names, current_ratios)}
        self.logger.info(f"현재 포트폴리오 비율: {current_ratios_with_names}")

        # self.logger.info target ratios with stock names and without decimal points
        target_ratios_with_names = {
            name: f"{int(ratio * 100)}%" for name, ratio in zip(stock_names, self.portfolio)}
        self.logger.info(f"목표 포트폴리오 비율: {target_ratios_with_names}")

        # self.logger.info current portfolio value with stock names and cash
        current_values = np.append(self.shares * current_price_assets, self.balance)
        current_values_with_names = {name: int(value)
                                     for name, value in zip(stock_names, current_values)}
        self.logger.info(f"현재 포트폴리오 가치: {current_values_with_names}")

        # Calculate the target values including cash
        target_values_with_names = {name: int(value)
                                    for name, value in zip(stock_names, target_values)}

        # Calculate the total value of the stocks
        total_stock_value = sum(target_values)

        # Calculate the cash value as the total portfolio value minus the total stock value
        cash_value = self._calculate_value() - total_stock_value

        # Add the cash value to the target values dictionary
        target_values_with_names['현금'] = int(cash_value)

        self.logger.info(f"목표 포트폴리오 가치: {target_values_with_names}")

        current_values_with_names = {name: int(value)
                                     for name, value in zip(stock_names, current_price_assets)}
        self.logger.info(f"현재주가   : {current_values_with_names}")

        # Calculate and self.logger.info target shares
        target_shares_with_names = {name: int(
            target_values[i] / current_price_assets[i]) for i, name in enumerate(self.data.columns[:self.n_assets])}
        # Calculate remaining cash after buying target shares

        self.logger.info(f"목표 주식수: {target_shares_with_names}")

        # Create a dictionary with stock names and their respective share counts
        shares_with_names = {name: int(share) for name, share in zip(
            self.data.columns[:self.n_assets], self.shares)}
        # self.logger.info the updated shares with stock names
        self.logger.info(f"갱신전 주식수: {shares_with_names}, 갱신전 현금 : {int(self.balance)}")
