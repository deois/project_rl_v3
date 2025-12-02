"""
데이터 처리 모듈
ETF, 거시경제 데이터 수집 및 병합
"""

from .merge import load_merged_data_v1
from .loader import (
    get_yf_assets_data_v2,
    get_FinanceDataReader_data,
    get_yf_market_data,
    get_FRED_data_v2
)

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    "load_merged_data_v1",
    "get_yf_assets_data_v2",
    "get_FinanceDataReader_data",
    "get_yf_market_data",
    "get_FRED_data_v2"
]
