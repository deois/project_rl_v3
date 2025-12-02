"""
데이터 로더 모듈
Yahoo Finance, FRED, FinanceDataReader를 통한 금융 데이터 수집
"""

import os
import pandas as pd
import yfinance as yf
from rich import print as rprint
from fredapi import Fred
import time
import FinanceDataReader as fdr  # FinanceDataReader 라이브러리 추가
from typing import List, Optional
import tempfile

# yfinance SQLite 에러 대응을 위한 설정
# try:
#     # SQLite 에러 방지를 위한 대안적 접근
#     # 임시 디렉토리를 사용하여 캐시 설정
#     temp_cache_dir = tempfile.mkdtemp()
#     yf.set_tz_cache_location(temp_cache_dir)
#     rprint("✅ yfinance 임시 캐시 디렉토리가 설정되었습니다 (SQLite 에러 방지)")
# except Exception as e:
#     rprint(f"⚠️ yfinance 캐시 설정 실패: {e}")
#     rprint("ℹ️ SQLite 에러가 발생할 수 있습니다. 최신 yfinance 버전으로 업그레이드를 권장합니다.")


# 데이터 가져오기 함수
def get_yf_assets_data_v2(tickers):
    """
    Yahoo Finance를 통해 주식 데이터를 가져옵니다.

    Args:
        tickers (list): 티커 심볼 리스트

    Returns:
        pandas.DataFrame: Close 가격 데이터
    """
    data = yf.download(tickers)
    data = data['Close']
    # 티커의 순서에 맞게 데이터의 컬럼을 재정렬
    data = data[tickers]
    rprint("get_stock_data_v2-------------------")
    rprint(data.head())
    rprint(data.tail())
    rprint("get_stock_data_v2----dropna--------")
    data_dropa = data.dropna()
    rprint(data_dropa.head())
    return data


def get_FinanceDataReader_data(tickers: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    FinanceDataReader를 사용하여 주식 데이터를 가져오는 함수

    Args:
        tickers: 주식 종목 코드 리스트 (예: ['005930', '000660', 'AAPL'])
        start_date: 시작 날짜 (YYYY-MM-DD 형식, 기본값: None - 모든 데이터)
        end_date: 종료 날짜 (YYYY-MM-DD 형식, 기본값: None - 모든 데이터)

    Returns:
        pandas.DataFrame: Close 가격 데이터가 포함된 DataFrame
    """
    data_frames = []
    successful_tickers = []

    # 날짜 범위 설정에 따른 메시지 출력
    if start_date and end_date:
        rprint(f"FinanceDataReader 데이터 수집 시작: {start_date} ~ {end_date}")
    elif start_date:
        rprint(f"FinanceDataReader 데이터 수집 시작: {start_date} ~ 현재")
    elif end_date:
        rprint(f"FinanceDataReader 데이터 수집 시작: 전체 ~ {end_date}")
    else:
        rprint("FinanceDataReader 전체 데이터 수집 시작")

    for ticker in tickers:
        try:
            # FinanceDataReader를 사용하여 데이터 가져오기 (조건부 매개변수 전달)
            if start_date and end_date:
                ticker_data = fdr.DataReader(ticker, start=start_date, end=end_date)
            elif start_date:
                ticker_data = fdr.DataReader(ticker, start=start_date)
            elif end_date:
                ticker_data = fdr.DataReader(ticker, end=end_date)
            else:
                ticker_data = fdr.DataReader(ticker)  # 모든 데이터 수집

            if not ticker_data.empty and 'Close' in ticker_data.columns:
                # Close 가격만 추출하고 컬럼명을 티커명으로 변경
                close_data = ticker_data['Close'].to_frame(name=ticker)
                data_frames.append(close_data)
                successful_tickers.append(ticker)
                rprint(f"✅ {ticker} 데이터 수집 완료: {len(ticker_data)} 행")
            else:
                rprint(f"❌ {ticker} 데이터 없음 또는 Close 컬럼 누락")

        except Exception as e:
            rprint(f"❌ {ticker} 데이터 수집 실패: {str(e)}")
            continue

    if not data_frames:
        rprint("⚠️  수집된 데이터가 없습니다.")
        return pd.DataFrame()

    # 모든 데이터를 하나의 DataFrame으로 결합
    combined_data = pd.concat(data_frames, axis=1)

    # 성공한 티커들의 순서에 맞게 컬럼 재정렬
    if successful_tickers:
        combined_data = combined_data[successful_tickers]

    rprint("get_FinanceDataReader_data-------------------")
    rprint(combined_data.head())
    rprint(combined_data.tail())
    rprint("get_FinanceDataReader_data----dropna--------")
    data_dropa = combined_data.dropna()
    rprint(data_dropa.head())
    rprint(f"원본 데이터: {len(combined_data)} 행, dropna 후: {len(data_dropa)} 행")

    return combined_data


def get_yf_market_data():
    """
    Yahoo Finance를 통해 시장 지수 데이터를 가져옵니다.

    Returns:
        pandas.DataFrame: 시장 지수 데이터
    """
    try:
        # Fetch VIX data
        vix_data = yf.download('^VIX')['Close']
        vix_data.name = 'VIX'  # Series 이름 설정

        # Fetch S&P 500 data
        sp500_data = yf.download('^GSPC')['Close']
        sp500_data.name = 'S&P_500'
        # Fetch Dow Jones data
        dji_data = yf.download('^DJI')['Close']
        dji_data.name = 'Dow_Jones'
        # Fetch NASDAQ-100 data
        nasdaq_100_data = yf.download('^NDX')['Close']
        nasdaq_100_data.name = 'NASDAQ_100'

        # Fetch KOSPI data
        kospi_data = yf.download('^KS11')['Close']
        kospi_data.name = 'KOSPI'
        # Fetch USD/KRW exchange rate data
        usd_krw_data = yf.download('KRW=X')['Close']
        usd_krw_data.name = 'USD/KRW'

        # Combine all indicators into a single DataFrame
        market_data = pd.concat([vix_data, sp500_data, dji_data,
                                nasdaq_100_data, kospi_data, usd_krw_data], axis=1)

        # Combine all indicators into a single DataFrame
        rprint("get_yf_market_data-------------------")
        rprint(market_data.head())
        rprint(market_data.tail())
        merged_data_dropa = market_data.dropna()
        rprint("get_yf_market_data----dropna--------")
        rprint(merged_data_dropa.head())

        return market_data

    except Exception as e:
        if "SQLite driver not installed" in str(e):
            rprint(f"❌ SQLite 드라이버 에러 발생: {e}")
            rprint("💡 해결책:")
            rprint("   1. pip install --upgrade yfinance (최신 버전 설치)")
            rprint("   2. sudo apt-get install sqlite3 libsqlite3-dev (Linux)")
            rprint("   3. FinanceDataReader 사용을 고려해보세요")

            # 빈 DataFrame 반환
            return pd.DataFrame()
        else:
            rprint(f"❌ yfinance 데이터 수집 중 에러 발생: {e}")
            return pd.DataFrame()


def get_FinanceDataReader_market_data(start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    FinanceDataReader를 사용하여 시장 지수 데이터를 가져옵니다.

    Args:
        start_date: 시작 날짜 (YYYY-MM-DD 형식, 기본값: None - 모든 데이터)
        end_date: 종료 날짜 (YYYY-MM-DD 형식, 기본값: None - 모든 데이터)

    Returns:
        pandas.DataFrame: 시장 지수 데이터
    """
    # 시장 지수 티커 목록 정의
    market_tickers = {
        '^VIX': 'VIX',           # VIX 지수
        '^GSPC': 'S&P_500',      # S&P 500
        '^DJI': 'Dow_Jones',     # 다우존스
        '^NDX': 'NASDAQ_100',    # 나스닥 100
        '^KS11': 'KOSPI',        # 코스피
        'KRW=X': 'USD/KRW'       # 달러/원 환율
    }

    data_frames = []
    successful_tickers = []

    # 날짜 범위 설정에 따른 메시지 출력
    if start_date and end_date:
        rprint(f"FinanceDataReader 시장 지수 데이터 수집 시작: {start_date} ~ {end_date}")
    elif start_date:
        rprint(f"FinanceDataReader 시장 지수 데이터 수집 시작: {start_date} ~ 현재")
    elif end_date:
        rprint(f"FinanceDataReader 시장 지수 데이터 수집 시작: 전체 ~ {end_date}")
    else:
        rprint("FinanceDataReader 전체 시장 지수 데이터 수집 시작")

    for ticker, name in market_tickers.items():
        try:
            # FinanceDataReader를 사용하여 데이터 가져오기 (조건부 매개변수 전달)
            if start_date and end_date:
                ticker_data = fdr.DataReader(ticker, start=start_date, end=end_date)
            elif start_date:
                ticker_data = fdr.DataReader(ticker, start=start_date)
            elif end_date:
                ticker_data = fdr.DataReader(ticker, end=end_date)
            else:
                ticker_data = fdr.DataReader(ticker)  # 모든 데이터 수집

            if not ticker_data.empty and 'Close' in ticker_data.columns:
                # Close 가격만 추출하고 컬럼명을 지수명으로 변경
                close_data = ticker_data['Close'].to_frame(name=name)
                data_frames.append(close_data)
                successful_tickers.append(name)
                rprint(f"✅ {name} ({ticker}) 데이터 수집 완료: {len(ticker_data)} 행")
            else:
                rprint(f"❌ {name} ({ticker}) 데이터 없음 또는 Close 컬럼 누락")

        except Exception as e:
            rprint(f"❌ {name} ({ticker}) 데이터 수집 실패: {str(e)}")
            continue

    if not data_frames:
        rprint("⚠️  수집된 시장 지수 데이터가 없습니다.")
        return pd.DataFrame()

    # 모든 데이터를 하나의 DataFrame으로 결합
    combined_data = pd.concat(data_frames, axis=1)

    # 성공한 지수들의 순서에 맞게 컬럼 재정렬
    if successful_tickers:
        combined_data = combined_data[successful_tickers]

    rprint("get_FinanceDataReader_market_data-------------------")
    rprint(combined_data.head())
    rprint(combined_data.tail())
    rprint("get_FinanceDataReader_market_data----dropna--------")
    data_dropa = combined_data.dropna()
    rprint(data_dropa.head())
    rprint(f"원본 데이터: {len(combined_data)} 행, dropna 후: {len(data_dropa)} 행")

    return combined_data


def get_FRED_data_v2(series_ids=['PPIACO', 'CPIAUCSL']):
    """
    FRED API를 통해 거시경제 데이터를 가져옵니다.

    Args:
        series_ids (list): FRED 데이터 시리즈 ID 리스트

    Returns:
        pandas.DataFrame: 거시경제 지표 데이터
    """
    api_key = os.getenv("API_KEY_FRED")
    rprint(f"API 키: {api_key}")
    fred = Fred(api_key=api_key)

    data_frames = []
    for series_id in series_ids:
        fred_data = fred.get_series(series_id)
        fred_df = fred_data.to_frame(name=series_id)

        fred_df.index.name = 'Date'

        # Rename the column if it matches 'IR3TIB01KRM156N'
        if series_id == 'IR3TIB01KRM156N':
            fred_df.rename(columns={'IR3TIB01KRM156N': 'BOK_RATE'}, inplace=True)

        # rprint(f"get_FRED_data_v2 - {series_id} -------------------")
        # rprint(fred_df.head())
        # rprint(fred_df.tail())
        data_frames.append(fred_df)

    combined_df = pd.concat(data_frames, axis=1)
    rprint("get_FRED_data_v2 - Combined Data -------------------")
    rprint(combined_df.head())
    rprint(combined_df.tail())
    rprint("get_FRED_data----dropna--------")
    combined_data_dropa = combined_df.dropna()
    rprint(combined_data_dropa.head())

    return combined_df
