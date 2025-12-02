"""
데이터 병합 모듈
다양한 소스의 금융 데이터를 병합하여 통합 데이터셋 생성
"""

import os
from typing import List, Optional

import pandas as pd
from rich import print as rprint

from src.data.loader import get_FRED_data_v2
from src.data.loader import get_yf_assets_data_v2
from src.data.loader import get_yf_market_data
from src.data.loader import get_FinanceDataReader_data
from src.data.loader import get_FinanceDataReader_market_data


def load_merged_data_v1(assets: List[str], filename: str, refresh: bool = False) -> pd.DataFrame:
    """
    여러 소스의 금융 데이터를 병합하여 통합 데이터셋을 생성합니다.

    Args:
        assets (List[str]): 투자 대상 ETF 목록
        filename (str): 저장할 파일명
        refresh (bool): 데이터 새로고침 여부

    Returns:
        pandas.DataFrame: 병합된 데이터프레임
    """
    # 선택된 ETF에 따른 고유한 파일명 생성
    etf_combination = "_".join(sorted(assets))
    csv_file_path = f'./data/{filename}_{etf_combination}.csv'

    # 데이터 수집이 필요한지 확인하는 변수
    need_to_collect_data = refresh or not os.path.exists(csv_file_path)

    if os.path.exists(csv_file_path) and not refresh:
        try:
            # 기존 CSV 파일 로드 시도
            merged_data = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)

            # 데이터가 비어있는지 확인
            if merged_data.empty or len(merged_data) == 0:
                rprint(f"⚠️ {csv_file_path} 파일이 비어있습니다. 새로운 데이터 수집을 진행합니다.")
                need_to_collect_data = True
            else:
                print(f"{csv_file_path} 기존 파일에서 병합 데이터 로드됨.")
                rprint(f"📂 로드된 데이터 형태: {merged_data.shape}")
                rprint(f"📊 ETF 조합: {', '.join(assets)}")

        except (pd.errors.EmptyDataError, pd.errors.ParserError, Exception) as e:
            rprint(f"⚠️ {csv_file_path} 파일 로드 중 오류 발생: {e}")
            rprint("새로운 데이터 수집을 진행합니다.")
            need_to_collect_data = True

    if need_to_collect_data:
        # 데이터 불러오기 및 전처리
        rprint(f"🔄 새로운 데이터 수집 시작 - ETF: {', '.join(assets)}")

        # Fetch FRED data
        market_fred_data = get_FRED_data_v2([
            'PPIACO', 'CPIAUCSL', 'PCEPI', 'UNRATE', 'PAYEMS', 'CIVPART',
            'FEDFUNDS', 'M2SL', 'GS10', 'CSUSHPISA', 'RSAFS', 'PSAVERT', 'TCU',
            'UMCSENT', 'IEABC', 'IR3TIB01KRM156N'
        ])

        # Fetch stock data
        # assets_data = get_yf_assets_data_v2(assets)
        assets_data = get_FinanceDataReader_data(assets)
        rprint(f"📈 ETF 데이터 수집 완료: {assets_data.shape}")

        # Fetch VIX data
        market_yf_data = get_FinanceDataReader_market_data()  # get_yf_market_data()
        rprint(f"📊 시장 데이터 수집 완료: {market_yf_data.shape}")

        # Ensure both DataFrames have timezone-naive DatetimeIndex
        if hasattr(assets_data.index, 'tz') and assets_data.index.tz is not None:
            assets_data.index = assets_data.index.tz_localize(None)
        if hasattr(market_yf_data.index, 'tz') and market_yf_data.index.tz is not None:
            market_yf_data.index = market_yf_data.index.tz_localize(None)
        if hasattr(market_fred_data.index, 'tz') and market_fred_data.index.tz is not None:
            market_fred_data.index = market_fred_data.index.tz_localize(None)

        # Concatenate stock data with VIX data
        merged_data = pd.concat(
            [assets_data, market_yf_data, market_fred_data], axis=1)

        rprint(f"🔗 병합 완료: {merged_data.shape}")

        # Create a dictionary to store the start date of each column
        start_dates = {}

        # Iterate through each column to find the first non-null value
        for column in merged_data.columns:
            first_valid_index = merged_data[column].first_valid_index()
            if first_valid_index is not None:
                start_dates[column] = first_valid_index

        # Sort the columns by their start dates in descending order
        sorted_start_dates = sorted(start_dates.items(),
                                    key=lambda x: x[1],
                                    reverse=True)

        # Print the columns and their start dates
        rprint("📅 컬럼별 시작 날짜 (최신순):")
        for column, start_date in sorted_start_dates:
            rprint(f"  {column}: {start_date}")

        # 결손값 처리
        merged_data = merged_data.ffill().dropna()
        rprint(f"🧹 결손값 처리 후: {merged_data.shape}")

        # data 디렉토리가 없으면 생성
        data_dir = os.path.dirname(csv_file_path)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Save merged_data to a CSV file
        merged_data.to_csv(csv_file_path, index=True)
        rprint(f"💾 {csv_file_path} 파일로 병합 데이터 저장 완료")

    return merged_data


def get_available_data_files() -> List[str]:
    """
    저장된 데이터 파일 목록을 반환합니다.

    Returns:
        List[str]: 사용 가능한 데이터 파일 목록
    """
    data_dir = './data'
    if not os.path.exists(data_dir):
        return []

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    return csv_files


def extract_etfs_from_filename(filename: str) -> Optional[List[str]]:
    """
    파일명에서 ETF 조합을 추출합니다.

    Args:
        filename (str): 데이터 파일명

    Returns:
        Optional[List[str]]: ETF 리스트 또는 None
    """
    if '_' not in filename:
        return None

    # 파일명에서 확장자 제거
    basename = filename.replace('.csv', '')

    # 마지막 '_' 이후가 ETF 조합
    parts = basename.split('_')
    if len(parts) < 2:
        return None

    # ETF 조합 부분 추출
    etf_combination = parts[-1]

    # 개별 ETF로 분리 (알파벳 순으로 정렬되어 있음을 가정)
    # 실제로는 더 정교한 파싱이 필요할 수 있음
    etfs = etf_combination.split('_') if '_' in etf_combination else [etf_combination]

    return etfs if len(etfs) <= 4 else None
