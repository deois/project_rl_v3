"""
ETF 관리 모듈
미국의 대표적인 ETF 리스트 관리 및 선택 기능
"""

from typing import List, Dict, Optional
import json
import os
from datetime import datetime

# 미국 대표 ETF 목록 (카테고리별 분류)
US_REPRESENTATIVE_ETFS = {
    "대형주": [
        {
            "symbol": "SPY",
            "name": "SPDR S&P 500 ETF",
            "description": "S&P 500 지수 추종",
            "category": "대형주",
        },
        {
            "symbol": "IVV",
            "name": "iShares Core S&P 500 ETF",
            "description": "S&P 500 지수 추종 (저비용)",
            "category": "대형주",
        },
        {
            "symbol": "VOO",
            "name": "Vanguard S&P 500 ETF",
            "description": "S&P 500 지수 추종 (초저비용)",
            "category": "대형주",
        },
        {
            "symbol": "VTI",
            "name": "Vanguard Total Stock Market ETF",
            "description": "미국 전체 주식시장",
            "category": "대형주",
        },
    ],
    "기술주": [
        {
            "symbol": "QQQ",
            "name": "Invesco QQQ Trust",
            "description": "나스닥 100 지수 추종",
            "category": "기술주",
        },
        {
            "symbol": "XLK",
            "name": "Technology Select Sector SPDR Fund",
            "description": "기술 섹터 ETF",
            "category": "기술주",
        },
        {
            "symbol": "VGT",
            "name": "Vanguard Information Technology ETF",
            "description": "정보기술 섹터",
            "category": "기술주",
        },
        {
            "symbol": "FTEC",
            "name": "Fidelity MSCI Information Technology ETF",
            "description": "IT 섹터 (저비용)",
            "category": "기술주",
        },
    ],
    "중소형주": [
        {
            "symbol": "IWM",
            "name": "iShares Russell 2000 ETF",
            "description": "러셀 2000 소형주",
            "category": "중소형주",
        },
        {
            "symbol": "VB",
            "name": "Vanguard Small-Cap ETF",
            "description": "미국 소형주",
            "category": "중소형주",
        },
        {
            "symbol": "VXF",
            "name": "Vanguard Extended Market ETF",
            "description": "S&P 500 제외 미국 주식",
            "category": "중소형주",
        },
        {
            "symbol": "IJH",
            "name": "iShares Core S&P Mid-Cap ETF",
            "description": "S&P 중형주",
            "category": "중소형주",
        },
    ],
    "배당주": [
        {
            "symbol": "SCHD",
            "name": "Schwab US Dividend Equity ETF",
            "description": "퀄리티 배당 성장주 ETF",
            "category": "배당주",
        },
        {
            "symbol": "DGRO",
            "name": "iShares Core Dividend Growth ETF",
            "description": "배당 성장주 집중 ETF",
            "category": "배당주",
        },
    ],
    "채권": [
        {
            "symbol": "AGG",
            "name": "iShares Core U.S. Aggregate Bond ETF",
            "description": "미국 총채권 지수",
            "category": "채권",
        },
        {
            "symbol": "BND",
            "name": "Vanguard Total Bond Market ETF",
            "description": "미국 전체 채권시장",
            "category": "채권",
        },
        {
            "symbol": "HYG",
            "name": "iShares iBoxx $ High Yield Corporate Bond ETF",
            "description": "하이일드 회사채",
            "category": "채권",
        },
        {
            "symbol": "LQD",
            "name": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
            "description": "투자적격 회사채",
            "category": "채권",
        },
        {
            "symbol": "TLT",
            "name": "iShares 20+ Year Treasury Bond ETF",
            "description": "장기 국채",
            "category": "채권",
        },
    ],
    "국제주식": [
        {
            "symbol": "VEA",
            "name": "Vanguard FTSE Developed Markets ETF",
            "description": "선진국 주식",
            "category": "국제주식",
        },
        {
            "symbol": "VWO",
            "name": "Vanguard FTSE Emerging Markets ETF",
            "description": "신흥국 주식",
            "category": "국제주식",
        },
        {
            "symbol": "EFA",
            "name": "iShares MSCI EAFE ETF",
            "description": "유럽, 아시아, 극동 선진국",
            "category": "국제주식",
        },
        {
            "symbol": "EWY",
            "name": "iShares MSCI South Korea ETF",
            "description": "한국 주식시장",
            "category": "국제주식",
        },
        {
            "symbol": "FXI",
            "name": "iShares China Large-Cap ETF",
            "description": "중국 대형주",
            "category": "국제주식",
        },
    ],
    "섹터별": [
        {
            "symbol": "XLE",
            "name": "Energy Select Sector SPDR Fund",
            "description": "에너지 섹터",
            "category": "섹터별",
        },
        {
            "symbol": "XLF",
            "name": "Financial Select Sector SPDR Fund",
            "description": "금융 섹터",
            "category": "섹터별",
        },
        {
            "symbol": "XLV",
            "name": "Health Care Select Sector SPDR Fund",
            "description": "헬스케어 섹터",
            "category": "섹터별",
        },
        {
            "symbol": "XLI",
            "name": "Industrial Select Sector SPDR Fund",
            "description": "산업 섹터",
            "category": "섹터별",
        },
        {
            "symbol": "XLY",
            "name": "Consumer Discretionary Select Sector SPDR Fund",
            "description": "소비재 섹터",
            "category": "섹터별",
        },
        {
            "symbol": "XLP",
            "name": "Consumer Staples Select Sector SPDR Fund",
            "description": "필수소비재 섹터",
            "category": "섹터별",
        },
    ],
    "기타": [
        {
            "symbol": "GLD",
            "name": "SPDR Gold Shares",
            "description": "금 ETF",
            "category": "기타",
        },
        {
            "symbol": "SLV",
            "name": "iShares Silver Trust",
            "description": "은 ETF",
            "category": "기타",
        },
        {
            "symbol": "VNQ",
            "name": "Vanguard Real Estate ETF",
            "description": "부동산 투자신탁",
            "category": "기타",
        },
        {
            "symbol": "ARKK",
            "name": "ARK Innovation ETF",
            "description": "혁신 기술 ETF",
            "category": "기타",
        },
    ],
}


class ETFManager:
    """ETF 선택 및 관리 클래스"""

    def __init__(self):
        self.selected_etfs: List[str] = []
        self.etf_info: Dict[str, Dict] = {}
        self._build_etf_lookup()

    def _build_etf_lookup(self):
        """ETF 조회용 딕셔너리 생성"""
        for category, etfs in US_REPRESENTATIVE_ETFS.items():
            for etf in etfs:
                self.etf_info[etf["symbol"]] = etf

    def get_all_etfs(self) -> Dict[str, List[Dict]]:
        """모든 ETF 목록 반환"""
        return US_REPRESENTATIVE_ETFS

    def get_etfs_by_category(self, category: str) -> List[Dict]:
        """카테고리별 ETF 목록 반환"""
        return US_REPRESENTATIVE_ETFS.get(category, [])

    def get_etf_info(self, symbol: str) -> Optional[Dict]:
        """특정 ETF 정보 반환"""
        return self.etf_info.get(symbol)

    def get_etf_options_for_dash(self) -> List[Dict]:
        """Dash 컴포넌트용 옵션 목록 생성"""
        options = []
        for category, etfs in US_REPRESENTATIVE_ETFS.items():
            # 카테고리 헤더 추가
            options.append(
                {
                    "label": f"--- {category} ---",
                    "value": f"category_{category}",
                    "disabled": True,
                }
            )
            # ETF 옵션 추가
            for etf in etfs:
                options.append(
                    {
                        "label": f"{etf['symbol']} - {etf['name']}",
                        "value": etf["symbol"],
                    }
                )
        return options

    def set_selected_etfs(self, symbols: List[str]) -> bool:
        """선택된 ETF 설정 (4개 제한)"""
        if len(symbols) > 4:
            return False

        # 모든 심볼이 유효한지 확인
        for symbol in symbols:
            if symbol not in self.etf_info:
                return False

        self.selected_etfs = symbols
        return True

    def get_selected_etfs(self) -> List[str]:
        """선택된 ETF 목록 반환"""
        return self.selected_etfs

    def get_selected_etf_info(self) -> List[Dict]:
        """선택된 ETF 정보 반환"""
        return [self.etf_info[symbol] for symbol in self.selected_etfs]

    def save_selection_to_file(self, filename: str, model_name: str = None) -> bool:
        """ETF 선택 정보를 파일로 저장"""
        try:
            selection_data = {
                "timestamp": datetime.now().isoformat(),
                "model_name": model_name,
                "selected_etfs": self.selected_etfs,
                "etf_details": self.get_selected_etf_info(),
            }

            # 선택 정보 디렉토리 생성
            selection_dir = "./data/etf_selections"
            os.makedirs(selection_dir, exist_ok=True)

            # 파일 저장
            file_path = os.path.join(selection_dir, f"{filename}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(selection_data, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            print(f"ETF 선택 정보 저장 실패: {e}")
            return False

    def load_selection_from_file(self, filename: str) -> bool:
        """파일에서 ETF 선택 정보 로드"""
        try:
            file_path = f"./data/etf_selections/{filename}.json"
            if not os.path.exists(file_path):
                return False

            with open(file_path, "r", encoding="utf-8") as f:
                selection_data = json.load(f)

            self.selected_etfs = selection_data.get("selected_etfs", [])
            return True
        except Exception as e:
            print(f"ETF 선택 정보 로드 실패: {e}")
            return False

    def get_default_etfs(self) -> List[str]:
        """기본 ETF 조합 반환 (균형잡힌 포트폴리오)"""
        return ["SPY", "DGRO", "SCHD", "EWY"]  # 대형주, 배당주 2종, 국제주식


# 전역 ETF 관리자 인스턴스
etf_manager = ETFManager()
