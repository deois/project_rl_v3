"""
Dash 인터페이스와 강화학습 모듈 통합 (리팩토링된 버전)
실제 DDPG 학습 로직을 Dash 콜백에서 실행

이 파일은 리팩토링되어 src/training/ 모듈로 분리되었습니다.
기존 호환성을 위해 임포트를 유지합니다.
"""

# 리팩토링된 모듈에서 import하여 기존 호환성 유지
from src.training.model_utils import calculate_model_hash
from src.training.training_manager import DashRealTrainingManager
from src.training.backtest_manager import DashBacktestManager

# 기존 코드와의 호환성을 위한 export
__all__ = [
    'calculate_model_hash',
    'DashRealTrainingManager',
    'DashBacktestManager'
]
