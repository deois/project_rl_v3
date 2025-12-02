"""
Training 모듈 통합 관리
학습 관리자, 백테스트 관리자, 모델 유틸리티를 통합 제공
"""

from .model_utils import (
    calculate_model_hash,
    validate_model_files,
    get_model_file_sizes
)

from .training_manager import DashRealTrainingManager
from .backtest_manager import DashBacktestManager

# 편의를 위한 alias
TrainingManager = DashRealTrainingManager
BacktestManager = DashBacktestManager

__all__ = [
    # 모델 유틸리티
    'calculate_model_hash',
    'validate_model_files',
    'get_model_file_sizes',

    # 관리자 클래스들
    'DashRealTrainingManager',
    'DashBacktestManager',

    # Alias
    'TrainingManager',
    'BacktestManager'
]
