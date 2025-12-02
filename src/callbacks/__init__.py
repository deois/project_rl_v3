"""
Dash 콜백 함수들 - 모듈화된 구조
각 기능별로 분리된 콜백들을 통합 관리
"""

from .training_callbacks import register_training_callbacks
from .backtest_callbacks import register_backtest_callbacks
from .model_callbacks import register_model_callbacks
from .chart_callbacks import register_chart_callbacks
from .etf_callbacks import register_etf_callbacks
from .monitoring_callbacks import register_monitoring_callbacks
from .config_callbacks import register_config_callbacks
from .log_callbacks import register_log_callbacks
from .ai_platform_callbacks import register_ai_platform_callbacks


def register_all_callbacks(app, dash_manager):
    """모든 콜백 함수들을 등록"""

    # 각 모듈별 콜백 등록
    register_training_callbacks(app, dash_manager)
    register_backtest_callbacks(app, dash_manager)
    register_model_callbacks(app, dash_manager)
    register_chart_callbacks(app, dash_manager)
    register_etf_callbacks(app, dash_manager)
    register_monitoring_callbacks(app, dash_manager)
    register_config_callbacks(app, dash_manager)
    register_log_callbacks(app, dash_manager)
    register_ai_platform_callbacks(app, dash_manager)
