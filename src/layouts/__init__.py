"""
Dash 레이아웃 모듈
탭별로 분리된 레이아웃 컴포넌트들
"""

from .about_layout import create_about_content
from .training_layout import create_training_content, create_training_metrics_cards, create_training_control_panel, create_training_charts_section
from .backtest_layout import create_backtest_content, create_backtest_settings_card, create_backtest_status_section, create_backtest_results_section
from .monitoring_layout import create_monitoring_content, create_monitoring_metrics_cards, create_monitoring_charts_section

__all__ = [
    "create_about_content",
    "create_training_content",
    "create_training_metrics_cards",
    "create_training_control_panel",
    "create_training_charts_section",
    "create_backtest_content",
    "create_backtest_settings_card",
    "create_backtest_status_section",
    "create_backtest_results_section",
    "create_monitoring_content",
    "create_monitoring_metrics_cards",
    "create_monitoring_charts_section"
]
