"""
차트 관련 콜백 함수들
성과 차트, 손실 차트 업데이트
"""

from typing import Any, Dict, List
from dash import Input, Output

from src.dash_charts import (
    create_performance_chart, create_loss_chart
)
from src.utils.logger import get_logger

logger = get_logger("chart_callbacks")


def register_chart_callbacks(app, dash_manager):
    """차트 관련 콜백 함수들을 등록"""

    @app.callback(
        Output("performance-chart", "figure"),
        [Input("chart-interval", "n_intervals"),
         Input("chart-data-store", "data")]
    )
    def update_performance_chart(n_intervals: int, chart_data: Dict[str, List[Any]]):
        """성과 차트 업데이트"""
        return create_performance_chart(chart_data)

    @app.callback(
        Output("loss-chart", "figure"),
        [Input("chart-interval", "n_intervals"),
         Input("chart-data-store", "data")]
    )
    def update_loss_chart(n_intervals: int, chart_data: Dict[str, List[Any]]):
        """손실 차트 업데이트"""
        return create_loss_chart(chart_data)
