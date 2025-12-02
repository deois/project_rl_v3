"""
로그 및 데이터 스토어 관련 콜백 함수들
로그 표시, 스토어 동기화
"""

from typing import Any, Tuple, List, Dict
from dash import callback_context, html, Input, Output, State
from src.utils.logger import get_logger

logger = get_logger("log_callbacks")


def register_log_callbacks(app, dash_manager):
    """로그 및 스토어 관련 콜백 함수들을 등록"""

    @app.callback(
        [Output("log-container", "children"),
         Output("log-count", "children")],
        [Input("logs-interval", "n_intervals"),
         Input("clear-logs-btn", "n_clicks")],
        [State("logs-store", "data")]
    )
    def update_logs(n_intervals: int, clear_clicks: int, logs_data: List[str]) -> Tuple[List[html.P], str]:
        """로그 업데이트"""

        ctx = callback_context
        if ctx.triggered and ctx.triggered[0]["prop_id"] == "clear-logs-btn.n_clicks" and clear_clicks:
            dash_manager.logs = []
            return [html.P("[Dash 대시보드] 로그 지워짐...",
                           style={'margin': '0', 'color': '#00ff41', 'opacity': '0.8'})], "0"

        if not dash_manager.logs:
            return [html.P("[Dash 대시보드] 시스템 초기화 완료...",
                           style={'margin': '0', 'color': '#00ff41', 'opacity': '0.8'})], "1"

        log_elements = []
        recent_logs = dash_manager.logs[-80:]  # 최근 80개만 표시

        for i, log in enumerate(recent_logs):
            # 로그 타입에 따른 색상 구분
            if "🚀" in log or "✅" in log:
                color = "#00ff41"  # 성공 - 밝은 녹색
            elif "❌" in log or "⚠️" in log:
                color = "#ff6b6b"  # 오류/경고 - 빨간색
            elif "🛑" in log:
                color = "#ffc107"  # 중지 - 노란색
            elif "📊" in log or "📈" in log:
                color = "#17a2b8"  # 정보 - 파란색
            else:
                color = "#b8f2ff"  # 기본 - 연한 파란색

            log_elements.append(
                html.P(log,
                       style={
                           'margin': '3px 0',
                           'color': color,
                           'opacity': max(0.4, (i + 1) / len(recent_logs)),  # 페이드 효과
                           'font-size': '13px',
                           'line-height': '1.4'
                       })
            )

        return log_elements, str(len(dash_manager.logs))

    @app.callback(
        [Output("logs-store", "data"),
         Output("chart-data-store", "data"),
         Output("backtest-data-store", "data")],
        [Input("logs-interval", "n_intervals")]
    )
    def sync_stores(n_intervals: int) -> Tuple[List[str], Dict[str, List[Any]], Dict[str, Any]]:
        """스토어 동기화"""
        return (dash_manager.logs,
                dash_manager.chart_data,
                dash_manager.backtest_data)
