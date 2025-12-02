"""
AI기반 통합투자분석플랫폼 콜백
외부 플랫폼 연동 및 상태 관리 콜백 함수들
"""

import dash
from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import requests
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def register_ai_platform_callbacks(app, dash_manager):
    """AI 플랫폼 관련 콜백 함수들 등록"""

    @app.callback(
        Output("ai-platform-iframe", "src"),
        Input("refresh-platform-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def refresh_platform(n_clicks):
        """플랫폼 새로고침"""
        if n_clicks:
            logger.info("🔄 AI 플랫폼 iframe 새로고침")
            # 캐시 방지를 위해 timestamp 추가
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            return f"http://211.53.251.130:8080?refresh={timestamp}"

        raise PreventUpdate

    @app.callback(
        [Output("platform-status-alert", "children"),
         Output("platform-status-alert", "color")],
        Input("main-tabs", "active_tab"),
        prevent_initial_call=True
    )
    def check_platform_status(active_tab):
        """플랫폼 연결 상태 확인"""
        if active_tab != "ai-platform-tab":
            raise PreventUpdate

        try:
            # 플랫폼 연결 상태 확인
            response = requests.get("http://211.53.251.130:8080", timeout=5)
            if response.status_code == 200:
                logger.info("✅ AI 플랫폼 연결 성공")
                return [
                    [
                        dash.html.I(className="bi bi-check-circle me-2"),
                        "플랫폼이 정상적으로 연결되었습니다. 실시간 데이터 분석이 가능합니다."
                    ],
                    "success"
                ]
            else:
                logger.warning(f"⚠️ AI 플랫폼 응답 코드: {response.status_code}")
                return [
                    [
                        dash.html.I(className="bi bi-exclamation-triangle me-2"),
                        f"플랫폼 응답 오류 (코드: {response.status_code}). 서버 상태를 확인해주세요."
                    ],
                    "warning"
                ]

        except requests.ConnectionError:
            logger.error("❌ AI 플랫폼 연결 실패: 연결 거부")
            return [
                [
                    dash.html.I(className="bi bi-x-circle me-2"),
                    "플랫폼에 연결할 수 없습니다. localhost:8080 서버가 실행 중인지 확인해주세요."
                ],
                "danger"
            ]

        except requests.Timeout:
            logger.error("❌ AI 플랫폼 연결 실패: 시간 초과")
            return [
                [
                    dash.html.I(className="bi bi-clock me-2"),
                    "플랫폼 연결 시간이 초과되었습니다. 네트워크 상태를 확인해주세요."
                ],
                "warning"
            ]

        except Exception as e:
            logger.error(f"❌ AI 플랫폼 연결 실패: {str(e)}")
            return [
                [
                    dash.html.I(className="bi bi-bug me-2"),
                    f"플랫폼 연결 중 오류가 발생했습니다: {str(e)}"
                ],
                "danger"
            ]

    @app.callback(
        Output("connection-status", "children"),
        Input("main-tabs", "active_tab"),
        prevent_initial_call=True
    )
    def update_connection_status(active_tab):
        """연결 상태 실시간 업데이트"""
        if active_tab != "ai-platform-tab":
            raise PreventUpdate

        try:
            response = requests.head("http://211.53.251.130:8080", timeout=3)
            if response.status_code == 200:
                return [
                    dash.html.I(className="bi bi-circle-fill", style={"color": "#28a745"}),
                    dash.html.Span(" 연결됨", className="ms-2 text-success fw-bold")
                ]
            else:
                return [
                    dash.html.I(className="bi bi-circle-fill", style={"color": "#ffc107"}),
                    dash.html.Span(" 불안정", className="ms-2 text-warning fw-bold")
                ]

        except:
            return [
                dash.html.I(className="bi bi-circle-fill", style={"color": "#dc3545"}),
                dash.html.Span(" 연결 안됨", className="ms-2 text-danger fw-bold")
            ]

    @app.callback(
        Output("last-update-time", "children"),
        Input("main-tabs", "active_tab"),
        prevent_initial_call=True
    )
    def update_last_update_time(active_tab):
        """마지막 업데이트 시간 표시"""
        if active_tab != "ai-platform-tab":
            raise PreventUpdate

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return current_time

    logger.info("🤖 AI 플랫폼 콜백 함수들이 등록되었습니다.")
