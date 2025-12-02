"""
모니터링 탭 레이아웃
시스템 상태 모니터링 및 성능 지표 컴포넌트
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
from src.dash_utils import CARD_STYLE, METRIC_CARD_STYLE


def create_monitoring_content() -> list:
    """모니터링 탭 콘텐츠 생성 - 시스템 상태 중심"""
    return [
        # 시스템 상태 요약
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="bi bi-display me-2"),
                            "시스템 상태 모니터링"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.P([
                            "실시간 시스템 성능 지표 및 리소스 사용량을 모니터링합니다. ",
                            "학습 및 백테스팅 작업의 안정적 실행을 위한 시스템 상태를 확인할 수 있습니다."
                        ], className="text-muted mb-3"),
                        dbc.Alert([
                            html.I(className="bi bi-info-circle me-2"),
                            "GPU 온도는 GPUtil 라이브러리가 설치된 경우에만 표시됩니다."
                        ], color="info", className="mb-0")
                    ])
                ], style=CARD_STYLE)
            ])
        ], className="mb-4"),

        # 실시간 상태 카드들
        *create_monitoring_metrics_cards(),

        # 실시간 차트들
        *create_monitoring_charts_section()
    ]


def create_monitoring_metrics_cards() -> list:
    """모니터링 메트릭 카드들 생성"""
    return [
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className="bi bi-server",
                               style={"font-size": "24px", "color": "#17a2b8"}),
                        html.H5(id="system-status", children="🟢 정상",
                                className="mt-2 mb-1", style={"font-weight": "600"}),
                        html.P("시스템 상태", className="text-muted mb-0")
                    ])
                ], style=METRIC_CARD_STYLE, color="info", outline=True)
            ], lg=3, md=6, sm=12),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className="bi bi-clock",
                               style={"font-size": "24px", "color": "#28a745"}),
                        html.H5(id="uptime", children="00:00:00",
                                className="mt-2 mb-1", style={"font-weight": "600"}),
                        html.P("운영 시간", className="text-muted mb-0")
                    ])
                ], style=METRIC_CARD_STYLE, color="success", outline=True)
            ], lg=3, md=6, sm=12),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className="bi bi-memory",
                               style={"font-size": "24px", "color": "#ffc107"}),
                        html.H5(id="memory-usage", children="0 MB",
                                className="mt-2 mb-1", style={"font-weight": "600"}),
                        html.P("메모리 사용량", className="text-muted mb-0")
                    ])
                ], style=METRIC_CARD_STYLE, color="warning", outline=True)
            ], lg=3, md=6, sm=12),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className="bi bi-thermometer-half",
                               style={"font-size": "24px", "color": "#dc3545"}),
                        html.H5(id="gpu-temp", children="N/A",
                                className="mt-2 mb-1", style={"font-weight": "600"}),
                        html.P("GPU 온도", className="text-muted mb-0")
                    ])
                ], style=METRIC_CARD_STYLE, color="danger", outline=True)
            ], lg=3, md=6, sm=12)
        ], className="mb-4")
    ]


def create_monitoring_charts_section() -> list:
    """모니터링 차트 섹션 생성"""
    return [
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="bi bi-graph-up me-2"),
                            "실시간 시스템 모니터링"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id="system-monitoring-chart",
                            config={
                                'displayModeBar': True,
                                'displaylogo': False
                            },
                            style={'height': '400px'}
                        )
                    ])
                ], style=CARD_STYLE)
            ], lg=12)
        ], className="mb-4")
    ]
