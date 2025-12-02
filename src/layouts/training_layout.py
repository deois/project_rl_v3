"""
학습 탭 레이아웃
DDPG 강화학습 학습 제어 및 모니터링 컴포넌트
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
from src.dash_utils import CARD_STYLE, METRIC_CARD_STYLE


def create_training_content() -> list:
    """학습 탭 콘텐츠 생성"""
    return [
        # 학습 모드 선택
        create_mode_selection(),

        # 학습 메트릭 카드들
        *create_training_metrics_cards(),

        # 학습 컨트롤 패널
        create_training_control_panel(),

        # 학습 차트 영역
        *create_training_charts_section(),

        # 로그 영역
        create_logs_section()
    ]


def create_mode_selection() -> dbc.Row:
    """학습 모드 선택 컴포넌트 생성"""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="bi bi-gear me-2"),
                        "학습 모드 선택"
                    ], className="mb-0")
                ]),
                dbc.CardBody([
                    dbc.RadioItems(
                        id="training-mode",
                        options=[
                            {"label": "🎮 시뮬레이션 모드 (빠른 테스트)", "value": "simulation"},
                            {"label": "🚀 실제 DDPG 학습 모드", "value": "real"}
                        ],
                        value="real",
                        inline=True,
                        style={"font-size": "16px"}
                    ),
                    html.Hr(),
                    html.Div(id="mode-description", className="text-muted")
                ])
            ], style=CARD_STYLE)
        ])
    ], className="mb-4")


def create_training_metrics_cards() -> list:
    """학습 메트릭 카드들 생성"""
    return [
        # 첫 번째 행 - 학습 상태
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className="bi bi-activity",
                               style={"font-size": "20px", "color": "#17a2b8"}),
                        html.H6(id="training-status-text", children="⚪ 대기 중",
                                className="mt-1 mb-0", style={"font-weight": "600", "font-size": "14px"}),
                        html.P("학습 상태", className="text-muted mb-0", style={"font-size": "11px"}),
                        html.Small(id="detailed-status", children="",
                                   className="text-warning d-block",
                                   style={"font-size": "9px", "line-height": "1.0",
                                          "overflow": "hidden", "text-overflow": "ellipsis",
                                          "white-space": "nowrap"})
                    ], style={"padding": "8px"})
                ], style=METRIC_CARD_STYLE, color="info", outline=True)
            ], lg=3, md=6, sm=12),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className="bi bi-graph-up",
                               style={"font-size": "20px", "color": "#28a745"}),
                        html.H6(id="current-episode", children="0",
                                className="mt-1 mb-0", style={"font-weight": "600", "font-size": "14px"}),
                        html.P("현재 에피소드", className="text-muted mb-0", style={"font-size": "11px"}),
                        html.Small(id="episode-progress", children="",
                                   className="text-info d-block",
                                   style={"font-size": "9px", "line-height": "1.0",
                                          "overflow": "hidden", "text-overflow": "ellipsis",
                                          "white-space": "nowrap"})
                    ], style={"padding": "8px"})
                ], style=METRIC_CARD_STYLE, color="success", outline=True)
            ], lg=3, md=6, sm=12),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className="bi bi-currency-dollar",
                               style={"font-size": "24px", "color": "#ffc107"}),
                        html.H6(id="current-reward", children="0.00",
                                className="mt-2 mb-1", style={"font-weight": "600", "font-size": "14px"}),
                        html.P("현재 보상", className="text-muted mb-0", style={"font-size": "12px"})
                    ])
                ], style=METRIC_CARD_STYLE, color="warning", outline=True)
            ], lg=3, md=6, sm=12),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className="bi bi-hourglass-split",
                               style={"font-size": "24px", "color": "#dc3545"}),
                        html.H6(id="progress-percent", children="0%",
                                className="mt-2 mb-1", style={"font-weight": "600", "font-size": "14px"}),
                        html.P("전체 진행률", className="text-muted mb-0", style={"font-size": "12px"}),
                        html.Div([
                            dbc.Progress(
                                id="episode-progress-bar",
                                value=0,
                                style={"height": "6px"},
                                color="info",
                                striped=True,
                                animated=True
                            )
                        ], className="mt-1")
                    ])
                ], style=METRIC_CARD_STYLE, color="danger", outline=True)
            ], lg=3, md=6, sm=12)
        ], className="mb-3"),

        # 두 번째 행 - 상세 메트릭
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className="bi bi-briefcase",
                               style={"font-size": "24px", "color": "#6f42c1"}),
                        html.H5(id="portfolio-value", children="$0.00",
                                className="mt-2 mb-1", style={"font-weight": "600"}),
                        html.P("포트폴리오", className="text-muted mb-0")
                    ])
                ], style=METRIC_CARD_STYLE, color="purple", outline=True)
            ], lg=3, md=6, sm=12),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className="bi bi-tag",
                               style={"font-size": "24px", "color": "#6c757d"}),
                        html.H5(id="task-id", children="-",
                                className="mt-2 mb-1", style={"font-weight": "600"}),
                        html.P("작업 ID", className="text-muted mb-0")
                    ])
                ], style=METRIC_CARD_STYLE, color="secondary", outline=True)
            ], lg=3, md=6, sm=12),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className="bi bi-cpu",
                               style={"font-size": "24px", "color": "#fd7e14"}),
                        html.H5(id="actor-loss", children="0.0000",
                                className="mt-2 mb-1", style={"font-weight": "600"}),
                        html.P("Actor Loss", className="text-muted mb-0")
                    ])
                ], style=METRIC_CARD_STYLE, color="orange", outline=True)
            ], lg=3, md=6, sm=12),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className="bi bi-speedometer2",
                               style={"font-size": "24px", "color": "#20c997"}),
                        html.H5(id="critic-loss", children="0.0000",
                                className="mt-2 mb-1", style={"font-weight": "600"}),
                        html.P("Critic Loss", className="text-muted mb-0")
                    ])
                ], style=METRIC_CARD_STYLE, color="teal", outline=True)
            ], lg=3, md=6, sm=12)
        ], className="mb-4")
    ]


def create_training_control_panel() -> dbc.Row:
    """학습 컨트롤 패널 생성"""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="bi bi-joystick me-2"),
                        "학습 컨트롤 패널"
                    ], className="mb-0")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                [html.I(className="bi bi-play-circle-fill me-2"), "학습 시작"],
                                id="start-training-btn",
                                color="success",
                                size="lg",
                                className="w-100",
                                style={"font-weight": "600"}
                            )
                        ], lg=3, md=6, sm=12, className="mb-2"),

                        dbc.Col([
                            dbc.Button(
                                [html.I(className="bi bi-stop-circle-fill me-2"), "학습 중지"],
                                id="stop-training-btn",
                                color="danger",
                                size="lg",
                                disabled=True,
                                className="w-100",
                                style={"font-weight": "600"}
                            )
                        ], lg=3, md=6, sm=12, className="mb-2"),

                        dbc.Col([
                            dbc.Button(
                                [html.I(className="bi bi-arrow-clockwise me-2"), "새로고침"],
                                id="refresh-training-btn",
                                color="secondary",
                                size="lg",
                                className="w-100",
                                style={"font-weight": "600"}
                            )
                        ], lg=3, md=6, sm=12, className="mb-2"),

                        dbc.Col([
                            dbc.Button(
                                [html.I(className="bi bi-gear me-2"), "학습 설정"],
                                id="training-config-btn",
                                color="outline-info",
                                size="lg",
                                className="w-100",
                                style={"font-weight": "600"}
                            )
                        ], lg=3, md=6, sm=12, className="mb-2"),

                        dbc.Col([
                            dbc.Button(
                                [html.I(className="bi bi-download me-2"), "모델 저장"],
                                id="save-model-btn",
                                color="outline-primary",
                                size="lg",
                                className="w-100",
                                style={"font-weight": "600"}
                            )
                        ], lg=3, md=6, sm=12, className="mb-2")
                    ])
                ])
            ], style=CARD_STYLE)
        ])
    ], className="mb-4")


def create_training_charts_section() -> list:
    """학습 차트 섹션 생성"""
    return [
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="bi bi-graph-up-arrow me-2"),
                            "실시간 성과 차트"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id="performance-chart",
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
                            },
                            style={'height': '450px'}
                        )
                    ])
                ], style=CARD_STYLE)
            ], lg=8),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="bi bi-cpu me-2"),
                            "학습 손실 차트"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id="loss-chart",
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
                            },
                            style={'height': '450px'}
                        )
                    ])
                ], style=CARD_STYLE)
            ], lg=4)
        ], className="mb-4")
    ]


def create_logs_section() -> dbc.Row:
    """로그 섹션 생성"""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col([
                            html.H5([
                                html.I(className="bi bi-terminal me-2"),
                                "실시간 로그",
                                dbc.Badge(id="log-count", children="0", color="light",
                                          className="ms-2")
                            ], className="mb-0")
                        ], md=8),
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button(
                                    [html.I(className="bi bi-trash3 me-1"), "지우기"],
                                    id="clear-logs-btn",
                                    color="outline-danger",
                                    size="sm"
                                ),
                                dbc.Button(
                                    [html.I(className="bi bi-download me-1"), "저장"],
                                    id="save-logs-btn",
                                    color="outline-primary",
                                    size="sm"
                                )
                            ])
                        ], md=4, className="text-end")
                    ])
                ]),
                dbc.CardBody([
                    html.Div(
                        id="log-container",
                        style={
                            'height': '350px',
                            'overflow-y': 'auto',
                            'background': 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
                            'color': '#00ff41',
                            'padding': '20px',
                            'font-family': "'Fira Code', 'Courier New', monospace",
                            'font-size': '14px',
                            'border-radius': '8px',
                            'border': '1px solid #333'
                        }
                    )
                ])
            ], style=CARD_STYLE)
        ])
    ], className="mb-4")
