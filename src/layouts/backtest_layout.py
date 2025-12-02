"""
백테스팅 탭 레이아웃
포트폴리오 백테스팅 및 결과 분석 컴포넌트
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
from src.dash_utils import get_available_models, CARD_STYLE


def create_backtest_content() -> list:
    """백테스팅 탭 콘텐츠 생성"""
    return [
        # 백테스트 설정 카드
        create_backtest_settings_card(),
        # 백테스트 상태 영역
        *create_backtest_status_section(),
        # 백테스트 결과 차트 영역
        *create_backtest_results_section(),
        # 모델 정보 모달
        create_model_info_modal(),
        # 모델 삭제 확인 모달
        create_model_delete_modal(),
    ]


def create_backtest_settings_card() -> dbc.Row:
    """백테스트 설정 카드 생성"""
    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.H5(
                                        [
                                            html.I(className="bi bi-gear-fill me-2"),
                                            "백테스트 설정",
                                        ],
                                        className="mb-0",
                                    ),
                                    html.Small(
                                        [
                                            html.Strong("투자 방식: "),
                                            "초기자본 $10,000 → 매월 투자 $300 및 리벨런싱 → 4개 ETF 동적 배분",
                                        ],
                                        className="text-muted mt-1",
                                    ),
                                ]
                            ),
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            # 모델 선택
                                            dbc.Col(
                                                [
                                                    dbc.Label(
                                                        "모델 선택",
                                                        html_for="backtest-model-dropdown",
                                                    ),
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(
                                                                id="backtest-model-dropdown",
                                                                options=get_available_models(),
                                                                value="./model/rl_ddpg",
                                                                placeholder="사용할 모델을 선택하세요",
                                                                style={"color": "#000"},
                                                            ),
                                                            dbc.Button(
                                                                [
                                                                    html.I(
                                                                        className="bi bi-arrow-clockwise me-1"
                                                                    ),
                                                                    "새로고침",
                                                                ],
                                                                id="refresh-backtest-models-btn",
                                                                color="outline-secondary",
                                                                size="sm",
                                                                className="mt-2",
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                md=6,
                                                className="mb-3",
                                            ),
                                            # 모델 정보 보기 버튼
                                            dbc.Col(
                                                [
                                                    dbc.Label(
                                                        "선택된 모델 정보",
                                                        className="mb-2",
                                                    ),
                                                    dbc.Button(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.I(
                                                                        className="bi bi-info-circle me-2"
                                                                    ),
                                                                    html.Span(
                                                                        id="model-metadata-preview",
                                                                        children="모델을 선택하면 정보가 표시됩니다",
                                                                        style={
                                                                            "color": "#6c757d"
                                                                        },
                                                                    ),
                                                                ]
                                                            )
                                                        ],
                                                        id="model-info-btn",
                                                        color="light",
                                                        className="w-100 text-start",
                                                        style={
                                                            "height": "80px",
                                                            "border": "2px dashed #dee2e6",
                                                            "background": "#f8f9fa",
                                                        },
                                                    ),
                                                ],
                                                md=6,
                                                className="mb-3",
                                            ),
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            # 컨트롤 버튼들 (전체 너비로 확장)
                                            dbc.Col(
                                                [
                                                    dbc.Label(
                                                        "실행 제어", className="mb-2"
                                                    ),
                                                    html.Div(
                                                        [
                                                            dbc.Button(
                                                                [
                                                                    html.I(
                                                                        className="bi bi-play-fill me-2"
                                                                    ),
                                                                    "백테스트 시작",
                                                                ],
                                                                id="backtest-btn",
                                                                color="info",
                                                                size="lg",
                                                                className="w-100 mb-2",
                                                                style={
                                                                    "font-weight": "600"
                                                                },
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Button(
                                                                                [
                                                                                    html.I(
                                                                                        className="bi bi-gear me-2"
                                                                                    ),
                                                                                    "고급 설정",
                                                                                ],
                                                                                id="backtest-config-btn",
                                                                                color="outline-info",
                                                                                size="sm",
                                                                                className="w-100",
                                                                                style={
                                                                                    "font-weight": "600"
                                                                                },
                                                                            )
                                                                        ],
                                                                        md=4,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Button(
                                                                                [
                                                                                    html.I(
                                                                                        className="bi bi-bookmark-star me-2"
                                                                                    ),
                                                                                    "기본모델로 저장",
                                                                                ],
                                                                                id="save-as-default-model-btn",
                                                                                color="outline-success",
                                                                                size="sm",
                                                                                className="w-100",
                                                                                style={
                                                                                    "font-weight": "600"
                                                                                },
                                                                            )
                                                                        ],
                                                                        md=4,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Button(
                                                                                [
                                                                                    html.I(
                                                                                        className="bi bi-trash3 me-2"
                                                                                    ),
                                                                                    "모델 삭제",
                                                                                ],
                                                                                id="delete-model-btn",
                                                                                color="outline-danger",
                                                                                size="sm",
                                                                                className="w-100",
                                                                                style={
                                                                                    "font-weight": "600"
                                                                                },
                                                                            )
                                                                        ],
                                                                        md=4,
                                                                    ),
                                                                ]
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                md=12,
                                                className="mb-3",
                                            )
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        style=CARD_STYLE,
                    )
                ]
            )
        ],
        className="mb-4",
    )


def create_backtest_status_section() -> list:
    """백테스트 상태 섹션 생성"""
    return [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H5(
                                            [
                                                html.I(
                                                    className="bi bi-bar-chart-line me-2"
                                                ),
                                                "백테스트 상태",
                                            ],
                                            className="mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "상태:",
                                                            className="text-muted mb-1",
                                                        ),
                                                        html.H5(
                                                            id="backtest-status-text",
                                                            children="⚪ 대기 중",
                                                            className="mb-2",
                                                            style={"fontWeight": "600"},
                                                        ),
                                                    ],
                                                    md=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "진행률:",
                                                            className="text-muted mb-1",
                                                        ),
                                                        dbc.Progress(
                                                            id="backtest-progress-bar",
                                                            value=0,
                                                            style={
                                                                "height": "30px",
                                                                "fontSize": "14px",
                                                            },
                                                            striped=True,
                                                            animated=True,
                                                            color="info",
                                                            className="mb-2",
                                                        ),
                                                        html.Small(
                                                            id="backtest-progress-text",
                                                            children="0.0%",
                                                            className="text-muted fw-bold d-block text-center",
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "작업 ID:",
                                                            className="text-muted mb-1",
                                                        ),
                                                        html.P(
                                                            id="backtest-task-id",
                                                            children="-",
                                                            className="mb-0 font-monospace fw-bold",
                                                            style={
                                                                "fontSize": "0.9rem"
                                                            },
                                                        ),
                                                    ],
                                                    md=3,
                                                ),
                                            ]
                                        ),
                                        # 추가적인 상태 정보 (진행 중일 때만 표시)
                                        html.Div(
                                            [
                                                dbc.Alert(
                                                    [
                                                        html.I(
                                                            className="bi bi-info-circle me-2"
                                                        ),
                                                        "백테스트가 진행 중입니다. 완료까지 잠시 기다려주세요.",
                                                    ],
                                                    color="info",
                                                    className="mt-3 mb-0",
                                                    style={"fontSize": "0.9rem"},
                                                )
                                            ],
                                            id="backtest-running-alert",
                                            style={"display": "none"},
                                        ),
                                    ],
                                    style={"padding": "20px"},
                                ),
                            ],
                            style=CARD_STYLE,
                            className="border-info",
                        )
                    ]
                )
            ],
            className="mb-4",
            id="backtest-status-row",
        )
    ]


def create_backtest_results_section() -> list:
    """백테스트 결과 섹션 생성"""
    return [
        # 백테스트 결과 차트 (전체 가로 화면 사용)
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H5(
                                            [
                                                html.I(className="bi bi-graph-up me-2"),
                                                "백테스트 결과",
                                            ],
                                            className="mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="backtest-results-chart",
                                            config={
                                                "displayModeBar": True,
                                                "displaylogo": False,
                                                "modeBarButtonsToRemove": [
                                                    "pan2d",
                                                    "lasso2d",
                                                    "select2d",
                                                ],
                                            },
                                            style={"height": "500px"},
                                        )
                                    ]
                                ),
                            ],
                            style=CARD_STYLE,
                        )
                    ],
                    lg=12,
                )
            ],
            className="mb-4",
            id="backtest-results-row",
            style={"display": "none"},
        ),
        # 포트폴리오 분석 메트릭 (별도 행)
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H5(
                                            [
                                                html.I(
                                                    className="bi bi-pie-chart me-2"
                                                ),
                                                "포트폴리오 분석",
                                            ],
                                            className="mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            id="backtest-metrics-display",
                                            style={
                                                "height": "300px",
                                                "overflow-y": "auto",
                                            },
                                        )
                                    ]
                                ),
                            ],
                            style=CARD_STYLE,
                        )
                    ],
                    lg=12,
                )
            ],
            className="mb-4",
            id="backtest-metrics-row",
            style={"display": "none"},
        ),
        # 포트폴리오 자산 배분 차트 섹션
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H5(
                                            [
                                                html.I(
                                                    className="bi bi-pie-chart-fill me-2"
                                                ),
                                                "포트폴리오 자산 배분 추이",
                                            ],
                                            className="mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="portfolio-allocation-chart",
                                            config={
                                                "displayModeBar": True,
                                                "displaylogo": False,
                                                "modeBarButtonsToRemove": [
                                                    "pan2d",
                                                    "lasso2d",
                                                    "select2d",
                                                ],
                                            },
                                            style={"height": "400px"},
                                        )
                                    ]
                                ),
                            ],
                            style=CARD_STYLE,
                        )
                    ],
                    lg=12,
                )
            ],
            className="mb-4",
            id="portfolio-allocation-row",
            style={"display": "none"},
        ),
        # 상세 분석 차트들 - 1x2 그리드
        dbc.Row(
            [
                # 연환산 수익률
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H5(
                                            [
                                                html.I(className="bi bi-graph-up me-2"),
                                                "연환산 수익률",
                                            ],
                                            className="mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="annualized-returns-chart",
                                            config={
                                                "displayModeBar": True,
                                                "displaylogo": False,
                                                "modeBarButtonsToRemove": [
                                                    "pan2d",
                                                    "lasso2d",
                                                    "select2d",
                                                ],
                                            },
                                            style={"height": "350px"},
                                        )
                                    ]
                                ),
                            ],
                            style=CARD_STYLE,
                        )
                    ],
                    lg=12,
                    md=12,
                    sm=12,
                ),
                # 누적 수익률
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H5(
                                            [
                                                html.I(className="bi bi-percent me-2"),
                                                "누적 수익률",
                                            ],
                                            className="mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="cumulative-returns-chart",
                                            config={
                                                "displayModeBar": True,
                                                "displaylogo": False,
                                                "modeBarButtonsToRemove": [
                                                    "pan2d",
                                                    "lasso2d",
                                                    "select2d",
                                                ],
                                            },
                                            style={"height": "350px"},
                                        )
                                    ]
                                ),
                            ],
                            style=CARD_STYLE,
                        )
                    ],
                    lg=12,
                    md=12,
                    sm=12,
                ),
            ],
            className="mb-4",
            id="detailed-analysis-row-1",
            style={"display": "none"},
        ),
    ]


def create_model_info_modal() -> dbc.Modal:
    """모델 정보 상세 모달 생성"""
    return dbc.Modal(
        [
            dbc.ModalHeader(
                [
                    html.H4(
                        [
                            html.I(className="bi bi-info-circle-fill me-2"),
                            "모델 상세 정보",
                        ],
                        className="mb-0",
                    )
                ]
            ),
            dbc.ModalBody(
                [
                    html.Div(
                        id="model-info-modal-content",
                        children=[
                            html.P(
                                "모델을 선택하면 상세 정보가 표시됩니다.",
                                className="text-muted text-center",
                            )
                        ],
                    )
                ]
            ),
            dbc.ModalFooter(
                [dbc.Button("닫기", id="model-info-modal-close", color="secondary")]
            ),
        ],
        id="model-info-modal",
        size="lg",
        is_open=False,
    )


def create_backtest_config_modal() -> dbc.Modal:
    """백테스트 고급 설정 모달 생성"""
    return dbc.Modal(
        [
            dbc.ModalHeader(
                [
                    html.H4(
                        [html.I(className="bi bi-gear-fill me-2"), "백테스트 고급 설정"]
                    )
                ]
            ),
            dbc.ModalBody(
                [
                    # 기간 설정
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "시작 날짜 (선택사항)",
                                        html_for="start-date-picker",
                                    ),
                                    dbc.Input(
                                        id="start-date-picker",
                                        type="text",
                                        placeholder="YYYY-MM-DD",
                                    ),
                                ],
                                md=6,
                                className="mb-3",
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "종료 날짜 (선택사항)",
                                        html_for="end-date-picker",
                                    ),
                                    dbc.Input(
                                        id="end-date-picker",
                                        type="text",
                                        placeholder="YYYY-MM-DD",
                                    ),
                                ],
                                md=6,
                                className="mb-3",
                            ),
                        ]
                    ),
                    # 추가 설정들
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "초기 자본금 (USD)",
                                        html_for="initial-capital-input",
                                    ),
                                    dbc.Input(
                                        id="initial-capital-input",
                                        type="number",
                                        value=10000,
                                        min=1000,
                                        step=1000,
                                    ),
                                    html.Small(
                                        "권장: $10,000 (백테스트 신뢰도 확보)",
                                        className="text-muted",
                                    ),
                                ],
                                md=6,
                                className="mb-3",
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "거래 수수료 (%)",
                                        html_for="transaction-fee-input",
                                    ),
                                    dbc.Input(
                                        id="transaction-fee-input",
                                        type="number",
                                        value=0.1,
                                        min=0,
                                        max=5,
                                        step=0.01,
                                    ),
                                    html.Small(
                                        "일반적: 0.1% (ETF 평균 수수료)",
                                        className="text-muted",
                                    ),
                                ],
                                md=6,
                                className="mb-3",
                            ),
                        ]
                    ),
                    # ETF 정보 표시 (동적으로 업데이트)
                    html.Div(
                        id="backtest-etf-info-display",
                        children=[
                            dbc.Alert(
                                [
                                    html.H6("📊 투자 대상 ETF", className="mb-2"),
                                    html.P(
                                        "모델을 선택하면 해당 모델이 학습된 ETF 정보가 표시됩니다.",
                                        className="text-muted",
                                    ),
                                ],
                                color="light",
                                className="mb-3",
                            )
                        ],
                    ),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "취소",
                        id="backtest-config-cancel-btn",
                        color="secondary",
                        className="me-2",
                    ),
                    dbc.Button(
                        "설정 저장", id="backtest-config-save-btn", color="primary"
                    ),
                ]
            ),
        ],
        id="backtest-config-modal",
        is_open=False,
        size="lg",
    )


def create_model_delete_modal() -> dbc.Modal:
    """모델 삭제 확인 모달 생성"""
    return dbc.Modal(
        [
            dbc.ModalHeader(
                [
                    html.H4(
                        [
                            html.I(
                                className="bi bi-exclamation-triangle-fill me-2",
                                style={"color": "#dc3545"},
                            ),
                            "모델 삭제 확인",
                        ],
                        className="text-danger",
                    )
                ]
            ),
            dbc.ModalBody(
                [
                    dbc.Alert(
                        [
                            html.I(className="bi bi-shield-exclamation me-2"),
                            html.Strong("⚠️ 주의: 이 작업은 되돌릴 수 없습니다!"),
                            html.Hr(className="my-2"),
                            html.P(
                                [
                                    "선택된 모델 폴더와 모든 관련 파일들이 ",
                                    html.Strong("영구적으로 삭제"),
                                    "됩니다.",
                                ],
                                className="mb-2",
                            ),
                            html.Ul(
                                [
                                    html.Li("모델 가중치 파일 (.pth)"),
                                    html.Li("메타데이터 파일 (.json)"),
                                    html.Li("모든 체크포인트 파일"),
                                    html.Li("전체 모델 폴더"),
                                ],
                                className="mb-0",
                            ),
                        ],
                        color="danger",
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.H6("🗂️ 삭제될 모델:", className="text-muted mb-2"),
                            html.Div(
                                id="delete-model-path-display",
                                className="p-3",
                                style={
                                    "background": "#f8f9fa",
                                    "border-radius": "8px",
                                    "border": "1px solid #dee2e6",
                                },
                            ),
                            html.Hr(className="my-3"),
                            html.H6("📝 삭제 확인:", className="text-muted mb-2"),
                            html.P(
                                "계속하려면 아래 확인란을 체크하고 삭제 버튼을 클릭하세요.",
                                className="small text-muted mb-2",
                            ),
                            dbc.Checklist(
                                id="delete-confirmation-checkbox",
                                options=[
                                    {
                                        "label": "네, 이 모델을 영구적으로 삭제하겠습니다.",
                                        "value": "confirmed",
                                    }
                                ],
                                value=[],
                                style={"color": "#dc3545"},
                            ),
                        ]
                    ),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "취소",
                        id="delete-model-cancel-btn",
                        color="secondary",
                        className="me-2",
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-trash3-fill me-2"), "삭제 실행"],
                        id="delete-model-confirm-btn",
                        color="danger",
                        disabled=True,  # 기본적으로 비활성화
                    ),
                ]
            ),
        ],
        id="model-delete-modal",
        size="lg",
        is_open=False,
    )
