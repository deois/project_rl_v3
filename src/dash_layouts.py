"""
Dash UI 레이아웃 모듈
메인 레이아웃 및 공통 컴포넌트 정의 - 탭 기반 구조
"""

from dash import dcc, html
import dash_bootstrap_components as dbc  # type: ignore[import]
from src.layouts import (
    create_about_content,
    create_training_content,
    create_backtest_content,
    create_monitoring_content,
)


def create_header() -> dbc.Row:
    """헤더 컴포넌트 생성"""
    return dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(
                        [
                            html.H1(
                                [
                                    html.I(
                                        className="bi bi-brain me-3",
                                        style={"color": "#667eea"},
                                    ),
                                    "ETF 포트폴리오 동적 자산배분 강화학습 콘솔",
                                ],
                                className="text-center mb-3 mt-4",
                                style={"font-weight": "700", "color": "#2c3e50"},
                            ),
                            dbc.Alert(
                                [
                                    html.I(className="bi bi-target me-2"),
                                    "AI 기반 ETF 포트폴리오 동적 자산배분 강화학습 콘솔: 하나의 파이프라인으로 학습·백테스트·모니터링",
                                    html.Br(),
                                    html.Small(
                                        "학습 · 백테스트 · 모니터링 전 과정을 하나의 강화학습 파이프라인으로 단순화",
                                        className="text-muted",
                                    ),
                                    html.Hr(className="my-2"),
                                    # DDPG 시스템
                                    html.Div(
                                        [
                                            html.I(
                                                className="bi bi-robot me-2",
                                                style={"color": "#667eea"},
                                            ),
                                            html.Strong(
                                                "DDPG 강화학습 시스템",
                                                className="text-primary me-3",
                                            ),
                                            "40여개 ETF 중 선택 → 포트폴리오 최적화 → 월별 리밸런싱",
                                        ],
                                        className="small mb-2",
                                    ),
                                    html.Hr(className="my-2"),
                                    html.Div(
                                        [
                                            html.Strong(
                                                "🎯 대상: ", className="text-primary"
                                            ),
                                            "장기 ETF 포트폴리오 투자자 | ",
                                            html.Strong(
                                                "🤖 AI 기술: ", className="text-success"
                                            ),
                                            "DDPG (Actor-Critic) | ",
                                            html.Strong(
                                                "📈 전략: ", className="text-info"
                                            ),
                                            "ETF 동적 자산배분 + 월별 리밸런싱",
                                        ],
                                        className="small text-muted",
                                    ),
                                ],
                                color="info",
                                className="text-center",
                                style={"border": "none"},
                            ),
                        ],
                        style={
                            "background": "white",
                            "border-radius": "15px",
                            "padding": "20px",
                            "box-shadow": "0 4px 20px rgba(0, 0, 0, 0.1)",
                        },
                    )
                ]
            )
        ],
        className="mb-4",
    )


def create_main_tabs() -> dbc.Tabs:
    """메인 탭 컴포넌트 생성"""
    return dbc.Tabs(
        [
            dbc.Tab(
                label="📋 프로젝트 설명",
                tab_id="about-tab",
                activeTabClassName="fw-bold",
                children=[
                    html.Div(create_about_content(), style={"padding": "20px 0"})
                ],
            ),
            dbc.Tab(
                label="🚀 강화학습",
                tab_id="training-tab",
                activeTabClassName="fw-bold",
                children=[
                    html.Div(create_training_content(), style={"padding": "20px 0"})
                ],
            ),
            dbc.Tab(
                label="📈 강화학습_백테스팅",
                tab_id="backtest-tab",
                activeTabClassName="fw-bold",
                children=[
                    html.Div(create_backtest_content(), style={"padding": "20px 0"})
                ],
            ),
            dbc.Tab(
                label="📊 모니터링",
                tab_id="monitoring-tab",
                activeTabClassName="fw-bold",
                children=[
                    html.Div(create_monitoring_content(), style={"padding": "20px 0"})
                ],
            ),
        ],
        id="main-tabs",
        active_tab="about-tab",
        className="mb-4",
    )


def create_training_config_modal() -> dbc.Modal:
    """학습 설정 모달 생성"""
    return dbc.Modal(
        [
            dbc.ModalHeader(
                [
                    html.H4(
                        [
                            html.I(className="bi bi-gear-fill me-2"),
                            "DDPG 학습 파라미터 설정",
                        ]
                    )
                ]
            ),
            dbc.ModalBody(
                [
                    # 프리셋 설정
                    html.H6("⚡ 빠른 프리셋", className="text-danger mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Button(
                                        "🚀 빠른 테스트",
                                        id="preset-fast-btn",
                                        color="primary",
                                        size="sm",
                                        className="me-2",
                                    ),
                                    dbc.Button(
                                        "🎯 균형잡힌",
                                        id="preset-balanced-btn",
                                        color="success",
                                        size="sm",
                                        className="me-2",
                                    ),
                                    dbc.Button(
                                        "💪 고성능",
                                        id="preset-high-performance-btn",
                                        color="warning",
                                        size="sm",
                                    ),
                                ],
                                md=12,
                                className="mb-3",
                            )
                        ]
                    ),
                    html.Hr(),
                    # 기본 학습 설정
                    html.H6("🎯 기본 학습 설정", className="text-primary mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "총 에피소드 수",
                                        html_for="training-episodes-input",
                                    ),
                                    dbc.Input(
                                        id="training-episodes-input",
                                        type="number",
                                        value=100,
                                        min=10,
                                        max=1000,
                                        step=10,
                                    ),
                                    html.Small(
                                        "권장: 100-500 (시뮬레이션은 50)",
                                        className="text-muted",
                                    ),
                                ],
                                md=6,
                                className="mb-3",
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "저장 주기",
                                        html_for="training-save-episodes-input",
                                    ),
                                    dbc.Input(
                                        id="training-save-episodes-input",
                                        type="number",
                                        value=10,
                                        min=1,
                                        max=50,
                                        step=1,
                                    ),
                                    html.Small(
                                        "매 N 에피소드마다 모델 저장",
                                        className="text-muted",
                                    ),
                                ],
                                md=6,
                                className="mb-3",
                            ),
                        ]
                    ),
                    # 신경망 구조 설정
                    html.Hr(),
                    html.H6("🧠 신경망 구조 설정", className="text-success mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "배치 크기",
                                        html_for="training-batch-size-input",
                                    ),
                                    dbc.Input(
                                        id="training-batch-size-input",
                                        type="number",
                                        value=128,
                                        min=32,
                                        max=512,
                                        step=32,
                                    ),
                                    html.Small(
                                        "메모리 사용량과 관련 (32, 64, 128, 256)",
                                        className="text-muted",
                                    ),
                                ],
                                md=6,
                                className="mb-3",
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "히든 레이어 차원",
                                        html_for="training-hidden-dim-input",
                                    ),
                                    dbc.Input(
                                        id="training-hidden-dim-input",
                                        type="number",
                                        value=256,
                                        min=64,
                                        max=1024,
                                        step=64,
                                    ),
                                    html.Small(
                                        "신경망 복잡도 결정 (64, 128, 256, 512)",
                                        className="text-muted",
                                    ),
                                ],
                                md=6,
                                className="mb-3",
                            ),
                        ]
                    ),
                    # 학습률 설정
                    html.Hr(),
                    html.H6("📈 학습률 설정", className="text-warning mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Actor 학습률",
                                        html_for="training-actor-lr-input",
                                    ),
                                    dbc.Input(
                                        id="training-actor-lr-input",
                                        type="number",
                                        value=0.0003,
                                        min=0.00001,
                                        max=0.01,
                                        step=0.00001,
                                    ),
                                    html.Small(
                                        "정책 네트워크 학습률 (권장: 0.0001-0.001)",
                                        className="text-muted",
                                    ),
                                ],
                                md=6,
                                className="mb-3",
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Critic 학습률",
                                        html_for="training-critic-lr-input",
                                    ),
                                    dbc.Input(
                                        id="training-critic-lr-input",
                                        type="number",
                                        value=0.0003,
                                        min=0.00001,
                                        max=0.01,
                                        step=0.00001,
                                    ),
                                    html.Small(
                                        "가치 네트워크 학습률 (권장: 0.0001-0.001)",
                                        className="text-muted",
                                    ),
                                ],
                                md=6,
                                className="mb-3",
                            ),
                        ]
                    ),
                    # Loss 함수 설정
                    html.Hr(),
                    html.H6("📉 Loss 함수 설정", className="text-success mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Critic Loss 함수",
                                        html_for="training-critic-loss-type-input",
                                    ),
                                    dcc.Dropdown(
                                        id="training-critic-loss-type-input",
                                        options=[
                                            {
                                                "label": "MSE Loss (기본값)",
                                                "value": "mse",
                                            },
                                            {
                                                "label": "Smooth L1 Loss",
                                                "value": "smooth_l1",
                                            },
                                        ],
                                        value="mse",
                                        clearable=False,
                                    ),
                                    html.Small(
                                        "MSE: 일반적인 상황에 적합, Smooth L1: 이상치에 robust",
                                        className="text-muted",
                                    ),
                                ],
                                md=12,
                                className="mb-3",
                            ),
                        ]
                    ),
                    # 데이터 설정
                    html.Hr(),
                    html.H6("📊 데이터 설정", className="text-info mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "윈도우 크기",
                                        html_for="training-window-size-input",
                                    ),
                                    dbc.Input(
                                        id="training-window-size-input",
                                        type="number",
                                        value=60,
                                        min=20,
                                        max=120,
                                        step=10,
                                    ),
                                    html.Small(
                                        "과거 N일 데이터 사용 (20-120일)",
                                        className="text-muted",
                                    ),
                                ],
                                md=6,
                                className="mb-3",
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "재시작 에피소드",
                                        html_for="training-resume-episodes-input",
                                    ),
                                    dbc.Input(
                                        id="training-resume-episodes-input",
                                        type="number",
                                        value=0,
                                        min=0,
                                        max=1000,
                                        step=1,
                                    ),
                                    html.Small(
                                        "학습 재개시 시작 에피소드 (0=처음부터)",
                                        className="text-muted",
                                    ),
                                ],
                                md=6,
                                className="mb-3",
                            ),
                        ]
                    ),
                    # ETF 선택 설정 (새로 추가)
                    html.Hr(),
                    html.H6("🏛️ ETF 선택 설정", className="text-purple mb-3"),
                    dbc.Alert(
                        [
                            html.Strong("📍 중요: "),
                            "학습에 사용할 ETF 4개를 선택하세요. 선택된 ETF 조합에 따라 모델의 성능이 달라집니다.",
                        ],
                        color="info",
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "ETF 선택 (4개 필수)",
                                        html_for="training-etf-selection",
                                    ),
                                    dcc.Dropdown(
                                        id="training-etf-selection",
                                        options=[],  # 콜백에서 동적으로 설정
                                        value=["SPY", "DGRO", "SCHD", "EWY"],  # 기본값
                                        multi=True,
                                        placeholder="ETF를 선택하세요 (최대 4개)",
                                        style={"color": "black"},
                                    ),
                                    html.Small(
                                        "카테고리별로 균형있게 선택하는 것이 좋습니다",
                                        className="text-muted",
                                    ),
                                ],
                                md=12,
                                className="mb-3",
                            )
                        ]
                    ),
                    # 선택된 ETF 정보 표시
                    dbc.Row(
                        [
                            dbc.Col(
                                [html.Div(id="selected-etf-info", className="mb-3")],
                                md=12,
                            )
                        ]
                    ),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "취소",
                        id="training-config-cancel-btn",
                        color="secondary",
                        className="me-2",
                    ),
                    dbc.Button(
                        "설정 적용", id="training-config-save-btn", color="primary"
                    ),
                ]
            ),
        ],
        id="training-config-modal",
        is_open=False,
        size="lg",
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


def create_hidden_components(dash_manager) -> list:
    """숨겨진 컴포넌트들 생성"""
    return [
        # 데이터 저장소들
        dcc.Store(id="training-state-store", data=dash_manager.training_status),
        dcc.Store(
            id="training-config-store",
            data={
                "episodes": 100,
                "episodes_save": 10,
                "episodes_resume": 0,
                "batch_size": 128,
                "hidden_dim": 256,
                "actor_lr": 0.0003,
                "critic_lr": 0.0003,
                "window_size": 60,
            },
        ),
        dcc.Store(id="backtest-state-store", data=dash_manager.backtest_status),
        dcc.Store(id="backtest-data-store", data=dash_manager.backtest_data),
        dcc.Store(id="backtest-config-store", data=dash_manager.backtest_config),
        dcc.Store(id="chart-data-store", data=dash_manager.chart_data),
        dcc.Store(id="logs-store", data=dash_manager.logs),
        # 인터벌 컴포넌트들
        dcc.Interval(
            id="status-interval",
            interval=200,  # 0.2초마다 상태 업데이트 (더 빠른 실시간 반응)
            n_intervals=0,
        ),
        dcc.Interval(
            id="chart-interval", interval=1500, n_intervals=0  # 1.5초마다 차트 업데이트
        ),
        dcc.Interval(
            id="logs-interval", interval=1000, n_intervals=0  # 1초마다 로그 업데이트
        ),
        dcc.Interval(
            id="backtest-interval",
            interval=1000,  # 1초마다 백테스트 상태 업데이트
            n_intervals=0,
        ),
        dcc.Interval(
            id="monitoring-interval",
            interval=2000,  # 2초마다 모니터링 업데이트
            n_intervals=0,
        ),
    ]
