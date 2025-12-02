"""
AI기반 통합투자분석플랫폼 레이아웃
외부 투자 분석 플랫폼과의 통합 인터페이스
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_ai_platform_content():
    """AI기반 통합투자분석플랫폼 콘텐츠 생성"""

    return [
        # 📊 AI 플랫폼 헤더
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H4([
                        html.I(className="bi bi-robot me-3", style={"color": "#6f42c1"}),
                        "AI기반 통합투자분석플랫폼"
                    ], className="text-center mb-3", style={"color": "#2c3e50", "font-weight": "600"}),

                    dbc.Alert([
                        html.I(className="bi bi-info-circle me-2"),
                        "외부 AI 투자분석 시스템과 연동하여 포괄적인 투자 인사이트를 제공합니다.",
                        html.Br(),
                        html.Small("DDPG 강화학습 모델과 함께 다각적 분석을 통한 최적의 투자 전략 수립",
                                   className="text-muted")
                    ], color="info", className="text-center mb-3"),

                    # 🔗 플랫폼 정보
                    dbc.Row([
                        dbc.Col([
                            dbc.Badge([
                                html.I(className="bi bi-graph-up me-1"),
                                "실시간 데이터 분석"
                            ], color="success", className="me-2"),
                            dbc.Badge([
                                html.I(className="bi bi-cpu me-1"),
                                "AI 기반 예측"
                            ], color="warning")
                        ], className="text-center")
                    ], className="mb-3")
                ])
            ])
        ], className="mb-4", style={
            "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "color": "white",
            "border": "none",
            "border-radius": "15px",
            "box-shadow": "0 8px 32px rgba(102, 126, 234, 0.2)"
        }),

        # 🔍 플랫폼 상태 카드
        create_platform_status_card(),

        # 🖼️ AI 플랫폼 iframe
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="bi bi-window me-2"),
                    "통합투자분석 대시보드"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                # 플랫폼 상태 확인
                dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    "플랫폼 연결 상태를 확인하는 중..."
                ], id="platform-status-alert", color="warning", className="mb-3"),

                # iframe 컨테이너
                html.Div([
                    html.Iframe(
                        id="ai-platform-iframe",
                        src="http://211.53.251.130:8080",
                        style={
                            "width": "100%",
                            "height": "80vh",
                            "border": "1px solid #dee2e6",
                            "border-radius": "10px",
                            "box-shadow": "0 4px 20px rgba(0, 0, 0, 0.1)"
                        }
                    )
                ], style={"min-height": "80vh"}),

                # 🔄 새로고침 버튼
                html.Div([
                    dbc.Button([
                        html.I(className="bi bi-arrow-clockwise me-2"),
                        "플랫폼 새로고침"
                    ], id="refresh-platform-btn", color="primary", size="sm", className="me-2"),

                    dbc.Button([
                        html.I(className="bi bi-box-arrow-up-right me-2"),
                        "새 창에서 열기"
                    ], id="open-external-btn", color="outline-secondary", size="sm",
                        href="http://211.53.251.130:8080", target="_blank")
                ], className="text-center mt-3")
            ])
        ], style={
            "border-radius": "15px",
            "box-shadow": "0 4px 20px rgba(0, 0, 0, 0.1)"
        }),


    ]


def create_platform_status_card():
    """플랫폼 상태 카드 생성"""
    return dbc.Card([
        dbc.CardBody([
            html.H6("연결 상태", className="text-muted mb-3"),

            # 상태 표시
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="bi bi-circle-fill", style={"color": "#28a745"}),
                        html.Span(" 연결됨", className="ms-2")
                    ], id="connection-status")
                ], md=6),

                dbc.Col([
                    html.Div([
                        html.Small("마지막 업데이트: ", className="text-muted"),
                        html.Span(id="last-update-time", className="fw-bold")
                    ])
                ], md=6)
            ])
        ])
    ], className="mb-3")
