"""
백테스트 관련 콜백 함수들
백테스트 실행, 상태 관리, 결과 차트, 메트릭 표시
"""

import time
import uuid
from typing import Any, Tuple, List, Dict
from dash import callback_context, html, Input, Output, State
import dash_bootstrap_components as dbc

from src.dash_charts import (
    create_backtest_results_chart,
    create_portfolio_allocation_chart,
    create_annualized_returns_chart,
    create_cumulative_returns_chart,
)
from src.dash_utils import get_available_models, load_model_training_config
from src.utils.etf_manager import etf_manager
from src.utils.logger import get_logger

logger = get_logger("backtest_callbacks")


def register_backtest_callbacks(app, dash_manager):
    """백테스트 관련 콜백 함수들을 등록"""

    @app.callback(
        [
            Output("backtest-config-modal", "is_open"),
            Output("backtest-config-store", "data"),
        ],
        [
            Input("backtest-config-btn", "n_clicks"),
            Input("backtest-config-cancel-btn", "n_clicks"),
            Input("backtest-config-save-btn", "n_clicks"),
        ],
        [
            State("backtest-config-modal", "is_open"),
            State("backtest-model-dropdown", "value"),
            State("start-date-picker", "value"),
            State("end-date-picker", "value"),
        ],
    )
    def handle_backtest_config_modal(
        config_clicks: int,
        cancel_clicks: int,
        save_clicks: int,
        is_open: bool,
        model_path: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """백테스트 설정 모달 관리"""
        ctx = callback_context
        if not ctx.triggered:
            return is_open, dash_manager.backtest_config

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "backtest-config-btn":
            return True, dash_manager.backtest_config
        elif trigger_id in ["backtest-config-cancel-btn"]:
            return False, dash_manager.backtest_config
        elif trigger_id == "backtest-config-save-btn":
            # 모델의 ETF 정보 확인
            model_assets = []
            if model_path:
                training_config = load_model_training_config(model_path)
                if training_config and "assets" in training_config:
                    model_assets = training_config["assets"]
                    dash_manager.add_log(
                        f"📊 모델의 ETF 정보 로드: {', '.join(model_assets)}"
                    )
                else:
                    dash_manager.add_log(
                        "⚠️ 모델의 ETF 정보를 찾을 수 없어 기본값을 사용합니다."
                    )

            # ETF 설정 (모델 정보 우선, 없으면 기본값)
            selected_assets = (
                model_assets if model_assets else etf_manager.get_default_etfs()
            )

            new_config = {
                "model_path": model_path or "./model/rl_ddpg",
                "episode": 0,  # 기본값으로 0 설정
                "assets": selected_assets,
                "start_date": start_date,
                "end_date": end_date,
            }
            dash_manager.backtest_config.update(new_config)
            dash_manager.add_log(
                f"⚙️ 백테스트 설정이 저장되었습니다: {', '.join(selected_assets)}"
            )
            return False, dash_manager.backtest_config

        return is_open, dash_manager.backtest_config

    @app.callback(
        Output("backtest-model-dropdown", "options"),
        [
            Input("refresh-backtest-models-btn", "n_clicks"),
            Input("backtest-config-btn", "n_clicks"),
            Input("main-tabs", "active_tab"),
        ],
    )
    def refresh_model_options(
        refresh_clicks: int, config_clicks: int, active_tab: str
    ) -> List[Dict[str, str]]:
        """모델 목록 새로고침 (백테스팅 탭 활성화 시 자동 갱신)"""
        ctx = callback_context

        # 백테스팅 탭이 활성화되었을 때도 모델 리스트 갱신
        if active_tab == "backtest-tab":
            dash_manager.add_log("🔄 백테스팅 탭 활성화 - 모델 목록 자동 갱신")

        models = get_available_models()

        # # 사용 가능한 모델 수 로깅
        # if models and len(models) > 0 and models[0]["value"]:  # 빈 값이 아닌 실제 모델이 있는 경우
        #     dash_manager.add_log(f"📊 백테스팅 가능한 모델 {len(models)}개 발견")
        # else:
        #     dash_manager.add_log(
        #         "⚠️ 백테스팅 가능한 모델이 없습니다. checkpoint_last.pth와 metadata_last.json 파일을 확인하세요.")

        return models

    # 백테스트 설정 동기화 콜백
    @app.callback(
        Output("backtest-model-dropdown", "value"),
        [Input("main-tabs", "active_tab")],
        [State("backtest-config-store", "data")],
    )
    def sync_backtest_settings(active_tab: str, backtest_config: Dict[str, Any]) -> str:
        """백테스팅 탭 활성화 시 설정 동기화"""
        if active_tab == "backtest-tab":
            return backtest_config.get("model_path", "./model/rl_ddpg")
        # 기본값 반환
        return "./model/rl_ddpg"

    @app.callback(
        Output("backtest-config-store", "data", allow_duplicate=True),
        [Input("backtest-model-dropdown", "value")],
        [State("backtest-config-store", "data")],
        prevent_initial_call=True,
    )
    def update_backtest_config_from_tab(
        model_path: str, current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """탭에서 백테스트 설정 업데이트 (모델의 ETF 정보 자동 로드)"""
        updated_config = current_config.copy()

        # 모델의 ETF 정보 확인
        model_assets = []
        if model_path:
            training_config = load_model_training_config(model_path)
            if training_config and "assets" in training_config:
                model_assets = training_config["assets"]
                dash_manager.add_log(
                    f"🔄 모델 변경 - ETF 자동 업데이트: {', '.join(model_assets)}"
                )

        # ETF 설정 (모델 정보 우선, 없으면 기본값)
        selected_assets = (
            model_assets if model_assets else etf_manager.get_default_etfs()
        )

        updated_config.update(
            {
                "model_path": model_path or "./model/rl_ddpg",
                "episode": 0,  # 기본값으로 0 설정
                "assets": selected_assets,
            }
        )
        return updated_config

    @app.callback(
        [
            Output("backtest-state-store", "data"),
            Output("backtest-status-row", "style"),
            Output("backtest-results-row", "style"),
            Output("backtest-metrics-row", "style"),
            Output("portfolio-allocation-row", "style"),
            Output("detailed-analysis-row-1", "style"),
        ],
        [Input("backtest-btn", "n_clicks"), Input("backtest-interval", "n_intervals")],
        [
            State("backtest-state-store", "data"),
            State("backtest-config-store", "data"),
            State("training-state-store", "data"),
        ],
    )
    def handle_backtest_execution(
        backtest_clicks: int,
        interval_n: int,
        backtest_state: Dict[str, Any],
        backtest_config: Dict[str, Any],
        training_state: Dict[str, Any],
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, str],
        Dict[str, str],
        Dict[str, str],
        Dict[str, str],
        Dict[str, str],
    ]:
        """백테스트 실행 및 상태 관리"""
        ctx = callback_context
        if not ctx.triggered:
            return (
                backtest_state,
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # 백테스트 시작
        if trigger_id == "backtest-btn" and backtest_clicks:
            # 상태 확인 로그 추가
            dash_manager.add_log(
                f"🔍 백테스트 시작 요청 - 학습 중: {training_state['is_training']}, 백테스트 실행 중: {backtest_state['is_running']}"
            )

            if not training_state["is_training"] and not backtest_state["is_running"]:
                task_id = str(uuid.uuid4())[:8]

                # 백테스트 상태 업데이트 (완료 상태 초기화)
                dash_manager.update_backtest_status(
                    is_running=True,
                    task_id=task_id,
                    start_time=time.time(),
                    progress=0,
                    current_step=0,
                    total_steps=1000,  # 예상 스텝 수
                    error=None,
                    is_completed=False,  # 새 백테스트 시작 시 완료 상태 초기화
                )

                # 백테스트 시작
                dash_manager.add_log(f"📈 백테스트 시작됨 (ID: {task_id})")
                dash_manager.add_log(
                    f"🎯 선택된 모델: {backtest_config.get('model_path', 'N/A')}"
                )
                dash_manager.add_log(f"📊 설정: {backtest_config}")
                dash_manager.reset_backtest_data()

                # 백테스트 관리자로 실행
                success = dash_manager.backtest_manager.start_backtest(backtest_config)
                if not success:
                    dash_manager.update_backtest_status(
                        is_running=False, error="백테스트 시작 실패"
                    )
                    dash_manager.add_log("❌ 백테스트 시작 실패")

                return (
                    dash_manager.backtest_status,
                    {"display": "block"},
                    {"display": "none"},
                    {"display": "none"},
                    {"display": "none"},
                    {"display": "none"},
                )
            else:
                # 백테스트 시작할 수 없는 이유 로깅
                if training_state["is_training"]:
                    dash_manager.add_log(
                        "⚠️ 학습이 진행 중이므로 백테스트를 시작할 수 없습니다. 학습을 먼저 종료해주세요."
                    )
                elif backtest_state["is_running"]:
                    dash_manager.add_log("⚠️ 이미 백테스트가 실행 중입니다.")

                return (
                    backtest_state,
                    {"display": "block"},
                    {"display": "none"},
                    {"display": "none"},
                    {"display": "none"},
                    {"display": "none"},
                )

        # 백테스트 상태 업데이트 (주기적)
        elif trigger_id == "backtest-interval":
            # 실제 dash_manager의 백테스트 상태를 가져옴
            current_status = dash_manager.backtest_status.copy()

            # 현재 백테스트 매니저의 상태를 확인
            if current_status.get("is_running", False):
                # 백테스트 완료 확인
                if not dash_manager.backtest_manager.is_running:
                    dash_manager.update_backtest_status(
                        is_running=False,
                        progress=100,
                        is_completed=True,  # 완료 상태 플래그 추가
                    )
                    dash_manager.add_log("✅ 백테스트가 완료되었습니다")

                    # 업데이트된 상태를 다시 가져옴
                    current_status = dash_manager.backtest_status.copy()

                    # 결과 영역 표시
                    results_style = (
                        {"display": "block"}
                        if dash_manager.backtest_data["portfolio_values"]
                        else {"display": "none"}
                    )
                    metrics_style = (
                        {"display": "block"}
                        if dash_manager.backtest_data["portfolio_values"]
                        else {"display": "none"}
                    )
                    allocation_style = (
                        {"display": "block"}
                        if dash_manager.backtest_data["portfolio_values"]
                        else {"display": "none"}
                    )
                    detailed_style = (
                        {"display": "block"}
                        if dash_manager.backtest_data["portfolio_values"]
                        else {"display": "none"}
                    )
                    return (
                        current_status,
                        {"display": "block"},
                        results_style,
                        metrics_style,
                        allocation_style,
                        detailed_style,
                    )
                else:
                    # 진행 중인 백테스트의 현재 상태 반환
                    return (
                        current_status,
                        {"display": "block"},
                        {"display": "none"},
                        {"display": "none"},
                        {"display": "none"},
                        {"display": "none"},
                    )
            else:
                # 백테스트가 실행 중이 아닌 경우 - 완료된 백테스트 결과 확인
                if (
                    current_status.get("is_completed", False)
                    and dash_manager.backtest_data["portfolio_values"]
                ):
                    # 완료된 백테스트 결과 표시
                    results_style = {"display": "block"}
                    metrics_style = {"display": "block"}
                    allocation_style = {"display": "block"}
                    detailed_style = {"display": "block"}
                    return (
                        current_status,
                        {"display": "block"},
                        results_style,
                        metrics_style,
                        allocation_style,
                        detailed_style,
                    )
                else:
                    # 백테스트가 완료되지 않았거나 결과가 없는 경우
                    return (
                        current_status,
                        {"display": "block"},
                        {"display": "none"},
                        {"display": "none"},
                        {"display": "none"},
                        {"display": "none"},
                    )

        # 기본 케이스: 현재 상태를 반환 - 상태 섹션은 항상 표시
        current_status = dash_manager.backtest_status.copy()
        status_style = {"display": "block"}

        # 완료된 백테스트가 있는지 확인하여 결과 표시 결정
        has_results = dash_manager.backtest_data["portfolio_values"]
        is_completed = current_status.get("is_completed", False)

        if is_completed and has_results:
            # 완료된 백테스트 결과가 있으면 모든 결과 섹션 표시
            results_style = {"display": "block"}
            metrics_style = {"display": "block"}
            allocation_style = {"display": "block"}
            detailed_style = {"display": "block"}
        else:
            # 완료되지 않았거나 결과가 없으면 숨김
            results_style = {"display": "none"}
            metrics_style = {"display": "none"}
            allocation_style = {"display": "none"}
            detailed_style = {"display": "none"}

        return (
            current_status,
            status_style,
            results_style,
            metrics_style,
            allocation_style,
            detailed_style,
        )

    @app.callback(
        [
            Output("backtest-status-text", "children"),
            Output("backtest-progress-bar", "value"),
            Output("backtest-progress-text", "children"),
            Output("backtest-task-id", "children"),
            Output("backtest-running-alert", "style"),
            Output("backtest-progress-bar", "color"),
        ],
        [Input("backtest-state-store", "data")],
    )
    def update_backtest_status_display(
        backtest_state: Dict[str, Any],
    ) -> Tuple[str, float, str, str, Dict[str, str], str]:
        """백테스트 상태 표시 업데이트"""

        # 디버깅 로그
        logger.debug(f"백테스트 상태 업데이트: {backtest_state}")

        # 기본값 설정
        is_running = backtest_state.get("is_running", False)
        progress = backtest_state.get("progress", 0)
        current_step = backtest_state.get("current_step", 0)
        total_steps = backtest_state.get("total_steps", 1)

        # 기본 색상과 스타일
        bar_color = "secondary"
        alert_style = {"display": "none"}

        if is_running:
            status = backtest_state.get("status", "진행 중")

            # 진행률에 따른 색상 변경
            if progress < 25:
                bar_color = "info"
            elif progress < 50:
                bar_color = "primary"
            elif progress < 75:
                bar_color = "warning"
            else:
                bar_color = "success"

            # 진행 중 알림 표시
            alert_style = {"display": "block"}

            if progress < 100:
                status_text = f"🔄 {status}"
                if current_step > 0 and total_steps > 0:
                    progress_text = (
                        f"{progress:.1f}% ({current_step:,}/{total_steps:,})"
                    )
                else:
                    progress_text = f"{progress:.1f}%"
            else:
                status_text = "⏳ 완료 중"
                progress_text = "100.0%"
                bar_color = "success"

        elif backtest_state.get("error"):
            status_text = f"❌ 오류: {backtest_state['error']}"
            progress = 0
            progress_text = "0.0%"
            bar_color = "danger"
            alert_style = {"display": "none"}
        elif backtest_state.get("is_completed", False):
            status_text = "✅ 완료"
            progress = 100
            progress_text = "100.0%"
            bar_color = "success"
            alert_style = {"display": "none"}
        else:
            status_text = "⚪ 대기 중"
            progress = 0
            progress_text = "0.0%"
            bar_color = "secondary"
            alert_style = {"display": "none"}

        task_id = backtest_state.get("task_id", "-")

        return status_text, progress, progress_text, task_id, alert_style, bar_color

    # 백테스트 결과 차트들
    @app.callback(
        Output("backtest-results-chart", "figure"),
        [Input("backtest-data-store", "data")],
    )
    def update_backtest_results_chart(backtest_data: Dict[str, Any]):
        """백테스트 결과 차트 업데이트"""
        return create_backtest_results_chart(backtest_data)

    @app.callback(
        Output("portfolio-allocation-chart", "figure"),
        [Input("backtest-data-store", "data")],
    )
    def update_portfolio_allocation_chart(backtest_data: Dict[str, Any]):
        """포트폴리오 자산 배분 차트 업데이트"""
        return create_portfolio_allocation_chart(backtest_data)

    @app.callback(
        Output("annualized-returns-chart", "figure"),
        [Input("backtest-data-store", "data")],
    )
    def update_annualized_returns_chart(backtest_data: Dict[str, Any]):
        """연환산 수익률 차트 업데이트"""
        return create_annualized_returns_chart(backtest_data)

    @app.callback(
        Output("cumulative-returns-chart", "figure"),
        [Input("backtest-data-store", "data")],
    )
    def update_cumulative_returns_chart(backtest_data: Dict[str, Any]):
        """누적 수익률 차트 업데이트"""
        return create_cumulative_returns_chart(backtest_data)

    @app.callback(
        Output("backtest-metrics-display", "children"),
        [Input("backtest-data-store", "data")],
    )
    def update_backtest_metrics_display(backtest_data: Dict[str, Any]):
        """백테스트 메트릭 표시 업데이트 - 강화학습 vs 균등투자 비교"""

        if not backtest_data.get("metrics"):
            return html.Div(
                [
                    html.H6("📊 백테스트 메트릭", className="text-muted mb-3"),
                    html.P("백테스트 결과가 없습니다.", className="text-muted"),
                ]
            )

        metrics = backtest_data["metrics"]

        return html.Div(
            [
                html.H6("📊 성과 비교 메트릭", className="mb-3"),
                # 강화학습 vs 균등투자 전략 메트릭 - 가로 배치
                dbc.Row(
                    [
                        # 강화학습 전략 메트릭
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H6(
                                                    "🤖 강화학습 전략",
                                                    className="text-primary mb-2",
                                                ),
                                                html.P(
                                                    f"최종 가치: ${metrics.get('rl_final_portfolio_value', 0):,.2f}",
                                                    className="mb-1",
                                                ),
                                                html.P(
                                                    f"총 수익률: {metrics.get('rl_total_return', 0):.2f}%",
                                                    className="mb-1",
                                                ),
                                                html.P(
                                                    f"연환산 수익률: {metrics.get('rl_annualized_return', 0):.2f}%",
                                                    className="mb-0",
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=6,
                        ),
                        # 균등투자 전략 메트릭
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H6(
                                                    "⚖️ 균등투자 전략",
                                                    className="text-success mb-2",
                                                ),
                                                html.P(
                                                    f"최종 가치: ${metrics.get('equal_final_portfolio_value', 0):,.2f}",
                                                    className="mb-1",
                                                ),
                                                html.P(
                                                    f"총 수익률: {metrics.get('equal_total_return', 0):.2f}%",
                                                    className="mb-1",
                                                ),
                                                html.P(
                                                    f"연환산 수익률: {metrics.get('equal_annualized_return', 0):.2f}%",
                                                    className="mb-0",
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-3",
                ),
                # 성능 비교 및 공통 정보 - 가로 배치
                dbc.Row(
                    [
                        # 성능 비교
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H6(
                                                    "🏆 성능 비교",
                                                    className="text-warning mb-2",
                                                ),
                                                html.P(
                                                    f"가치 차이: ${metrics.get('value_difference', 0):,.2f}",
                                                    className="mb-1",
                                                ),
                                                html.P(
                                                    f"수익률 차이: {metrics.get('return_difference', 0):.2f}%p",
                                                    className="mb-1",
                                                ),
                                                html.P(
                                                    f"연환산 수익률 차이: {metrics.get('annualized_return_difference', 0):.2f}%p",
                                                    className="mb-1",
                                                ),
                                                html.P(
                                                    [
                                                        html.Strong(
                                                            "우수 전략: ",
                                                            className="me-1",
                                                        ),
                                                        html.Span(
                                                            (
                                                                "강화학습"
                                                                if metrics.get(
                                                                    "value_difference",
                                                                    0,
                                                                )
                                                                > 0
                                                                else "균등투자"
                                                            ),
                                                            className=(
                                                                "text-primary"
                                                                if metrics.get(
                                                                    "value_difference",
                                                                    0,
                                                                )
                                                                > 0
                                                                else "text-success"
                                                            ),
                                                        ),
                                                    ],
                                                    className="mb-0",
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=6,
                        ),
                        # 공통 정보
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H6(
                                                    "📋 공통 정보",
                                                    className="text-info mb-2",
                                                ),
                                                html.P(
                                                    f"총 투자금: ${metrics.get('total_invested', 0):,.2f}",
                                                    className="mb-1",
                                                ),
                                                html.P(
                                                    f"평가 기간: {metrics.get('evaluation_days', 0)}일",
                                                    className="mb-1",
                                                ),
                                                html.P(
                                                    f"총 스텝: {metrics.get('total_steps', 0):,}",
                                                    className="mb-0",
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=6,
                        ),
                    ]
                ),
            ]
        )

    @app.callback(
        Output("backtest-etf-info-display", "children"),
        [Input("backtest-model-dropdown", "value")],
    )
    def update_backtest_etf_info(model_path: str):
        """선택된 모델의 ETF 정보를 백테스트 모달에 표시"""
        if not model_path:
            return [
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
            ]

        # 모델의 ETF 정보 로드
        model_assets = []
        training_config = load_model_training_config(model_path)
        if training_config and "assets" in training_config:
            model_assets = training_config["assets"]

        if not model_assets:
            return [
                dbc.Alert(
                    [
                        html.H6("📊 투자 대상 ETF", className="mb-2"),
                        html.P(
                            "⚠️ 모델의 ETF 정보를 찾을 수 없습니다.",
                            className="text-warning",
                        ),
                    ],
                    color="warning",
                    className="mb-3",
                )
            ]

        # ETF 세부 정보 가져오기
        etf_details = []
        for asset in model_assets:
            etf_info = etf_manager.get_etf_info(asset)
            if etf_info:
                etf_details.append(
                    dbc.Col(
                        [
                            html.Strong(asset),
                            f" - {etf_info['name']}",
                            html.Br(),
                            html.Small(etf_info["description"], className="text-muted"),
                        ],
                        md=3 if len(model_assets) == 4 else 6,
                        className="mb-2",
                    )
                )
            else:
                etf_details.append(
                    dbc.Col(
                        [
                            html.Strong(asset),
                            " - ETF 정보 없음",
                            html.Br(),
                            html.Small(
                                "상세 정보를 찾을 수 없습니다.", className="text-muted"
                            ),
                        ],
                        md=3 if len(model_assets) == 4 else 6,
                        className="mb-2",
                    )
                )

        return [
            dbc.Alert(
                [
                    html.H6("📊 모델 학습 ETF 정보", className="mb-2"),
                    html.P(
                        f"이 모델은 {len(model_assets)}개 ETF로 학습되었습니다.",
                        className="text-info mb-3",
                    ),
                    dbc.Row(etf_details),
                ],
                color="light",
                className="mb-3",
            )
        ]
