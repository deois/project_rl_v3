"""
Dash 콜백 함수들
학습, 백테스트, UI 상태 관리 콜백 - 탭 기반 구조
"""

import time
import uuid
from typing import Any, Tuple, List, Dict
import numpy as np
import dash
from dash import callback_context, html, Input, Output, State
import dash_bootstrap_components as dbc

from src.dash_charts import (
    create_performance_chart,
    create_loss_chart,
    create_backtest_results_chart,
    create_portfolio_allocation_chart,
    create_annualized_returns_chart,
    create_cumulative_returns_chart,
)
from src.dash_utils import (
    get_available_models,
    delete_model_folder,
    get_model_deletion_info,
)
from src.dash_simulation import start_simulation_training
from src.utils.logger import get_logger
from src.utils.etf_manager import etf_manager

# 로거 설정
logger = get_logger("dash_callbacks")


def register_callbacks(app, dash_manager):
    """모든 콜백 함수들을 등록"""

    @app.callback(
        Output("mode-description", "children"), [Input("training-mode", "value")]
    )
    def update_mode_description(mode: str) -> dbc.Alert:
        """모드 설명 업데이트"""
        if mode == "simulation":
            return dbc.Alert(
                [
                    html.I(className="bi bi-info-circle me-2"),
                    html.Strong("시뮬레이션 모드: "),
                    "빠른 가상 데이터로 UI 테스트 및 시스템 검증",
                    html.Br(),
                    html.Small(
                        [
                            "• 가상 데이터 생성으로 빠른 테스트 | ",
                            "• UI 반응성 확인 | ",
                            "• 시스템 안정성 검증",
                        ],
                        className="text-muted",
                    ),
                ],
                color="primary",
                className="mb-0",
            )
        else:
            return dbc.Alert(
                [
                    html.I(className="bi bi-rocket me-2"),
                    html.Strong("실제 DDPG 학습 모드: "),
                    "Deep Deterministic Policy Gradient 알고리즘 기반 포트폴리오 최적화",
                    html.Br(),
                    html.Small(
                        [
                            "• Gym 환경: 연속 행동공간(포트폴리오 비율 0~1) | ",
                            "• 상태공간: 60일 가격이동평균, 변동성, 모멘텀 | ",
                            "• 보상함수: 샤프비율 + 리스크조정수익률 | ",
                            "• 리밸런싱: 매일 포트폴리오 비율 재조정",
                        ],
                        className="text-muted",
                    ),
                ],
                color="success",
                className="mb-0",
            )

    @app.callback(
        [
            Output("training-state-store", "data"),
            Output("start-training-btn", "disabled"),
            Output("stop-training-btn", "disabled"),
        ],
        [
            Input("start-training-btn", "n_clicks"),
            Input("stop-training-btn", "n_clicks"),
            Input("status-interval", "n_intervals"),
        ],
        [
            State("training-state-store", "data"),
            State("training-mode", "value"),
            State("training-config-store", "data"),
        ],
    )
    def handle_training_controls(
        start_clicks: int,
        stop_clicks: int,
        interval_n: int,
        current_state: Dict[str, Any],
        training_mode: str,
        training_config: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool, bool]:
        """학습 시작/중지 및 상태 관리"""

        ctx = callback_context
        if not ctx.triggered:
            return current_state, False, True

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # 학습 시작
        if trigger_id == "start-training-btn" and start_clicks:
            if not current_state["is_training"]:
                task_id = str(uuid.uuid4())[:8]

                # 저장된 설정 사용
                config = training_config.copy()

                # 시뮬레이션 모드일 때는 에피소드 수 조정
                if training_mode == "simulation":
                    config["episodes"] = min(50, config.get("episodes", 100))

                # 선택된 ETF 사용 (기본값 설정)
                if "assets" not in config or not config["assets"]:
                    config["assets"] = etf_manager.get_default_etfs()

                # 전역 설정에도 저장
                dash_manager.training_config = config.copy()

                dash_manager.training_status.update(
                    {
                        "is_training": True,
                        "can_stop": True,
                        "task_id": task_id,
                        "start_time": time.time(),
                        "current_episode": 0,
                        "total_episodes": config["episodes"],
                        "mode": training_mode,
                        "current_step": 0,
                        "total_steps_per_episode": 0,
                        "learning_phase": "학습 시작 중",
                    }
                )

                dash_manager.add_log(
                    f"🚀 {'시뮬레이션' if training_mode == 'simulation' else '실제 DDPG'} 학습 시작됨 (ID: {task_id})"
                )
                dash_manager.reset_chart_data()

                # 모드에 따라 다른 학습 시작
                if training_mode == "simulation":
                    start_simulation_training(dash_manager, task_id, config)
                else:
                    dash_manager.real_training_manager.start_real_training(
                        task_id, config
                    )

                return dash_manager.training_status, True, False

        # 학습 중지
        elif trigger_id == "stop-training-btn" and stop_clicks:
            if current_state["is_training"] and current_state["can_stop"]:
                dash_manager.training_status.update({"can_stop": False})

                # 모드에 따라 다른 중지 방법
                if current_state.get("mode") == "simulation":
                    if dash_manager.simulation_stop_event:
                        dash_manager.simulation_stop_event.set()
                else:
                    dash_manager.real_training_manager.stop_training()

                dash_manager.add_log(
                    f"🛑 학습 중지 요청됨 (ID: {current_state['task_id']})"
                )

                return dash_manager.training_status, True, True

        # 상태 간격 업데이트에서 학습 완료 확인
        elif trigger_id == "status-interval":
            # 학습이 실제로 완료되었는지 확인하고 상태 업데이트
            if current_state["is_training"]:
                # 시뮬레이션 모드: stop_event가 설정되었는지 확인
                if (
                    current_state.get("mode") == "simulation"
                    and dash_manager.simulation_stop_event
                    and dash_manager.simulation_stop_event.is_set()
                ):

                    dash_manager.training_status.update(
                        {"is_training": False, "can_stop": False}
                    )
                    dash_manager.add_log("✅ 시뮬레이션 학습이 완전히 종료되었습니다")

                # 실제 모드: training_manager 상태 확인
                elif (
                    current_state.get("mode") != "simulation"
                    and not dash_manager.real_training_manager.is_training
                ):

                    dash_manager.training_status.update(
                        {"is_training": False, "can_stop": False}
                    )
                    dash_manager.add_log("✅ 실제 학습이 완전히 종료되었습니다")

        # 상태 업데이트만
        return (
            dash_manager.training_status,
            current_state["is_training"],
            not (current_state["is_training"] and current_state["can_stop"]),
        )

    @app.callback(
        [
            Output("training-status-text", "children"),
            Output("current-episode", "children"),
            Output("current-reward", "children"),
            Output("portfolio-value", "children"),
            Output("task-id", "children"),
            Output("progress-percent", "children"),
            Output("actor-loss", "children"),
            Output("critic-loss", "children"),
            Output("episode-progress", "children"),
            Output("detailed-status", "children"),
            Output("episode-progress-bar", "value"),
        ],
        [Input("training-state-store", "data")],
    )
    def update_status_display(
        training_state: Dict[str, Any],
    ) -> Tuple[str, str, str, str, str, str, str, str, str, str, float]:
        """상태 표시 업데이트"""

        if training_state["is_training"]:
            if training_state["can_stop"]:
                mode_icon = "🎮" if training_state.get("mode") == "simulation" else "🚀"
                status_text = f"{mode_icon} 학습 중"
            else:
                status_text = "🟡 중지 중"
        else:
            status_text = "⚪ 대기 중"

        # 전체 진행률 계산
        progress = 0
        if training_state["total_episodes"] > 0:
            progress = (
                training_state["current_episode"] / training_state["total_episodes"]
            ) * 100

            # 에피소드 내 진행률 정보
        episode_progress_text = ""
        detailed_status_text = ""
        episode_progress_value = 0

        if training_state["is_training"]:
            current_episode = training_state.get("current_episode", 0)
            current_step = training_state.get("current_step", 0)
            total_steps = training_state.get("total_steps_per_episode", 0)
            learning_phase = training_state.get("learning_phase", "")

            if total_steps > 0 and current_step > 0:
                episode_progress_value = (current_step / total_steps) * 100
                episode_progress_text = f"{current_step}/{total_steps}"
                detailed_status_text = (
                    f"EP{current_episode} ({episode_progress_value:.0f}%)"
                )

                # 학습 단계 정보 추가 (짧게)
                if learning_phase:
                    phase_short = learning_phase.replace("에피소드 ", "").replace(
                        " 중", ""
                    )
                    detailed_status_text = f"{detailed_status_text} • {phase_short}"
            else:
                # 에피소드 시작 단계
                if current_episode > 0:
                    episode_progress_text = f"EP{current_episode} 시작"
                    detailed_status_text = f"EP{current_episode} 준비 중"
                    if learning_phase:
                        phase_short = learning_phase.replace("에피소드 ", "")
                        detailed_status_text = f"EP{current_episode} • {phase_short}"
                else:
                    episode_progress_text = "준비 중"
                    detailed_status_text = "시스템 초기화"

        return (
            status_text,
            f"{training_state['current_episode']:,}",
            f"{training_state['current_reward']:.2f}",
            f"${training_state['portfolio_value']:,.2f}",
            training_state["task_id"] or "-",
            f"{progress:.1f}%",
            f"{training_state['actor_loss']:.4f}",
            f"{training_state['critic_loss']:.4f}",
            episode_progress_text,
            detailed_status_text,
            episode_progress_value,
        )

    @app.callback(
        Output("performance-chart", "figure"),
        [Input("chart-interval", "n_intervals"), Input("chart-data-store", "data")],
    )
    def update_performance_chart(n_intervals: int, chart_data: Dict[str, List[Any]]):
        """성과 차트 업데이트"""
        return create_performance_chart(chart_data)

    @app.callback(
        Output("loss-chart", "figure"),
        [Input("chart-interval", "n_intervals"), Input("chart-data-store", "data")],
    )
    def update_loss_chart(n_intervals: int, chart_data: Dict[str, List[Any]]):
        """손실 차트 업데이트"""
        return create_loss_chart(chart_data)

    @app.callback(
        [Output("log-container", "children"), Output("log-count", "children")],
        [Input("logs-interval", "n_intervals"), Input("clear-logs-btn", "n_clicks")],
        [State("logs-store", "data")],
    )
    def update_logs(
        n_intervals: int, clear_clicks: int, logs_data: List[str]
    ) -> Tuple[List[html.P], str]:
        """로그 업데이트"""

        ctx = callback_context
        if (
            ctx.triggered
            and ctx.triggered[0]["prop_id"] == "clear-logs-btn.n_clicks"
            and clear_clicks
        ):
            dash_manager.logs = []
            return [
                html.P(
                    "[Dash 대시보드] 로그 지워짐...",
                    style={"margin": "0", "color": "#00ff41", "opacity": "0.8"},
                )
            ], "0"

        if not dash_manager.logs:
            return [
                html.P(
                    "[Dash 대시보드] 시스템 초기화 완료...",
                    style={"margin": "0", "color": "#00ff41", "opacity": "0.8"},
                )
            ], "1"

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
                html.P(
                    log,
                    style={
                        "margin": "3px 0",
                        "color": color,
                        "opacity": max(0.4, (i + 1) / len(recent_logs)),  # 페이드 효과
                        "font-size": "13px",
                        "line-height": "1.4",
                    },
                )
            )

        return log_elements, str(len(dash_manager.logs))

    @app.callback(
        [
            Output("logs-store", "data"),
            Output("chart-data-store", "data"),
            Output("backtest-data-store", "data"),
        ],
        [Input("logs-interval", "n_intervals")],
    )
    def sync_stores(
        n_intervals: int,
    ) -> Tuple[List[str], Dict[str, List[Any]], Dict[str, Any]]:
        """스토어 동기화"""
        return (dash_manager.logs, dash_manager.chart_data, dash_manager.backtest_data)

    # ETF 선택 관련 콜백들
    @app.callback(
        Output("training-etf-selection", "options"),
        [Input("training-config-modal", "is_open")],
    )
    def update_etf_options(is_open: bool) -> List[Dict[str, Any]]:
        """ETF 선택 드롭다운 옵션 업데이트"""
        if is_open:
            return etf_manager.get_etf_options_for_dash()
        return []

    @app.callback(
        [
            Output("selected-etf-info", "children"),
            Output("training-etf-selection", "style"),
        ],
        [Input("training-etf-selection", "value")],
    )
    def update_selected_etf_info(
        selected_etfs: List[str],
    ) -> Tuple[List, Dict[str, str]]:
        """선택된 ETF 정보 표시 및 유효성 검증"""
        if not selected_etfs:
            return [
                dbc.Alert("ETF를 선택해주세요.", color="warning", className="mt-2")
            ], {"color": "black"}

        # 카테고리 헤더 제거 (disabled 옵션들)
        filtered_etfs = [
            etf for etf in selected_etfs if not etf.startswith("category_")
        ]

        # 4개 초과 선택 검증
        if len(filtered_etfs) > 4:
            return [
                dbc.Alert(
                    f"최대 4개의 ETF만 선택할 수 있습니다. 현재 {len(filtered_etfs)}개 선택됨.",
                    color="danger",
                    className="mt-2",
                )
            ], {"color": "black", "border": "2px solid red"}

        # 4개 미만 선택시 경고
        if len(filtered_etfs) < 4:
            return [
                dbc.Alert(
                    f"정확히 4개의 ETF를 선택해야 합니다. 현재 {len(filtered_etfs)}개 선택됨.",
                    color="warning",
                    className="mt-2",
                )
            ], {"color": "black", "border": "2px solid orange"}

        # 선택된 ETF 정보 표시
        etf_info_cards = []
        for etf_symbol in filtered_etfs:
            etf_info = etf_manager.get_etf_info(etf_symbol)
            if etf_info:
                etf_info_cards.append(
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H6(
                                                f"📊 {etf_info['symbol']}",
                                                className="text-primary mb-1",
                                            ),
                                            html.P(
                                                etf_info["name"], className="small mb-1"
                                            ),
                                            html.P(
                                                etf_info["description"],
                                                className="small text-muted mb-0",
                                            ),
                                            dbc.Badge(
                                                etf_info["category"],
                                                color="light",
                                                className="mt-1",
                                            ),
                                        ],
                                        style={"padding": "10px"},
                                    )
                                ],
                                style={"height": "100%"},
                            )
                        ],
                        md=3,
                        className="mb-2",
                    )
                )

        return [
            dbc.Alert(
                [html.Strong("✅ 선택 완료: "), f"4개의 ETF가 선택되었습니다."],
                color="success",
                className="mt-2",
            ),
            dbc.Row(etf_info_cards, className="mt-2"),
        ], {"color": "black", "border": "2px solid green"}

    @app.callback(
        Output("training-etf-selection", "value"),
        [
            Input("preset-fast-btn", "n_clicks"),
            Input("preset-balanced-btn", "n_clicks"),
            Input("preset-high-performance-btn", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def update_etf_selection_on_preset(
        fast_clicks: int, balanced_clicks: int, high_perf_clicks: int
    ) -> List[str]:
        """프리셋 버튼 클릭시 ETF 선택 업데이트"""
        return ["SPY", "DGRO", "SCHD", "EWY"]

    # 백테스트 관련 콜백들
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
            from src.dash_utils import load_model_training_config

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

        # 사용 가능한 모델 수 로깅
        if (
            models and len(models) > 0 and models[0]["value"]
        ):  # 빈 값이 아닌 실제 모델이 있는 경우
            dash_manager.add_log(f"📊 백테스팅 가능한 모델 {len(models)}개 발견")
        else:
            dash_manager.add_log(
                "⚠️ 백테스팅 가능한 모델이 없습니다. checkpoint_last.pth와 metadata_last.json 파일을 확인하세요."
            )

        return models

    @app.callback(
        Output("model-metadata-preview", "children"),
        [Input("backtest-model-dropdown", "value")],
    )
    def update_model_metadata_preview(model_path: str):
        """선택된 모델의 미리보기 텍스트 표시"""
        if not model_path:
            return "모델을 선택하면 정보가 표시됩니다"

        from src.dash_utils import get_model_metadata, load_model_training_config

        # 메타데이터 로드
        metadata_info = get_model_metadata(model_path)
        training_config = load_model_training_config(model_path)

        if metadata_info and training_config:
            episode = training_config.get("current_episode", 0)
            total_episodes = training_config.get(
                "total_episodes", training_config.get("episodes", 0)
            )
            assets = training_config.get("assets", [])
            return f"에피소드 {episode}/{total_episodes} • {len(assets)}개 자산 • 클릭하여 상세보기"
        elif metadata_info:
            return f"에피소드 {metadata_info.get('episode', 0)} • 클릭하여 상세보기"
        else:
            return "메타데이터 없음 • 클릭하여 확인"

    @app.callback(
        [
            Output("model-info-modal", "is_open"),
            Output("model-info-modal-content", "children"),
        ],
        [
            Input("model-info-btn", "n_clicks"),
            Input("model-info-modal-close", "n_clicks"),
            Input("backtest-model-dropdown", "value"),
        ],
        [State("model-info-modal", "is_open")],
    )
    def handle_model_info_modal(
        info_clicks: int, close_clicks: int, model_path: str, is_open: bool
    ):
        """모델 정보 모달 관리"""
        ctx = callback_context

        # 모달 내용 업데이트
        modal_content = []

        if model_path:
            from src.dash_utils import get_model_metadata, load_model_training_config

            # 메타데이터 로드
            metadata_info = get_model_metadata(model_path)
            training_config = load_model_training_config(model_path)

            if metadata_info or training_config:
                # 모델 경로 정보
                modal_content.append(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.H5(
                                        [
                                            html.I(className="bi bi-folder2-open me-2"),
                                            "모델 경로",
                                        ],
                                        className="mb-0 text-primary",
                                    )
                                ]
                            ),
                            dbc.CardBody(
                                [
                                    html.Code(
                                        model_path,
                                        className="d-block p-2 bg-light rounded",
                                    )
                                ]
                            ),
                        ],
                        className="mb-3",
                    )
                )

                # 학습 설정 정보
                if training_config:
                    modal_content.append(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H5(
                                            [
                                                html.I(
                                                    className="bi bi-gear-fill me-2"
                                                ),
                                                "학습 설정",
                                            ],
                                            className="mb-0 text-success",
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
                                                            "📊 기본 설정",
                                                            className="text-info mb-2",
                                                        ),
                                                        html.P(
                                                            [
                                                                html.Strong(
                                                                    "총 에피소드: "
                                                                ),
                                                                f"{training_config.get('total_episodes', training_config.get('episodes', 'N/A'))}",
                                                            ],
                                                            className="mb-1",
                                                        ),
                                                        html.P(
                                                            [
                                                                html.Strong(
                                                                    "현재 에피소드: "
                                                                ),
                                                                f"{training_config.get('current_episode', 0)}",
                                                            ],
                                                            className="mb-1",
                                                        ),
                                                        html.P(
                                                            [
                                                                html.Strong(
                                                                    "배치 크기: "
                                                                ),
                                                                f"{training_config.get('batch_size', 128)}",
                                                            ],
                                                            className="mb-1",
                                                        ),
                                                        html.P(
                                                            [
                                                                html.Strong(
                                                                    "저장 주기: "
                                                                ),
                                                                f"{training_config.get('episodes_save', 10)} 에피소드",
                                                            ],
                                                            className="mb-1",
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "🧠 신경망 구조",
                                                            className="text-warning mb-2",
                                                        ),
                                                        html.P(
                                                            [
                                                                html.Strong(
                                                                    "히든 차원: "
                                                                ),
                                                                f"{training_config.get('hidden_dim', 256)}",
                                                            ],
                                                            className="mb-1",
                                                        ),
                                                        html.P(
                                                            [
                                                                html.Strong(
                                                                    "Actor 학습률: "
                                                                ),
                                                                f"{training_config.get('actor_lr', 0.0003)}",
                                                            ],
                                                            className="mb-1",
                                                        ),
                                                        html.P(
                                                            [
                                                                html.Strong(
                                                                    "Critic 학습률: "
                                                                ),
                                                                f"{training_config.get('critic_lr', 0.0003)}",
                                                            ],
                                                            className="mb-1",
                                                        ),
                                                        html.P(
                                                            [
                                                                html.Strong(
                                                                    "윈도우 크기: "
                                                                ),
                                                                f"{training_config.get('window_size', 60)}일",
                                                            ],
                                                            className="mb-1",
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                            ],
                            className="mb-3",
                        )
                    )

                # 투자 자산 정보
                if training_config and "assets" in training_config:
                    assets = training_config["assets"]
                    modal_content.append(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H5(
                                            [
                                                html.I(
                                                    className="bi bi-briefcase-fill me-2"
                                                ),
                                                "투자 자산",
                                            ],
                                            className="mb-0 text-info",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            f"총 {len(assets)}개 자산으로 포트폴리오 구성",
                                            className="mb-2",
                                        ),
                                        html.Div(
                                            [
                                                dbc.Badge(
                                                    asset,
                                                    color="primary",
                                                    className="me-2 mb-1",
                                                    pill=True,
                                                )
                                                for asset in assets
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        )
                    )

                # 성과 정보
                if training_config and "average_reward" in training_config:
                    modal_content.append(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H5(
                                            [
                                                html.I(className="bi bi-graph-up me-2"),
                                                "학습 성과",
                                            ],
                                            className="mb-0 text-success",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            [
                                                html.Strong("평균 보상: "),
                                                f"{training_config.get('average_reward', 0.0):.4f}",
                                            ],
                                            className="mb-1",
                                        ),
                                        html.P(
                                            [
                                                html.Strong("작업 ID: "),
                                                f"{training_config.get('task_id', 'N/A')}",
                                            ],
                                            className="mb-1",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        )
                    )

                # 시간 정보
                if metadata_info:
                    modal_content.append(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H5(
                                            [
                                                html.I(
                                                    className="bi bi-clock-fill me-2"
                                                ),
                                                "저장 정보",
                                            ],
                                            className="mb-0 text-secondary",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            [
                                                html.Strong("저장 시간: "),
                                                metadata_info.get("date", "N/A"),
                                            ],
                                            className="mb-1",
                                        ),
                                        html.P(
                                            [
                                                html.Strong("에피소드: "),
                                                f"{metadata_info.get('episode', 0)}",
                                            ],
                                            className="mb-1",
                                        ),
                                    ]
                                ),
                            ]
                        )
                    )
            else:
                modal_content = [
                    dbc.Alert(
                        [
                            html.I(className="bi bi-exclamation-triangle me-2"),
                            "선택된 모델에서 메타데이터를 찾을 수 없습니다.",
                        ],
                        color="warning",
                    )
                ]
        else:
            modal_content = [
                dbc.Alert(
                    [
                        html.I(className="bi bi-info-circle me-2"),
                        "모델을 먼저 선택해주세요.",
                    ],
                    color="info",
                )
            ]

        # 모달 열기/닫기 처리
        if not ctx.triggered:
            return False, modal_content

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "model-info-btn":
            return True, modal_content
        elif trigger_id == "model-info-modal-close":
            return False, modal_content
        elif trigger_id == "backtest-model-dropdown":
            return (
                is_open,
                modal_content,
            )  # 모델 변경 시 모달 상태 유지하고 내용만 업데이트

        return is_open, modal_content

    # 모델 저장 콜백 수정
    @app.callback(
        [Output("save-model-btn", "disabled"), Output("save-model-btn", "children")],
        [Input("save-model-btn", "n_clicks")],
        [State("training-state-store", "data"), State("training-config-store", "data")],
    )
    def handle_manual_model_save(
        save_clicks: int,
        training_state: Dict[str, Any],
        training_config: Dict[str, Any],
    ) -> Tuple[bool, List]:
        """수동 모델 저장 처리 - 실제 저장 로직 포함"""

        # 기본 버튼 상태
        default_button = [html.I(className="bi bi-download me-2"), "모델 저장"]

        if not save_clicks:
            return False, default_button

        # 학습 중이 아닌 경우
        if not training_state["is_training"]:
            dash_manager.add_log("⚠️ 현재 학습 중이 아닙니다. 모델 저장을 건너뜁니다.")
            return False, default_button

        # 현재 에피소드가 0 이하인 경우
        current_episode = training_state.get("current_episode", 0)
        if current_episode <= 0:
            dash_manager.add_log("⚠️ 저장할 수 있는 모델이 없습니다. (에피소드 0)")
            return False, default_button

        # 실제 모델 저장 실행
        dash_manager.add_log(f"💾 수동 모델 저장 시작: 에피소드 {current_episode}")

        # 실제 저장 요청
        success = dash_manager.real_training_manager.manual_save_model()

        if success:
            # 저장 성공 시 버튼 일시적 변경
            success_button = [html.I(className="bi bi-check-circle me-2"), "저장 완료!"]
            return True, success_button  # 일시적으로 비활성화
        else:
            # 저장 실패 시
            error_button = [
                html.I(className="bi bi-exclamation-triangle me-2"),
                "저장 실패",
            ]
            return False, error_button

    # 저장 버튼 상태 복원 콜백 추가
    @app.callback(
        [
            Output("save-model-btn", "disabled", allow_duplicate=True),
            Output("save-model-btn", "children", allow_duplicate=True),
        ],
        [Input("status-interval", "n_intervals")],
        [State("save-model-btn", "disabled"), State("save-model-btn", "children")],
        prevent_initial_call=True,
    )
    def restore_save_button_state(
        n_intervals: int, is_disabled: bool, current_children: List
    ) -> Tuple[bool, List]:
        """저장 버튼 상태 복원 (저장 완료 후 정상 상태로)"""

        # 기본 버튼 상태
        default_button = [html.I(className="bi bi-download me-2"), "모델 저장"]

        # 현재 버튼이 "저장 완료!" 또는 "저장 실패" 상태인 경우 복원
        if is_disabled and current_children:
            if any(
                "저장 완료" in str(child) or "저장 실패" in str(child)
                for child in current_children
                if hasattr(child, "children") or isinstance(child, str)
            ):
                return False, default_button

        return is_disabled, current_children

    # 백테스트 설정 동기화 콜백 추가
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

    # 백테스트 설정 업데이트 콜백
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
        from src.dash_utils import load_model_training_config

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
                {"display": "none"},
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

                # 백테스트 상태 업데이트
                dash_manager.update_backtest_status(
                    is_running=True,
                    task_id=task_id,
                    start_time=time.time(),
                    progress=0,
                    current_step=0,
                    total_steps=1000,  # 예상 스텝 수
                    error=None,
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
                    {"display": "none"},
                    {"display": "none"},
                    {"display": "none"},
                    {"display": "none"},
                    {"display": "none"},
                )

        # 백테스트 상태 업데이트 (주기적)
        elif trigger_id == "backtest-interval":
            # 현재 백테스트 매니저의 상태를 백테스트 상태에 반영
            if backtest_state["is_running"]:
                current_status = dash_manager.backtest_status.copy()

                # 백테스트 완료 확인
                if not dash_manager.backtest_manager.is_running:
                    dash_manager.update_backtest_status(is_running=False, progress=100)
                    dash_manager.add_log("✅ 백테스트가 완료되었습니다")

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
                        dash_manager.backtest_status,
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

        # 상태에 따른 표시 결정
        status_style = (
            {"display": "block"}
            if backtest_state["is_running"]
            else {"display": "none"}
        )
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
            backtest_state,
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
        else:
            status_text = "⚪ 대기 중"
            progress = 0
            progress_text = "0.0%"
            bar_color = "secondary"
            alert_style = {"display": "none"}

        task_id = backtest_state.get("task_id", "-")

        return status_text, progress, progress_text, task_id, alert_style, bar_color

    @app.callback(
        Output("backtest-results-chart", "figure"),
        [Input("backtest-data-store", "data")],
    )
    def update_backtest_results_chart(backtest_data: Dict[str, Any]):
        """백테스트 결과 차트 업데이트"""
        # 디버깅: 받은 데이터 구조 확인
        # if backtest_data:
        #     # logger.info(f"🎨 차트 콜백 - 받은 데이터 키들: {list(backtest_data.keys())}")
        #     # if 'dates' in backtest_data:
        #     #     dates = backtest_data['dates']
        #     #     # logger.info(f"🎨 차트 콜백 - 날짜 데이터: {len(dates)}개")
        #     #     # if dates:
        #     #     #     logger.info(f"🎨 차트 콜백 - 날짜 범위: {dates[0]} ~ {dates[-1]}")
        #     #     #     logger.info(f"🎨 차트 콜백 - 첫 5개 날짜: {dates[:5]}")

        #     # if 'portfolio_values' in backtest_data:
        #     #     values = backtest_data['portfolio_values']
        #     #     logger.info(f"🎨 차트 콜백 - 포트폴리오 값: {len(values)}개")
        #     #     if values:
        #     #         logger.info(f"🎨 차트 콜백 - 값 범위: ${min(values):,.2f} ~ ${max(values):,.2f}")

        #     # if 'equal_strategy' in backtest_data:
        #     #     equal_data = backtest_data['equal_strategy']
        #     #     logger.info(f"🎨 차트 콜백 - 균등투자 데이터 키들: {list(equal_data.keys())}")
        #     #     if 'dates' in equal_data:
        #     #         equal_dates = equal_data['dates']
        #     #         logger.info(f"🎨 차트 콜백 - 균등투자 날짜: {len(equal_dates)}개")
        # else:
        #     logger.info("🎨 차트 콜백 - 받은 데이터가 비어있음")

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

    # 학습 설정 관련 콜백들
    @app.callback(
        [
            Output("training-config-modal", "is_open"),
            Output("training-config-store", "data"),
        ],
        [
            Input("training-config-btn", "n_clicks"),
            Input("training-config-cancel-btn", "n_clicks"),
            Input("training-config-save-btn", "n_clicks"),
            Input("preset-fast-btn", "n_clicks"),
            Input("preset-balanced-btn", "n_clicks"),
            Input("preset-high-performance-btn", "n_clicks"),
        ],
        [
            State("training-config-modal", "is_open"),
            State("training-episodes-input", "value"),
            State("training-save-episodes-input", "value"),
            State("training-batch-size-input", "value"),
            State("training-hidden-dim-input", "value"),
            State("training-actor-lr-input", "value"),
            State("training-critic-lr-input", "value"),
            State("training-window-size-input", "value"),
            State("training-resume-episodes-input", "value"),
            State("training-etf-selection", "value"),
            State("training-config-store", "data"),
        ],
    )
    def handle_training_config_modal(
        config_clicks: int,
        cancel_clicks: int,
        save_clicks: int,
        fast_clicks: int,
        balanced_clicks: int,
        high_perf_clicks: int,
        is_open: bool,
        episodes: int,
        save_episodes: int,
        batch_size: int,
        hidden_dim: int,
        actor_lr: float,
        critic_lr: float,
        window_size: int,
        resume_episodes: int,
        selected_etfs: List[str],
        current_config: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """학습 설정 모달 관리"""
        ctx = callback_context
        if not ctx.triggered:
            return is_open, current_config

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # 모달 열기
        if trigger_id == "training-config-btn":
            return True, current_config

        # 모달 닫기
        elif trigger_id == "training-config-cancel-btn":
            return False, current_config

        # 설정 저장
        elif trigger_id == "training-config-save-btn":
            # ETF 선택 유효성 검증
            if not selected_etfs or len(selected_etfs) != 4:
                dash_manager.add_log("❌ 정확히 4개의 ETF를 선택해야 합니다.")
                return True, current_config  # 모달을 열린 상태로 유지

            # 카테고리 헤더 제거
            filtered_etfs = [
                etf for etf in selected_etfs if not etf.startswith("category_")
            ]
            if len(filtered_etfs) != 4:
                dash_manager.add_log("❌ 정확히 4개의 ETF를 선택해야 합니다.")
                return True, current_config

            # ETF 선택 정보를 etf_manager에 저장
            if not etf_manager.set_selected_etfs(filtered_etfs):
                dash_manager.add_log("❌ 유효하지 않은 ETF가 선택되었습니다.")
                return True, current_config

            new_config = {
                "episodes": episodes or 100,
                "episodes_save": save_episodes or 10,
                "episodes_resume": resume_episodes or 0,
                "batch_size": batch_size or 128,
                "hidden_dim": hidden_dim or 256,
                "actor_lr": actor_lr or 0.0003,
                "critic_lr": critic_lr or 0.0003,
                "window_size": window_size or 60,
                "assets": filtered_etfs,  # 선택된 ETF 추가
            }
            # 전역 설정 업데이트
            if hasattr(dash_manager, "training_config"):
                dash_manager.training_config.update(new_config)
            else:
                dash_manager.training_config = new_config

            dash_manager.add_log(
                f"⚙️ 학습 설정이 저장되었습니다: Episodes={episodes}, Batch={batch_size}, ETFs={', '.join(filtered_etfs)}"
            )
            return False, new_config

        # 프리셋 적용
        elif trigger_id == "preset-fast-btn":
            preset_config = {
                "episodes": 50,
                "episodes_save": 5,
                "episodes_resume": 0,
                "batch_size": 64,
                "hidden_dim": 128,
                "actor_lr": 0.001,
                "critic_lr": 0.001,
                "window_size": 30,
            }
            dash_manager.add_log("🚀 빠른 테스트 프리셋이 적용되었습니다")
            return True, preset_config

        elif trigger_id == "preset-balanced-btn":
            preset_config = {
                "episodes": 100,
                "episodes_save": 10,
                "episodes_resume": 0,
                "batch_size": 128,
                "hidden_dim": 256,
                "actor_lr": 0.0003,
                "critic_lr": 0.0003,
                "window_size": 60,
            }
            dash_manager.add_log("🎯 균형잡힌 프리셋이 적용되었습니다")
            return True, preset_config

        elif trigger_id == "preset-high-performance-btn":
            preset_config = {
                "episodes": 300,
                "episodes_save": 20,
                "episodes_resume": 0,
                "batch_size": 256,
                "hidden_dim": 512,
                "actor_lr": 0.0001,
                "critic_lr": 0.0001,
                "window_size": 90,
            }
            dash_manager.add_log("💪 고성능 프리셋이 적용되었습니다")
            return True, preset_config

        return is_open, current_config

    @app.callback(
        [
            Output("training-episodes-input", "value"),
            Output("training-save-episodes-input", "value"),
            Output("training-batch-size-input", "value"),
            Output("training-hidden-dim-input", "value"),
            Output("training-actor-lr-input", "value"),
            Output("training-critic-lr-input", "value"),
            Output("training-window-size-input", "value"),
            Output("training-resume-episodes-input", "value"),
        ],
        [Input("training-config-store", "data")],
    )
    def update_training_config_inputs(
        config: Dict[str, Any],
    ) -> Tuple[int, int, int, int, float, float, int, int]:
        """학습 설정 입력 필드 업데이트"""
        return (
            config.get("episodes", 100),
            config.get("episodes_save", 10),
            config.get("batch_size", 128),
            config.get("hidden_dim", 256),
            config.get("actor_lr", 0.0003),
            config.get("critic_lr", 0.0003),
            config.get("window_size", 60),
            config.get("episodes_resume", 0),
        )

    # 모니터링 관련 콜백들
    @app.callback(
        [
            Output("system-status", "children"),
            Output("uptime", "children"),
            Output("memory-usage", "children"),
            Output("gpu-temp", "children"),
        ],
        [Input("monitoring-interval", "n_intervals")],
    )
    def update_monitoring_metrics(n_intervals: int) -> Tuple[str, str, str, str]:
        """모니터링 메트릭 업데이트"""
        import psutil
        import datetime

        try:
            # 시스템 상태
            cpu_percent = psutil.cpu_percent()
            if cpu_percent < 70:
                system_status = "🟢 정상"
            elif cpu_percent < 90:
                system_status = "🟡 주의"
            else:
                system_status = "🔴 과부하"

            # 업타임 (대략적인 값)
            uptime_seconds = time.time() - dash_manager.training_status.get(
                "start_time", time.time()
            )
            uptime = str(datetime.timedelta(seconds=int(uptime_seconds)))

            # 메모리 사용량
            memory = psutil.virtual_memory()
            memory_usage = f"{memory.used // (1024**2)} MB"

            # GPU 온도 (사용 가능한 경우)
            gpu_temp = "N/A"
            try:
                # GPUtil은 선택적 의존성이므로 import 시도
                import GPUtil

                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_temp = f"{gpus[0].temperature}°C"
            except ImportError:
                # GPUtil이 설치되지 않은 경우
                gpu_temp = "N/A (GPUtil 미설치)"
            except Exception:
                # 기타 오류
                gpu_temp = "N/A"

            return system_status, uptime, memory_usage, gpu_temp
        except:
            return "🟡 모니터링 오류", "00:00:00", "0 MB", "N/A"

    @app.callback(
        Output("system-monitoring-chart", "figure"),
        [Input("monitoring-interval", "n_intervals")],
    )
    def update_monitoring_chart(n_intervals: int):
        """시스템 모니터링 차트 업데이트"""
        import plotly.graph_objs as go
        import psutil

        try:
            # CPU 사용률 가져오기
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent

            # 간단한 게이지 차트 생성
            fig = go.Figure()

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=cpu_percent,
                    domain={"x": [0, 0.5], "y": [0, 1]},
                    title={"text": "CPU 사용률 (%)"},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 80], "color": "yellow"},
                            {"range": [80, 100], "color": "red"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                )
            )

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=memory_percent,
                    domain={"x": [0.5, 1], "y": [0, 1]},
                    title={"text": "메모리 사용률 (%)"},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkgreen"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 80], "color": "yellow"},
                            {"range": [80, 100], "color": "red"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                )
            )

            fig.update_layout(title="실시간 시스템 모니터링", height=400)

            return fig
        except:
            # 오류 시 빈 차트 반환
            return go.Figure().add_annotation(
                text="모니터링 데이터를 사용할 수 없습니다",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=16),
            )

    # 기본 모델로 저장 콜백 추가
    @app.callback(
        [
            Output("save-as-default-model-btn", "disabled"),
            Output("save-as-default-model-btn", "children"),
        ],
        [Input("save-as-default-model-btn", "n_clicks")],
        [State("backtest-model-dropdown", "value")],
    )
    def handle_save_as_default_model(save_clicks: int, selected_model_path: str):
        """선택된 모델을 기본 DDPG 모델로 저장"""

        # 기본 버튼 상태
        default_button = [
            html.I(className="bi bi-bookmark-star me-2"),
            "기본모델로 저장",
        ]

        if not save_clicks:
            return False, default_button

        if not selected_model_path:
            dash_manager.add_log("⚠️ 저장할 모델을 먼저 선택해주세요")
            return False, default_button

        # 자기 자신으로 저장하는 것을 방지
        if selected_model_path == "./model/rl_ddpg":
            dash_manager.add_log("⚠️ 이미 기본 모델입니다")
            return False, default_button

        try:
            from src.dash_utils import save_as_default_model

            dash_manager.add_log(f"💾 기본 모델로 저장 시작: {selected_model_path}")

            # 모델 저장 실행
            result = save_as_default_model(selected_model_path)

            if result["success"]:
                # 성공 메시지
                dash_manager.add_log(f"✅ {result['message']}")
                dash_manager.add_log(f"📁 복사된 파일: {len(result['copied_files'])}개")
                if result.get("backup_created"):
                    dash_manager.add_log("💼 기존 기본 모델이 백업되었습니다")

                # 성공 버튼 상태
                success_button = [
                    html.I(className="bi bi-check-circle me-2"),
                    "저장 완료!",
                ]
                return True, success_button
            else:
                # 실패 메시지
                dash_manager.add_log(f"❌ {result['message']}")
                error_button = [
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    "저장 실패",
                ]
                return False, error_button

        except Exception as e:
            error_msg = f"❌ 기본 모델 저장 중 예상치 못한 오류: {str(e)}"
            dash_manager.add_log(error_msg)
            error_button = [
                html.I(className="bi bi-exclamation-triangle me-2"),
                "저장 실패",
            ]
            return False, error_button

    # 기본 모델 저장 버튼 상태 복원 콜백
    @app.callback(
        [
            Output("save-as-default-model-btn", "disabled", allow_duplicate=True),
            Output("save-as-default-model-btn", "children", allow_duplicate=True),
        ],
        [Input("backtest-interval", "n_intervals")],
        [
            State("save-as-default-model-btn", "disabled"),
            State("save-as-default-model-btn", "children"),
        ],
        prevent_initial_call=True,
    )
    def restore_default_save_button_state(
        n_intervals: int, is_disabled: bool, current_children: List
    ):
        """기본 모델 저장 버튼 상태 복원"""

        # 기본 버튼 상태
        default_button = [
            html.I(className="bi bi-bookmark-star me-2"),
            "기본모델로 저장",
        ]

        # 현재 버튼이 "저장 완료!" 또는 "저장 실패" 상태인 경우 복원
        if is_disabled and current_children:
            if any(
                "저장 완료" in str(child) or "저장 실패" in str(child)
                for child in current_children
                if hasattr(child, "children") or isinstance(child, str)
            ):
                return False, default_button

        return is_disabled, current_children

    # 모델 삭제 관련 콜백들
    @app.callback(
        [
            Output("model-delete-modal", "is_open"),
            Output("delete-model-path-display", "children"),
        ],
        [
            Input("delete-model-btn", "n_clicks"),
            Input("delete-model-cancel-btn", "n_clicks"),
        ],
        [
            State("model-delete-modal", "is_open"),
            State("backtest-model-dropdown", "value"),
        ],
    )
    def handle_delete_model_modal(
        delete_clicks: int, cancel_clicks: int, is_open: bool, selected_model_path: str
    ):
        """모델 삭제 모달 관리"""
        ctx = callback_context
        if not ctx.triggered:
            return is_open, ""

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "delete-model-btn" and delete_clicks:
            # 선택된 모델이 없는 경우
            if not selected_model_path:
                dash_manager.add_log("❌ 삭제할 모델을 먼저 선택해주세요")
                return False, ""

            # 모델 정보 가져오기
            model_info = get_model_deletion_info(selected_model_path)

            if not model_info.get("exists"):
                dash_manager.add_log(
                    f"❌ 모델이 존재하지 않습니다: {selected_model_path}"
                )
                return False, ""

            # 보호된 모델인지 확인
            if model_info.get("is_protected"):
                dash_manager.add_log(
                    f"🔒 기본 모델은 삭제할 수 없습니다: {model_info['model_name']}"
                )
                return False, ""

            # 모델 정보 표시
            model_name = model_info["model_name"]
            size_mb = model_info["size_mb"]
            file_count = model_info["file_count"]
            metadata = model_info.get("metadata")

            info_display = html.Div(
                [
                    html.P([html.Strong("📁 폴더명: "), model_name], className="mb-1"),
                    html.P(
                        [html.Strong("📂 경로: "), selected_model_path],
                        className="mb-1",
                    ),
                    html.P(
                        [
                            html.Strong("📊 크기: "),
                            f"{size_mb} MB ({file_count}개 파일)",
                        ],
                        className="mb-1",
                    ),
                    html.P(
                        [
                            html.Strong("📅 메타데이터: "),
                            (
                                f"에피소드 {metadata['episode']}, {metadata['date']}"
                                if metadata
                                else "정보 없음"
                            ),
                        ],
                        className="mb-0",
                    ),
                ]
            )

            dash_manager.add_log(f"🗑️ 모델 삭제 확인 대화상자 열림: {model_name}")
            return True, info_display

        elif trigger_id == "delete-model-cancel-btn":
            return False, ""

        return is_open, ""

    @app.callback(
        Output("delete-model-confirm-btn", "disabled"),
        [Input("delete-confirmation-checkbox", "value")],
    )
    def update_delete_confirm_button(checkbox_values: List[str]):
        """삭제 확인 체크박스에 따라 삭제 버튼 활성화/비활성화"""
        return "confirmed" not in (checkbox_values or [])

    @app.callback(
        [
            Output("delete-model-btn", "disabled"),
            Output("delete-model-btn", "children"),
            Output("backtest-model-dropdown", "options", allow_duplicate=True),
            Output("model-delete-modal", "is_open", allow_duplicate=True),
            Output("delete-confirmation-checkbox", "value", allow_duplicate=True),
        ],
        [Input("delete-model-confirm-btn", "n_clicks")],
        [
            State("backtest-model-dropdown", "value"),
            State("model-delete-modal", "is_open"),
        ],
        prevent_initial_call=True,
    )
    def handle_model_deletion(
        confirm_clicks: int, selected_model_path: str, modal_is_open: bool
    ):
        """실제 모델 삭제 실행"""
        if not confirm_clicks or not modal_is_open or not selected_model_path:
            return (
                False,
                [html.I(className="bi bi-trash3 me-2"), "모델 삭제"],
                get_available_models(),
                modal_is_open,
                [],
            )

        try:
            # 모델 삭제 실행
            import os

            model_name = os.path.basename(selected_model_path)
            dash_manager.add_log(f"🗑️ 모델 삭제 실행 중: {model_name}")

            result = delete_model_folder(selected_model_path)

            if result["success"]:
                # 성공 시 - 상세한 완료 메시지
                dash_manager.add_log(f"✅ {result['message']}")
                dash_manager.add_log(f"🎉 모델 '{model_name}' 삭제가 완료되었습니다!")
                dash_manager.add_log(f"📁 삭제된 경로: {selected_model_path}")

                # 버튼 상태 변경 (일시적)
                success_children = [
                    html.I(className="bi bi-check-circle-fill me-2"),
                    "삭제 완료!",
                ]

                # 모델 목록 새로고침
                updated_models = get_available_models()

                # 즉시 모달 닫기 및 체크박스 초기화
                return True, success_children, updated_models, False, []
            else:
                # 실패 시
                dash_manager.add_log(f"❌ 모델 삭제 실패: {result['message']}")
                dash_manager.add_log(
                    f"⚠️ 모델 '{model_name}' 삭제에 실패했습니다. 다시 시도하거나 수동으로 확인해주세요."
                )

                error_children = [
                    html.I(className="bi bi-exclamation-triangle-fill me-2"),
                    "삭제 실패",
                ]

                # 실패 시에도 모달 닫기
                return True, error_children, get_available_models(), False, []

        except Exception as e:
            error_msg = f"모델 삭제 중 예외 발생: {str(e)}"
            dash_manager.add_log(f"❌ {error_msg}")
            dash_manager.add_log(
                f"💥 예상치 못한 오류로 모델 삭제에 실패했습니다. 시스템 관리자에게 문의하세요."
            )

            error_children = [
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                "오류 발생",
            ]

            # 오류 시에도 모달 닫기
            return True, error_children, get_available_models(), False, []

    @app.callback(
        [
            Output("delete-model-btn", "disabled", allow_duplicate=True),
            Output("delete-model-btn", "children", allow_duplicate=True),
            Output("model-delete-modal", "is_open", allow_duplicate=True),
            Output("delete-confirmation-checkbox", "value"),
        ],
        [Input("backtest-interval", "n_intervals")],
        [
            State("delete-model-btn", "disabled"),
            State("delete-model-btn", "children"),
            State("model-delete-modal", "is_open"),
            State("delete-confirmation-checkbox", "value"),
        ],
        prevent_initial_call=True,
    )
    def restore_delete_button_state(
        n_intervals: int,
        is_disabled: bool,
        current_children: List,
        modal_is_open: bool,
        current_checkbox_value: List,
    ):
        """삭제 버튼 상태 복원 및 모달 닫기"""
        ctx = callback_context

        # 모달이 열려있으면 현재 체크박스 상태 유지
        if modal_is_open:
            return (
                is_disabled,
                current_children,
                modal_is_open,
                current_checkbox_value or [],
            )

        # 삭제 버튼이 "완료" 상태인 경우만 복원 (몇 초 후)
        if current_children and len(current_children) > 1:
            button_text = " ".join(
                str(item) for item in current_children if isinstance(item, str)
            )
            if "삭제 완료" in button_text:
                # 삭제 완료 상태를 잠시 유지한 후 복원
                import time

                if hasattr(dash_manager, "_delete_completion_time"):
                    if (
                        time.time() - dash_manager._delete_completion_time > 3
                    ):  # 3초 후 복원
                        # 상태 초기화
                        delattr(dash_manager, "_delete_completion_time")
                        return (
                            False,
                            [html.I(className="bi bi-trash3 me-2"), "모델 삭제"],
                            False,
                            [],
                        )
                else:
                    # 삭제 완료 시간 기록
                    dash_manager._delete_completion_time = time.time()

            elif any(
                "실패" in str(item) or "오류" in str(item)
                for item in current_children
                if isinstance(item, str)
            ):
                # 실패/오류 상태는 즉시 복원
                return (
                    False,
                    [html.I(className="bi bi-trash3 me-2"), "모델 삭제"],
                    False,
                    [],
                )

        return (
            is_disabled,
            current_children,
            modal_is_open,
            current_checkbox_value or [],
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

        from src.dash_utils import load_model_training_config

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
