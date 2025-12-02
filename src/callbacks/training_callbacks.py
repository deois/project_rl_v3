"""
학습 관련 콜백 함수들
학습 시작/중지, 상태 표시, 진행률 관리
"""

import time
import uuid
from typing import Any, Tuple, Dict
import dash
from dash import callback_context, html, Input, Output, State
import dash_bootstrap_components as dbc

from src.dash_simulation import start_simulation_training
from src.utils.logger import get_logger
from src.utils.etf_manager import etf_manager

logger = get_logger("training_callbacks")


def register_training_callbacks(app, dash_manager):
    """학습 관련 콜백 함수들을 등록"""

    @app.callback(
        Output("mode-description", "children"),
        [Input("training-mode", "value")]
    )
    def update_mode_description(mode: str) -> dbc.Alert:
        """모드 설명 업데이트"""
        if mode == "simulation":
            return dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                html.Strong("시뮬레이션 모드: "),
                "빠른 가상 데이터로 UI 테스트 및 시스템 검증",
                html.Br(),
                html.Small([
                    "• 가상 데이터 생성으로 빠른 테스트 | ",
                    "• UI 반응성 확인 | ",
                    "• 시스템 안정성 검증"
                ], className="text-muted")
            ], color="primary", className="mb-0")
        else:
            return dbc.Alert([
                html.I(className="bi bi-rocket me-2"),
                html.Strong("실제 DDPG 학습 모드: "),
                "Deep Deterministic Policy Gradient 알고리즘 기반 포트폴리오 최적화",
                html.Br(),
                html.Small([
                    "• Gym 환경: 연속 행동공간(포트폴리오 비율 0~1) | ",
                    "• 상태공간: 60일 가격이동평균, 변동성, 모멘텀 | ",
                    "• 보상함수: 샤프비율 + 리스크조정수익률 | ",
                    "• 리밸런싱: 매일 포트폴리오 비율 재조정"
                ], className="text-muted")
            ], color="success", className="mb-0")

    @app.callback(
        [Output("training-state-store", "data"),
         Output("start-training-btn", "disabled"),
         Output("stop-training-btn", "disabled")],
        [Input("start-training-btn", "n_clicks"),
         Input("stop-training-btn", "n_clicks"),
         Input("status-interval", "n_intervals")],
        [State("training-state-store", "data"),
         State("training-mode", "value"),
         State("training-config-store", "data")]
    )
    def handle_training_controls(start_clicks: int, stop_clicks: int,
                                 interval_n: int, current_state: Dict[str, Any],
                                 training_mode: str, training_config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, bool]:
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

                dash_manager.training_status.update({
                    "is_training": True,
                    "can_stop": True,
                    "task_id": task_id,
                    "start_time": time.time(),
                    "current_episode": 0,
                    "total_episodes": config["episodes"],
                    "mode": training_mode,
                    "current_step": 0,
                    "total_steps_per_episode": 0,
                    "learning_phase": "학습 시작 중"
                })

                dash_manager.add_log(
                    f"🚀 {'시뮬레이션' if training_mode == 'simulation' else '실제 DDPG'} 학습 시작됨 (ID: {task_id})")
                dash_manager.reset_chart_data()

                # 모드에 따라 다른 학습 시작
                if training_mode == "simulation":
                    start_simulation_training(dash_manager, task_id, config)
                else:
                    dash_manager.real_training_manager.start_real_training(task_id, config)

                return dash_manager.training_status, True, False

        # 학습 중지
        elif trigger_id == "stop-training-btn" and stop_clicks:
            if current_state["is_training"] and current_state["can_stop"]:
                dash_manager.training_status.update({
                    "can_stop": False
                })

                # 모드에 따라 다른 중지 방법
                if current_state.get("mode") == "simulation":
                    if dash_manager.simulation_stop_event:
                        dash_manager.simulation_stop_event.set()
                else:
                    dash_manager.real_training_manager.stop_training()

                dash_manager.add_log(f"🛑 학습 중지 요청됨 (ID: {current_state['task_id']})")

                return dash_manager.training_status, True, True

        # 상태 간격 업데이트에서 학습 완료 확인
        elif trigger_id == "status-interval":
            # 학습이 실제로 완료되었는지 확인하고 상태 업데이트
            if current_state["is_training"]:
                # 시뮬레이션 모드: stop_event가 설정되었는지 확인
                if (current_state.get("mode") == "simulation" and
                    dash_manager.simulation_stop_event and
                        dash_manager.simulation_stop_event.is_set()):

                    dash_manager.training_status.update({
                        "is_training": False,
                        "can_stop": False
                    })
                    dash_manager.add_log("✅ 시뮬레이션 학습이 완전히 종료되었습니다")

                # 실제 모드: training_manager 상태 확인
                elif (current_state.get("mode") != "simulation" and
                      not dash_manager.real_training_manager.is_training):

                    dash_manager.training_status.update({
                        "is_training": False,
                        "can_stop": False
                    })
                    dash_manager.add_log("✅ 실제 학습이 완전히 종료되었습니다")

        # 상태 업데이트만
        return (dash_manager.training_status,
                current_state["is_training"],
                not (current_state["is_training"] and current_state["can_stop"]))

    @app.callback(
        [Output("training-status-text", "children"),
         Output("current-episode", "children"),
         Output("current-reward", "children"),
         Output("portfolio-value", "children"),
         Output("task-id", "children"),
         Output("progress-percent", "children"),
         Output("actor-loss", "children"),
         Output("critic-loss", "children"),
         Output("episode-progress", "children"),
         Output("detailed-status", "children"),
         Output("episode-progress-bar", "value")],
        [Input("training-state-store", "data")]
    )
    def update_status_display(training_state: Dict[str, Any]) -> Tuple[str, str, str, str, str, str, str, str, str, str, float]:
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
            progress = (training_state["current_episode"] / training_state["total_episodes"]) * 100

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
                detailed_status_text = f"EP{current_episode} ({episode_progress_value:.0f}%)"

                # 학습 단계 정보 추가 (짧게)
                if learning_phase:
                    phase_short = learning_phase.replace("에피소드 ", "").replace(" 중", "")
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
            episode_progress_value
        )
