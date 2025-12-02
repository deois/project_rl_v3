"""
설정 관련 콜백 함수들
학습 설정 모달, 입력 필드 업데이트
"""

from typing import Any, Tuple, List, Dict
from dash import callback_context, Input, Output, State

from src.utils.etf_manager import etf_manager
from src.utils.logger import get_logger

logger = get_logger("config_callbacks")


def register_config_callbacks(app, dash_manager):
    """설정 관련 콜백 함수들을 등록"""

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
            State("training-critic-loss-type-input", "value"),
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
        critic_loss_type: str,
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
                "critic_loss_type": critic_loss_type or "mse",
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
                "critic_loss_type": "mse",
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
                "critic_loss_type": "mse",
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
                "critic_loss_type": "mse",
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
            Output("training-critic-loss-type-input", "value"),
            Output("training-window-size-input", "value"),
            Output("training-resume-episodes-input", "value"),
        ],
        [Input("training-config-store", "data")],
    )
    def update_training_config_inputs(
        config: Dict[str, Any],
    ) -> Tuple[int, int, int, int, float, float, str, int, int]:
        """학습 설정 입력 필드 업데이트"""
        return (
            config.get("episodes", 100),
            config.get("episodes_save", 10),
            config.get("batch_size", 128),
            config.get("hidden_dim", 256),
            config.get("actor_lr", 0.0003),
            config.get("critic_lr", 0.0003),
            config.get("critic_loss_type", "mse"),
            config.get("window_size", 60),
            config.get("episodes_resume", 0),
        )
