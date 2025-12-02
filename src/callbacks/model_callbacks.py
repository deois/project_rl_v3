"""
모델 관리 관련 콜백 함수들
모델 정보 표시, 저장, 삭제, 기본 모델 설정
"""

import time
import os
from typing import Any, Tuple, List, Dict
from dash import callback_context, html, Input, Output, State
import dash_bootstrap_components as dbc

from src.dash_utils import (
    get_model_metadata, load_model_training_config,
    save_as_default_model, delete_model_folder,
    get_model_deletion_info, get_available_models
)
from src.utils.logger import get_logger

logger = get_logger("model_callbacks")


def register_model_callbacks(app, dash_manager):
    """모델 관리 관련 콜백 함수들을 등록"""

    @app.callback(
        Output("model-metadata-preview", "children"),
        [Input("backtest-model-dropdown", "value")]
    )
    def update_model_metadata_preview(model_path: str):
        """선택된 모델의 미리보기 텍스트 표시"""
        if not model_path:
            return "모델을 선택하면 정보가 표시됩니다"

        # 메타데이터 로드
        metadata_info = get_model_metadata(model_path)
        training_config = load_model_training_config(model_path)

        if metadata_info and training_config:
            episode = training_config.get('current_episode', 0)
            total_episodes = training_config.get(
                'total_episodes', training_config.get('episodes', 0))
            assets = training_config.get('assets', [])
            return f"에피소드 {episode}/{total_episodes} • {len(assets)}개 자산 • 클릭하여 상세보기"
        elif metadata_info:
            return f"에피소드 {metadata_info.get('episode', 0)} • 클릭하여 상세보기"
        else:
            return "메타데이터 없음 • 클릭하여 확인"

    @app.callback(
        [Output("model-info-modal", "is_open"),
         Output("model-info-modal-content", "children")],
        [Input("model-info-btn", "n_clicks"),
         Input("model-info-modal-close", "n_clicks"),
         Input("backtest-model-dropdown", "value")],
        [State("model-info-modal", "is_open")]
    )
    def handle_model_info_modal(info_clicks: int, close_clicks: int, model_path: str, is_open: bool):
        """모델 정보 모달 관리"""
        ctx = callback_context

        # 모달 내용 업데이트
        modal_content = []

        if model_path:
            # 메타데이터 로드
            metadata_info = get_model_metadata(model_path)
            training_config = load_model_training_config(model_path)

            if metadata_info or training_config:
                # 모델 경로 정보
                modal_content.append(
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="bi bi-folder2-open me-2"),
                                "모델 경로"
                            ], className="mb-0 text-primary")
                        ]),
                        dbc.CardBody([
                            html.Code(model_path, className="d-block p-2 bg-light rounded")
                        ])
                    ], className="mb-3")
                )

                # 학습 설정 정보
                if training_config:
                    critic_loss_type = training_config.get("critic_loss_type", "mse")
                    critic_loss_label = (
                        "MSE Loss" if critic_loss_type == "mse" else "Smooth L1 Loss"
                    )
                    max_grad_norm = training_config.get("max_grad_norm", 0.5)
                    modal_content.append(
                        dbc.Card([
                            dbc.CardHeader([
                                html.H5([
                                    html.I(className="bi bi-gear-fill me-2"),
                                    "학습 설정"
                                ], className="mb-0 text-success")
                            ]),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H6("📊 기본 설정", className="text-info mb-2"),
                                        html.P([
                                            html.Strong("총 에피소드: "),
                                            f"{training_config.get('total_episodes', training_config.get('episodes', 'N/A'))}"
                                        ], className="mb-1"),
                                        html.P([
                                            html.Strong("현재 에피소드: "),
                                            f"{training_config.get('current_episode', 0)}"
                                        ], className="mb-1"),
                                        html.P([
                                            html.Strong("배치 크기: "),
                                            f"{training_config.get('batch_size', 128)}"
                                        ], className="mb-1"),
                                        html.P([
                                            html.Strong("저장 주기: "),
                                            f"{training_config.get('episodes_save', 10)} 에피소드"
                                        ], className="mb-1")
                                    ], md=6),
                                    dbc.Col([
                                        html.H6("🧠 신경망 구조", className="text-warning mb-2"),
                                        html.P([
                                            html.Strong("히든 차원: "),
                                            f"{training_config.get('hidden_dim', 256)}"
                                        ], className="mb-1"),
                                        html.P([
                                            html.Strong("Actor 학습률: "),
                                            f"{training_config.get('actor_lr', 0.0003)}"
                                        ], className="mb-1"),
                                        html.P([
                                            html.Strong("Critic 학습률: "),
                                            f"{training_config.get('critic_lr', 0.0003)}"
                                        ], className="mb-1"),
                                        html.P([
                                            html.Strong("Critic Loss 함수: "),
                                            critic_loss_label
                                        ], className="mb-1"),
                                        html.P([
                                            html.Strong("Gradient Clipping: "),
                                            f"{max_grad_norm}"
                                        ], className="mb-1"),
                                        html.P([
                                            html.Strong("윈도우 크기: "),
                                            f"{training_config.get('window_size', 60)}일"
                                        ], className="mb-1")
                                    ], md=6)
                                ])
                            ])
                        ], className="mb-3")
                    )

                # 투자 자산 정보
                if training_config and 'assets' in training_config:
                    assets = training_config['assets']
                    modal_content.append(
                        dbc.Card([
                            dbc.CardHeader([
                                html.H5([
                                    html.I(className="bi bi-briefcase-fill me-2"),
                                    "투자 자산"
                                ], className="mb-0 text-info")
                            ]),
                            dbc.CardBody([
                                html.P(f"총 {len(assets)}개 자산으로 포트폴리오 구성", className="mb-2"),
                                html.Div([
                                    dbc.Badge(asset, color="primary",
                                              className="me-2 mb-1", pill=True)
                                    for asset in assets
                                ])
                            ])
                        ], className="mb-3")
                    )

                # 성과 정보
                if training_config and 'average_reward' in training_config:
                    modal_content.append(
                        dbc.Card([
                            dbc.CardHeader([
                                html.H5([
                                    html.I(className="bi bi-graph-up me-2"),
                                    "학습 성과"
                                ], className="mb-0 text-success")
                            ]),
                            dbc.CardBody([
                                html.P([
                                    html.Strong("평균 보상: "),
                                    f"{training_config.get('average_reward', 0.0):.4f}"
                                ], className="mb-1"),
                                html.P([
                                    html.Strong("작업 ID: "),
                                    f"{training_config.get('task_id', 'N/A')}"
                                ], className="mb-1")
                            ])
                        ], className="mb-3")
                    )

                # 시간 정보
                if metadata_info:
                    modal_content.append(
                        dbc.Card([
                            dbc.CardHeader([
                                html.H5([
                                    html.I(className="bi bi-clock-fill me-2"),
                                    "저장 정보"
                                ], className="mb-0 text-secondary")
                            ]),
                            dbc.CardBody([
                                html.P([
                                    html.Strong("저장 시간: "),
                                    metadata_info.get('date', 'N/A')
                                ], className="mb-1"),
                                html.P([
                                    html.Strong("에피소드: "),
                                    f"{metadata_info.get('episode', 0)}"
                                ], className="mb-1")
                            ])
                        ])
                    )
            else:
                modal_content = [
                    dbc.Alert([
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        "선택된 모델에서 메타데이터를 찾을 수 없습니다."
                    ], color="warning")
                ]
        else:
            modal_content = [
                dbc.Alert([
                    html.I(className="bi bi-info-circle me-2"),
                    "모델을 먼저 선택해주세요."
                ], color="info")
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
            return is_open, modal_content  # 모델 변경 시 모달 상태 유지하고 내용만 업데이트

        return is_open, modal_content

    # 모델 저장 콜백
    @app.callback(
        [Output("save-model-btn", "disabled"),
         Output("save-model-btn", "children")],
        [Input("save-model-btn", "n_clicks")],
        [State("training-state-store", "data"),
         State("training-config-store", "data")]
    )
    def handle_manual_model_save(save_clicks: int, training_state: Dict[str, Any],
                                 training_config: Dict[str, Any]) -> Tuple[bool, List]:
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
            error_button = [html.I(className="bi bi-exclamation-triangle me-2"), "저장 실패"]
            return False, error_button

    # 저장 버튼 상태 복원 콜백
    @app.callback(
        [Output("save-model-btn", "disabled", allow_duplicate=True),
         Output("save-model-btn", "children", allow_duplicate=True)],
        [Input("status-interval", "n_intervals")],
        [State("save-model-btn", "disabled"),
         State("save-model-btn", "children")],
        prevent_initial_call=True
    )
    def restore_save_button_state(n_intervals: int, is_disabled: bool,
                                  current_children: List) -> Tuple[bool, List]:
        """저장 버튼 상태 복원 (저장 완료 후 정상 상태로)"""

        # 기본 버튼 상태
        default_button = [html.I(className="bi bi-download me-2"), "모델 저장"]

        # 현재 버튼이 "저장 완료!" 또는 "저장 실패" 상태인 경우 복원
        if is_disabled and current_children:
            if any("저장 완료" in str(child) or "저장 실패" in str(child)
                   for child in current_children if hasattr(child, 'children') or isinstance(child, str)):
                return False, default_button

        return is_disabled, current_children

    # 기본 모델로 저장 콜백
    @app.callback(
        [Output("save-as-default-model-btn", "disabled"),
         Output("save-as-default-model-btn", "children")],
        [Input("save-as-default-model-btn", "n_clicks")],
        [State("backtest-model-dropdown", "value")]
    )
    def handle_save_as_default_model(save_clicks: int, selected_model_path: str):
        """선택된 모델을 기본 DDPG 모델로 저장"""

        # 기본 버튼 상태
        default_button = [html.I(className="bi bi-bookmark-star me-2"), "기본모델로 저장"]

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
                success_button = [html.I(className="bi bi-check-circle me-2"), "저장 완료!"]
                return True, success_button
            else:
                # 실패 메시지
                dash_manager.add_log(f"❌ {result['message']}")
                error_button = [html.I(className="bi bi-exclamation-triangle me-2"), "저장 실패"]
                return False, error_button

        except Exception as e:
            error_msg = f"❌ 기본 모델 저장 중 예상치 못한 오류: {str(e)}"
            dash_manager.add_log(error_msg)
            error_button = [html.I(className="bi bi-exclamation-triangle me-2"), "저장 실패"]
            return False, error_button

    # 기본 모델 저장 버튼 상태 복원 콜백
    @app.callback(
        [Output("save-as-default-model-btn", "disabled", allow_duplicate=True),
         Output("save-as-default-model-btn", "children", allow_duplicate=True)],
        [Input("backtest-interval", "n_intervals")],
        [State("save-as-default-model-btn", "disabled"),
         State("save-as-default-model-btn", "children")],
        prevent_initial_call=True
    )
    def restore_default_save_button_state(n_intervals: int, is_disabled: bool, current_children: List):
        """기본 모델 저장 버튼 상태 복원"""

        # 기본 버튼 상태
        default_button = [html.I(className="bi bi-bookmark-star me-2"), "기본모델로 저장"]

        # 현재 버튼이 "저장 완료!" 또는 "저장 실패" 상태인 경우 복원
        if is_disabled and current_children:
            if any("저장 완료" in str(child) or "저장 실패" in str(child)
                   for child in current_children if hasattr(child, 'children') or isinstance(child, str)):
                return False, default_button

        return is_disabled, current_children

    # 모델 삭제 관련 콜백들
    @app.callback(
        [Output("model-delete-modal", "is_open"),
         Output("delete-model-path-display", "children")],
        [Input("delete-model-btn", "n_clicks"),
         Input("delete-model-cancel-btn", "n_clicks")],
        [State("model-delete-modal", "is_open"),
         State("backtest-model-dropdown", "value")]
    )
    def handle_delete_model_modal(delete_clicks: int, cancel_clicks: int,
                                  is_open: bool, selected_model_path: str):
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
                dash_manager.add_log(f"❌ 모델이 존재하지 않습니다: {selected_model_path}")
                return False, ""

            # 보호된 모델인지 확인
            if model_info.get("is_protected"):
                dash_manager.add_log(f"🔒 기본 모델은 삭제할 수 없습니다: {model_info['model_name']}")
                return False, ""

            # 모델 정보 표시
            model_name = model_info["model_name"]
            size_mb = model_info["size_mb"]
            file_count = model_info["file_count"]
            metadata = model_info.get("metadata")

            info_display = html.Div([
                html.P([
                    html.Strong("📁 폴더명: "), model_name
                ], className="mb-1"),
                html.P([
                    html.Strong("📂 경로: "), selected_model_path
                ], className="mb-1"),
                html.P([
                    html.Strong("📊 크기: "), f"{size_mb} MB ({file_count}개 파일)"
                ], className="mb-1"),
                html.P([
                    html.Strong("📅 메타데이터: "),
                    f"에피소드 {metadata['episode']}, {metadata['date']}" if metadata else "정보 없음"
                ], className="mb-0")
            ])

            dash_manager.add_log(f"🗑️ 모델 삭제 확인 대화상자 열림: {model_name}")
            return True, info_display

        elif trigger_id == "delete-model-cancel-btn":
            return False, ""

        return is_open, ""

    @app.callback(
        Output("delete-model-confirm-btn", "disabled"),
        [Input("delete-confirmation-checkbox", "value")]
    )
    def update_delete_confirm_button(checkbox_values: List[str]):
        """삭제 확인 체크박스에 따라 삭제 버튼 활성화/비활성화"""
        return "confirmed" not in (checkbox_values or [])

    @app.callback(
        [Output("delete-model-btn", "disabled"),
         Output("delete-model-btn", "children"),
         Output("backtest-model-dropdown", "options", allow_duplicate=True),
         Output("model-delete-modal", "is_open", allow_duplicate=True),
         Output("delete-confirmation-checkbox", "value", allow_duplicate=True)],
        [Input("delete-model-confirm-btn", "n_clicks")],
        [State("backtest-model-dropdown", "value"),
         State("model-delete-modal", "is_open")],
        prevent_initial_call=True
    )
    def handle_model_deletion(confirm_clicks: int, selected_model_path: str, modal_is_open: bool):
        """실제 모델 삭제 실행"""
        if not confirm_clicks or not modal_is_open or not selected_model_path:
            return False, [
                html.I(className="bi bi-trash3 me-2"),
                "모델 삭제"
            ], get_available_models(), modal_is_open, []

        try:
            # 모델 삭제 실행
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
                    "삭제 완료!"
                ]

                # 모델 목록 새로고침
                updated_models = get_available_models()

                # 즉시 모달 닫기 및 체크박스 초기화
                return True, success_children, updated_models, False, []
            else:
                # 실패 시
                dash_manager.add_log(f"❌ 모델 삭제 실패: {result['message']}")
                dash_manager.add_log(f"⚠️ 모델 '{model_name}' 삭제에 실패했습니다. 다시 시도하거나 수동으로 확인해주세요.")

                error_children = [
                    html.I(className="bi bi-exclamation-triangle-fill me-2"),
                    "삭제 실패"
                ]

                # 실패 시에도 모달 닫기
                return True, error_children, get_available_models(), False, []

        except Exception as e:
            error_msg = f"모델 삭제 중 예외 발생: {str(e)}"
            dash_manager.add_log(f"❌ {error_msg}")
            dash_manager.add_log(f"💥 예상치 못한 오류로 모델 삭제에 실패했습니다. 시스템 관리자에게 문의하세요.")

            error_children = [
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                "오류 발생"
            ]

            # 오류 시에도 모달 닫기
            return True, error_children, get_available_models(), False, []

    @app.callback(
        [Output("delete-model-btn", "disabled", allow_duplicate=True),
         Output("delete-model-btn", "children", allow_duplicate=True),
         Output("model-delete-modal", "is_open", allow_duplicate=True),
         Output("delete-confirmation-checkbox", "value")],
        [Input("backtest-interval", "n_intervals")],
        [State("delete-model-btn", "disabled"),
         State("delete-model-btn", "children"),
         State("model-delete-modal", "is_open"),
         State("delete-confirmation-checkbox", "value")],
        prevent_initial_call=True
    )
    def restore_delete_button_state(n_intervals: int, is_disabled: bool,
                                    current_children: List, modal_is_open: bool,
                                    current_checkbox_value: List):
        """삭제 버튼 상태 복원 및 모달 닫기"""
        ctx = callback_context

        # 모달이 열려있으면 현재 체크박스 상태 유지
        if modal_is_open:
            return is_disabled, current_children, modal_is_open, current_checkbox_value or []

        # 삭제 버튼이 "완료" 상태인 경우만 복원 (몇 초 후)
        if current_children and len(current_children) > 1:
            button_text = " ".join(str(item) for item in current_children if isinstance(item, str))
            if "삭제 완료" in button_text:
                # 삭제 완료 상태를 잠시 유지한 후 복원
                if hasattr(dash_manager, '_delete_completion_time'):
                    if time.time() - dash_manager._delete_completion_time > 3:  # 3초 후 복원
                        # 상태 초기화
                        delattr(dash_manager, '_delete_completion_time')
                        return False, [
                            html.I(className="bi bi-trash3 me-2"),
                            "모델 삭제"
                        ], False, []
                else:
                    # 삭제 완료 시간 기록
                    dash_manager._delete_completion_time = time.time()

            elif any("실패" in str(item) or "오류" in str(item)
                     for item in current_children if isinstance(item, str)):
                # 실패/오류 상태는 즉시 복원
                return False, [
                    html.I(className="bi bi-trash3 me-2"),
                    "모델 삭제"
                ], False, []

        return is_disabled, current_children, modal_is_open, current_checkbox_value or []
