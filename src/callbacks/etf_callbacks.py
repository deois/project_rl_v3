"""
ETF 선택 관련 콜백 함수들
ETF 드롭다운, 프리셋 선택, 유효성 검증
"""

from typing import Any, Tuple, List, Dict
from dash import html, Input, Output
import dash_bootstrap_components as dbc

from src.utils.etf_manager import etf_manager
from src.utils.logger import get_logger

logger = get_logger("etf_callbacks")


def register_etf_callbacks(app, _dash_manager):
    """ETF 선택 관련 콜백 함수들을 등록"""

    @app.callback(
        Output("training-etf-selection", "options"),
        [Input("training-config-modal", "is_open")],
    )
    def update_etf_options(_is_open: bool) -> List[Dict[str, Any]]:
        """ETF 선택 드롭다운 옵션 업데이트"""
        return etf_manager.get_etf_options_for_dash()

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
                [html.Strong("✅ 선택 완료: "), "4개의 ETF가 선택되었습니다."],
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
        _fast_clicks: int, _balanced_clicks: int, _high_perf_clicks: int
    ) -> List[str]:
        """프리셋 버튼 클릭시 ETF 선택 업데이트"""
        return ["SPY", "DGRO", "SCHD", "EWY"]
