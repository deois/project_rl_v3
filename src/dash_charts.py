"""
Dash 차트 모듈
Plotly 차트 생성 및 업데이트 함수들
"""

from typing import Dict, Any, List
import plotly.graph_objs as go
from src.utils.logger import get_logger

# 로거 설정
logger = get_logger("dash_charts")


def create_performance_chart(chart_data: Dict[str, List[Any]]) -> go.Figure:
    """성과 차트 생성"""
    fig = go.Figure()

    if not chart_data["episodes"]:
        # 빈 차트
        fig.add_trace(
            go.Scatter(
                x=[], y=[], name="포트폴리오 가치", line=dict(color="#28a745", width=3)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name="에피소드 보상",
                yaxis="y2",
                line=dict(color="#ffc107", width=3),
            )
        )
    else:
        # 포트폴리오 가치
        fig.add_trace(
            go.Scatter(
                x=chart_data["episodes"],
                y=chart_data["portfolio_values"],
                name="포트폴리오 가치",
                line=dict(color="#28a745", width=3),
                mode="lines+markers",
                marker=dict(size=6),
            )
        )

        # 에피소드 보상 (2차 Y축)
        fig.add_trace(
            go.Scatter(
                x=chart_data["episodes"],
                y=chart_data["rewards"],
                name="에피소드 보상",
                line=dict(color="#ffc107", width=3),
                mode="lines+markers",
                marker=dict(size=6),
                yaxis="y2",
            )
        )

    fig.update_layout(
        title={
            "text": "실시간 포트폴리오 성과",
            "x": 0.5,
            "font": {"size": 18, "family": "Inter", "color": "#2c3e50"},
        },
        xaxis_title="에피소드",
        yaxis=dict(title="포트폴리오 가치 (USD)", side="left", tickformat="$,.0f"),
        yaxis2=dict(
            title="에피소드 보상", side="right", overlaying="y", tickformat=".2f"
        ),
        template="plotly_white",
        height=400,
        margin=dict(l=70, r=70, t=60, b=60),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
        plot_bgcolor="rgba(0, 0, 0, 0.02)",
    )

    return fig


def create_loss_chart(chart_data: Dict[str, List[Any]]) -> go.Figure:
    """손실 차트 생성"""
    fig = go.Figure()

    if not chart_data["episodes"]:
        # 빈 차트
        fig.add_trace(
            go.Scatter(
                x=[], y=[], name="Actor Loss", line=dict(color="#dc3545", width=2)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[], y=[], name="Critic Loss", line=dict(color="#6f42c1", width=2)
            )
        )
    else:
        # Actor Loss
        fig.add_trace(
            go.Scatter(
                x=chart_data["episodes"],
                y=chart_data["actor_losses"],
                name="Actor Loss",
                line=dict(color="#dc3545", width=2),
                mode="lines",
            )
        )

        # Critic Loss
        fig.add_trace(
            go.Scatter(
                x=chart_data["episodes"],
                y=chart_data["critic_losses"],
                name="Critic Loss",
                line=dict(color="#6f42c1", width=2),
                mode="lines",
            )
        )

    fig.update_layout(
        title={
            "text": "학습 손실 추이",
            "x": 0.5,
            "font": {"size": 16, "family": "Inter", "color": "#2c3e50"},
        },
        xaxis_title="에피소드",
        yaxis_title="손실값",
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
        plot_bgcolor="rgba(0, 0, 0, 0.02)",
    )

    return fig


def create_backtest_results_chart(backtest_data: Dict[str, Any]) -> go.Figure:
    """백테스트 결과 차트 생성 - 강화학습 vs 균등투자 비교"""
    fig = go.Figure()

    if not backtest_data.get("portfolio_values"):
        # 빈 차트
        fig.add_trace(
            go.Scatter(
                x=[], y=[], name="강화학습 전략", line=dict(color="#17a2b8", width=3)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[], y=[], name="균등투자 전략", line=dict(color="#28a745", width=3)
            )
        )
        fig.update_layout(
            title="백테스트 결과 - 포트폴리오 가치 비교",
            xaxis_title="날짜",
            yaxis_title="포트폴리오 가치 ($)",
            height=450,
            showlegend=True,
            hovermode="x unified",
            template="plotly_white",
            margin=dict(l=50, r=50, t=50, b=50),
        )
    else:
        # 날짜 데이터 처리
        dates = backtest_data["dates"]
        portfolio_values = backtest_data["portfolio_values"]

        # 데이터 길이 확인 로그 (디버깅용)
        # print(f"Chart Debug: 날짜 데이터 {len(dates)}개, 포트폴리오 값 {len(portfolio_values)}개")
        # if dates:
        #     print(f"Chart Debug: 첫 날짜 {dates[0]}, 마지막 날짜 {dates[-1]}")
        #     print(f"Chart Debug: 첫 10개 날짜: {dates[:10]}")

        # 강화학습 전략 포트폴리오 가치 차트
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=portfolio_values,
                name="강화학습 전략",
                line=dict(color="#17a2b8", width=3),
                fill="tonexty",
                fillcolor="rgba(23, 162, 184, 0.1)",
                hovertemplate="날짜: %{x}<br>강화학습 가치: $%{y:,.2f}<extra></extra>",
            )
        )

        # 균등투자 전략 데이터가 있는 경우 추가
        comparison_data = backtest_data.get("equal_strategy")
        if comparison_data and comparison_data.get("portfolio_values"):
            equal_dates = comparison_data.get("dates", dates)
            equal_values = comparison_data["portfolio_values"]

            # 균등투자 전략 데이터 로그 (필요시 주석 해제)
            # print(f"Chart Debug: 균등투자 날짜 {len(equal_dates)}개, 값 {len(equal_values)}개")

            fig.add_trace(
                go.Scatter(
                    x=equal_dates,
                    y=equal_values,
                    name="균등투자 전략",
                    line=dict(color="#28a745", width=3, dash="dash"),
                    hovertemplate="날짜: %{x}<br>균등투자 가치: $%{y:,.2f}<extra></extra>",
                )
            )

        # 초기 투자금 기준선 추가 (있는 경우)
        if backtest_data.get("initial_investment"):
            initial_investment = backtest_data["initial_investment"]
            fig.add_hline(
                y=initial_investment,
                line_dash="dot",
                line_color="gray",
                annotation_text=f"초기 투자금: ${initial_investment:,.2f}",
                annotation_position="bottom right",
            )

        # 레이아웃 설정
        fig.update_layout(
            title="백테스트 결과 - 강화학습 vs 균등투자 전략 비교",
            xaxis_title="날짜",
            yaxis_title="포트폴리오 가치 ($)",
            height=450,
            showlegend=True,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            template="plotly_white",
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(
                type="date",  # 날짜 타입으로 명시적 설정
                tickformat="%Y-%m-%d",  # 날짜 형식 지정
                showgrid=True,
            ),
            yaxis=dict(showgrid=True, tickformat="$,.0f"),  # 달러 형식으로 표시
        )

    return fig


def create_portfolio_allocation_chart(backtest_data: Dict[str, Any]) -> go.Figure:
    """포트폴리오 자산 배분 차트 생성 (스택형 영역 차트)"""
    fig = go.Figure()

    # 데이터 구조 확인 및 allocations 키 찾기
    allocations = None
    dates = backtest_data.get("allocation_dates", [])  # 배분 전용 날짜 사용

    # 배분 전용 날짜가 없으면 일반 날짜 사용
    if not dates:
        dates = backtest_data.get("dates", [])

    # 여러 가능한 키를 확인
    if backtest_data.get("allocations"):
        allocations = backtest_data["allocations"]
    elif backtest_data.get("portfolio_allocations"):
        allocations = backtest_data["portfolio_allocations"]

    # 디버깅을 위한 로그 출력
    # logger.info(f"백테스트 데이터 키들: {list(backtest_data.keys())}")
    # logger.info(f"할당 데이터 존재 여부: {allocations is not None}")
    # if allocations:
    #     logger.info(f"할당 데이터 길이: {len(allocations)}")
    #     if len(allocations) > 0:
    #         logger.info(f"첫 번째 할당 데이터 예시: {allocations[0]}")

    if not allocations or not dates or len(allocations) == 0:
        # 빈 차트
        fig.update_layout(
            title="포트폴리오 자산 배분 (데이터 없음)",
            xaxis_title="날짜",
            yaxis_title="비율 (%)",
            height=400,
            showlegend=True,
            template="plotly_white",
        )
        return fig

    # 자산별 색상 정의 (더 명확하고 구분되는 선명한 색상 사용)
    # 회색 계열 제거, 모든 색상을 뚜렷하게 구분
    asset_colors = {
        "QQQ": "#0066CC",  # 선명한 파란색 (기존보다 더 밝고 명확)
        "SPY": "#FF6600",  # 선명한 주황색 (기존보다 더 밝고 명확)
        "EWY": "#00AA44",  # 선명한 녹색 (기존보다 더 밝고 명확)
        "HYG": "#CC0000",  # 선명한 빨간색 (기존보다 더 밝고 명확)
        "VEA": "#00CCCC",  # 선명한 청록색
        "Cash": "#9933CC",  # 선명한 보라색 (영문, 기존보다 더 밝고 명확)
        "현금": "#9933CC",  # 선명한 보라색 (한글, 기존보다 더 밝고 명확)
    }

    # 기본 색상 팔레트 (자산 이름이 딕셔너리에 없을 경우 사용)
    # 회색 계열 완전 제거, 모든 색상을 뚜렷하고 선명하게
    default_color_palette = [
        "#FF33CC",  # 선명한 분홍색
        "#FFCC00",  # 선명한 노란색
        "#00FF99",  # 선명한 민트색
        "#FF3366",  # 선명한 로즈색
        "#33CCFF",  # 선명한 하늘색
        "#CC66FF",  # 선명한 라벤더색
        "#FF9933",  # 선명한 오렌지색
        "#66FF33",  # 선명한 라임색
        "#FF0066",  # 선명한 핑크색
        "#0099FF",  # 선명한 시안색
    ]

    if allocations and len(allocations) > 0:
        # 각 자산별로 데이터 추출
        asset_names = list(allocations[0].keys()) if allocations else []
        # logger.info(f"자산 이름들: {asset_names}")

        # 자산 이름별로 색상 인덱스 추적 (기본 색상 팔레트 사용을 위해)
        asset_color_index = {}
        default_color_idx = 0

        for asset_name in asset_names:
            asset_values = []
            valid_dates = []

            # 배분 데이터와 날짜 데이터의 길이가 같은지 확인
            max_len = min(len(allocations), len(dates))

            for i in range(max_len):
                allocation = allocations[i]
                if asset_name in allocation:
                    # 값이 비율(0-1)인지 퍼센트(0-100)인지 확인하고 변환
                    value = float(allocation[asset_name])
                    if value <= 1.0:  # 0-1 범위라면 퍼센트로 변환
                        value = value * 100
                    asset_values.append(value)
                    valid_dates.append(dates[i])

            if asset_values:  # 데이터가 있는 경우만 추가
                # 색상 할당: 딕셔너리에 있으면 사용, 없으면 기본 팔레트에서 순차적으로 할당
                if asset_name in asset_colors:
                    color = asset_colors[asset_name]
                else:
                    # 기본 색상 팔레트에서 순차적으로 할당
                    if asset_name not in asset_color_index:
                        asset_color_index[asset_name] = default_color_idx
                        default_color_idx = (default_color_idx + 1) % len(
                            default_color_palette
                        )
                    color = default_color_palette[asset_color_index[asset_name]]

                # logger.info(
                #     f"{asset_name}: {len(asset_values)}개 데이터 포인트, 첫 번째 값: {asset_values[0]:.2f}%, 색상: {color}")

                # 마지막 몇 개 값도 로깅하여 변화 확인
                # if len(asset_values) > 5:
                #     recent_values = asset_values[-5:]
                #     logger.info(f"{asset_name} 최근 5개 값: {[f'{v:.2f}%' for v in recent_values]}")

                fig.add_trace(
                    go.Scatter(
                        x=valid_dates,
                        y=asset_values,
                        mode="lines",
                        stackgroup="one",  # 스택형 차트
                        name=asset_name,
                        line=dict(color=color, width=0),
                        fillcolor=color,
                        hovertemplate=f"{asset_name}: %{{y:.1f}}%<br>날짜: %{{x}}<extra></extra>",
                    )
                )

    # 레이아웃 설정
    fig.update_layout(
        title="포트폴리오 자산 배분 추이",
        xaxis_title="날짜",
        yaxis_title="비율 (%)",
        xaxis=dict(
            type="date",  # 날짜 타입으로 명시적 설정
            tickformat="%Y-%m-%d",  # 날짜 형식 지정
            showgrid=True,
        ),
        yaxis=dict(range=[0, 100], tickformat=".0f", ticksuffix="%"),
        height=400,
        showlegend=True,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


def create_annualized_returns_chart(backtest_data: Dict[str, Any]) -> go.Figure:
    """연환산 수익률 차트 생성 - 강화학습 vs 균등투자 비교"""
    fig = go.Figure()

    if not backtest_data.get("returns_data"):
        # 빈 차트
        fig.update_layout(
            title="연환산 수익률 비교 (데이터 없음)",
            xaxis_title="날짜",
            yaxis_title="연환산 수익률 (%)",
            height=400,
            showlegend=True,
            template="plotly_white",
            margin=dict(l=50, r=50, t=50, b=50),
        )
        return fig

    dates = backtest_data.get("dates", [])
    returns_data = backtest_data["returns_data"]
    rl_annualized_returns = returns_data.get("annualized_returns", [])

    if not rl_annualized_returns:
        fig.update_layout(
            title="연환산 수익률 비교 (데이터 없음)",
            xaxis_title="날짜",
            yaxis_title="연환산 수익률 (%)",
            height=400,
            showlegend=True,
            template="plotly_white",
            margin=dict(l=50, r=50, t=50, b=50),
        )
        return fig

    # 강화학습 연환산 수익률 차트 - 30일 이후 데이터만 사용
    valid_dates = []
    valid_returns = []
    for i, (date, return_val) in enumerate(zip(dates, rl_annualized_returns)):
        if i >= 30:  # 30일 이후 데이터만 사용 (연환산 수익률이 의미가 있는 구간)
            valid_dates.append(date)
            valid_returns.append(return_val)

    if valid_dates and valid_returns:
        fig.add_trace(
            go.Scatter(
                x=valid_dates,
                y=valid_returns,
                name="강화학습 전략",
                line=dict(color="#17a2b8", width=3),
                fill="tonexty",
                fillcolor="rgba(23, 162, 184, 0.1)",
                hovertemplate="날짜: %{x}<br>강화학습 연환산 수익률: %{y:.2f}%<extra></extra>",
            )
        )

    # 균등투자 전략 데이터가 있는 경우 추가 - 30일 이후 데이터만 사용
    equal_strategy = backtest_data.get("equal_strategy")
    if equal_strategy and equal_strategy.get("annualized_returns"):
        equal_annualized_returns = equal_strategy["annualized_returns"]
        equal_dates = equal_strategy.get("dates", dates)

        # 균등투자 전략도 30일 이후 데이터만 사용
        equal_valid_dates = []
        equal_valid_returns = []
        for i, (date, return_val) in enumerate(
            zip(equal_dates, equal_annualized_returns)
        ):
            if i >= 30:  # 30일 이후 데이터만 사용
                equal_valid_dates.append(date)
                equal_valid_returns.append(return_val)

        if equal_valid_dates and equal_valid_returns:
            fig.add_trace(
                go.Scatter(
                    x=equal_valid_dates,
                    y=equal_valid_returns,
                    name="균등투자 전략",
                    line=dict(color="#28a745", width=3, dash="dash"),
                    hovertemplate="날짜: %{x}<br>균등투자 연환산 수익률: %{y:.2f}%<extra></extra>",
                )
            )

    # 0% 기준선 추가
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
        annotation_text="기준점 (0%)",
        annotation_position="bottom right",
    )

    # 레이아웃 설정
    fig.update_layout(
        title="연환산 수익률 비교 - 강화학습 vs 균등투자",
        xaxis_title="날짜",
        yaxis_title="연환산 수익률 (%)",
        height=400,
        showlegend=True,
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_cumulative_returns_chart(backtest_data: Dict[str, Any]) -> go.Figure:
    """누적 수익률 차트 생성 - 강화학습 vs 균등투자 비교"""
    fig = go.Figure()

    if not backtest_data.get("returns_data"):
        # 빈 차트
        fig.update_layout(
            title="누적 수익률 비교 (데이터 없음)",
            xaxis_title="날짜",
            yaxis_title="누적 수익률 (%)",
            height=400,
            showlegend=True,
            template="plotly_white",
            margin=dict(l=50, r=50, t=50, b=50),
        )
        return fig

    dates = backtest_data.get("dates", [])
    returns_data = backtest_data["returns_data"]
    rl_cumulative_returns = returns_data.get("cumulative_returns", [])

    if not rl_cumulative_returns:
        fig.update_layout(
            title="누적 수익률 비교 (데이터 없음)",
            xaxis_title="날짜",
            yaxis_title="누적 수익률 (%)",
            height=400,
            showlegend=True,
            template="plotly_white",
            margin=dict(l=50, r=50, t=50, b=50),
        )
        return fig

    # 강화학습 누적 수익률 차트
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rl_cumulative_returns,
            name="강화학습 전략",
            line=dict(color="#17a2b8", width=3),
            fill="tonexty",
            fillcolor="rgba(23, 162, 184, 0.1)",
            hovertemplate="날짜: %{x}<br>강화학습 누적 수익률: %{y:.2f}%<extra></extra>",
        )
    )

    # 균등투자 전략 데이터가 있는 경우 추가
    equal_strategy = backtest_data.get("equal_strategy")
    if equal_strategy and equal_strategy.get("cumulative_returns"):
        equal_cumulative_returns = equal_strategy["cumulative_returns"]
        equal_dates = equal_strategy.get("dates", dates)

        fig.add_trace(
            go.Scatter(
                x=equal_dates,
                y=equal_cumulative_returns,
                name="균등투자 전략",
                line=dict(color="#28a745", width=3, dash="dash"),
                hovertemplate="날짜: %{x}<br>균등투자 누적 수익률: %{y:.2f}%<extra></extra>",
            )
        )

    # 0% 기준선 추가
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
        annotation_text="기준점 (0%)",
        annotation_position="bottom right",
    )

    # 레이아웃 설정
    fig.update_layout(
        title="누적 수익률 비교 - 강화학습 vs 균등투자",
        xaxis_title="날짜",
        yaxis_title="누적 수익률 (%)",
        height=400,
        showlegend=True,
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
