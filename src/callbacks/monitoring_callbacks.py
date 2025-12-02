"""
모니터링 관련 콜백 함수들
시스템 상태, 메모리 사용량, GPU 온도, 모니터링 차트
"""

import time
import datetime
from typing import Tuple
from dash import Input, Output
import plotly.graph_objs as go

from src.utils.logger import get_logger

logger = get_logger("monitoring_callbacks")


def register_monitoring_callbacks(app, dash_manager):
    """모니터링 관련 콜백 함수들을 등록"""

    @app.callback(
        [Output("system-status", "children"),
         Output("uptime", "children"),
         Output("memory-usage", "children"),
         Output("gpu-temp", "children")],
        [Input("monitoring-interval", "n_intervals")]
    )
    def update_monitoring_metrics(n_intervals: int) -> Tuple[str, str, str, str]:
        """모니터링 메트릭 업데이트"""
        import psutil

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
            uptime_seconds = time.time() - dash_manager.training_status.get("start_time", time.time())
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
        [Input("monitoring-interval", "n_intervals")]
    )
    def update_monitoring_chart(n_intervals: int):
        """시스템 모니터링 차트 업데이트"""
        import psutil

        try:
            # CPU 사용률 가져오기
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent

            # 간단한 게이지 차트 생성
            fig = go.Figure()

            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=cpu_percent,
                domain={'x': [0, 0.5], 'y': [0, 1]},
                title={'text': "CPU 사용률 (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))

            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=memory_percent,
                domain={'x': [0.5, 1], 'y': [0, 1]},
                title={'text': "메모리 사용률 (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))

            fig.update_layout(
                title="실시간 시스템 모니터링",
                height=400
            )

            return fig
        except:
            # 오류 시 빈 차트 반환
            return go.Figure().add_annotation(
                text="모니터링 데이터를 사용할 수 없습니다",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
