"""
Dash 강화학습 트레이딩 봇 - 구조화된 메인 파일
재구성된 모듈들을 사용하여 탭 기반 구조로 정리
환경변수 기반 설정 시스템 적용
"""

from src.utils.logger import get_logger
from src.dash_layouts import (
    create_header, create_main_tabs, create_training_config_modal,
    create_backtest_config_modal, create_hidden_components
)
from src.dash_utils import CUSTOM_CSS
from src.dash_manager import CompleteDashManager
from src.callbacks import register_all_callbacks
from src.dash_config import get_config
import os
import sys
import dash
from dash import dcc
import dash_bootstrap_components as dbc

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 설정 로드 (환경변수 DASH_CONFIG_FILE 기반)
config = get_config()

# 로거 설정
logger = get_logger("dash_main")

# 전역 관리자 인스턴스
dash_manager = CompleteDashManager()

# 🎨 Bootstrap 테마 적용
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        ("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.7.2/"
         "font/bootstrap-icons.css"),
        ("https://fonts.googleapis.com/css2?"
         "family=Inter:wght@300;400;500;600;700&display=swap")
    ],
    suppress_callback_exceptions=True,
    title="🤖 DDPG 포트폴리오 최적화 시스템"
)

# 📱 메인 레이아웃
app.layout = dbc.Container([
    # 📝 커스텀 CSS
    dcc.Store(id="custom-css", data=CUSTOM_CSS),

    # 🎯 헤더
    create_header(),

    # 📱 메인 탭들
    create_main_tabs(),

    # ⚙️ 학습 설정 모달
    create_training_config_modal(),

    # 📊 백테스트 설정 모달
    create_backtest_config_modal(),

    # 🔄 숨겨진 컴포넌트들
    *create_hidden_components(dash_manager)

], fluid=True, style={
    "background": "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
    "min-height": "100vh"
})


# 콜백 함수 등록
register_all_callbacks(app, dash_manager)


# 🚀 메인 실행 함수
def run_complete_dash_app():
    """완전한 Dash 앱 실행 - 환경변수 기반 설정"""
    try:
        logger.info("🚀 DDPG 포트폴리오 최적화 시스템 시작...")

        # 설정 요약 출력
        config.print_config_summary()

        # 서버 정보 출력
        print("🤖 DDPG 포트폴리오 최적화 시스템 (탭 기반 구조)")
        print("="*60)
        print("🚀 학습 탭: 실시간 포트폴리오 모니터링")
        print("📈 백테스팅 탭: DDPG 모델 성과 분석")
        print("📊 모니터링 탭: 시스템 상태 모니터링")
        print("🔧 완전히 구조화된 탭 기반 코드베이스")
        print("="*60)
        print(f"🌐 로컬 URL: http://127.0.0.1:{config.port}")
        print(f"🌍 네트워크 URL: http://{config.host}:{config.port}")
        print(f"📱 모바일: http://[로컬IP]:{config.port}")
        print("="*60)

        # 환경변수 기반 설정으로 앱 실행
        run_config = config.get_dash_run_config()
        logger.info(f"🎛️ 앱 실행 설정: {run_config}")

        app.run(**run_config)

    except Exception as e:
        logger.error(f"❌ 서버 시작 실패: {e}")
        raise


if __name__ == "__main__":
    run_complete_dash_app()
