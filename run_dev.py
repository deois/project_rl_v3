#!/usr/bin/env python3
"""
개발 모드로 Dash 앱 실행
자동 리로드 및 개발 도구가 활성화된 상태로 실행됩니다.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 개발 환경 설정 파일 지정
os.environ['DASH_CONFIG_FILE'] = 'config.development.env'

# 실행 타임스탬프 생성 및 환경변수로 설정
startup_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.environ['APP_STARTUP_TIMESTAMP'] = startup_timestamp

# 메인 앱 모듈에서 설정을 다시 로드하도록 환경변수 설정
if __name__ == "__main__":
    # logs 디렉토리 생성
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    # 앱 실행 전용 로거 설정
    from src.utils.logger import get_logger, get_current_log_file
    app_logger = get_logger(f"dev_startup_{startup_timestamp}")

    # 통합 로그 파일 경로
    unified_log_file = get_current_log_file()

    # HTTP 요청 로그 억제 설정
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)

    # 시작 정보 로그 기록
    app_logger.info("🚀 개발 모드로 Dash 앱을 시작합니다...")
    app_logger.info(f"📁 설정 파일: config.development.env")
    app_logger.info(f"🔄 자동 리로드: 활성화")
    app_logger.info(f"🛠️ 개발 도구: 활성화")
    app_logger.info(f"🔇 HTTP 요청 로그: 비활성화")
    app_logger.info(f"⏰ 시작 시간: {startup_timestamp}")
    app_logger.info(f"📄 통합 로그 파일: {unified_log_file}")

    print("🚀 개발 모드로 Dash 앱을 시작합니다...")
    print("📁 설정 파일: config.development.env")
    print("🔄 자동 리로드: 활성화")
    print("🛠️ 개발 도구: 활성화")
    print("🔇 HTTP 요청 로그: 비활성화")
    print(f"⏰ 시작 시간: {startup_timestamp}")
    print(f"📄 통합 로그 파일: {unified_log_file}")
    print("-" * 50)

    try:
        # 메인 앱 실행
        from dash_interface_complete_refactored import run_complete_dash_app
        app_logger.info("✅ Dash 앱 모듈 로드 완료")
        run_complete_dash_app()
    except Exception as e:
        app_logger.error(f"❌ 앱 실행 중 오류 발생: {str(e)}")
        print(f"❌ 앱 실행 중 오류 발생: {str(e)}")
        raise
    finally:
        app_logger.info("�� 개발 모드 Dash 앱 종료")
