"""
로깅 시스템
고유한 로거 인스턴스 생성 및 관리 - 통합 로그 파일 사용
"""

import datetime
import logging
import os
from typing import Dict, Optional


# 전역 로거 레지스트리 (중복 방지)
_logger_registry: Dict[str, logging.Logger] = {}

# 통합 로그 파일 핸들러 (전역 공유)
_shared_file_handler: Optional[logging.FileHandler] = None
_shared_console_handler: Optional[logging.StreamHandler] = None


def _get_unified_log_filename() -> str:
    """통합 로그 파일명 생성"""
    # 환경변수에서 앱 시작 타임스탬프 가져오기 (run_dev.py, run_prod.py에서 설정)
    startup_timestamp = os.environ.get('APP_STARTUP_TIMESTAMP')

    if startup_timestamp:
        # 개발/운영 모드 구분
        if 'DASH_CONFIG_FILE' in os.environ:
            config_file = os.environ['DASH_CONFIG_FILE']
            if 'development' in config_file:
                mode = 'dev'
            elif 'production' in config_file:
                mode = 'prod'
            else:
                mode = 'app'
        else:
            mode = 'app'

        log_filename = f'logs/log_{mode}_unified_{startup_timestamp}.log'
    else:
        # 기본 타임스탬프 사용
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f'logs/log_app_unified_{current_time}.log'

    return log_filename


def _get_shared_handlers() -> tuple[logging.FileHandler, logging.StreamHandler]:
    """공유 핸들러 생성 및 반환"""
    global _shared_file_handler, _shared_console_handler

    if _shared_file_handler is None or _shared_console_handler is None:
        # logs 디렉토리 생성
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # 통합 로그 파일명 생성
        log_filename = _get_unified_log_filename()

        # 파일 핸들러 생성 (UTF-8 인코딩, append 모드)
        _shared_file_handler = logging.FileHandler(log_filename, encoding='utf-8', mode='a')
        _shared_file_handler.setLevel(logging.INFO)

        # 콘솔 핸들러 생성
        _shared_console_handler = logging.StreamHandler()
        _shared_console_handler.setLevel(logging.INFO)

        # 포매터 생성 및 적용 (모듈명 포함)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        _shared_file_handler.setFormatter(formatter)
        _shared_console_handler.setFormatter(formatter)

    return _shared_file_handler, _shared_console_handler


def get_logger(fine_name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    통합 로그 파일을 사용하는 고유한 로거 인스턴스를 생성합니다.

    Args:
        fine_name (str): 로거 이름에 사용될 고유 식별자 (모듈 구분용)
        log_level (int): 로깅 레벨 (기본값: logging.INFO)

    Returns:
        logging.Logger: 설정된 로거 인스턴스
    """
    # 이미 생성된 로거가 있다면 재사용
    if fine_name in _logger_registry:
        return _logger_registry[fine_name]

    # 로거 이름 생성 (모듈 구분을 위한 고유 이름)
    logger_name = f"{fine_name}"
    logger = logging.getLogger(logger_name)

    # 로거 레벨 설정
    logger.setLevel(log_level)

    # 이미 핸들러가 설정되어 있다면 재사용 (중복 핸들러 방지)
    if logger.handlers:
        _logger_registry[fine_name] = logger
        return logger

    # 공유 핸들러 가져오기
    file_handler, console_handler = _get_shared_handlers()

    # 핸들러를 로거에 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 부모 로거로의 전파 방지 (중복 로그 방지)
    logger.propagate = False

    # 전역 레지스트리에 등록
    _logger_registry[fine_name] = logger

    return logger


def clear_logger_registry() -> None:
    """로거 레지스트리를 초기화합니다 (테스트용)"""
    global _logger_registry, _shared_file_handler, _shared_console_handler

    # 기존 핸들러 정리
    if _shared_file_handler:
        _shared_file_handler.close()
        _shared_file_handler = None

    if _shared_console_handler:
        _shared_console_handler = None

    # 레지스트리 초기화
    _logger_registry.clear()


def get_current_log_file() -> str:
    """현재 사용 중인 로그 파일 경로 반환"""
    return _get_unified_log_filename()
