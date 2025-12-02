"""
Dash 애플리케이션 설정 관리 모듈
환경변수를 통한 설정 제어 및 개발/운영 모드 분리
"""

import os
from typing import Dict, Any, Union, Optional
from pathlib import Path


def str_to_bool(value: Union[str, bool]) -> bool:
    """문자열을 불린값으로 변환"""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('true', '1', 'yes', 'on', 'enabled')


def load_env_file(env_file: str = "config.env") -> None:
    """환경변수 파일 로드"""
    env_path = Path(env_file)
    if env_path.exists():
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # 값에서 인라인 주석 제거 (# 이후 모든 내용 제거)
                        if '#' in value:
                            value = value.split('#')[0].strip()
                        # 기존 환경변수가 없는 경우에만 설정
                        if key not in os.environ:
                            os.environ[key] = value
        except Exception as e:
            print(f"⚠️ 환경변수 파일 로드 실패 ({env_file}): {e}")


class DashConfig:
    """Dash 애플리케이션 설정 클래스"""

    def __init__(self, env_file: Optional[str] = None):
        """
        Args:
            env_file: 환경변수 설정 파일 경로 (None일 때 자동 선택)
        """
        # 환경변수에서 설정 파일 경로 확인
        if env_file is None:
            env_file = os.getenv('DASH_CONFIG_FILE', 'config.env')

        # 환경변수 파일 로드
        load_env_file(env_file)

        # Dash 관련 설정
        self.debug = str_to_bool(os.getenv('DASH_DEBUG', 'false'))
        self.auto_reload = str_to_bool(os.getenv('DASH_AUTO_RELOAD', 'true'))
        self.dev_tools_ui = str_to_bool(os.getenv('DASH_DEV_TOOLS_UI', 'true'))
        self.dev_tools_props_check = str_to_bool(os.getenv('DASH_DEV_TOOLS_PROPS_CHECK', 'true'))
        self.hot_reload = str_to_bool(os.getenv('DASH_HOT_RELOAD', 'true'))
        self.serve_dev_bundles = str_to_bool(os.getenv('DASH_SERVE_DEV_BUNDLES', 'true'))

        # 서버 설정
        self.host = os.getenv('DASH_HOST', '0.0.0.0')
        self.port = int(os.getenv('DASH_PORT', '8050'))

        # API 키 설정
        self.api_key_fred = os.getenv('API_KEY_FRED', '')
        self.telegram_api_key = os.getenv('TELEGRAM_API_KEY', '')
        self.telegram_chat_id = os.getenv('TELECRAM_CHAT_ID', '')

    def get_dash_run_config(self) -> Dict[str, Any]:
        """Dash app.run() 메서드에 사용할 설정 반환"""
        config = {
            'host': self.host,
            'port': self.port,
            'debug': self.debug,
        }

        # debug 모드일 때만 개발 도구 설정 적용
        if self.debug:
            config.update({
                'dev_tools_ui': self.dev_tools_ui,
                'dev_tools_props_check': self.dev_tools_props_check,
                'dev_tools_hot_reload': self.auto_reload and self.hot_reload,
                'dev_tools_serve_dev_bundles': self.serve_dev_bundles,
            })

        return config

    def print_config_summary(self) -> None:
        """현재 설정 요약 출력"""
        print("\n" + "="*60)
        print("🔧 DASH 애플리케이션 설정")
        print("="*60)
        print(f"🌐 서버: {self.host}:{self.port}")
        print(f"🐛 디버그 모드: {'✅' if self.debug else '❌'}")
        print(f"🔄 자동 리로드: {'✅' if self.auto_reload else '❌'}")
        print(f"🛠️ 개발 도구 UI: {'✅' if self.dev_tools_ui else '❌'}")
        print(f"📊 속성 검사: {'✅' if self.dev_tools_props_check else '❌'}")
        print(f"🔥 핫 리로드: {'✅' if self.hot_reload else '❌'}")
        print(f"📦 개발 번들 제공: {'✅' if self.serve_dev_bundles else '❌'}")
        print("="*60)

        # 개발 모드 안내
        if self.debug:
            print("💡 개발 모드: 파일 변경 시 자동 리로드됩니다")
            if not self.auto_reload:
                print("⚠️ 자동 리로드가 비활성화되었습니다")
        else:
            print("🚀 운영 모드: 최적화된 성능으로 실행됩니다")
        print("="*60)


def get_config(env_file: Optional[str] = None) -> DashConfig:
    """설정 인스턴스 반환"""
    return DashConfig(env_file)
