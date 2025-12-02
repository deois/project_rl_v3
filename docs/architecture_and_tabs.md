# 프로젝트 구조 및 탭 문서

## 🏗️ 프로젝트 구조

```text
/
├── dash_interface_complete_refactored.py  # 메인 애플리케이션
├── run_dev.py                            # 개발 모드 실행 스크립트
├── run_prod.py                           # 운영 모드 실행 스크립트
├── config.env                            # 기본 환경 설정
├── config.development.env                # 개발 환경 설정
├── config.production.env                 # 운영 환경 설정
├── src/                                   # 모듈화된 소스 코드
│   ├── __init__.py
│   ├── dash_config.py                    # 🆕 환경변수 설정 관리
│   ├── ddpg_algorithm/                   # DDPG 알고리즘 모듈
│   │   ├── __init__.py
│   │   ├── ddpg_agent.py                # DDPGAgent 클래스
│   │   ├── ddpg_models.py               # Actor/Critic 신경망 모델
│   │   └── ddpg_noise.py                # OUNoise 클래스
│   ├── utils/                            # 🆕 유틸리티 모듈
│   │   ├── __init__.py
│   │   └── logger.py                    # 로깅 시스템
│   ├── data/                             # 🆕 데이터 처리 모듈
│   │   ├── __init__.py
│   │   ├── merge.py                     # 데이터 병합
│   │   └── loader.py                    # 데이터 로더 (Yahoo Finance, FRED)
│   ├── environment/                      # 🆕 거래 환경 모듈
│   │   ├── __init__.py
│   │   ├── common.py                    # 기본 거래 환경
│   │   └── trading_env.py               # DDPG용 거래 환경
│   ├── dash_layouts.py                   # UI 레이아웃 (탭 기반)
│   ├── dash_callbacks.py                 # 콜백 함수들
│   ├── dash_charts.py                    # 차트 생성
│   ├── dash_manager.py                   # 상태 관리
│   ├── dash_simulation.py                # 시뮬레이션
│   ├── dash_training_integration.py      # 실제 학습 통합
│   └── dash_utils.py                     # 유틸리티 함수
├── README_TAB_STRUCTURE.md               # 탭 구조 문서
└── README_ENV_CONFIG.md                  # 🆕 환경변수 설정 가이드
```

### 🏗️ 모듈 구조

```text
src/
├── utils/                    # 유틸리티 모듈
│   ├── __init__.py
│   └── logger.py            # 고유 로거 인스턴스 생성 및 관리
├── data/                     # 데이터 처리 모듈
│   ├── __init__.py
│   ├── loader.py            # Yahoo Finance, FRED, FinanceDataReader 데이터 수집
│   └── merge.py             # 다양한 소스의 금융 데이터 병합
├── environment/              # 거래 환경 모듈
│   ├── __init__.py
│   ├── common.py            # OpenAI Gym 기반 기본 거래 환경
│   └── trading_env.py       # DDPG용 MinMaxScaler 정규화 환경
└── ddpg_algorithm/           # DDPG 알고리즘 모듈
    ├── __init__.py
    ├── ddpg_agent.py        # DDPGAgent 클래스
    ├── ddpg_models.py       # Actor/Critic 신경망 모델
    └── ddpg_noise.py        # OUNoise 클래스
```

#### ✅ 모듈 통합의 장점

- **일관된 import 경로**: 모든 모듈이 `src.` 접두사로 시작
- **명확한 책임 분리**: 기능별로 명확하게 구분된 모듈
- **향상된 가독성**: 직관적인 모듈명과 구조
- **쉬운 유지보수**: 관련 기능들의 논리적 그룹화
- **확장성**: 새로운 기능 추가 시 적절한 모듈 위치 명확

#### 🔧 Import 경로 업데이트

모든 기존 import 구문이 새로운 경로로 자동 업데이트되었습니다:

```python
# 이전
from sub.log import get_logger
from sub_data.merge import load_merged_data_v1
from sub_enviroment.trading_enviroment_079 import TradingEnvironment

# 현재
from src.utils.logger import get_logger
from src.data.merge import load_merged_data_v1
from src.environment.trading_env import TradingEnvironment
```

## 📱 탭 상세 설명

### 🚀 강화학습 탭 (Training Tab)

**목적**: 단일 DDPG 모델의 학습·탐험·체크포인트를 실시간 관리

**주요 구성요소**:

- **모드 선택**: 시뮬레이션, 실제 학습, 모델 재개
- **파라미터 모달**: 에피소드 수, 배치 크기, Actor/Critic LR, ETF 조합, 윈도우 크기 등 통합 설정
- **메트릭 카드**: 학습 상태, 현재 에피소드/스텝, 누적 보상, 포트폴리오 가치
- **컨트롤 패널**: 학습 시작/중지, 체크포인트 저장/로드, 로그 초기화
- **성과 차트**: 포트폴리오 가치, 보상, 정책/가치 손실
- **실시간 로그**: 에이전트 이벤트, 예외, 데이터 큐 상태

**사용법**:

1. ETF 4종을 선택하고 하이퍼파라미터를 조정
2. "학습 시작" 버튼으로 에이전트 실행
3. 차트·메트릭·로그로 안정성 확인
4. 필요 시 "학습 중지"로 안전하게 종료 후 체크포인트 저장

![강화학습 탭 화면](images/page_agent.JPG)

*강화학습 탭의 실제 화면. DDPG 에이전트의 학습 파라미터 설정, 실시간 메트릭 모니터링, 성과 차트 및 손실 그래프를 확인할 수 있습니다.*

### 📈 백테스팅 탭 (Backtesting Tab)

**목적**: 동일한 DDPG 정책으로 시뮬레이션 시장 데이터를 재생하고 KPI를 확인

**주요 구성요소**:

- **백테스트 설정 카드**: 체크포인트 선택, 기간·초기자본·수수료 입력
- **실행 제어**: 백테스트 시작/취소, 고급 설정 모달
- **상태 모니터링**: 진행률, 실행 상태, 작업 ID, 남은 시간 추정
- **결과 섹션**: 포트폴리오 가치, MDD, 샤프 비율, 분기별 수익률 차트
- **로그 패널**: 리밸런싱 이벤트와 에러 확인

**사용법**:

1. 학습 탭에서 저장한 모델을 선택
2. 기간·ETF·초기 자본을 지정
3. "백테스트 시작"을 누르고 진행률과 차트를 관찰
4. 결과 카드/차트/로그로 전략 적합성을 평가

![백테스팅 탭 화면](images/page_backtest.JPG)

*백테스팅 탭의 실제 화면. 체크포인트 선택, 백테스트 설정, 진행 상황 모니터링 및 결과 분석을 수행할 수 있습니다.*

### 📊 모니터링 탭 (Monitoring Tab)

**목적**: 학습/백테스트 중 발생하는 시스템 부하를 추적하고 병목을 빠르게 감지

**주요 구성요소**:

- **시스템 메트릭 카드**: CPU, RAM, 디스크, 업타임
- **GPU 패널**: GPU 사용률/온도/메모리(사용 가능한 경우)
- **실시간 차트**: CPU/메모리 게이지, 프로세스별 사용률
- **로그 뷰어**: `src/utils/logger`에서 수집한 운영 로그

**사용법**:

1. 탭 진입과 동시에 Interval 기반 모니터링이 동작
2. 자원 한계가 감지되면 학습/백테스트 탭에서 즉시 조정
3. 로그를 통해 비정상 상태를 빠르게 역추적

![모니터링 탭 화면](images/page_util.JPG)

*모니터링 탭의 실제 화면. 시스템 리소스(CPU, 메모리, 디스크) 사용률, GPU 상태, 실시간 차트 및 로그를 확인할 수 있습니다.*
