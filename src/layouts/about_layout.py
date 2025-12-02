"""
프로젝트 설명 탭 레이아웃
DDPG 포트폴리오 최적화 시스템에 대한 상세 설명
"""

from dash import html
import dash_bootstrap_components as dbc
from src.dash_utils import CARD_STYLE


def create_about_content() -> list:
    """프로젝트 설명 탭 콘텐츠 생성"""
    return [
        # 전체 프로젝트 개요
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H4(
                                            [
                                                html.I(className="bi bi-bullseye me-2"),
                                                "문제 정의 및 최적화 목표",
                                            ],
                                            className="mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            "강화학습 파이프라인은 장기 ETF 포트폴리오 가치를 극대화한다는 명확한 목표 하에 학습·검증·백테스트 단계를 동일 자산 구성과 하이퍼파라미터로 유지하도록 설계되어 있습니다. 초기 자본 $300, 월 투자금 $300으로 시작하여 약 30일마다 포트폴리오를 리밸런싱하며, 각 자산과 현금의 최소 비중 7.5% 제약을 보장하는 Softmax-Affine 변환을 통해 강화학습 기반 동적 자산배분 전략과 균등 투자 전략을 동일 조건에서 비교합니다.",
                                            className="mb-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "🎯 최적화 대상",
                                                            className="text-primary mb-2",
                                                        ),
                                                        html.Ul(
                                                            [
                                                                html.Li(
                                                                    "ETF 4종 + 현금 비중을 약 30일(1개월)마다 동적으로 리밸런싱 (각 자산 및 현금 최소 비중 7.5% 제약 보장)"
                                                                ),
                                                                html.Li(
                                                                    "초기 자본 $300, 월 투자금 $300, 거래 수수료(매도 0.265%, 매수 0.015%)를 포함한 실제 시나리오 반영"
                                                                ),
                                                                html.Li(
                                                                    "강화학습 기반 전략 vs 균등 투자 전략(각 ETF 25%, 현금 0%)을 동일 조건에서 비교"
                                                                ),
                                                                html.Li(
                                                                    "샤프 비율·위험 조정 수익률·장기 성장률을 핵심 성능 지표로 사용"
                                                                ),
                                                            ],
                                                            className="small",
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "🔗 파이프라인 일관성",
                                                            className="text-success mb-2",
                                                        ),
                                                        html.Ul(
                                                            [
                                                                html.Li(
                                                                    "학습, 체크포인트 저장, 백테스트 워커가 동일 구성(`training-config-store`, `backtest-config-store`)을 공유"
                                                                ),
                                                                html.Li(
                                                                    "DashRealTrainingManager가 생성한 모델은 즉시 DashBacktestManager에서 재사용"
                                                                ),
                                                                html.Li(
                                                                    "로그·KPI·차트 업데이트가 모두 동일 모델 상태를 참조하여 운영 오류를 최소화"
                                                                ),
                                                            ],
                                                            className="small",
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            style=CARD_STYLE,
                        )
                    ]
                )
            ],
            className="mb-4",
        ),
        # State / Action / Reward 정의
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H4(
                                            [
                                                html.I(
                                                    className="bi bi-diagram-3 me-2"
                                                ),
                                                "State / Action / Reward 정의",
                                            ],
                                            className="mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "📡 State",
                                                            className="text-primary mb-2",
                                                        ),
                                                        html.Ul(
                                                            [
                                                                html.Li(
                                                                    "관찰 공간: (정규화된 시계열 × window_size) + 포트폴리오 비율(n_assets+1) + 현금 + 보유 주식수(n_assets) + 총가치"
                                                                ),
                                                                html.Li(
                                                                    "MinMaxScaler로 window_size일 히스토리를 정규화 (Dash 기본값 60일, 최소 30일)"
                                                                ),
                                                                html.Li(
                                                                    "TradingEnvironment가 `_next_observation()`에서 시계열·비중·잔액·보유량·총가치를 하나의 벡터로 결합"
                                                                ),
                                                            ],
                                                            className="small",
                                                        ),
                                                    ],
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "🎛️ Action",
                                                            className="text-success mb-2",
                                                        ),
                                                        html.Ul(
                                                            [
                                                                html.Li(
                                                                    "공간: `spaces.Box(low=0, high=1, shape=(n_assets+1,))` (ETF 4종 + 현금)"
                                                                ),
                                                                html.Li(
                                                                    "Actor가 Softmax-Affine 변환을 통해 출력한 연속 비율은 최소 비중 7.5% 제약을 자동으로 보장하며, 노이즈 추가 후에도 이 제약을 유지하도록 클리핑 및 정규화"
                                                                ),
                                                                html.Li(
                                                                    "`DashRealTrainingManager._run_episode_steps`에서 월별 리밸런싱에 적용, 매수 0.015% / 매도 0.265% 수수료 반영"
                                                                ),
                                                            ],
                                                            className="small",
                                                        ),
                                                    ],
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "💎 Reward",
                                                            className="text-warning mb-2",
                                                        ),
                                                        html.Ul(
                                                            [
                                                                html.Li(
                                                                    "30일마다 리밸런싱 발생: 에이전트와 균등 투자 전략 모두 독립적으로 월 투자금($300) 추가"
                                                                ),
                                                                html.Li(
                                                                    "리밸런싱 프로세스: 매도(먼저) → 매수(나중) 순서로 주식 수 조정, 거래 수수료 차감"
                                                                ),
                                                                html.Li(
                                                                    "`reward_monthly_agent`: 직전 리밸런싱 대비 에이전트 포트폴리오 가치 증분 (%)"
                                                                ),
                                                                html.Li(
                                                                    "`reward_monthly_equal`: 직전 리밸런싱 대비 균등 투자 포트폴리오 가치 증분 (%)"
                                                                ),
                                                                html.Li(
                                                                    "리밸런싱 스텝만 `verification=True`로 표시되어 리플레이 버퍼에 저장, 절대/상대 성과를 동시에 최적화"
                                                                ),
                                                            ],
                                                            className="small mb-0",
                                                        ),
                                                    ],
                                                    md=4,
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                            ],
                            style=CARD_STYLE,
                        )
                    ]
                )
            ],
            className="mb-4",
        ),
        # DDPG 기법 선택 근거
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H4(
                                            [
                                                html.I(className="bi bi-cpu me-2"),
                                                "DDPG 기법 선택 근거",
                                            ],
                                            className="mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            "포트폴리오 비중 결정은 연속 행동 공간에서의 최적화 문제이므로, Actor-Critic 기반의 DDPG가 정책 출력과 가치 평가를 동시에 다루기에 적합합니다.",
                                            className="mb-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "🔑 핵심 특징",
                                                            className="text-primary mb-2",
                                                        ),
                                                        html.Ul(
                                                            [
                                                                html.Li(
                                                                    "Actor-Critic 구조: Actor는 ETF 비중을 직접 출력, Critic은 해당 정책의 가치를 평가하여 안정적인 업데이트 제공"
                                                                ),
                                                                html.Li(
                                                                    "연속 행동 공간 지원: ETF 4종 + 현금을 0~1 범위의 연속 값으로 표현"
                                                                ),
                                                                html.Li(
                                                                    "Off-policy 학습: 리플레이 버퍼와 타깃 네트워크로 희소한 월별 보상에도 효율적 학습"
                                                                ),
                                                            ],
                                                            className="small",
                                                        ),
                                                    ],
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "🧠 Actor/Critic 설계",
                                                            className="text-success mb-2",
                                                        ),
                                                        html.Ul(
                                                            [
                                                                html.Li(
                                                                    "Actor: Linear(400) → LayerNorm → ReLU → Linear(300) → LayerNorm → ReLU → Linear(5) → Softmax → Affine Scaling"
                                                                ),
                                                                html.Li(
                                                                    "Critic: 상태·행동 벡터 결합 후 Linear(400) → LayerNorm → ReLU → Linear(300) → LayerNorm → ReLU → Linear(1)"
                                                                ),
                                                                html.Li(
                                                                    "Softmax-Affine 변환으로 자산 비중을 자연스럽게 정규화하며 최소 비중 7.5% 제약을 보장, LayerNorm으로 다양한 스케일의 입력 안정화"
                                                                ),
                                                            ],
                                                            className="small",
                                                        ),
                                                    ],
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "🛡️ 안정화 메커니즘",
                                                            className="text-warning mb-2",
                                                        ),
                                                        html.Ul(
                                                            [
                                                                html.Li(
                                                                    "타깃 네트워크 소프트 업데이트(τ=0.001)로 Q-value 변동 완화"
                                                                ),
                                                                html.Li(
                                                                    "Gradient Clipping, Smooth L1 Loss(옵션)로 이상치 대응"
                                                                ),
                                                                html.Li(
                                                                    "Ornstein-Uhlenbeck 노이즈(θ=0.15, σ=0.2)로 시간상관 탐험 제공"
                                                                ),
                                                            ],
                                                            className="small",
                                                        ),
                                                    ],
                                                    md=4,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            style=CARD_STYLE,
                        )
                    ]
                )
            ],
            className="mb-4",
        ),
        # DDPG 파이프라인 핵심 구성
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H4(
                                            [
                                                html.I(
                                                    className="bi bi-diagram-2 me-2"
                                                ),
                                                "DDPG 강화학습 파이프라인",
                                            ],
                                            className="mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        html.H5(
                                            "🧠 하나의 알고리즘, 세 단계 운영",
                                            className="text-primary mb-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.H6(
                                                                    "1️⃣ 데이터 & 환경",
                                                                    className="text-info mb-2",
                                                                ),
                                                                html.Ul(
                                                                    [
                                                                        html.Li(
                                                                            "Yahoo Finance · FRED · FinanceDataReader 통합"
                                                                        ),
                                                                        html.Li(
                                                                            "MinMax 정규화 및 포트폴리오 상태 피쳐 구성"
                                                                        ),
                                                                        html.Li(
                                                                            "월별 리밸런싱과 거래 비용 시뮬레이션"
                                                                        ),
                                                                    ],
                                                                    className="small mb-0",
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.H6(
                                                                    "2️⃣ 학습 & 탐험",
                                                                    className="text-success mb-2",
                                                                ),
                                                                html.Ul(
                                                                    [
                                                                        html.Li(
                                                                            "Actor/Critic 듀얼 네트워크"
                                                                        ),
                                                                        html.Li(
                                                                            "OUNoise 기반 탐험 전략"
                                                                        ),
                                                                        html.Li(
                                                                            "리플레이 버퍼 및 타깃 네트워크 동기화"
                                                                        ),
                                                                    ],
                                                                    className="small mb-0",
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.H6(
                                                                    "3️⃣ 분석 & 운영",
                                                                    className="text-warning mb-2",
                                                                ),
                                                                html.Ul(
                                                                    [
                                                                        html.Li(
                                                                            "체크포인트 기반 백테스트"
                                                                        ),
                                                                        html.Li(
                                                                            "Dash 탭 전반에서 KPI 공유"
                                                                        ),
                                                                        html.Li(
                                                                            "GPU/시스템 모니터링으로 운영 안전성 확보"
                                                                        ),
                                                                    ],
                                                                    className="small mb-0",
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    md=4,
                                                ),
                                            ]
                                        ),
                                        html.Hr(),
                                        html.Div(
                                            [
                                                html.H6(
                                                    "⚙️ Python 구현 및 실험 파이프라인",
                                                    className="text-primary mb-2",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Ul(
                                                                    [
                                                                        html.Li(
                                                                            "비동기 워커: `DashRealTrainingManager` · `DashBacktestManager`가 별도 스레드에서 학습/백테스트를 실행하고 `threading.Event`로 안전하게 중지"
                                                                        ),
                                                                        html.Li(
                                                                            "학습 워커는 월별 리밸런싱 시점 경험만 리플레이 버퍼(`deque`, maxlen=100000)에 저장 후 배치 학습 수행"
                                                                        ),
                                                                        html.Li(
                                                                            "백테스트 워커는 체크포인트를 로드해 총 수익률/연환산 수익률/최대 낙폭/샤프 비율을 계산"
                                                                        ),
                                                                    ],
                                                                    className="small mb-0",
                                                                )
                                                            ],
                                                            md=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Ul(
                                                                    [
                                                                        html.Li(
                                                                            "데이터 파이프라인: `load_merged_data_v1`가 ETF 가격·시장 지표·거시 지표를 병합하고 ETF 조합별 CSV 캐시 생성"
                                                                        ),
                                                                        html.Li(
                                                                            "모듈화된 콜백: `register_all_callbacks`가 9개 콜백 모듈을 통합 등록하여 `dcc.Store` 기반 상태 공유 및 `dcc.Interval` 주기 갱신"
                                                                        ),
                                                                        html.Li(
                                                                            "통합 로깅: `src/utils/logger.py`가 전역 로거 레지스트리와 UTF-8 로그 파일(`logs/log_{mode}_unified_*.log`)을 관리"
                                                                        ),
                                                                    ],
                                                                    className="small mb-0",
                                                                )
                                                            ],
                                                            md=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Ul(
                                                                    [
                                                                        html.Li(
                                                                            "체크포인트 & 메타데이터: `model/rl_ddpg_{task_id}` / `model/rl_ddpg_latest`에 Actor/Critic 상태와 JSON 메타데이터를 저장"
                                                                        ),
                                                                        html.Li(
                                                                            "Dash UI 실행 스크립트(`run_dev.py`, `run_prod.py`)가 `DASH_CONFIG_FILE`과 `APP_STARTUP_TIMESTAMP`로 환경 설정"
                                                                        ),
                                                                        html.Li(
                                                                            "재현성 보장: 작업 ID 기반 디렉터리와 메타데이터로 동일 ETF 조합·하이퍼파라미터를 손쉽게 복원"
                                                                        ),
                                                                    ],
                                                                    className="small mb-0",
                                                                )
                                                            ],
                                                            md=4,
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            style=CARD_STYLE,
                        )
                    ]
                )
            ],
            className="mb-4",
        ),
        # 실험 결과 및 보고 체계
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H4(
                                            [
                                                html.I(
                                                    className="bi bi-graph-up-arrow me-2"
                                                ),
                                                "실험 결과 및 보고 체계",
                                            ],
                                            className="mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.H6(
                                                                    "📈 학습 중 실시간 시각화",
                                                                    className="text-primary mb-2",
                                                                ),
                                                                html.Ul(
                                                                    [
                                                                        html.Li(
                                                                            "`create_performance_chart` / `create_loss_chart`가 에피소드별 포트폴리오 가치와 Actor·Critic 손실을 이중 Y축으로 표시"
                                                                        ),
                                                                        html.Li(
                                                                            "`chart_callbacks.py`의 `dcc.Interval` 콜백이 차트 데이터를 주기적으로 갱신하여 실시간 모니터링 구현"
                                                                        ),
                                                                        html.Li(
                                                                            "`DashRealTrainingManager.update_status`와 `training_callbacks.py`가 KPI 카드·진행률 바를 업데이트"
                                                                        ),
                                                                    ],
                                                                    className="small mb-0",
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.H6(
                                                                    "🧐 손실 그래프 해석 가이드",
                                                                    className="text-danger mb-2",
                                                                ),
                                                                html.Ul(
                                                                    [
                                                                        html.Li(
                                                                            "Actor Loss(빨강): 일반적으로 음수, 값이 점진적으로 더 음수로 이동하면 정책이 높은 Q값을 학습 중이라는 신호"
                                                                        ),
                                                                        html.Li(
                                                                            "Critic Loss(보라): 양수 범위(0~60)에서 안정적으로 유지되면 가치 함수가 정확하게 추정되고 있다는 의미"
                                                                        ),
                                                                        html.Li(
                                                                            "두 곡선이 안정적인 추세를 보이면 학습이 정상 진행, 급격한 변동이 반복되면 학습 설정을 재점검"
                                                                        ),
                                                                    ],
                                                                    className="small mb-0",
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    md=6,
                                                ),
                                            ]
                                        ),
                                        html.Hr(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.H6(
                                                                    "📊 백테스트 시각화",
                                                                    className="text-success mb-2",
                                                                ),
                                                                html.Ul(
                                                                    [
                                                                        html.Li(
                                                                            "포트폴리오 가치 비교 차트: 강화학습 vs 균등투자 곡선을 동일 축에 표시"
                                                                        ),
                                                                        html.Li(
                                                                            "자산 배분 차트: 리밸런싱 시점별 ETF 비중 스택 영역"
                                                                        ),
                                                                        html.Li(
                                                                            "연환산/누적 수익률 차트: 장기 성과를 직관적으로 비교"
                                                                        ),
                                                                        html.Li(
                                                                            "`backtest_callbacks.py`가 KPI 카드와 Plotly `go.Figure`를 업데이트"
                                                                        ),
                                                                    ],
                                                                    className="small mb-0",
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.H6(
                                                                    "🗃️ 로그 및 메트릭 저장",
                                                                    className="text-warning mb-2",
                                                                ),
                                                                html.Ul(
                                                                    [
                                                                        html.Li(
                                                                            "통합 로그 파일 `logs/log_{mode}_unified_{timestamp}.log`에 UTF-8로 학습·백테스트 로그 기록"
                                                                        ),
                                                                        html.Li(
                                                                            "메타데이터 JSON(`metadata_{episode}.json`, `metadata_last.json`)에 하이퍼파라미터·성과 지표·학습 시간 등 저장"
                                                                        ),
                                                                        html.Li(
                                                                            "차트 데이터 구조와 `dcc.Store`를 통해 학습/백테스트 메트릭을 UI에 즉시 반영"
                                                                        ),
                                                                    ],
                                                                    className="small mb-0",
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    md=6,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            style=CARD_STYLE,
                        )
                    ]
                )
            ],
            className="mb-4",
        ),
        # Dash UI 기반 운영 흐름
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H4(
                                            [
                                                html.I(
                                                    className="bi bi-ui-checks-grid me-2"
                                                ),
                                                "Dash UI 기반 운영 흐름",
                                            ],
                                            className="mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "🎯 ETF 선택·저장 체계",
                                                            className="text-primary mb-2",
                                                        ),
                                                        html.Ul(
                                                            [
                                                                html.Li(
                                                                    "`src/utils/etf_manager.py`는 40여개 ETF를 8개 카테고리(대형주, 기술주, 중소형주, 배당주, 채권, 국제주식, 섹터별, 기타)로 관리"
                                                                ),
                                                                html.Li(
                                                                    "기본 조합을 `['SPY', 'DGRO', 'SCHD', 'EWY']`로 고정 (S&P500 + 배당 성장/가치 + 한국 주식)"
                                                                ),
                                                                html.Li(
                                                                    "초기자금 $10,000, 월 투자금 $300, 월별 리밸런싱으로 운영"
                                                                ),
                                                            ],
                                                            className="small",
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "⚙️ 학습 파라미터 모달 & 프리셋",
                                                            className="text-success mb-2",
                                                        ),
                                                        html.Ul(
                                                            [
                                                                html.Li(
                                                                    "`src/dash_layouts.py`의 학습 설정 모달 상단에는 '빠른 테스트', '균형잡힌', '고성능' 프리셋 버튼 배치"
                                                                ),
                                                                html.Li(
                                                                    "`src/callbacks/etf_callbacks.py`는 어떤 프리셋을 누르더라도 ETF 값이 항상 기본 조합으로 되돌아가도록 강제"
                                                                ),
                                                                html.Li(
                                                                    "선택 값 검증 및 미리보기 카드(`selected-etf-info`)를 동시에 갱신"
                                                                ),
                                                            ],
                                                            className="small",
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                            ]
                                        ),
                                        html.Hr(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "📊 백테스트 고급 설정 연동",
                                                            className="text-info mb-2",
                                                        ),
                                                        html.Ul(
                                                            [
                                                                html.Li(
                                                                    "`src/callbacks/backtest_callbacks.py`는 `backtest-etf-info-display`에 현재 선택된 모델/학습 설정에서 사용한 ETF 정보를 실시간으로 투영"
                                                                ),
                                                                html.Li(
                                                                    "백테스트 실행 시 학습된 모델과 동일한 ETF 조합 사용을 보장"
                                                                ),
                                                            ],
                                                            className="small",
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "🔄 비동기 학습·백테스트 관리자",
                                                            className="text-warning mb-2",
                                                        ),
                                                        html.Ul(
                                                            [
                                                                html.Li(
                                                                    "`DashRealTrainingManager`와 `DashBacktestManager`는 `threading.Thread` 기반 비동기 워커로 장시간 작업을 UI 블로킹 없이 처리"
                                                                ),
                                                                html.Li(
                                                                    "`CompleteDashManager`의 상태 저장소(`dcc.Store`)와 연동되어 학습 → 체크포인트 → 백테스트가 하나의 파이프라인으로 이어짐"
                                                                ),
                                                            ],
                                                            className="small",
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            style=CARD_STYLE,
                        )
                    ]
                )
            ],
            className="mb-4",
        ),
        # 실행 및 재현 절차
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H4(
                                            [
                                                html.I(
                                                    className="bi bi-play-circle me-2"
                                                ),
                                                "실행 및 재현 절차",
                                            ],
                                            className="mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "🚀 실행 스크립트",
                                                            className="text-primary mb-2",
                                                        ),
                                                        html.Ul(
                                                            [
                                                                html.Li(
                                                                    "개발 모드: `python run_dev.py` (자동 리로드, 디버그 활성화)"
                                                                ),
                                                                html.Li(
                                                                    "운영 모드: `python run_prod.py` (성능 최적화 모드)"
                                                                ),
                                                                html.Li(
                                                                    "환경변수: `DASH_CONFIG_FILE`, `APP_STARTUP_TIMESTAMP`로 설정 자동화"
                                                                ),
                                                            ],
                                                            className="small",
                                                        ),
                                                    ],
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "💾 체크포인트 시스템",
                                                            className="text-success mb-2",
                                                        ),
                                                        html.Ul(
                                                            [
                                                                html.Li(
                                                                    "작업 ID 기반: `model/rl_ddpg_{task_id}/checkpoint_{episode:04d}.pth`"
                                                                ),
                                                                html.Li(
                                                                    "최신 모델: `model/rl_ddpg_latest/checkpoint_last.pth`"
                                                                ),
                                                                html.Li(
                                                                    "Actor/Critic 메인 및 타깃 네트워크, 옵티마이저 상태 저장"
                                                                ),
                                                            ],
                                                            className="small",
                                                        ),
                                                    ],
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            "📋 메타데이터",
                                                            className="text-info mb-2",
                                                        ),
                                                        html.Ul(
                                                            [
                                                                html.Li(
                                                                    "하이퍼파라미터, ETF 조합, 윈도우 크기 저장"
                                                                ),
                                                                html.Li(
                                                                    "학습 시간 통계, 평균 보상, 모델 해시 기록"
                                                                ),
                                                                html.Li(
                                                                    "JSON 파일로 실험 추적성 확보"
                                                                ),
                                                            ],
                                                            className="small",
                                                        ),
                                                    ],
                                                    md=4,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            style=CARD_STYLE,
                        )
                    ]
                )
            ],
            className="mb-4",
        ),
        html.Hr(),
        # 시스템 기술 스택
        html.H6("🔍 시스템 기술 스택", className="text-primary mb-3"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [html.H6("🛠️ 기술 스택", className="mb-0")]
                                ),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            [
                                                html.Strong("딥러닝:"),
                                                " PyTorch, OpenAI Gymnasium",
                                            ],
                                            className="small mb-2",
                                        ),
                                        html.P(
                                            [
                                                html.Strong("데이터:"),
                                                " pandas, numpy, scikit-learn",
                                            ],
                                            className="small mb-2",
                                        ),
                                        html.P(
                                            [
                                                html.Strong("금융 API:"),
                                                " yfinance, fredapi, FinanceDataReader",
                                            ],
                                            className="small",
                                        ),
                                    ]
                                ),
                            ],
                            className="h-100",
                        )
                    ],
                    md=12,
                ),
            ],
            className="mb-3",
        ),
        html.Div(
            [
                html.P(
                    [
                        "💡 ",
                        html.Strong("핵심 가치"),
                        ": 강화학습을 통해 시장 변화에 적응하는 동적 포트폴리오 최적화로 장기 투자 수익률 극대화",
                    ],
                    className="text-center text-primary",
                )
            ],
            className="mt-3 p-3 bg-light rounded",
        ),
    ]
