# 프로젝트 개요 및 실행 가이드

## 📋 개요

본 프로젝트는 **DDPG(Deep Deterministic Policy Gradient)** 에이전트로 구현된 **ETF 포트폴리오 동적 자산배분 강화학습 콘솔**입니다. UI는 3개의 핵심 탭(강화학습 · 백테스팅 · 모니터링)으로 단순화되어 있으며, 모든 화면과 동작이 동일한 강화학습 파이프라인을 중심으로 구성됩니다.

## 🎯 주요 특징

### 📱 탭 기반 구조

- **🚀 강화학습 탭**: DDPG 학습 파라미터 설정, 실시간 진행 상황 모니터링, 모델 저장 제어

![강화학습 탭 화면](images/page_agent.JPG)

- **📈 백테스팅 탭**: 동일한 DDPG 모델을 활용한 성과 재현 및 기간별 분석

![백테스팅 탭 화면](images/page_backtest.JPG)

- **📊 모니터링 탭**: 학습·백테스트 과정의 시스템 리소스 상태 추적

![모니터링 탭 화면](images/page_util.JPG)

### ⚡ 핵심 기능

- **DDPG 학습 전용 워크플로우**: ETF 선택 → 환경 구성 → 학습/탐험 → 체크포인트 관리
- **시뮬레이션 & 실거래 모드**: 빠른 UI 검증과 실제 학습을 동일 화면에서 전환
- **일관된 자산 데이터 파이프라인**: `src/data` 및 `src/environment` 모듈을 통한 정규화 입력
- **운영 가시성**: GPU/CPU, 메모리, 로그 스트림을 Dash 내에서 실시간 확인

## 🚀 실행 방법

> **참고**: 상세한 실행 방법은 [README.md](../README.md#6-실행-방법)를 참조하세요.

### 간단한 실행 가이드

```bash
# 개발 모드 (자동 리로드 활성화)
python run_dev.py

# 운영 모드 (자동 리로드 비활성화)
python run_prod.py
```

**웹 브라우저 접속**: <http://127.0.0.1:8050>

자세한 환경 설정, 의존성 설치, 모델 저장 및 다운로드 방법은 README.md의 "6. 실행 방법" 및 "6.4. 학습된 모델 저장 및 다운로드" 섹션을 참조하세요.

## 🎨 주요 개선사항

### ✅ 구조적 개선

- **모듈화**: 기능별로 분리된 소스 파일
- **DDPG 알고리즘 분리**: 강화학습 모듈을 독립적으로 구조화
- **🆕 완전한 모듈 구조화**: `sub/`, `sub_data/`, `sub_enviroment/` → `src/` 통합
- **타입 힌트 추가**: 코드 안정성 및 가독성 향상
- **탭 기반 UI**: 직관적인 기능 분류
- **재사용성**: 컴포넌트 기반 설계
- **유지보수성**: 코드 구조 명확화

### ✅ 사용자 경험 개선

- **DDPG 전용 흐름**: 학습 → 백테스트 → 모니터링 전환이 끊김 없이 동작
- **전문화된 인터페이스**: 각 탭이 동일 모델 정보를 공유해 컨텍스트 전환 감소
- **실시간 동기화**: Dash Store + Interval 구조로 KPI가 즉시 반영
- **반응형 디자인**: 모바일/데스크탑에서도 동일 기능 제공

### ✅ 기능적 개선

- **독립적 백테스트**: 학습과 분리된 백테스트 실행
- **향상된 모니터링**: 시스템 리소스 추적
- **설정 관리**: 탭별 설정 저장/복원
- **오류 처리**: 개선된 에러 핸들링
- **환경변수 설정**: `.env` 파일로 auto-reload 및 개발 도구 제어

## 🧠 DDPG 알고리즘 모듈 (`src/ddpg_algorithm/`)

### 모듈 구조

- **`ddpg_agent.py`**: DDPG 에이전트 메인 클래스
  - `DDPGAgent`: 포트폴리오 최적화를 위한 DDPG 구현
  - 학습, 행동 선택, 모델 저장/로드 기능

- **`ddpg_models.py`**: 신경망 모델 정의
  - `Actor`: 연속적 행동 공간에서 포트폴리오 비율 결정
  - `Critic`: 상태-행동 쌍의 가치 평가
  - `fanin_init`: Fan-in 기반 가중치 초기화

- **`ddpg_noise.py`**: 탐험 노이즈 생성
  - `OUNoise`: Ornstein-Uhlenbeck 프로세스 기반 노이즈
  - 연속적 행동 공간에서 효과적인 탐험

### 사용 예시

```python
from src.ddpg_algorithm import DDPGAgent, Actor, Critic, OUNoise

# 에이전트 생성
agent = DDPGAgent(
    logger=logger,
    state_dim=state_dimension,
    action_dim=action_dimension,
    hidden_dim=256,
    actor_lr=0.0003,
    critic_lr=0.0003,
    device='cuda'
)

# 행동 선택
action, raw_action = agent.select_action(state, add_noise=True)

# 모델 업데이트
actor_loss, critic_loss = agent.update(batch)
```

## 🔧 개발자 가이드

### 새로운 탭 추가

1. `src/dash_layouts.py`에 새 탭 콘텐츠 함수 추가
2. `create_main_tabs()` 함수에 탭 정의 추가
3. `src/dash_callbacks.py`에 관련 콜백 함수 추가

### 컴포넌트 커스터마이징

- **레이아웃**: `src/dash_layouts.py` 수정
- **차트**: `src/dash_charts.py` 수정
- **콜백**: `src/dash_callbacks.py` 수정
- **스타일**: `src/dash_utils.py`의 스타일 상수 수정

### 상태 관리

- **글로벌 상태**: `src/dash_manager.py`의 `CompleteDashManager` 클래스
- **탭간 데이터 공유**: Dash Store 컴포넌트 사용
- **실시간 업데이트**: Interval 컴포넌트로 주기적 갱신

### DDPG 알고리즘 확장

- **새로운 네트워크 구조**: `ddpg_models.py`에 새 모델 클래스 추가
- **다른 노이즈 전략**: `ddpg_noise.py`에 새 노이즈 클래스 추가
- **하이퍼파라미터 튜닝**: `ddpg_agent.py`의 기본값 수정

## 📈 성능 최적화

- **비동기 처리**: 학습/백테스트 별도 스레드 실행
- **메모리 관리**: 차트 데이터 포인트 제한 (1000개)
- **로그 관리**: 최대 150개 로그 유지
- **업데이트 최적화**: 컴포넌트별 적절한 업데이트 주기

## 🛠️ 기술 스택

- **Frontend**: Dash, Plotly, Bootstrap
- **Backend**: Python, Threading
- **ML**: PyTorch, DDPG 알고리즘
- **Data**: Pandas, NumPy
- **Monitoring**: psutil, GPUtil

## 📞 지원

프로젝트 관련 문의나 이슈는 해당 레포지토리의 Issues 섹션을 이용해 주세요.

---

**주의**: 이 시스템은 교육 및 연구 목적으로 설계되었습니다. 실제 투자에 사용하기 전 충분한 검증이 필요합니다.
