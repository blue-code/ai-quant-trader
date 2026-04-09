# AI Quant Trader

[AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)의 트리 탐색 방법론을 활용한 자율 퀀트 트레이딩 시스템.

## 개요

LLM이 트레이딩 전략을 자동으로 생성·검증·최적화하고, 최종적으로 실거래까지 수행하는 엔드투엔드 퀀트 시스템이다.

```
전략 생성 (Ideation) → 백테스트 (Backtest) → 최적화 (Tree Search) → 실거래 (Execution)
```

### AI-Scientist-v2 적용 포인트

| AI-Scientist-v2 | AI Quant Trader |
|---|---|
| ML 연구 가설 생성 | 트레이딩 전략 가설 생성 |
| 실험 코드 실행 | 백테스트 엔진 실행 |
| Best-First Tree Search | 전략 파라미터 최적화 |
| 4단계 실험 진행 | 기본전략 → 파라미터튜닝 → 팩터엔지니어링 → 로버스트니스 검증 |
| 논문 작성 | 성과 리포트 생성 |

## 4단계 파이프라인

### Stage 1: 전략 생성 (Ideation)
- LLM이 팩터 조합, 진입/청산 규칙, 포지션 사이징 전략을 자동 생성
- 기존 전략과의 차별성 검증

### Stage 2: 백테스트 & 파라미터 최적화
- 생성된 전략을 과거 데이터로 검증
- Best-First Tree Search로 최적 파라미터 탐색
- 다중 시장 환경(상승/하락/횡보)에서 교차 검증

### Stage 3: 팩터 엔지니어링 & 앙상블
- 개별 전략의 신호를 결합하여 앙상블 전략 구축
- 매크로 지표(VIX, 금리 스프레드) 통합
- 섹터 로테이션 전략 탐색

### Stage 4: 로버스트니스 검증 & 실거래
- Walk-Forward 검증으로 과적합 방지
- 스트레스 테스트 (시장 급락, 레짐 전환)
- 실거래 연동 (한국투자증권 API)

## 빠른 시작

### 1. 설치

```bash
# setup.bat 더블클릭 또는:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 실행

각 단계별 bat 파일을 더블클릭하거나, 인자를 넣어 실행한다.

| bat 파일 | 설명 | 사용법 |
|---|---|---|
| `setup.bat` | venv 생성 + 패키지 설치 (최초 1회) | 더블클릭 |
| `ideate.bat` | LLM 전략 생성 | `ideate.bat [유니버스] [개수]` |
| `backtest.bat` | 백테스트 | `backtest.bat [전략파일] [유니버스]` |
| `optimize.bat` | 트리 탐색 최적화 | `optimize.bat [유니버스] [전략파일]` |
| `trade.bat` | 실거래 / 모의투자 | `trade.bat [전략파일] [paper\|live]` |
| `run_all.bat` | 전체 파이프라인 | `run_all.bat [유니버스] [개수]` |

인자 없이 실행하면 기본값(`kospi_bluechip`, 10개)으로 동작한다.

### 3. CLI 직접 실행

```bash
# venv 활성화 후
.venv\Scripts\activate

# 전략 생성
python -m src.main ideate --universe kospi_bluechip --num-strategies 10

# 백테스트
python -m src.main backtest --strategies ./results/ideas.json --universe kospi_bluechip

# 전체 파이프라인 (생성 → 최적화 → 검증)
python -m src.main optimize --universe kospi_bluechip

# 모의투자
python -m src.main trade --strategy ./results/best_strategy.json --mode paper
```

## LLM 연동

### 방법 1: Codex OAuth (기본값, API 키 불필요)

로컬에 설치된 [OpenAI Codex CLI](https://github.com/openai/codex)의 OAuth 인증을 그대로 활용한다.
ChatGPT Plus/Pro 구독이 있으면 별도 API 키 없이 사용 가능하다.

```bash
# Codex CLI 로그인 (최초 1회)
codex
```

`config/default.yaml`의 기본 설정:

```yaml
llm:
  provider: "codex"    # ~/.codex/auth.json의 OAuth 토큰 사용
  model: ""            # Codex 설정의 기본 모델 사용 (gpt-5.4 등)
```

### 방법 2: API 키 직접 설정

```bash
# Anthropic
export ANTHROPIC_API_KEY="your-key"

# 또는 OpenAI
export OPENAI_API_KEY="your-key"
```

`config/default.yaml`에서 `provider`를 `"anthropic"` 또는 `"openai"`로 변경한다.

## 투자 유니버스

사전 정의된 유니버스를 사용하거나, 커스텀 유니버스를 구성할 수 있다.

| 유니버스 | 시장 | 설명 |
|---|---|---|
| `kospi_bluechip` | 한국 | KOSPI 대표 블루칩 30종목 |
| `kosdaq_growth` | 한국 | KOSDAQ 성장주 20종목 |
| `us_tech` | 미국 | 미국 대형 기술주 15종목 |
| `sp500_sector` | 미국 | S&P500 섹터별 대표 11종목 |

## 환경 변수

```bash
# 실거래 시 (한국투자증권 Open API)
export KIS_APP_KEY="your-key"
export KIS_APP_SECRET="your-secret"
export KIS_ACCOUNT_NO="your-account-number"
```

## 프로젝트 구조

```
ai-quant-trader/
├── config/
│   └── default.yaml          # 기본 설정 (LLM, 트레이딩, 백테스트)
├── src/
│   ├── main.py               # CLI 진입점 (click 기반)
│   ├── llm/
│   │   ├── client.py         # LLM 통합 클라이언트 (Codex/Anthropic/OpenAI)
│   │   └── prompts.py        # 전략 생성·분석·개선 프롬프트
│   ├── ideation/
│   │   └── generator.py      # LLM 기반 전략 가설 자동 생성
│   ├── data/
│   │   ├── loader.py         # 시장 데이터 로더 (yfinance/pykrx)
│   │   └── universe.py       # 종목 유니버스 관리
│   ├── backtest/
│   │   ├── engine.py         # 벡터화 백테스트 엔진
│   │   ├── metrics.py        # 성과 지표 (Sharpe, MDD, 승률 등)
│   │   └── interpreter.py    # LLM 생성 전략 코드 실행기
│   ├── treesearch/
│   │   ├── node.py           # 전략 노드 (트리 탐색 단위)
│   │   ├── journal.py        # 탐색 결과 저널
│   │   ├── parallel_agent.py # 병렬 전략 최적화 에이전트
│   │   └── agent_manager.py  # 4단계 파이프라인 오케스트레이터
│   ├── execution/
│   │   ├── broker.py         # 브로커 인터페이스 (KIS/모의투자)
│   │   └── risk.py           # 리스크 관리 (서킷 브레이커)
│   └── reporting/
│       └── report.py         # HTML/마크다운 성과 리포트
├── examples/
│   ├── sample_strategy.py    # 예제: 이동평균 크로스오버 + RSI
│   └── momentum_breakout.py  # 예제: 모멘텀 돌파 전략
├── tests/
│   ├── test_metrics.py       # 성과 지표 테스트
│   └── test_backtest.py      # 백테스트 엔진 테스트
├── setup.bat                 # 환경 설치
├── ideate.bat                # 전략 생성
├── backtest.bat              # 백테스트
├── optimize.bat              # 트리 탐색 최적화
├── trade.bat                 # 실거래/모의투자
└── run_all.bat               # 전체 파이프라인
```

## 기술 스택

- **LLM**: GPT-5.4 (Codex OAuth) / Claude / GPT-4o
- **데이터**: yfinance, pykrx (한국 시장)
- **백테스트**: 자체 벡터화 엔진
- **최적화**: AI-Scientist-v2 Best-First Tree Search
- **실거래**: 한국투자증권 Open API
- **시각화**: matplotlib, plotly
- **CLI**: click

## 라이선스

MIT License
