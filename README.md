# AI Quant Trader

AI-Scientist-v2의 트리 탐색 방법론을 활용한 자율 퀀트 트레이딩 시스템.

## 개요

LLM이 트레이딩 전략을 자동으로 생성·검증·최적화하고, 최종적으로 실거래까지 수행하는 엔드투엔드 퀀트 시스템이다.

### 핵심 파이프라인

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
- 실거래 연동 (한국투자증권 API / Interactive Brokers)

## 설치

```bash
pip install -r requirements.txt
```

## 환경 변수

```bash
# LLM API (택 1)
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# 브로커 API (실거래 시)
export KIS_APP_KEY="your-key"        # 한국투자증권
export KIS_APP_SECRET="your-secret"
```

## 사용법

### 1. 전략 생성
```bash
python -m src.main ideate --universe kospi200 --num-strategies 20
```

### 2. 백테스트
```bash
python -m src.main backtest --strategies strategies.json --start 2020-01-01 --end 2024-12-31
```

### 3. 최적화 (Tree Search)
```bash
python -m src.main optimize --strategy best_strategy.json --stages 4
```

### 4. 실거래
```bash
python -m src.main trade --strategy optimized_strategy.json --mode paper
```

## 프로젝트 구조

```
ai-quant-trader/
├── config/                  # 설정 파일
│   ├── default.yaml         # 기본 설정
│   └── strategies/          # 전략별 설정
├── src/
│   ├── main.py              # CLI 진입점
│   ├── llm/                 # LLM 인터페이스
│   ├── ideation/            # 전략 가설 생성
│   ├── data/                # 시장 데이터 로더
│   ├── backtest/            # 백테스트 엔진
│   ├── treesearch/          # 트리 탐색 최적화
│   ├── execution/           # 실거래 연동
│   └── reporting/           # 성과 리포트
├── tests/                   # 테스트
└── examples/                # 예제
```

## 기술 스택

- **LLM**: Claude / GPT-4o (전략 생성·분석)
- **데이터**: yfinance, pykrx (한국 시장)
- **백테스트**: 자체 벡터화 엔진
- **최적화**: AI-Scientist-v2 Best-First Tree Search
- **실거래**: 한국투자증권 API, Interactive Brokers
- **시각화**: matplotlib, plotly

## 라이선스

MIT License
