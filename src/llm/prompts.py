"""트레이딩 전략 생성용 프롬프트 템플릿

AI-Scientist-v2가 ML 연구 가설을 생성하듯,
여기서는 퀀트 트레이딩 전략 가설을 생성하는 프롬프트를 관리한다.
"""

from __future__ import annotations


class StrategyPrompts:
    """전략 생성·분석·개선에 사용하는 프롬프트 모음"""

    # 시스템 프롬프트: 퀀트 전략가 페르소나
    SYSTEM = """당신은 20년 경력의 시니어 퀀트 트레이더다.
한국 및 미국 주식 시장에 정통하며, 팩터 투자, 통계적 차익거래,
모멘텀/평균회귀 전략에 깊은 전문성을 보유하고 있다.

핵심 원칙:
1. 모든 전략은 수학적 근거가 있어야 한다
2. 거래비용(수수료 + 슬리피지)을 반드시 고려한다
3. 과적합 방지를 최우선으로 한다
4. 룩어헤드 바이어스를 절대 허용하지 않는다
5. 실거래 가능성(유동성, 체결 현실성)을 감안한다"""

    # Stage 1: 전략 아이디어 생성
    IDEATION = """다음 조건에 맞는 트레이딩 전략을 {num_strategies}개 생성하라.

## 투자 유니버스
{universe_description}

## 사용 가능한 데이터
- OHLCV (시가, 고가, 저가, 종가, 거래량)
- 기술적 지표 (이동평균, RSI, MACD, 볼린저밴드 등)
- 거래량 프로파일
- 섹터/업종 분류

## 전략 유형 (다양하게 조합)
- 모멘텀: 추세 추종, 상대 강도
- 평균 회귀: 과매도/과매수, 볼린저밴드
- 변동성: 변동성 돌파, 변동성 매매
- 팩터: 가치, 성장, 퀄리티 팩터 결합
- 계절성: 월별/요일별 패턴
- 상관관계: 페어 트레이딩, 섹터 로테이션

## 출력 형식 (JSON 배열)
```json
[
  {{
    "name": "전략명 (영문 snake_case)",
    "description": "전략 설명 (한글)",
    "hypothesis": "이 전략이 작동하는 이유 (경제적 근거)",
    "signal_type": "momentum | mean_reversion | volatility | factor | seasonal | correlation",
    "indicators": ["사용하는 지표 목록"],
    "entry_rule": "진입 조건 설명",
    "exit_rule": "청산 조건 설명",
    "position_sizing": "포지션 사이징 방법",
    "expected_holding_days": 5,
    "expected_sharpe": 1.0,
    "risk_level": "low | medium | high",
    "code": "파이썬 전략 코드 (함수 형태)"
  }}
]
```

## 전략 코드 규칙
- 함수 시그니처: `def strategy(df: pd.DataFrame, params: dict) -> pd.Series`
- 입력 df 컬럼: Date, Open, High, Low, Close, Volume
- 반환: pd.Series (1=매수, -1=매도, 0=관망)
- numpy, pandas, ta 라이브러리 사용 가능
- 룩어헤드 바이어스 절대 금지 (미래 데이터 참조 불가)
"""

    # Stage 2: 전략 코드 생성 (Draft)
    STRATEGY_DRAFT = """다음 전략 아이디어를 실행 가능한 파이썬 코드로 구현하라.

## 전략 아이디어
{strategy_idea}

## 구현 요구사항
1. `strategy(df, params)` 함수 구현
2. df 컬럼: Date, Open, High, Low, Close, Volume
3. 반환: pd.Series (1=매수, -1=매도, 0=관망)
4. params dict로 모든 하이퍼파라미터를 외부에서 조절 가능하게
5. 기본 params 값을 DEFAULT_PARAMS dict로 제공

## 코드 템플릿
```python
import numpy as np
import pandas as pd

DEFAULT_PARAMS = {{
    # 여기에 기본 파라미터
}}

def strategy(df: pd.DataFrame, params: dict = None) -> pd.Series:
    '''
    {strategy_name}

    Args:
        df: OHLCV 데이터프레임
        params: 전략 파라미터

    Returns:
        시그널 시리즈 (1=매수, -1=매도, 0=관망)
    '''
    p = {{**DEFAULT_PARAMS, **(params or {{}})}}

    # 지표 계산

    # 시그널 생성

    # 리스크 필터

    return signals
```

반드시 ```python 코드 블록으로 감싸서 응답하라.
"""

    # Stage 2: 디버그
    STRATEGY_DEBUG = """다음 전략 코드가 실행 중 오류가 발생했다. 수정하라.

## 전략 코드
```python
{strategy_code}
```

## 오류 메시지
```
{error_message}
```

## 입력 데이터 샘플
```
{data_sample}
```

수정된 전체 코드를 ```python 블록으로 응답하라.
오류 원인과 수정 내용을 주석으로 간략히 설명하라.
"""

    # Stage 2: 파라미터 최적화 제안
    PARAMETER_TUNING = """다음 전략의 성과를 분석하고 파라미터 개선안을 제시하라.

## 전략 코드
```python
{strategy_code}
```

## 현재 파라미터
{current_params}

## 백테스트 성과
{backtest_metrics}

## 개선 요구사항
- Sharpe Ratio를 높이되, 최대 낙폭은 -20% 이내 유지
- 과적합 위험이 적은 방향으로 조정
- 3가지 파라미터 세트를 제안 (보수적/균형/공격적)

```json
[
  {{
    "label": "보수적",
    "params": {{}},
    "rationale": "변경 이유"
  }},
  {{
    "label": "균형",
    "params": {{}},
    "rationale": "변경 이유"
  }},
  {{
    "label": "공격적",
    "params": {{}},
    "rationale": "변경 이유"
  }}
]
```
"""

    # Stage 3: 팩터 엔지니어링 / 앙상블
    FACTOR_ENGINEERING = """다음 개별 전략들의 성과를 분석하고 앙상블 전략을 설계하라.

## 개별 전략 성과
{individual_results}

## 요구사항
1. 상관관계가 낮은 전략들을 조합하여 분산 효과를 극대화
2. 각 전략의 가중치를 시장 환경에 따라 동적으로 조절
3. 매크로 레짐(상승/하락/횡보) 감지 로직 포함
4. 전체 앙상블의 리스크 예산(Risk Budget) 배분

앙상블 전략 코드를 ```python 블록으로 응답하라.
"""

    # Stage 4: 로버스트니스 분석
    ROBUSTNESS_ANALYSIS = """다음 전략의 로버스트니스를 분석하라.

## 전략 코드
```python
{strategy_code}
```

## Walk-Forward 검증 결과
{walk_forward_results}

## 분석 항목
1. 시간대별 성과 안정성
2. 시장 환경(상승/하락/횡보)별 성과
3. 파라미터 민감도 (±20% 변동 시 성과 변화)
4. 과적합 위험도 평가 (IS vs OOS Sharpe 비율)
5. 실거래 적합성 판단

```json
{{
  "robustness_score": 0.0,
  "overfitting_risk": "low | medium | high",
  "market_regime_analysis": {{}},
  "parameter_sensitivity": {{}},
  "recommendation": "실거래 투입 | 추가 검증 필요 | 폐기",
  "rationale": "판단 근거"
}}
```
"""

    # 성과 리포트 생성
    REPORT_GENERATION = """다음 백테스트 결과를 기반으로 투자 성과 리포트를 작성하라.

## 전략 정보
{strategy_info}

## 백테스트 결과
{backtest_results}

## 리포트 포함 항목
1. 전략 개요 (가설, 로직, 투자 유니버스)
2. 수익률 분석 (연간/월간 수익률, 벤치마크 대비)
3. 리스크 분석 (변동성, 최대낙폭, VaR)
4. 거래 분석 (승률, 손익비, 평균 보유기간)
5. 시장 환경별 분석
6. 결론 및 실거래 권고사항

마크다운 형식으로 작성하라.
"""
