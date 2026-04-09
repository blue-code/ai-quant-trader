"""예제 전략: 이동평균 크로스오버 + RSI 필터

가장 기본적인 퀀트 전략 예제.
단기 이동평균이 장기 이동평균을 상향 돌파하면 매수,
하향 돌파하면 매도한다. RSI로 과매수/과매도 필터를 적용한다.
"""

import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "fast_period": 20,  # 단기 이동평균 기간
    "slow_period": 60,  # 장기 이동평균 기간
    "rsi_period": 14,  # RSI 기간
    "rsi_overbought": 70,  # RSI 과매수 임계값
    "rsi_oversold": 30,  # RSI 과매도 임계값
}


def strategy(df: pd.DataFrame, params: dict = None) -> pd.Series:
    """이동평균 크로스오버 + RSI 필터 전략

    Args:
        df: OHLCV 데이터프레임 (Date, Open, High, Low, Close, Volume)
        params: 전략 파라미터

    Returns:
        시그널 시리즈 (1=매수, -1=매도, 0=관망)
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    close = df["Close"]

    # 이동평균 계산
    sma_fast = close.rolling(window=p["fast_period"]).mean()
    sma_slow = close.rolling(window=p["slow_period"]).mean()

    # RSI 계산
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=p["rsi_period"]).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=p["rsi_period"]).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # 기본 시그널: 이동평균 크로스오버
    signals = pd.Series(0, index=df.index)
    signals[sma_fast > sma_slow] = 1
    signals[sma_fast < sma_slow] = -1

    # RSI 필터: 과매수 구간에서 매수 차단, 과매도 구간에서 매도 차단
    signals[(signals == 1) & (rsi > p["rsi_overbought"])] = 0
    signals[(signals == -1) & (rsi < p["rsi_oversold"])] = 0

    # 워밍업 기간은 관망
    warmup = max(p["fast_period"], p["slow_period"], p["rsi_period"])
    signals.iloc[:warmup] = 0

    return signals


if __name__ == "__main__":
    # 단독 실행 테스트
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from src.data.loader import MarketDataLoader
    from src.backtest.engine import BacktestEngine
    from src.backtest.interpreter import StrategyInterpreter

    # 삼성전자 데이터 로드
    loader = MarketDataLoader(market="kr")
    df = loader.load("005930", "2020-01-01", "2024-12-31")

    if not df.empty:
        # 전략 실행
        signals = strategy(df)
        print(f"시그널 분포: 매수={sum(signals==1)}, 매도={sum(signals==-1)}, 관망={sum(signals==0)}")

        # 백테스트
        engine = BacktestEngine()
        result = engine.run_single(df, signals, "MA_Crossover_RSI")
        print(f"\n=== 백테스트 결과 ===")
        print(result.metrics.summary())
