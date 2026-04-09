"""예제 전략: 모멘텀 돌파 전략

변동성 돌파 기반 전략.
전일 변동폭의 일정 비율을 당일 시가에 더한 값을 돌파하면 매수한다.
래리 윌리엄스의 변동성 돌파 전략을 기반으로 한다.
"""

import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "k": 0.5,  # 변동폭 배수 (0.3 ~ 0.7)
    "lookback": 20,  # 모멘텀 확인 기간
    "volume_ma": 20,  # 거래량 이동평균 기간
    "volume_threshold": 1.2,  # 거래량 필터 (평균 대비 배수)
    "atr_period": 14,  # ATR 기간
    "atr_stop_mult": 2.0,  # ATR 기반 손절 배수
}


def strategy(df: pd.DataFrame, params: dict = None) -> pd.Series:
    """모멘텀 돌파 전략

    Args:
        df: OHLCV 데이터프레임
        params: 전략 파라미터

    Returns:
        시그널 시리즈 (1=매수, -1=매도, 0=관망)
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    open_ = df["Open"]
    volume = df["Volume"]

    # 전일 변동폭
    prev_range = (high.shift(1) - low.shift(1))

    # 돌파 기준가: 당일 시가 + 전일 변동폭 * k
    breakout_price = open_ + prev_range * p["k"]

    # 모멘텀 확인: N일 수익률 양수
    momentum = close.pct_change(p["lookback"])

    # 거래량 필터: 평균 대비 일정 배수 이상
    vol_ma = volume.rolling(window=p["volume_ma"]).mean()
    vol_ratio = volume / vol_ma.replace(0, np.nan)

    # ATR (Average True Range) 계산
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=p["atr_period"]).mean()

    # ATR 기반 손절가
    stop_price = close - atr * p["atr_stop_mult"]

    # 시그널 생성
    signals = pd.Series(0, index=df.index)

    # 매수: 종가가 돌파 기준가 이상 + 모멘텀 양수 + 거래량 충분
    buy_condition = (
        (close >= breakout_price)
        & (momentum > 0)
        & (vol_ratio >= p["volume_threshold"])
    )
    signals[buy_condition] = 1

    # 매도: 종가가 손절가 이하
    sell_condition = close <= stop_price
    signals[sell_condition] = -1

    # 워밍업 기간
    warmup = max(p["lookback"], p["volume_ma"], p["atr_period"]) + 1
    signals.iloc[:warmup] = 0

    return signals
