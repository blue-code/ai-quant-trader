"""백테스트 엔진 테스트"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.interpreter import StrategyInterpreter, ExecutionResult


class TestBacktestEngine:
    """백테스트 엔진 테스트"""

    def _make_ohlcv(self, n: int = 252) -> pd.DataFrame:
        """테스트용 OHLCV 데이터 생성"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 50000 + np.cumsum(np.random.randn(n) * 500)
        close = np.maximum(close, 1000)  # 음수 방지

        return pd.DataFrame({
            "Date": dates,
            "Open": close * (1 + np.random.randn(n) * 0.005),
            "High": close * (1 + abs(np.random.randn(n) * 0.01)),
            "Low": close * (1 - abs(np.random.randn(n) * 0.01)),
            "Close": close,
            "Volume": np.random.randint(100000, 1000000, n),
        })

    def test_buy_and_hold(self):
        """매수 후 보유 전략"""
        df = self._make_ohlcv()
        signals = pd.Series(1, index=df.index)  # 항상 매수
        signals.iloc[0] = 0  # 첫날 관망

        engine = BacktestEngine(BacktestConfig(initial_capital=100_000_000))
        result = engine.run_single(df, signals, "buy_and_hold")

        assert result.metrics is not None
        assert len(result.equity_curve) == len(df)
        assert result.equity_curve.iloc[0] == 100_000_000

    def test_no_trade(self):
        """거래 없는 전략"""
        df = self._make_ohlcv()
        signals = pd.Series(0, index=df.index)

        engine = BacktestEngine()
        result = engine.run_single(df, signals, "no_trade")

        assert result.metrics.total_trades == 0
        assert result.metrics.total_return == 0

    def test_walk_forward(self):
        """Walk-Forward 검증"""
        df = self._make_ohlcv(500)
        signals = pd.Series(0, index=df.index)
        signals[df["Close"] > df["Close"].rolling(20).mean()] = 1

        engine = BacktestEngine()
        results = engine.walk_forward(df, signals, n_splits=5)

        assert len(results) > 0
        for r in results:
            assert r.metrics is not None


class TestStrategyInterpreter:
    """전략 인터프리터 테스트"""

    def _make_ohlcv(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 50000 + np.cumsum(np.random.randn(n) * 500)
        return pd.DataFrame({
            "Date": dates,
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.random.randint(100000, 1000000, n),
        })

    def test_valid_strategy(self):
        """유효한 전략 코드 실행"""
        code = """
import numpy as np
import pandas as pd

DEFAULT_PARAMS = {"period": 20}

def strategy(df, params=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    sma = df["Close"].rolling(p["period"]).mean()
    signals = pd.Series(0, index=df.index)
    signals[df["Close"] > sma] = 1
    signals[df["Close"] < sma] = -1
    signals.iloc[:p["period"]] = 0
    return signals
"""
        df = self._make_ohlcv()
        interpreter = StrategyInterpreter()
        result = interpreter.execute(code, df)

        assert result.success is True
        assert result.signals is not None
        assert len(result.signals) == len(df)

    def test_invalid_strategy(self):
        """잘못된 코드 실행"""
        code = "def strategy(df, params=None): raise ValueError('테스트 오류')"
        df = self._make_ohlcv()
        interpreter = StrategyInterpreter()
        result = interpreter.execute(code, df)

        assert result.success is False
        assert "테스트 오류" in result.error

    def test_code_block_extraction(self):
        """```python 블록에서 코드 추출"""
        code = """여기는 설명입니다.

```python
import pandas as pd

DEFAULT_PARAMS = {}

def strategy(df, params=None):
    return pd.Series(0, index=df.index)
```

위 코드는...
"""
        df = self._make_ohlcv()
        interpreter = StrategyInterpreter()
        result = interpreter.execute(code, df)

        assert result.success is True
