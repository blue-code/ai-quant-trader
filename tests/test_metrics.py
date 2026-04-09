"""백테스트 메트릭 계산 테스트"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import calculate_metrics, PerformanceMetrics


class TestCalculateMetrics:
    """calculate_metrics 함수 테스트"""

    def _make_equity_curve(self, returns: list[float]) -> pd.Series:
        """수익률 리스트에서 수익곡선 생성"""
        values = [100_000_000]  # 1억 시작
        for r in returns:
            values.append(values[-1] * (1 + r))
        dates = pd.date_range("2024-01-01", periods=len(values), freq="B")
        return pd.Series(values, index=dates)

    def test_positive_returns(self):
        """양의 수익률 시나리오"""
        # 매일 0.1% 수익
        returns = [0.001] * 252
        equity = self._make_equity_curve(returns)
        metrics = calculate_metrics(equity)

        assert metrics.total_return > 0
        assert metrics.sharpe_ratio > 0
        assert metrics.max_drawdown == 0  # 낙폭 없음

    def test_negative_returns(self):
        """음의 수익률 시나리오"""
        returns = [-0.001] * 100
        equity = self._make_equity_curve(returns)
        metrics = calculate_metrics(equity)

        assert metrics.total_return < 0
        assert metrics.max_drawdown < 0

    def test_mixed_returns(self):
        """혼합 수익률 시나리오"""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, 252).tolist()
        equity = self._make_equity_curve(returns)
        metrics = calculate_metrics(equity)

        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert metrics.volatility > 0

    def test_empty_equity(self):
        """빈 데이터"""
        equity = pd.Series([], dtype=float)
        metrics = calculate_metrics(equity)
        assert metrics.total_return == 0
        assert metrics.sharpe_ratio == 0

    def test_metrics_comparison(self):
        """메트릭 비교 연산"""
        m1 = PerformanceMetrics(sharpe_ratio=1.5)
        m2 = PerformanceMetrics(sharpe_ratio=0.8)
        assert m1 > m2

    def test_is_acceptable(self):
        """임계값 충족 검증"""
        m = PerformanceMetrics(
            sharpe_ratio=1.0,
            max_drawdown=-0.15,
            win_rate=0.55,
        )
        thresholds = {"min_sharpe": 0.5, "max_drawdown": -0.20, "min_win_rate": 0.45}
        assert m.is_acceptable(thresholds) is True

        # Sharpe 미달
        m2 = PerformanceMetrics(sharpe_ratio=0.3, max_drawdown=-0.10, win_rate=0.50)
        assert m2.is_acceptable(thresholds) is False
