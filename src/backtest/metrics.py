"""백테스트 성과 지표

Sharpe Ratio, 최대낙폭, 승률 등 핵심 퀀트 메트릭을 계산한다.
AI-Scientist-v2의 MetricValue 패턴을 참고하여,
복수 메트릭을 구조화된 형태로 관리한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class PerformanceMetrics:
    """백테스트 성과 지표 모음

    AI-Scientist-v2의 MetricValue처럼 복수 메트릭을 관리하고,
    전략 간 비교를 지원한다.
    """

    # 수익률 지표
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_returns: list[float] = field(default_factory=list)

    # 리스크 지표
    volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # 최대 낙폭 지속일

    # 위험조정 수익률
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # 거래 지표
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_days: float = 0.0

    # 벤치마크 대비
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0

    def __gt__(self, other: PerformanceMetrics) -> bool:
        """전략 비교: Sharpe Ratio 기준 (AI-Scientist-v2의 Node 비교 패턴)"""
        return self.sharpe_ratio > other.sharpe_ratio

    def is_acceptable(self, thresholds: dict) -> bool:
        """성과가 임계값을 충족하는지 확인"""
        if self.sharpe_ratio < thresholds.get("min_sharpe", 0.5):
            return False
        if self.max_drawdown < thresholds.get("max_drawdown", -0.20):
            return False
        if self.win_rate < thresholds.get("min_win_rate", 0.45):
            return False
        return True

    def to_dict(self) -> dict:
        """딕셔너리 변환 (직렬화용)"""
        return {
            "total_return": round(self.total_return, 4),
            "annual_return": round(self.annual_return, 4),
            "volatility": round(self.volatility, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "max_drawdown_duration": self.max_drawdown_duration,
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "avg_holding_days": round(self.avg_holding_days, 1),
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "information_ratio": round(self.information_ratio, 4),
        }

    def summary(self) -> str:
        """요약 문자열 (LLM 프롬프트 삽입용)"""
        return (
            f"총수익률: {self.total_return:.2%} | "
            f"연수익률: {self.annual_return:.2%} | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"MDD: {self.max_drawdown:.2%} | "
            f"승률: {self.win_rate:.2%} | "
            f"거래수: {self.total_trades}"
        )


def calculate_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.035,  # 한국 무위험이자율 3.5%
    trading_days: int = 252,
) -> PerformanceMetrics:
    """수익곡선에서 전체 성과 지표를 계산

    Args:
        equity_curve: 일별 포트폴리오 가치 시리즈
        trades: 거래 내역 DataFrame (선택)
        benchmark: 벤치마크 수익곡선 (선택)
        risk_free_rate: 무위험이자율
        trading_days: 연간 거래일

    Returns:
        PerformanceMetrics 객체
    """
    metrics = PerformanceMetrics()

    if equity_curve.empty or len(equity_curve) < 2:
        return metrics

    # 일별 수익률
    returns = equity_curve.pct_change().dropna()
    if returns.empty:
        return metrics

    # 수익률 지표
    metrics.total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_years = len(returns) / trading_days
    if n_years > 0:
        metrics.annual_return = (1 + metrics.total_return) ** (1 / n_years) - 1

    # 월별 수익률
    if hasattr(equity_curve.index, "to_period"):
        monthly = equity_curve.resample("ME").last().pct_change().dropna()
        metrics.monthly_returns = monthly.tolist()

    # 변동성
    metrics.volatility = returns.std() * np.sqrt(trading_days)

    # 최대 낙폭 (MDD)
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    metrics.max_drawdown = drawdown.min()

    # MDD 지속 기간
    is_dd = drawdown < 0
    if is_dd.any():
        dd_groups = (~is_dd).cumsum()
        dd_lengths = is_dd.groupby(dd_groups).sum()
        metrics.max_drawdown_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0

    # Sharpe Ratio
    excess_return = returns.mean() - (risk_free_rate / trading_days)
    if returns.std() > 0:
        metrics.sharpe_ratio = (excess_return / returns.std()) * np.sqrt(trading_days)

    # Sortino Ratio (하방 변동성만 사용)
    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        metrics.sortino_ratio = (
            excess_return / downside.std()
        ) * np.sqrt(trading_days)

    # Calmar Ratio
    if metrics.max_drawdown < 0:
        metrics.calmar_ratio = metrics.annual_return / abs(metrics.max_drawdown)

    # 거래 지표
    if trades is not None and not trades.empty:
        metrics.total_trades = len(trades)
        if "pnl" in trades.columns:
            wins = trades[trades["pnl"] > 0]
            losses = trades[trades["pnl"] <= 0]
            metrics.win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
            metrics.avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
            metrics.avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0

            total_win = wins["pnl"].sum() if len(wins) > 0 else 0
            total_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
            metrics.profit_factor = (
                total_win / total_loss if total_loss > 0 else float("inf")
            )

        if "holding_days" in trades.columns:
            metrics.avg_holding_days = trades["holding_days"].mean()

    # 벤치마크 대비 지표
    if benchmark is not None and len(benchmark) > 1:
        bench_returns = benchmark.pct_change().dropna()
        # 길이 맞추기
        common_len = min(len(returns), len(bench_returns))
        r = returns.iloc[:common_len].values
        b = bench_returns.iloc[:common_len].values

        if len(r) > 1 and np.std(b) > 0:
            # Beta
            covariance = np.cov(r, b)[0, 1]
            metrics.beta = covariance / np.var(b)

            # Alpha (Jensen's Alpha)
            metrics.alpha = (
                np.mean(r) - risk_free_rate / trading_days
                - metrics.beta * (np.mean(b) - risk_free_rate / trading_days)
            ) * trading_days

            # Information Ratio
            tracking_error = np.std(r - b) * np.sqrt(trading_days)
            if tracking_error > 0:
                metrics.information_ratio = (
                    (np.mean(r) - np.mean(b)) * trading_days / tracking_error
                )

    return metrics
