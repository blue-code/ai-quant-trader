"""벡터화 백테스트 엔진

시그널 기반으로 포트폴리오를 시뮬레이션하고 성과를 계산한다.
거래비용(수수료+슬리피지)과 포지션 제한을 반영한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .metrics import PerformanceMetrics, calculate_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """백테스트 설정"""

    initial_capital: float = 100_000_000  # 1억원
    commission_bps: float = 10  # 수수료 (basis points)
    slippage_bps: float = 5  # 슬리피지
    max_position_pct: float = 0.05  # 종목당 최대 비중
    max_positions: int = 20  # 최대 보유 종목 수
    max_leverage: float = 1.0
    risk_free_rate: float = 0.035  # 무위험이자율


@dataclass
class BacktestResult:
    """백테스트 결과"""

    metrics: PerformanceMetrics
    equity_curve: pd.Series
    trades: pd.DataFrame
    positions: pd.DataFrame
    daily_returns: pd.Series
    config: BacktestConfig
    strategy_name: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "metrics": self.metrics.to_dict(),
            "total_trades": len(self.trades),
            "period": {
                "start": str(self.equity_curve.index[0]) if len(self.equity_curve) > 0 else "",
                "end": str(self.equity_curve.index[-1]) if len(self.equity_curve) > 0 else "",
                "days": len(self.equity_curve),
            },
        }


class BacktestEngine:
    """벡터화 백테스트 엔진

    단일 종목 및 포트폴리오 수준의 백테스트를 지원한다.
    벡터 연산 기반으로 대량 종목도 빠르게 처리한다.
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run_single(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        strategy_name: str = "",
    ) -> BacktestResult:
        """단일 종목 백테스트

        Args:
            df: OHLCV 데이터
            signals: 시그널 시리즈 (1=매수, -1=매도, 0=관망)
            strategy_name: 전략명

        Returns:
            BacktestResult
        """
        capital = self.config.initial_capital
        commission_rate = self.config.commission_bps / 10000
        slippage_rate = self.config.slippage_bps / 10000

        prices = df["Close"].values
        dates = df["Date"] if "Date" in df.columns else df.index
        sig = signals.values

        n = len(prices)
        equity = np.zeros(n)
        position = np.zeros(n)  # 보유 주수
        cash = np.full(n, capital)
        trade_records = []

        equity[0] = capital

        for i in range(1, n):
            position[i] = position[i - 1]
            cash[i] = cash[i - 1]

            # 시그널 변경 시 거래 실행
            if sig[i] != sig[i - 1]:
                price = prices[i]

                if sig[i] == 1 and sig[i - 1] <= 0:
                    # 매수: 자본의 일정 비율만큼
                    available = cash[i] * self.config.max_position_pct * (1 / self.config.max_position_pct)
                    available = min(available, cash[i] * 0.95)  # 현금 5% 유보
                    cost_per_share = price * (1 + commission_rate + slippage_rate)
                    shares = int(available / cost_per_share) if cost_per_share > 0 else 0

                    if shares > 0:
                        total_cost = shares * cost_per_share
                        position[i] = shares
                        cash[i] -= total_cost
                        trade_records.append({
                            "date": dates.iloc[i] if hasattr(dates, "iloc") else dates[i],
                            "action": "buy",
                            "price": price,
                            "shares": shares,
                            "cost": total_cost,
                        })

                elif sig[i] <= 0 and position[i - 1] > 0:
                    # 매도: 전량 청산
                    shares = position[i - 1]
                    proceeds = shares * price * (1 - commission_rate - slippage_rate)
                    position[i] = 0
                    cash[i] += proceeds

                    # 손익 계산
                    buy_trade = None
                    for t in reversed(trade_records):
                        if t["action"] == "buy":
                            buy_trade = t
                            break

                    pnl = proceeds - (buy_trade["cost"] if buy_trade else 0)
                    holding_days = 0
                    if buy_trade:
                        try:
                            buy_date = pd.Timestamp(buy_trade["date"])
                            sell_date = pd.Timestamp(
                                dates.iloc[i] if hasattr(dates, "iloc") else dates[i]
                            )
                            holding_days = (sell_date - buy_date).days
                        except Exception:
                            pass

                    trade_records.append({
                        "date": dates.iloc[i] if hasattr(dates, "iloc") else dates[i],
                        "action": "sell",
                        "price": price,
                        "shares": shares,
                        "proceeds": proceeds,
                        "pnl": pnl,
                        "holding_days": holding_days,
                    })

            # 자산 평가
            equity[i] = cash[i] + position[i] * prices[i]

        # 결과 구성
        equity_series = pd.Series(equity, index=dates)
        trades_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()

        metrics = calculate_metrics(
            equity_curve=equity_series,
            trades=trades_df,
            risk_free_rate=self.config.risk_free_rate,
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_series,
            trades=trades_df,
            positions=pd.Series(position, index=dates),
            daily_returns=equity_series.pct_change().fillna(0),
            config=self.config,
            strategy_name=strategy_name,
        )

    def run_portfolio(
        self,
        data: dict[str, pd.DataFrame],
        signals: dict[str, pd.Series],
        strategy_name: str = "",
    ) -> BacktestResult:
        """포트폴리오 백테스트 (복수 종목)

        Args:
            data: {ticker: OHLCV DataFrame}
            signals: {ticker: 시그널 Series}
            strategy_name: 전략명

        Returns:
            통합 BacktestResult
        """
        if not data or not signals:
            raise ValueError("데이터와 시그널이 비어있음")

        # 공통 날짜 범위 추출
        all_dates = set()
        for ticker, df in data.items():
            dates = df["Date"] if "Date" in df.columns else df.index
            all_dates.update(pd.to_datetime(dates))
        common_dates = sorted(all_dates)

        n = len(common_dates)
        capital = self.config.initial_capital
        per_stock_capital = capital / min(len(data), self.config.max_positions)

        # 종목별 백테스트 실행
        total_equity = pd.Series(0.0, index=common_dates)
        all_trades = []

        for ticker in data:
            if ticker not in signals:
                continue

            # 종목별 자본 배분
            stock_config = BacktestConfig(
                initial_capital=per_stock_capital,
                commission_bps=self.config.commission_bps,
                slippage_bps=self.config.slippage_bps,
                max_position_pct=1.0,  # 종목별 엔진에서는 100%
                risk_free_rate=self.config.risk_free_rate,
            )
            engine = BacktestEngine(stock_config)
            result = engine.run_single(data[ticker], signals[ticker], ticker)

            # 날짜 기준으로 합산
            for date, eq in result.equity_curve.items():
                date = pd.Timestamp(date)
                if date in total_equity.index:
                    total_equity[date] += eq

            if not result.trades.empty:
                result.trades["ticker"] = ticker
                all_trades.append(result.trades)

        trades_df = pd.concat(all_trades) if all_trades else pd.DataFrame()

        metrics = calculate_metrics(
            equity_curve=total_equity,
            trades=trades_df,
            risk_free_rate=self.config.risk_free_rate,
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=total_equity,
            trades=trades_df,
            positions=pd.DataFrame(),
            daily_returns=total_equity.pct_change().fillna(0),
            config=self.config,
            strategy_name=strategy_name,
        )

    def walk_forward(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        n_splits: int = 5,
        oos_pct: float = 0.2,
    ) -> list[BacktestResult]:
        """Walk-Forward 검증

        과적합 방지를 위해 시계열을 분할하여 순차적으로 검증한다.

        Args:
            df: OHLCV 데이터
            signals: 시그널 시리즈
            n_splits: 분할 수
            oos_pct: Out-of-Sample 비율

        Returns:
            분할별 BacktestResult 리스트
        """
        n = len(df)
        results = []

        for i in range(n_splits):
            # 각 분할의 시작/끝 인덱스
            split_size = n // n_splits
            start = i * split_size
            end = min((i + 1) * split_size, n)

            # OOS 구간
            oos_start = start + int(split_size * (1 - oos_pct))
            if oos_start >= end:
                continue

            df_oos = df.iloc[oos_start:end].reset_index(drop=True)
            sig_oos = signals.iloc[oos_start:end].reset_index(drop=True)

            result = self.run_single(df_oos, sig_oos, f"WF_split_{i + 1}")
            results.append(result)

        return results
