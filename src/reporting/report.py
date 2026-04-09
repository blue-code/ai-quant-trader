"""성과 리포트 생성기

백테스트 결과를 시각화하고, HTML/마크다운 리포트를 생성한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult
from ..backtest.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class ReportGenerator:
    """성과 리포트 생성기"""

    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown(
        self,
        result: BacktestResult,
        strategy_description: str = "",
    ) -> str:
        """마크다운 리포트 생성"""
        m = result.metrics
        report = f"""# 백테스트 성과 리포트

**전략명**: {result.strategy_name}
**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 1. 전략 개요

{strategy_description if strategy_description else '(전략 설명 없음)'}

## 2. 핵심 성과 지표

| 지표 | 값 |
|---|---|
| 총 수익률 | {m.total_return:.2%} |
| 연환산 수익률 | {m.annual_return:.2%} |
| 변동성 (연환산) | {m.volatility:.2%} |
| Sharpe Ratio | {m.sharpe_ratio:.2f} |
| Sortino Ratio | {m.sortino_ratio:.2f} |
| Calmar Ratio | {m.calmar_ratio:.2f} |
| 최대 낙폭 (MDD) | {m.max_drawdown:.2%} |
| MDD 지속일 | {m.max_drawdown_duration}일 |

## 3. 거래 분석

| 지표 | 값 |
|---|---|
| 총 거래 수 | {m.total_trades}회 |
| 승률 | {m.win_rate:.2%} |
| 손익비 (Profit Factor) | {m.profit_factor:.2f} |
| 평균 수익 (건당) | {m.avg_win:.2%} |
| 평균 손실 (건당) | {m.avg_loss:.2%} |
| 평균 보유기간 | {m.avg_holding_days:.1f}일 |

## 4. 벤치마크 대비

| 지표 | 값 |
|---|---|
| Alpha | {m.alpha:.4f} |
| Beta | {m.beta:.2f} |
| Information Ratio | {m.information_ratio:.2f} |

## 5. 수익곡선

```
기간: {result.equity_curve.index[0] if len(result.equity_curve) > 0 else 'N/A'} ~ {result.equity_curve.index[-1] if len(result.equity_curve) > 0 else 'N/A'}
거래일: {len(result.equity_curve)}일
```

## 6. 월별 수익률

{self._monthly_returns_table(result)}

---
*AI Quant Trader에 의해 자동 생성된 리포트*
"""
        return report

    def generate_html(
        self,
        result: BacktestResult,
        strategy_description: str = "",
    ) -> str:
        """HTML 리포트 생성 (차트 포함)"""
        m = result.metrics
        equity = result.equity_curve

        # 수익곡선 SVG 생성 (외부 의존성 없이)
        equity_chart = self._generate_svg_chart(equity, "수익곡선")
        drawdown_chart = self._generate_drawdown_svg(equity)

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>백테스트 리포트 - {result.strategy_name}</title>
    <style>
        body {{ font-family: 'Malgun Gothic', sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; }}
        h2 {{ color: #16213e; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #16213e; color: white; }}
        tr:hover {{ background: #f8f9fa; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #e94560; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .positive {{ color: #2ecc71; }}
        .negative {{ color: #e74c3c; }}
        .chart {{ margin: 20px 0; }}
        svg {{ width: 100%; height: auto; }}
    </style>
</head>
<body>
<div class="container">
    <h1>{result.strategy_name} - 백테스트 리포트</h1>
    <p>생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value {'positive' if m.total_return >= 0 else 'negative'}">{m.total_return:.2%}</div>
            <div class="metric-label">총 수익률</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{m.sharpe_ratio:.2f}</div>
            <div class="metric-label">Sharpe Ratio</div>
        </div>
        <div class="metric-card">
            <div class="metric-value negative">{m.max_drawdown:.2%}</div>
            <div class="metric-label">최대 낙폭</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{m.win_rate:.1%}</div>
            <div class="metric-label">승률</div>
        </div>
    </div>

    <h2>수익곡선</h2>
    <div class="chart">{equity_chart}</div>

    <h2>낙폭 (Drawdown)</h2>
    <div class="chart">{drawdown_chart}</div>

    <h2>상세 성과 지표</h2>
    <table>
        <tr><th>구분</th><th>지표</th><th>값</th></tr>
        <tr><td>수익률</td><td>총 수익률</td><td>{m.total_return:.2%}</td></tr>
        <tr><td>수익률</td><td>연환산 수익률</td><td>{m.annual_return:.2%}</td></tr>
        <tr><td>리스크</td><td>변동성 (연환산)</td><td>{m.volatility:.2%}</td></tr>
        <tr><td>리스크</td><td>최대 낙폭</td><td>{m.max_drawdown:.2%}</td></tr>
        <tr><td>리스크</td><td>MDD 지속일</td><td>{m.max_drawdown_duration}일</td></tr>
        <tr><td>위험조정</td><td>Sharpe Ratio</td><td>{m.sharpe_ratio:.2f}</td></tr>
        <tr><td>위험조정</td><td>Sortino Ratio</td><td>{m.sortino_ratio:.2f}</td></tr>
        <tr><td>위험조정</td><td>Calmar Ratio</td><td>{m.calmar_ratio:.2f}</td></tr>
        <tr><td>거래</td><td>총 거래 수</td><td>{m.total_trades}회</td></tr>
        <tr><td>거래</td><td>승률</td><td>{m.win_rate:.2%}</td></tr>
        <tr><td>거래</td><td>손익비</td><td>{m.profit_factor:.2f}</td></tr>
        <tr><td>거래</td><td>평균 보유기간</td><td>{m.avg_holding_days:.1f}일</td></tr>
    </table>
</div>
</body>
</html>"""
        return html

    def save(
        self,
        result: BacktestResult,
        strategy_description: str = "",
        fmt: str = "html",
    ) -> Path:
        """리포트 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = result.strategy_name or "strategy"
        safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)

        if fmt == "html":
            content = self.generate_html(result, strategy_description)
            path = self.output_dir / f"{safe_name}_{timestamp}.html"
        else:
            content = self.generate_markdown(result, strategy_description)
            path = self.output_dir / f"{safe_name}_{timestamp}.md"

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"리포트 저장: {path}")
        return path

    def _monthly_returns_table(self, result: BacktestResult) -> str:
        """월별 수익률 테이블 생성"""
        if result.equity_curve.empty:
            return "(데이터 없음)"

        try:
            eq = result.equity_curve.copy()
            eq.index = pd.to_datetime(eq.index)
            monthly = eq.resample("ME").last().pct_change().dropna()

            if monthly.empty:
                return "(월별 데이터 부족)"

            lines = ["| 연월 | 수익률 |", "|---|---|"]
            for date, ret in monthly.items():
                sign = "+" if ret >= 0 else ""
                lines.append(f"| {date.strftime('%Y-%m')} | {sign}{ret:.2%} |")
            return "\n".join(lines)

        except Exception:
            return "(월별 수익률 계산 불가)"

    @staticmethod
    def _generate_svg_chart(series: pd.Series, title: str) -> str:
        """간단한 SVG 라인 차트 생성"""
        if series.empty:
            return "<p>데이터 없음</p>"

        values = series.values
        n = len(values)
        if n < 2:
            return "<p>데이터 부족</p>"

        width, height = 800, 300
        padding = 50

        # 정규화
        vmin, vmax = np.min(values), np.max(values)
        if vmax == vmin:
            vmax = vmin + 1

        points = []
        for i, v in enumerate(values):
            x = padding + (i / (n - 1)) * (width - 2 * padding)
            y = height - padding - ((v - vmin) / (vmax - vmin)) * (height - 2 * padding)
            points.append(f"{x:.1f},{y:.1f}")

        polyline = " ".join(points)
        color = "#2ecc71" if values[-1] >= values[0] else "#e74c3c"

        svg = f"""<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" fill="#fafafa" rx="4"/>
  <text x="{width//2}" y="20" text-anchor="middle" font-size="14" fill="#333">{title}</text>
  <polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>
  <text x="{padding}" y="{height-10}" font-size="11" fill="#999">{vmin:,.0f}</text>
  <text x="{width-padding}" y="{height-10}" font-size="11" fill="#999" text-anchor="end">{vmax:,.0f}</text>
</svg>"""
        return svg

    @staticmethod
    def _generate_drawdown_svg(equity: pd.Series) -> str:
        """낙폭 SVG 차트"""
        if equity.empty or len(equity) < 2:
            return "<p>데이터 부족</p>"

        values = equity.values
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax

        n = len(drawdown)
        width, height = 800, 200
        padding = 50

        vmin = np.min(drawdown)
        if vmin == 0:
            vmin = -0.01

        points = [f"{padding:.1f},{padding:.1f}"]
        for i, v in enumerate(drawdown):
            x = padding + (i / (n - 1)) * (width - 2 * padding)
            y = padding + (v / vmin) * (height - 2 * padding)
            points.append(f"{x:.1f},{y:.1f}")
        points.append(f"{width-padding:.1f},{padding:.1f}")

        polygon = " ".join(points)

        svg = f"""<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" fill="#fafafa" rx="4"/>
  <text x="{width//2}" y="20" text-anchor="middle" font-size="14" fill="#333">Drawdown</text>
  <polygon points="{polygon}" fill="rgba(231,76,60,0.3)" stroke="#e74c3c" stroke-width="1"/>
  <text x="{padding}" y="{height-10}" font-size="11" fill="#999">MDD: {vmin:.2%}</text>
</svg>"""
        return svg
