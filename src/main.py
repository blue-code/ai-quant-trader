"""AI Quant Trader - CLI 진입점

전체 파이프라인을 CLI에서 실행한다.
  python -m src.main ideate    → 전략 생성
  python -m src.main backtest  → 백테스트
  python -m src.main optimize  → 트리 탐색 최적화
  python -m src.main trade     → 실거래
  python -m src.main run       → 전체 파이프라인
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ai_quant_trader.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ai_quant_trader")


def load_config(config_path: str = "config/default.yaml") -> dict:
    """설정 파일 로드"""
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


@click.group()
@click.option("--config", default="config/default.yaml", help="설정 파일 경로")
@click.pass_context
def cli(ctx, config):
    """AI Quant Trader - AI 기반 자율 퀀트 트레이딩 시스템"""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


@cli.command()
@click.option("--universe", default="kospi_bluechip", help="투자 유니버스")
@click.option("--num-strategies", default=10, help="생성할 전략 수")
@click.option("--output", default="./results/ideas.json", help="출력 경로")
@click.pass_context
def ideate(ctx, universe, num_strategies, output):
    """Stage 1: LLM 기반 트레이딩 전략 생성"""
    from .llm.client import LLMClient, LLMConfig
    from .ideation.generator import StrategyGenerator
    from .data.universe import StockUniverse

    cfg = ctx.obj["config"]
    llm_cfg = LLMConfig(**cfg["llm"])
    llm = LLMClient(llm_cfg)

    stock_universe = StockUniverse.from_preset(universe)
    generator = StrategyGenerator(llm)

    logger.info(f"전략 생성 시작: {universe}, {num_strategies}개")
    ideas = generator.generate(stock_universe, num_strategies)

    generator.save(ideas, output)
    logger.info(f"전략 {len(ideas)}개 저장: {output}")

    for i, idea in enumerate(ideas, 1):
        click.echo(f"  {i}. {idea.name}: {idea.description[:60]}")


@cli.command()
@click.option("--strategies", required=True, help="전략 JSON 파일 경로")
@click.option("--universe", default="kospi_bluechip", help="투자 유니버스")
@click.option("--start", default="2020-01-01", help="시작일")
@click.option("--end", default="2024-12-31", help="종료일")
@click.option("--output-dir", default="./results", help="결과 저장 경로")
@click.pass_context
def backtest(ctx, strategies, universe, start, end, output_dir):
    """Stage 2: 전략 백테스트 실행"""
    from .llm.client import LLMClient, LLMConfig
    from .ideation.generator import StrategyGenerator
    from .data.universe import StockUniverse
    from .data.loader import MarketDataLoader
    from .backtest.engine import BacktestEngine, BacktestConfig
    from .backtest.interpreter import StrategyInterpreter
    from .reporting.report import ReportGenerator

    cfg = ctx.obj["config"]

    # 전략 로드
    ideas = StrategyGenerator.load(strategies)
    logger.info(f"전략 {len(ideas)}개 로드")

    # 데이터 로드
    stock_universe = StockUniverse.from_preset(universe)
    loader = MarketDataLoader(market=stock_universe.market)

    logger.info("시장 데이터 로드 중...")
    data = loader.load_multiple(stock_universe.tickers[:5], start, end)

    if not data:
        click.echo("데이터 로드 실패")
        return

    primary_ticker = list(data.keys())[0]
    primary_df = data[primary_ticker]
    logger.info(f"대표 종목: {primary_ticker} ({len(primary_df)}일)")

    # 백테스트 실행
    bt_config = BacktestConfig(**{
        k: v for k, v in cfg.get("trading", {}).items()
        if k in BacktestConfig.__dataclass_fields__
    })
    engine = BacktestEngine(bt_config)
    interpreter = StrategyInterpreter()
    reporter = ReportGenerator(output_dir)

    results = []
    for idea in ideas:
        click.echo(f"\n  백테스트: {idea.name}")
        exec_result = interpreter.execute(idea.code, primary_df)

        if not exec_result.success:
            click.echo(f"    실패: {exec_result.error[:80]}")
            continue

        bt_result = engine.run_single(primary_df, exec_result.signals, idea.name)
        results.append((idea, bt_result))

        m = bt_result.metrics
        click.echo(
            f"    Sharpe: {m.sharpe_ratio:.2f} | "
            f"수익률: {m.total_return:.2%} | "
            f"MDD: {m.max_drawdown:.2%}"
        )

        # 리포트 생성
        reporter.save(bt_result, idea.description)

    # 결과 요약
    if results:
        best_idea, best_result = max(results, key=lambda x: x[1].metrics.sharpe_ratio)
        click.echo(f"\n  최고 전략: {best_idea.name}")
        click.echo(f"  Sharpe: {best_result.metrics.sharpe_ratio:.2f}")


@cli.command()
@click.option("--strategies", help="전략 JSON 파일 (없으면 자동 생성)")
@click.option("--universe", default="kospi_bluechip", help="투자 유니버스")
@click.option("--start", default="2020-01-01", help="시작일")
@click.option("--end", default="2024-12-31", help="종료일")
@click.option("--output-dir", default="./results", help="결과 저장 경로")
@click.pass_context
def optimize(ctx, strategies, universe, start, end, output_dir):
    """Stage 2-4: 트리 탐색 기반 전략 최적화 (전체 파이프라인)"""
    from .llm.client import LLMClient, LLMConfig
    from .ideation.generator import StrategyGenerator
    from .data.universe import StockUniverse
    from .data.loader import MarketDataLoader
    from .treesearch.agent_manager import AgentManager, PipelineConfig

    cfg = ctx.obj["config"]
    llm_cfg = LLMConfig(**cfg["llm"])
    llm = LLMClient(llm_cfg)

    # 데이터 준비
    stock_universe = StockUniverse.from_preset(universe)
    loader = MarketDataLoader(market=stock_universe.market)

    logger.info("시장 데이터 로드 중...")
    data = loader.load_multiple(stock_universe.tickers[:10], start, end)

    if not data:
        click.echo("데이터 로드 실패")
        return

    # 전략 아이디어 로드 또는 생성
    ideas = None
    if strategies:
        ideas = StrategyGenerator.load(strategies)

    # 파이프라인 설정
    pipeline_cfg = PipelineConfig(
        stages=AgentManager.DEFAULT_STAGES,
        num_workers=cfg.get("treesearch", {}).get("num_workers", 4),
        output_dir=output_dir,
    )

    # 파이프라인 실행
    manager = AgentManager(llm, pipeline_cfg)
    journal = manager.run(stock_universe, data, ideas)

    # 결과 출력
    best = journal.get_best()
    if best and best.metrics:
        click.echo(f"\n=== 최적화 완료 ===")
        click.echo(f"탐색 노드: {len(journal)}개")
        click.echo(f"최고 전략 Sharpe: {best.metrics.sharpe_ratio:.2f}")
        click.echo(f"최고 전략 수익률: {best.metrics.total_return:.2%}")
        click.echo(f"결과 저장: {output_dir}")


@cli.command()
@click.option("--strategy", required=True, help="전략 JSON 파일 경로")
@click.option("--mode", default="paper", type=click.Choice(["paper", "live"]))
@click.option("--universe", default="kospi_bluechip", help="투자 유니버스")
@click.pass_context
def trade(ctx, strategy, mode, universe):
    """실거래 / 모의투자 실행"""
    from .data.universe import StockUniverse
    from .data.loader import MarketDataLoader
    from .backtest.interpreter import StrategyInterpreter
    from .execution.broker import PaperBroker, KISBroker
    from .execution.risk import RiskManager, RiskLimits

    cfg = ctx.obj["config"]

    # 전략 로드
    with open(strategy, "r", encoding="utf-8") as f:
        strategy_data = json.load(f)

    strategy_code = strategy_data.get("strategy", {}).get("code", "")
    strategy_params = strategy_data.get("strategy", {}).get("params", {})

    if not strategy_code:
        click.echo("전략 코드가 없음")
        return

    # 브로커 설정
    if mode == "paper":
        broker = PaperBroker(
            initial_capital=cfg.get("trading", {}).get("initial_capital", 100_000_000)
        )
    else:
        broker = KISBroker(paper_mode=False)

    if not broker.connect():
        click.echo("브로커 연결 실패")
        return

    # 리스크 매니저
    risk_limits = RiskLimits(
        max_position_pct=cfg.get("trading", {}).get("max_position_pct", 0.05),
        max_positions=cfg.get("trading", {}).get("max_positions", 20),
    )
    risk_mgr = RiskManager(risk_limits)
    account = broker.get_account()
    risk_mgr.initialize(account)

    # 유니버스 데이터 로드 (최근 데이터)
    stock_universe = StockUniverse.from_preset(universe)
    loader = MarketDataLoader(market=stock_universe.market)

    click.echo(f"[{mode}] 트레이딩 시작")
    click.echo(risk_mgr.status_report(account))

    # 시그널 생성
    interpreter = StrategyInterpreter()
    signals = {}

    for ticker in stock_universe.tickers[:20]:
        try:
            df = loader.load(ticker, "2024-01-01", "2024-12-31")
            if df.empty:
                continue

            result = interpreter.execute(strategy_code, df, strategy_params)
            if result.success and result.signals is not None:
                last_signal = int(result.signals.iloc[-1])
                if last_signal != 0:
                    signals[ticker] = last_signal
        except Exception as e:
            logger.warning(f"{ticker} 시그널 생성 실패: {e}")

    click.echo(f"\n시그널: 매수 {sum(1 for v in signals.values() if v == 1)}개, "
               f"매도 {sum(1 for v in signals.values() if v == -1)}개")

    # 주문 실행
    for ticker, signal in signals.items():
        from .execution.broker import Order, OrderSide

        side = OrderSide.BUY if signal == 1 else OrderSide.SELL
        price = broker.get_current_price(ticker)
        if price <= 0:
            continue

        qty = int(account.total_equity * risk_limits.max_position_pct / price)
        order = Order(ticker=ticker, side=side, quantity=qty)

        # 리스크 체크
        approved, reason = risk_mgr.check_order(order, account)
        if not approved:
            click.echo(f"  [{ticker}] 주문 거부: {reason}")
            continue

        order = broker.place_order(order)
        risk_mgr.record_order(order)
        click.echo(
            f"  [{ticker}] {order.side.value} x{order.filled_quantity} "
            f"@ {order.filled_price:,.0f} - {order.status.value}"
        )

    # 최종 현황
    account = broker.get_account()
    risk_mgr.update_pnl(account)
    click.echo(f"\n{risk_mgr.status_report(account)}")


@cli.command()
@click.option("--universe", default="kospi_bluechip", help="투자 유니버스")
@click.option("--num-strategies", default=10, help="생성할 전략 수")
@click.option("--start", default="2020-01-01", help="시작일")
@click.option("--end", default="2024-12-31", help="종료일")
@click.option("--output-dir", default="./results", help="결과 저장 경로")
@click.pass_context
def run(ctx, universe, num_strategies, start, end, output_dir):
    """전체 파이프라인 실행 (생성 → 백테스트 → 최적화 → 리포트)"""
    click.echo("=== AI Quant Trader 전체 파이프라인 ===\n")

    # ideate → optimize 순서로 실행
    ctx.invoke(ideate, universe=universe, num_strategies=num_strategies,
               output=f"{output_dir}/ideas.json")
    ctx.invoke(optimize, strategies=f"{output_dir}/ideas.json",
               universe=universe, start=start, end=end, output_dir=output_dir)


if __name__ == "__main__":
    cli()
