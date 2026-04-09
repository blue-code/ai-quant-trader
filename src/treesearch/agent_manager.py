"""에이전트 매니저 (4단계 파이프라인 오케스트레이터)

AI-Scientist-v2의 AgentManager를 트레이딩에 적용.
4단계(기본전략→파라미터튜닝→팩터엔지니어링→로버스트니스검증)를
자동으로 진행한다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..llm.client import LLMClient
from ..llm.prompts import StrategyPrompts
from ..backtest.engine import BacktestEngine, BacktestConfig
from ..backtest.interpreter import StrategyInterpreter
from ..ideation.generator import StrategyGenerator, StrategyIdea
from ..data.universe import StockUniverse
from ..data.loader import MarketDataLoader
from .node import StrategyNode, NodeStatus
from .journal import StrategyJournal
from .parallel_agent import ParallelOptimizer

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """스테이지별 설정"""

    name: str
    num_iterations: int
    description: str


@dataclass
class PipelineConfig:
    """전체 파이프라인 설정"""

    stages: list[StageConfig]
    num_workers: int = 4
    max_nodes: int = 150
    output_dir: str = "./results"


class AgentManager:
    """4단계 전략 최적화 매니저

    AI-Scientist-v2의 AgentManager처럼 4단계를 순차적으로 진행하되,
    각 단계 내에서는 병렬 탐색을 수행한다.

    Stage 1: 기본 전략 생성 - 다양한 팩터 조합으로 초기 전략 탐색
    Stage 2: 파라미터 최적화 - 진입/청산 임계값, 룩백 기간 최적화
    Stage 3: 팩터 엔지니어링 - 신호 결합, 앙상블, 매크로 지표 통합
    Stage 4: 로버스트니스 검증 - Walk-Forward 검증, 스트레스 테스트
    """

    DEFAULT_STAGES = [
        StageConfig("기본 전략", 20, "기본 팩터 조합으로 초기 전략 생성"),
        StageConfig("파라미터 최적화", 12, "진입/청산 임계값, 룩백 기간 최적화"),
        StageConfig("팩터 엔지니어링", 12, "신호 결합, 앙상블, 매크로 지표 통합"),
        StageConfig("로버스트니스 검증", 18, "Walk-Forward 검증, 스트레스 테스트"),
    ]

    def __init__(
        self,
        llm: LLMClient,
        config: PipelineConfig | None = None,
    ):
        self.llm = llm
        self.config = config or PipelineConfig(stages=self.DEFAULT_STAGES)
        self.journal = StrategyJournal()
        self.generator = StrategyGenerator(llm)

        # 백테스트 인프라
        self.bt_engine = BacktestEngine()
        self.interpreter = StrategyInterpreter()
        self.optimizer = ParallelOptimizer(
            llm=llm,
            backtest_engine=self.bt_engine,
            interpreter=self.interpreter,
            num_workers=self.config.num_workers,
        )

        # 결과 저장 경로
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        universe: StockUniverse,
        data: dict[str, pd.DataFrame],
        ideas: list[StrategyIdea] | None = None,
    ) -> StrategyJournal:
        """전체 파이프라인 실행

        Args:
            universe: 투자 유니버스
            data: {ticker: OHLCV DataFrame}
            ideas: 사전 생성된 전략 아이디어 (None이면 자동 생성)

        Returns:
            최종 StrategyJournal
        """
        logger.info(f"=== 파이프라인 시작: {universe.name} ===")

        # 대표 종목 데이터 (전략 검증용)
        primary_ticker = universe.tickers[0]
        primary_df = data.get(primary_ticker)
        if primary_df is None:
            raise ValueError(f"대표 종목 {primary_ticker} 데이터 없음")

        # Stage 1: 기본 전략 생성
        if ideas is None:
            ideas = self.generator.generate(
                universe=universe,
                num_strategies=self.config.stages[0].num_iterations,
            )
        self._run_stage_1(ideas, primary_df)

        # Stage 2: 파라미터 최적화
        self._run_stage_2(primary_df)

        # Stage 3: 팩터 엔지니어링
        self._run_stage_3(primary_df, data)

        # Stage 4: 로버스트니스 검증
        self._run_stage_4(primary_df, data)

        # 결과 저장
        self.journal.save(self.output_dir / "journal.json")
        self._save_best_strategy()

        logger.info("=== 파이프라인 완료 ===")
        best = self.journal.get_best()
        if best and best.metrics:
            logger.info(f"최고 전략: Sharpe {best.metrics.sharpe_ratio:.2f}")

        return self.journal

    def _run_stage_1(
        self, ideas: list[StrategyIdea], df: pd.DataFrame
    ) -> None:
        """Stage 1: 기본 전략 생성

        AI-Scientist-v2의 Stage 1 (Basic Implementation)에 대응.
        다양한 전략 아이디어를 코드로 변환하고 백테스트한다.
        """
        stage_config = self.config.stages[0]
        logger.info(f"\n--- Stage 1: {stage_config.name} ({stage_config.num_iterations} iterations) ---")

        idea_descriptions = [
            f"{idea.name}: {idea.description}\n가설: {idea.hypothesis}\n"
            f"진입: {idea.entry_rule}\n청산: {idea.exit_rule}"
            for idea in ideas
        ]

        self.optimizer.run_parallel_exploration(
            ideas=idea_descriptions[:stage_config.num_iterations],
            df=df,
            journal=self.journal,
            stage=1,
        )

        # 체크포인트 저장
        self.journal.save(self.output_dir / "checkpoint_stage1.json")
        logger.info(self.journal.stage_summary(1))

    def _run_stage_2(self, df: pd.DataFrame) -> None:
        """Stage 2: 파라미터 최적화

        AI-Scientist-v2의 Stage 2 (Baseline Tuning)에 대응.
        Stage 1의 상위 전략들의 파라미터를 최적화한다.
        """
        stage_config = self.config.stages[1]
        logger.info(f"\n--- Stage 2: {stage_config.name} ({stage_config.num_iterations} iterations) ---")

        # Stage 1 상위 전략 선택
        top_strategies = self.journal.get_top_k(k=4, stage=1)
        if not top_strategies:
            logger.warning("Stage 1에서 성공한 전략 없음. Stage 2 건너뜀")
            return

        # 상위 전략별 파라미터 최적화
        iterations_per_strategy = max(
            1, stage_config.num_iterations // len(top_strategies)
        )

        for parent in top_strategies:
            for _ in range(iterations_per_strategy):
                children = self.optimizer.improve_strategy(
                    parent_node=parent,
                    journal=self.journal,
                    df=df,
                )
                for child in children:
                    child.stage = 2
                    self.journal.append(child)

        self.journal.save(self.output_dir / "checkpoint_stage2.json")
        logger.info(self.journal.stage_summary(2))

    def _run_stage_3(
        self, primary_df: pd.DataFrame, all_data: dict[str, pd.DataFrame]
    ) -> None:
        """Stage 3: 팩터 엔지니어링

        AI-Scientist-v2의 Stage 3 (Creative Research)에 대응.
        개별 전략을 결합하여 앙상블 전략을 구축한다.
        """
        stage_config = self.config.stages[2]
        logger.info(f"\n--- Stage 3: {stage_config.name} ({stage_config.num_iterations} iterations) ---")

        # 상위 전략들의 성과 요약
        top_all = self.journal.get_top_k(k=5)
        if not top_all:
            logger.warning("성공한 전략 없음. Stage 3 건너뜀")
            return

        individual_results = "\n".join([
            f"### {node.plan[:50]}\n"
            f"- Sharpe: {node.metrics.sharpe_ratio:.2f}\n"
            f"- 수익률: {node.metrics.total_return:.2%}\n"
            f"- MDD: {node.metrics.max_drawdown:.2%}\n"
            f"```python\n{node.code[:500]}\n```"
            for node in top_all if node.metrics
        ])

        # 앙상블 전략 생성
        prompt = StrategyPrompts.FACTOR_ENGINEERING.format(
            individual_results=individual_results,
        )

        for i in range(min(stage_config.num_iterations, 5)):
            response = self.llm.generate(
                prompt=prompt,
                system_message=StrategyPrompts.SYSTEM,
                temperature=0.7 + (i * 0.05),
            )

            node = StrategyNode(
                idx=-1,
                stage=3,
                code=response,
                plan=f"앙상블 전략 v{i + 1}",
            )
            node = self.journal.append(node)
            node = self.optimizer.execute_and_evaluate(node, primary_df)
            if node.status == NodeStatus.FAILED:
                node = self.optimizer.debug_strategy(node, primary_df)

        self.journal.save(self.output_dir / "checkpoint_stage3.json")
        logger.info(self.journal.stage_summary(3))

    def _run_stage_4(
        self, primary_df: pd.DataFrame, all_data: dict[str, pd.DataFrame]
    ) -> None:
        """Stage 4: 로버스트니스 검증

        AI-Scientist-v2의 Stage 4 (Ablation Studies)에 대응.
        Walk-Forward 검증과 다중 종목 테스트로 로버스트니스를 확인한다.
        """
        stage_config = self.config.stages[3]
        logger.info(f"\n--- Stage 4: {stage_config.name} ---")

        best = self.journal.get_best()
        if not best or not best.is_successful:
            logger.warning("검증할 전략 없음. Stage 4 건너뜀")
            return

        # Walk-Forward 검증
        result = self.interpreter.execute(best.code, primary_df, best.params)
        if result.success and result.signals is not None:
            wf_results = self.bt_engine.walk_forward(
                df=primary_df,
                signals=result.signals,
                n_splits=5,
            )

            wf_sharpes = [
                r.metrics.sharpe_ratio for r in wf_results
                if r.metrics.sharpe_ratio != 0
            ]

            if wf_sharpes:
                avg_sharpe = sum(wf_sharpes) / len(wf_sharpes)
                min_sharpe = min(wf_sharpes)
                logger.info(
                    f"Walk-Forward 결과: 평균 Sharpe {avg_sharpe:.2f}, "
                    f"최소 {min_sharpe:.2f}"
                )

        # 다중 종목 테스트
        multi_results = []
        for ticker, df in list(all_data.items())[:10]:
            result = self.interpreter.execute(best.code, df, best.params)
            if result.success and result.signals is not None:
                bt = self.bt_engine.run_single(df, result.signals, ticker)
                multi_results.append((ticker, bt.metrics.sharpe_ratio))

        if multi_results:
            avg = sum(s for _, s in multi_results) / len(multi_results)
            positive = sum(1 for _, s in multi_results if s > 0)
            logger.info(
                f"다중 종목 테스트: {len(multi_results)}종목 평균 Sharpe {avg:.2f}, "
                f"양수 비율 {positive}/{len(multi_results)}"
            )

        # 로버스트니스 분석 (LLM)
        wf_summary = "\n".join([
            f"- Split {i+1}: Sharpe {r.metrics.sharpe_ratio:.2f}, "
            f"수익률 {r.metrics.total_return:.2%}"
            for i, r in enumerate(wf_results)
        ]) if 'wf_results' in dir() else "Walk-Forward 미실행"

        analysis = self.llm.generate_json(
            prompt=StrategyPrompts.ROBUSTNESS_ANALYSIS.format(
                strategy_code=best.code,
                walk_forward_results=wf_summary,
            ),
            system_message=StrategyPrompts.SYSTEM,
        )

        # 분석 결과를 메타데이터에 저장
        best.metadata["robustness_analysis"] = analysis
        best.metadata["walk_forward_sharpes"] = wf_sharpes if 'wf_sharpes' in dir() else []
        best.metadata["multi_stock_results"] = multi_results

        self.journal.save(self.output_dir / "checkpoint_stage4.json")

    def _save_best_strategy(self) -> None:
        """최고 전략을 별도 파일로 저장"""
        best = self.journal.get_best()
        if not best:
            return

        output = {
            "strategy": best.to_dict(),
            "ancestry": [n.to_dict() for n in self.journal.get_ancestry(best)],
            "stage_summaries": {
                i: self.journal.stage_summary(i) for i in range(1, 5)
            },
        }

        path = self.output_dir / "best_strategy.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        # 전략 코드만 별도 저장
        code_path = self.output_dir / "best_strategy.py"
        with open(code_path, "w", encoding="utf-8") as f:
            # 코드 블록에서 순수 코드 추출
            code = best.code
            import re
            match = re.search(r"```python\s*([\s\S]*?)\s*```", code)
            if match:
                code = match.group(1)
            f.write(code)

        logger.info(f"최고 전략 저장: {path}")
