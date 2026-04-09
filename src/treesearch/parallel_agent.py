"""병렬 전략 최적화 에이전트

AI-Scientist-v2의 ParallelAgent + MinimalAgent를 트레이딩에 적용.
LLM이 전략 코드를 생성·디버그·개선하고,
백테스트로 검증하는 사이클을 병렬로 실행한다.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from ..llm.client import LLMClient
from ..llm.prompts import StrategyPrompts
from ..backtest.engine import BacktestEngine, BacktestConfig
from ..backtest.interpreter import StrategyInterpreter, ExecutionResult
from .node import StrategyNode, NodeStatus
from .journal import StrategyJournal

logger = logging.getLogger(__name__)


class ParallelOptimizer:
    """병렬 전략 탐색 및 최적화

    AI-Scientist-v2의 ParallelAgent처럼,
    복수 전략을 동시에 생성·실행·평가한다.

    핵심 사이클: Draft → Execute → Debug → Improve
    """

    def __init__(
        self,
        llm: LLMClient,
        backtest_engine: BacktestEngine,
        interpreter: StrategyInterpreter,
        num_workers: int = 4,
        max_debug_attempts: int = 3,
        debug_prob: float = 0.5,
    ):
        self.llm = llm
        self.engine = backtest_engine
        self.interpreter = interpreter
        self.num_workers = num_workers
        self.max_debug_attempts = max_debug_attempts
        self.debug_prob = debug_prob

    def draft_strategy(
        self,
        idea_description: str,
        journal: StrategyJournal,
        stage: int = 1,
    ) -> StrategyNode:
        """새로운 전략 코드 생성 (Draft Phase)

        AI-Scientist-v2의 MinimalAgent._draft()에 대응.
        """
        # 이전 스테이지 최고 성과를 컨텍스트로 제공
        context = ""
        if stage > 1:
            best = journal.get_best(stage=stage - 1)
            if best:
                context = f"\n\n## 이전 스테이지 최고 전략\n```python\n{best.code}\n```\n성과: {best.metrics.summary() if best.metrics else '없음'}"

        prompt = StrategyPrompts.STRATEGY_DRAFT.format(
            strategy_idea=idea_description + context,
            strategy_name=idea_description[:50],
        )

        response = self.llm.generate(
            prompt=prompt,
            system_message=StrategyPrompts.SYSTEM,
        )

        node = StrategyNode(
            idx=-1,  # journal.append에서 할당
            stage=stage,
            code=response,
            plan=idea_description,
        )
        return node

    def execute_and_evaluate(
        self,
        node: StrategyNode,
        df: pd.DataFrame,
    ) -> StrategyNode:
        """전략 실행 + 백테스트 (Execute Phase)

        코드를 실행하여 시그널을 생성하고, 백테스트로 성과를 평가한다.
        """
        node.status = NodeStatus.RUNNING

        # 전략 코드 실행
        result = self.interpreter.execute(node.code, df, node.params or None)

        if not result.success:
            node.status = NodeStatus.FAILED
            node.error = result.error
            return node

        # 백테스트 실행
        try:
            bt_result = self.engine.run_single(
                df=df,
                signals=result.signals,
                strategy_name=node.plan[:50],
            )
            node.metrics = bt_result.metrics
            node.params = result.params or {}
            node.status = NodeStatus.SUCCESS
            logger.info(
                f"[Node {node.idx}] 성공 - Sharpe: {node.metrics.sharpe_ratio:.2f}, "
                f"수익률: {node.metrics.total_return:.2%}"
            )
        except Exception as e:
            node.status = NodeStatus.FAILED
            node.error = str(e)

        return node

    def debug_strategy(
        self,
        node: StrategyNode,
        df: pd.DataFrame,
    ) -> StrategyNode:
        """전략 코드 디버그 (Debug Phase)

        AI-Scientist-v2의 MinimalAgent._debug()에 대응.
        실행 실패한 코드를 LLM으로 수정한다.
        """
        if node.status != NodeStatus.FAILED or not node.error:
            return node

        for attempt in range(self.max_debug_attempts):
            # 데이터 샘플 생성
            sample = df.head(5).to_string() if len(df) > 0 else "데이터 없음"

            prompt = StrategyPrompts.STRATEGY_DEBUG.format(
                strategy_code=node.code,
                error_message=node.error,
                data_sample=sample,
            )

            response = self.llm.generate(
                prompt=prompt,
                system_message=StrategyPrompts.SYSTEM,
            )

            # 수정된 코드로 재실행
            node.code = response
            node = self.execute_and_evaluate(node, df)

            if node.is_successful:
                logger.info(f"[Node {node.idx}] 디버그 성공 (시도 {attempt + 1})")
                return node

        logger.warning(f"[Node {node.idx}] 디버그 실패 ({self.max_debug_attempts}회 시도)")
        return node

    def improve_strategy(
        self,
        parent_node: StrategyNode,
        journal: StrategyJournal,
        df: pd.DataFrame,
    ) -> list[StrategyNode]:
        """전략 파라미터 개선 (Improve Phase)

        AI-Scientist-v2의 MinimalAgent._improve()에 대응.
        기존 전략의 파라미터를 LLM이 제안한 대로 수정한다.
        """
        if not parent_node.is_successful or parent_node.metrics is None:
            return []

        prompt = StrategyPrompts.PARAMETER_TUNING.format(
            strategy_code=parent_node.code,
            current_params=json.dumps(parent_node.params, ensure_ascii=False, indent=2) if parent_node.params else "{}",
            backtest_metrics=parent_node.metrics.summary(),
        )

        response = self.llm.generate_json(
            prompt=prompt,
            system_message=StrategyPrompts.SYSTEM,
        )

        # 제안된 파라미터 세트별로 자식 노드 생성
        children = []
        if isinstance(response, list):
            for suggestion in response:
                child = StrategyNode(
                    idx=-1,
                    parent_idx=parent_node.idx,
                    stage=parent_node.stage,
                    code=parent_node.code,
                    params=suggestion.get("params", {}),
                    plan=suggestion.get("rationale", "파라미터 최적화"),
                )
                child = self.execute_and_evaluate(child, df)
                if child.status == NodeStatus.FAILED:
                    child = self.debug_strategy(child, df)
                children.append(child)

        return children

    def run_parallel_exploration(
        self,
        ideas: list[str],
        df: pd.DataFrame,
        journal: StrategyJournal,
        stage: int = 1,
    ) -> list[StrategyNode]:
        """병렬 탐색 실행

        AI-Scientist-v2의 ParallelAgent._select_parallel_nodes()에 대응.
        복수 아이디어를 동시에 Draft → Execute → Debug 한다.
        """
        results = []

        def _process_idea(idea: str) -> StrategyNode:
            node = self.draft_strategy(idea, journal, stage)
            node = journal.append(node)
            node = self.execute_and_evaluate(node, df)
            if node.status == NodeStatus.FAILED:
                node = self.debug_strategy(node, df)
            return node

        # 병렬 실행
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(_process_idea, idea): idea for idea in ideas
            }
            for future in as_completed(futures):
                try:
                    node = future.result()
                    results.append(node)
                except Exception as e:
                    logger.error(f"병렬 탐색 실패: {e}")

        success_count = sum(1 for n in results if n.is_successful)
        logger.info(
            f"[Stage {stage}] 탐색 완료: "
            f"{len(results)}개 중 {success_count}개 성공"
        )
        return results


# json import (improve_strategy에서 사용)
import json
