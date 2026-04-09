"""전략 저널 (Strategy Journal)

AI-Scientist-v2의 Journal 클래스를 트레이딩에 적용.
탐색 트리의 전체 노드를 관리하고, 필터링·선택·직렬화를 제공한다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .node import StrategyNode, NodeStatus

logger = logging.getLogger(__name__)


class StrategyJournal:
    """전략 탐색 결과 저널

    AI-Scientist-v2의 Journal처럼 모든 탐색 노드를 보관하고,
    최고 성과 전략 선택, 스테이지별 필터링 등을 제공한다.
    """

    def __init__(self):
        self._nodes: list[StrategyNode] = []
        self._next_idx: int = 0

    def __len__(self) -> int:
        return len(self._nodes)

    def append(self, node: StrategyNode) -> StrategyNode:
        """노드 추가 (자동 인덱스 부여)"""
        node.idx = self._next_idx
        self._next_idx += 1
        self._nodes.append(node)

        # 부모-자식 관계 갱신
        if node.parent_idx >= 0:
            parent = self.get(node.parent_idx)
            if parent and node.idx not in parent.children:
                parent.children.append(node.idx)

        return node

    def get(self, idx: int) -> StrategyNode | None:
        """인덱스로 노드 조회"""
        for node in self._nodes:
            if node.idx == idx:
                return node
        return None

    def get_best(self, stage: int | None = None) -> StrategyNode | None:
        """최고 성과 노드 반환

        AI-Scientist-v2의 journal.get_best_node()에 대응.
        Sharpe Ratio 기준으로 최고 성과 전략을 선택한다.
        """
        candidates = self.filter(status=NodeStatus.SUCCESS, stage=stage)
        if not candidates:
            return None
        return max(candidates, key=lambda n: n.sharpe)

    def get_top_k(self, k: int = 5, stage: int | None = None) -> list[StrategyNode]:
        """상위 k개 노드 반환"""
        candidates = self.filter(status=NodeStatus.SUCCESS, stage=stage)
        return sorted(candidates, key=lambda n: n.sharpe, reverse=True)[:k]

    def filter(
        self,
        status: NodeStatus | None = None,
        stage: int | None = None,
    ) -> list[StrategyNode]:
        """조건별 노드 필터링"""
        result = self._nodes
        if status is not None:
            result = [n for n in result if n.status == status]
        if stage is not None:
            result = [n for n in result if n.stage == stage]
        return result

    @property
    def successful_nodes(self) -> list[StrategyNode]:
        return self.filter(status=NodeStatus.SUCCESS)

    @property
    def leaf_nodes(self) -> list[StrategyNode]:
        return [n for n in self._nodes if n.is_leaf]

    def stage_summary(self, stage: int) -> str:
        """스테이지별 요약 (LLM 프롬프트용)"""
        nodes = self.filter(stage=stage)
        success = [n for n in nodes if n.is_successful]
        failed = [n for n in nodes if n.status == NodeStatus.FAILED]
        best = self.get_best(stage=stage)

        lines = [
            f"## Stage {stage} 요약",
            f"- 전체: {len(nodes)}개 | 성공: {len(success)}개 | 실패: {len(failed)}개",
        ]
        if best:
            lines.append(f"- 최고 성과: {best.summary()}")

        # 상위 3개 전략
        top3 = self.get_top_k(3, stage=stage)
        if top3:
            lines.append("- 상위 3개:")
            for n in top3:
                lines.append(f"  {n.summary()}")

        return "\n".join(lines)

    def get_ancestry(self, node: StrategyNode) -> list[StrategyNode]:
        """루트부터 해당 노드까지의 경로 반환"""
        path = [node]
        current = node
        while current.parent_idx >= 0:
            parent = self.get(current.parent_idx)
            if parent is None:
                break
            path.insert(0, parent)
            current = parent
        return path

    def save(self, path: str | Path) -> None:
        """저널 전체를 JSON으로 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "total_nodes": len(self._nodes),
            "next_idx": self._next_idx,
            "nodes": [n.to_dict() for n in self._nodes],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"저널 저장: {path} ({len(self._nodes)}개 노드)")

    @classmethod
    def load(cls, path: str | Path) -> StrategyJournal:
        """JSON에서 저널 로드"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        journal = cls()
        journal._next_idx = data.get("next_idx", 0)
        for node_data in data.get("nodes", []):
            node = StrategyNode.from_dict(node_data)
            journal._nodes.append(node)

        logger.info(f"저널 로드: {path} ({len(journal._nodes)}개 노드)")
        return journal
