"""전략 노드 (Strategy Node)

AI-Scientist-v2의 Node 클래스를 트레이딩에 적용.
각 노드는 하나의 전략 변형(코드 + 파라미터 + 성과)을 나타낸다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..backtest.metrics import PerformanceMetrics


class NodeStatus(Enum):
    DRAFT = "draft"  # 아직 실행 전
    RUNNING = "running"  # 백테스트 실행 중
    SUCCESS = "success"  # 실행 성공, 성과 있음
    FAILED = "failed"  # 실행 실패 (코드 오류)
    PRUNED = "pruned"  # 기준 미달로 가지치기


@dataclass
class StrategyNode:
    """전략 탐색 트리의 노드

    AI-Scientist-v2의 Node와 동일한 트리 구조를 갖되,
    ML 실험 대신 트레이딩 전략의 변형을 추적한다.

    Attributes:
        idx: 노드 고유 ID
        parent_idx: 부모 노드 ID (-1이면 루트)
        stage: 소속 스테이지 (1~4)
        code: 전략 파이썬 코드
        params: 전략 파라미터
        plan: LLM이 제시한 변경 계획
        metrics: 백테스트 성과
        status: 노드 상태
        error: 실패 시 오류 메시지
        children: 자식 노드 ID 목록
    """

    idx: int
    parent_idx: int = -1
    stage: int = 1
    code: str = ""
    params: dict = field(default_factory=dict)
    plan: str = ""
    metrics: PerformanceMetrics | None = None
    status: NodeStatus = NodeStatus.DRAFT
    error: str = ""
    children: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __gt__(self, other: StrategyNode) -> bool:
        """노드 비교: Sharpe Ratio 기준

        AI-Scientist-v2의 Node.__gt__() 패턴.
        Best-First Search에서 우선순위 결정에 사용한다.
        """
        if self.metrics is None:
            return False
        if other.metrics is None:
            return True
        return self.metrics.sharpe_ratio > other.metrics.sharpe_ratio

    @property
    def is_successful(self) -> bool:
        return self.status == NodeStatus.SUCCESS

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def sharpe(self) -> float:
        return self.metrics.sharpe_ratio if self.metrics else float("-inf")

    def summary(self) -> str:
        """노드 요약 (LLM 프롬프트 삽입용)"""
        status_str = self.status.value
        metrics_str = self.metrics.summary() if self.metrics else "성과 없음"
        return (
            f"[Node {self.idx}] Stage {self.stage} | {status_str}\n"
            f"  계획: {self.plan[:100]}\n"
            f"  성과: {metrics_str}"
        )

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "parent_idx": self.parent_idx,
            "stage": self.stage,
            "code": self.code,
            "params": self.params,
            "plan": self.plan,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "status": self.status.value,
            "error": self.error,
            "children": self.children,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StrategyNode:
        metrics = None
        if data.get("metrics"):
            metrics = PerformanceMetrics(**data["metrics"])

        return cls(
            idx=data["idx"],
            parent_idx=data.get("parent_idx", -1),
            stage=data.get("stage", 1),
            code=data.get("code", ""),
            params=data.get("params", {}),
            plan=data.get("plan", ""),
            metrics=metrics,
            status=NodeStatus(data.get("status", "draft")),
            error=data.get("error", ""),
            children=data.get("children", []),
        )
