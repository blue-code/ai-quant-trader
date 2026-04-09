"""전략 가설 생성기

AI-Scientist-v2의 Ideation Phase를 트레이딩에 적용.
LLM이 다양한 트레이딩 전략 가설을 자동 생성하고,
기존 전략과의 차별성을 검증한다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..llm.client import LLMClient, LLMConfig
from ..llm.prompts import StrategyPrompts
from ..data.universe import StockUniverse

logger = logging.getLogger(__name__)


@dataclass
class StrategyIdea:
    """전략 가설 (AI-Scientist-v2의 연구 아이디어에 대응)"""

    name: str
    description: str
    hypothesis: str
    signal_type: str
    indicators: list[str]
    entry_rule: str
    exit_rule: str
    position_sizing: str
    expected_holding_days: int
    expected_sharpe: float
    risk_level: str
    code: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "signal_type": self.signal_type,
            "indicators": self.indicators,
            "entry_rule": self.entry_rule,
            "exit_rule": self.exit_rule,
            "position_sizing": self.position_sizing,
            "expected_holding_days": self.expected_holding_days,
            "expected_sharpe": self.expected_sharpe,
            "risk_level": self.risk_level,
            "code": self.code,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StrategyIdea:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class StrategyGenerator:
    """LLM 기반 전략 가설 생성기

    AI-Scientist-v2의 perform_ideation_temp_free.py에 대응.
    LLM에게 투자 유니버스와 조건을 제시하고,
    다양한 전략 가설을 생성한다.
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()
        self.generated_ideas: list[StrategyIdea] = []

    def generate(
        self,
        universe: StockUniverse,
        num_strategies: int = 10,
        constraints: dict | None = None,
    ) -> list[StrategyIdea]:
        """전략 가설 생성

        Args:
            universe: 투자 유니버스
            num_strategies: 생성할 전략 수
            constraints: 추가 제약 조건

        Returns:
            StrategyIdea 리스트
        """
        logger.info(f"전략 생성 시작: {universe.name}, {num_strategies}개")

        # 프롬프트 구성
        prompt = StrategyPrompts.IDEATION.format(
            num_strategies=num_strategies,
            universe_description=universe.get_description_for_llm(),
        )

        if constraints:
            prompt += f"\n\n## 추가 제약 조건\n{json.dumps(constraints, ensure_ascii=False, indent=2)}"

        # LLM 호출
        result = self.llm.generate_json(
            prompt=prompt,
            system_message=StrategyPrompts.SYSTEM,
            temperature=0.8,  # 다양성 확보를 위해 온도 높임
        )

        # 결과 파싱
        ideas = []
        if isinstance(result, list):
            for item in result:
                try:
                    idea = StrategyIdea.from_dict(item)
                    ideas.append(idea)
                except Exception as e:
                    logger.warning(f"전략 파싱 실패: {e}")

        logger.info(f"전략 {len(ideas)}개 생성 완료")
        self.generated_ideas.extend(ideas)
        return ideas

    def refine(
        self,
        idea: StrategyIdea,
        feedback: str = "",
    ) -> StrategyIdea:
        """전략 가설 개선

        AI-Scientist-v2의 reflection 단계에 대응.
        초기 가설을 피드백 기반으로 개선한다.
        """
        prompt = f"""다음 전략 가설을 개선하라.

## 원본 전략
{json.dumps(idea.to_dict(), ensure_ascii=False, indent=2)}

## 개선 요청
{feedback if feedback else "전략의 약점을 보완하고 실거래 가능성을 높여라."}

개선된 전략을 동일한 JSON 형식으로 응답하라.
"""
        result = self.llm.generate_json(
            prompt=prompt,
            system_message=StrategyPrompts.SYSTEM,
        )

        if isinstance(result, dict):
            return StrategyIdea.from_dict(result)
        return idea

    def check_novelty(
        self,
        idea: StrategyIdea,
        existing_ideas: list[StrategyIdea] | None = None,
    ) -> dict:
        """전략 차별성 검증

        AI-Scientist-v2의 SemanticScholar 기반 novelty check에 대응.
        기존 전략들과의 유사도를 확인한다.
        """
        existing = existing_ideas or self.generated_ideas
        existing_descriptions = [
            f"- {e.name}: {e.description}" for e in existing if e.name != idea.name
        ]

        prompt = f"""다음 전략의 기존 전략 대비 차별성을 평가하라.

## 평가 대상 전략
이름: {idea.name}
설명: {idea.description}
가설: {idea.hypothesis}
시그널: {idea.signal_type}
지표: {', '.join(idea.indicators)}

## 기존 전략 목록
{chr(10).join(existing_descriptions) if existing_descriptions else "(없음)"}

```json
{{
  "novelty_score": 0.0,
  "similar_strategies": [],
  "differentiating_factors": [],
  "recommendation": "신규 채택 | 유사 전략 존재 | 기각"
}}
```
"""
        return self.llm.generate_json(
            prompt=prompt,
            system_message=StrategyPrompts.SYSTEM,
        )

    def save(self, ideas: list[StrategyIdea], path: str | Path) -> None:
        """전략 아이디어를 JSON 파일로 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [idea.to_dict() for idea in ideas]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"전략 {len(ideas)}개 저장: {path}")

    @staticmethod
    def load(path: str | Path) -> list[StrategyIdea]:
        """JSON 파일에서 전략 아이디어 로드"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [StrategyIdea.from_dict(item) for item in data]
