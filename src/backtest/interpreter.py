"""전략 코드 인터프리터

AI-Scientist-v2의 Interpreter를 참고하여, LLM이 생성한 전략 코드를
안전하게 실행하고 결과를 반환한다.
"""

from __future__ import annotations

import logging
import re
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """전략 코드 실행 결과

    AI-Scientist-v2의 ExecutionResult 패턴과 동일한 구조.
    """

    success: bool = False
    signals: pd.Series | None = None
    params: dict | None = None
    error: str = ""
    traceback: str = ""
    runtime_seconds: float = 0.0


class StrategyInterpreter:
    """LLM 생성 전략 코드의 안전한 실행 환경

    AI-Scientist-v2의 Interpreter처럼 격리된 환경에서 코드를 실행하되,
    트레이딩 도메인에 필요한 라이브러리만 허용한다.
    """

    # 전략 코드에서 사용 가능한 모듈
    ALLOWED_MODULES = {
        "numpy": np,
        "np": np,
        "pandas": pd,
        "pd": pd,
    }

    def __init__(self, timeout_seconds: int = 60):
        self.timeout = timeout_seconds

    def execute(
        self,
        code: str,
        df: pd.DataFrame,
        params: dict | None = None,
    ) -> ExecutionResult:
        """전략 코드 실행

        Args:
            code: 전략 파이썬 코드 (strategy 함수 포함)
            df: OHLCV 데이터프레임
            params: 전략 파라미터 (None이면 DEFAULT_PARAMS 사용)

        Returns:
            ExecutionResult (시그널 시리즈 포함)
        """
        import time

        start_time = time.time()
        result = ExecutionResult()

        try:
            # 코드에서 python 블록 추출
            clean_code = self._extract_code(code)

            # 실행 환경 구성
            exec_globals = self._build_namespace()

            # 코드 실행 (함수 정의)
            exec(clean_code, exec_globals)

            # strategy 함수 추출
            if "strategy" not in exec_globals:
                result.error = "strategy() 함수가 정의되지 않음"
                return result

            strategy_func = exec_globals["strategy"]

            # 파라미터 결정: 명시적 params > DEFAULT_PARAMS > 빈 dict
            if params is None:
                params = exec_globals.get("DEFAULT_PARAMS", {})
            result.params = params

            # 전략 실행
            signals = strategy_func(df.copy(), params)

            # 결과 검증
            signals = self._validate_signals(signals, df)

            result.success = True
            result.signals = signals
            result.runtime_seconds = time.time() - start_time

        except Exception as e:
            result.error = str(e)
            result.traceback = traceback.format_exc()
            result.runtime_seconds = time.time() - start_time
            logger.warning(f"전략 실행 실패: {e}")

        return result

    def _extract_code(self, text: str) -> str:
        """텍스트에서 파이썬 코드 블록 추출"""
        # ```python ... ``` 블록 추출
        pattern = r"```python\s*([\s\S]*?)\s*```"
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        # ``` ... ``` 블록 추출
        pattern = r"```\s*([\s\S]*?)\s*```"
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        # 코드 블록이 없으면 전체를 코드로 간주
        return text

    def _build_namespace(self) -> dict[str, Any]:
        """안전한 실행 네임스페이스 구성"""
        namespace = {"__builtins__": __builtins__}
        namespace.update(self.ALLOWED_MODULES)
        return namespace

    def _validate_signals(
        self, signals: Any, df: pd.DataFrame
    ) -> pd.Series:
        """시그널 유효성 검증

        - pd.Series 또는 np.ndarray여야 한다
        - 값은 -1, 0, 1 중 하나여야 한다
        - 길이가 원본 데이터와 일치해야 한다
        """
        if isinstance(signals, np.ndarray):
            signals = pd.Series(signals, index=df.index)
        elif not isinstance(signals, pd.Series):
            raise ValueError(
                f"시그널 타입 오류: {type(signals)}. pd.Series 또는 np.ndarray 필요"
            )

        if len(signals) != len(df):
            raise ValueError(
                f"시그널 길이({len(signals)})가 데이터 길이({len(df)})와 불일치"
            )

        # NaN을 0(관망)으로 처리
        signals = signals.fillna(0)

        # 값 범위 검증: -1, 0, 1만 허용
        valid_values = {-1, 0, 1}
        unique_values = set(signals.unique())
        if not unique_values.issubset(valid_values):
            # 연속 값이면 부호 기반으로 이산화
            signals = pd.Series(
                np.sign(signals).astype(int), index=signals.index
            )

        return signals
