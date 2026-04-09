"""LLM 클라이언트 - Anthropic / OpenAI 통합 인터페이스

AI-Scientist-v2의 llm.py를 참고하되, 트레이딩 도메인에 맞게 단순화.
JSON 추출, 재시도 로직, 배치 요청을 지원한다.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import backoff

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM 설정"""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096


class LLMClient:
    """통합 LLM 클라이언트

    Anthropic Claude와 OpenAI GPT를 동일한 인터페이스로 사용한다.
    AI-Scientist-v2의 프로바이더 자동 감지 패턴을 적용했다.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """프로바이더에 맞는 클라이언트 초기화"""
        if self.config.provider == "anthropic":
            try:
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY")
                )
            except ImportError:
                raise ImportError("anthropic 패키지 설치 필요: pip install anthropic")
        elif self.config.provider == "openai":
            try:
                import openai

                self._client = openai.OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY")
                )
            except ImportError:
                raise ImportError("openai 패키지 설치 필요: pip install openai")
        else:
            raise ValueError(f"지원하지 않는 프로바이더: {self.config.provider}")

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate(
        self,
        prompt: str,
        system_message: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """단일 텍스트 응답 생성"""
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        if self.config.provider == "anthropic":
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "max_tokens": tokens,
                "temperature": temp,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_message:
                kwargs["system"] = system_message
            response = self._client.messages.create(**kwargs)
            return response.content[0].text

        else:  # openai
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            return response.choices[0].message.content

    def generate_json(
        self,
        prompt: str,
        system_message: str = "",
        temperature: float | None = None,
    ) -> dict | list:
        """JSON 형식 응답 생성 및 파싱

        AI-Scientist-v2의 JSON 추출 패턴: ```json 블록에서 추출한다.
        """
        json_instruction = (
            "\n\n반드시 ```json 코드 블록 안에 유효한 JSON으로 응답하라."
        )
        response = self.generate(
            prompt + json_instruction,
            system_message=system_message,
            temperature=temperature,
        )
        return self._extract_json(response)

    def generate_batch(
        self,
        prompt: str,
        num_responses: int = 3,
        system_message: str = "",
        temperature: float | None = None,
    ) -> list[str]:
        """복수 응답 생성 (전략 다양성 확보용)"""
        responses = []
        temp = temperature if temperature is not None else self.config.temperature
        for i in range(num_responses):
            # 다양성 확보를 위해 온도를 약간씩 조절
            adjusted_temp = min(1.0, temp + (i * 0.05))
            resp = self.generate(
                prompt,
                system_message=system_message,
                temperature=adjusted_temp,
            )
            responses.append(resp)
        return responses

    @staticmethod
    def _extract_json(text: str) -> dict | list:
        """텍스트에서 JSON 추출

        AI-Scientist-v2 패턴: ```json ... ``` 블록을 우선 탐색하고,
        없으면 전체 텍스트를 JSON으로 파싱 시도한다.
        """
        # ```json 블록에서 추출
        pattern = r"```json\s*([\s\S]*?)\s*```"
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 전체 텍스트에서 JSON 파싱 시도
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # { 또는 [ 로 시작하는 부분 추출
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"JSON 추출 실패. 원본 텍스트:\n{text[:500]}")
