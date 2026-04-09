"""LLM 클라이언트 - Anthropic / OpenAI / Codex 통합 인터페이스

AI-Scientist-v2의 llm.py를 참고하되, 트레이딩 도메인에 맞게 단순화.
JSON 추출, 재시도 로직, 배치 요청을 지원한다.

프로바이더:
- anthropic: Anthropic API 키 방식
- openai: OpenAI API 키 방식
- codex: 로컬 Codex CLI의 OAuth 토큰 활용 (API 키 불필요)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import backoff

logger = logging.getLogger(__name__)

# Codex OAuth 인증 파일 경로
CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"
CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"


def _load_codex_auth() -> dict:
    """Codex CLI의 OAuth 인증 정보 로드

    ~/.codex/auth.json에서 access_token, refresh_token을 읽는다.
    Codex CLI가 ChatGPT 계정으로 OAuth 로그인하면 이 파일이 생성된다.
    """
    if not CODEX_AUTH_PATH.exists():
        raise FileNotFoundError(
            f"Codex 인증 파일 없음: {CODEX_AUTH_PATH}\n"
            "터미널에서 'codex' 실행 후 로그인하라."
        )

    with open(CODEX_AUTH_PATH, "r", encoding="utf-8") as f:
        auth = json.load(f)

    tokens = auth.get("tokens", {})
    access_token = tokens.get("access_token")
    if not access_token:
        raise ValueError("Codex auth.json에 access_token이 없음")

    return {
        "access_token": access_token,
        "refresh_token": tokens.get("refresh_token", ""),
        "account_id": tokens.get("account_id", ""),
    }


def _get_codex_default_model() -> str:
    """Codex config.toml에서 기본 모델 읽기"""
    if not CODEX_CONFIG_PATH.exists():
        return "gpt-4o"

    try:
        # 간단한 TOML 파싱 (최상위 model 키만)
        with open(CODEX_CONFIG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("model") and "=" in line:
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    return value
    except Exception:
        pass
    return "gpt-4o"


def _refresh_codex_token(refresh_token: str) -> str | None:
    """Codex OAuth 토큰 갱신

    access_token이 만료되었을 때 refresh_token으로 새 토큰을 발급받는다.
    """
    try:
        import requests

        resp = requests.post(
            "https://auth.openai.com/oauth/token",
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
            },
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        new_access = data.get("access_token")

        if new_access:
            # auth.json 업데이트
            with open(CODEX_AUTH_PATH, "r", encoding="utf-8") as f:
                auth = json.load(f)
            auth["tokens"]["access_token"] = new_access
            if "refresh_token" in data:
                auth["tokens"]["refresh_token"] = data["refresh_token"]
            auth["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
            with open(CODEX_AUTH_PATH, "w", encoding="utf-8") as f:
                json.dump(auth, f, indent=2)

            logger.info("[Codex] OAuth 토큰 갱신 완료")
            return new_access

    except Exception as e:
        logger.error(f"[Codex] 토큰 갱신 실패: {e}")
    return None


@dataclass
class LLMConfig:
    """LLM 설정

    provider 옵션:
    - "anthropic": Anthropic API (ANTHROPIC_API_KEY 필요)
    - "openai": OpenAI API (OPENAI_API_KEY 필요)
    - "codex": 로컬 Codex CLI의 OAuth 토큰 (API 키 불필요)
    """

    provider: str = "codex"
    model: str = ""  # 빈 문자열이면 프로바이더 기본값 사용
    temperature: float = 0.7
    max_tokens: int = 4096


class LLMClient:
    """통합 LLM 클라이언트

    Anthropic Claude, OpenAI GPT, Codex OAuth를 동일한 인터페이스로 사용한다.
    AI-Scientist-v2의 프로바이더 자동 감지 패턴을 적용했다.

    Codex 모드: ~/.codex/auth.json의 OAuth 토큰으로 OpenAI API를 호출한다.
    별도 API 키 없이 ChatGPT Plus/Pro 구독으로 사용 가능하다.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = None
        self._codex_auth: dict = {}
        self._init_client()

    def _init_client(self) -> None:
        """프로바이더에 맞는 클라이언트 초기화"""
        if self.config.provider == "codex":
            self._init_codex()
        elif self.config.provider == "anthropic":
            self._init_anthropic()
        elif self.config.provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"지원하지 않는 프로바이더: {self.config.provider}")

    def _init_codex(self) -> None:
        """Codex OAuth 토큰으로 OpenAI 클라이언트 초기화"""
        try:
            import openai
        except ImportError:
            raise ImportError("openai 패키지 설치 필요: pip install openai")

        self._codex_auth = _load_codex_auth()

        # 기본 모델 설정 (config에 명시하지 않았으면 Codex 설정에서 읽기)
        if not self.config.model:
            self.config.model = _get_codex_default_model()

        self._client = openai.OpenAI(
            api_key=self._codex_auth["access_token"],
        )
        logger.info(
            f"[Codex] OAuth 인증으로 초기화 완료 (모델: {self.config.model})"
        )

    def _init_anthropic(self) -> None:
        """Anthropic API 클라이언트 초기화"""
        try:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        except ImportError:
            raise ImportError("anthropic 패키지 설치 필요: pip install anthropic")

    def _init_openai(self) -> None:
        """OpenAI API 클라이언트 초기화"""
        try:
            import openai

            self._client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        except ImportError:
            raise ImportError("openai 패키지 설치 필요: pip install openai")

    def _refresh_codex_if_needed(self) -> None:
        """Codex 토큰 만료 시 자동 갱신"""
        if self.config.provider != "codex" or not self._codex_auth:
            return

        refresh_token = self._codex_auth.get("refresh_token", "")
        if not refresh_token:
            return

        new_token = _refresh_codex_token(refresh_token)
        if new_token:
            import openai

            self._codex_auth["access_token"] = new_token
            self._client = openai.OpenAI(api_key=new_token)

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

        else:  # openai 또는 codex (둘 다 OpenAI API 호출)
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                )
                return response.choices[0].message.content

            except Exception as e:
                # Codex: 401 인증 오류 시 토큰 갱신 후 재시도
                if self.config.provider == "codex" and "401" in str(e):
                    logger.warning("[Codex] 토큰 만료, 갱신 시도...")
                    self._refresh_codex_if_needed()
                    response = self._client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=tokens,
                    )
                    return response.choices[0].message.content
                raise

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
