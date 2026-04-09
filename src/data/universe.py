"""종목 유니버스 관리

투자 대상 종목군을 정의하고 관리한다.
한국 시장(KOSPI/KOSDAQ)과 미국 시장(S&P500/NASDAQ)을 지원한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StockUniverse:
    """투자 유니버스 정의"""

    name: str
    market: str  # kr | us
    tickers: list[str] = field(default_factory=list)
    description: str = ""

    # 사전 정의된 유니버스
    PRESETS: dict[str, dict] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.PRESETS = {
            # 한국 시장 - 대표 블루칩
            "kospi_bluechip": {
                "market": "kr",
                "description": "KOSPI 대표 블루칩 30종목",
                "tickers": [
                    "005930",  # 삼성전자
                    "000660",  # SK하이닉스
                    "373220",  # LG에너지솔루션
                    "207940",  # 삼성바이오로직스
                    "005380",  # 현대자동차
                    "006400",  # 삼성SDI
                    "051910",  # LG화학
                    "003670",  # 포스코홀딩스
                    "005490",  # POSCO
                    "035420",  # NAVER
                    "000270",  # 기아
                    "068270",  # 셀트리온
                    "105560",  # KB금융
                    "055550",  # 신한지주
                    "012330",  # 현대모비스
                    "066570",  # LG전자
                    "028260",  # 삼성물산
                    "003550",  # LG
                    "032830",  # 삼성생명
                    "096770",  # SK이노베이션
                    "034730",  # SK
                    "030200",  # KT
                    "086790",  # 하나금융지주
                    "035720",  # 카카오
                    "316140",  # 우리금융지주
                    "017670",  # SK텔레콤
                    "009150",  # 삼성전기
                    "018260",  # 삼성에스디에스
                    "033780",  # KT&G
                    "011200",  # HMM
                ],
            },
            # 한국 시장 - 성장주
            "kosdaq_growth": {
                "market": "kr",
                "description": "KOSDAQ 성장주 20종목",
                "tickers": [
                    "247540",  # 에코프로비엠
                    "091990",  # 셀트리온헬스케어
                    "263750",  # 펄어비스
                    "293490",  # 카카오게임즈
                    "196170",  # 알테오젠
                    "403870",  # HPSP
                    "067310",  # 하나마이크론
                    "041510",  # 에스엠
                    "145020",  # 휴젤
                    "112040",  # 위메이드
                    "357780",  # 솔브레인
                    "086520",  # 에코프로
                    "328130",  # 루닛
                    "039030",  # 이오테크닉스
                    "036930",  # 주성엔지니어링
                    "058470",  # 리노공업
                    "131970",  # 테스나
                    "383220",  # F&F
                    "377300",  # 카카오페이
                    "352820",  # 하이브
                ],
            },
            # 미국 시장 - 대형 기술주
            "us_tech": {
                "market": "us",
                "description": "미국 대형 기술주 (Magnificent 7+)",
                "tickers": [
                    "AAPL",
                    "MSFT",
                    "GOOGL",
                    "AMZN",
                    "NVDA",
                    "META",
                    "TSLA",
                    "AVGO",
                    "AMD",
                    "CRM",
                    "ORCL",
                    "ADBE",
                    "NFLX",
                    "INTC",
                    "QCOM",
                ],
            },
            # 미국 시장 - S&P500 섹터 대표
            "sp500_sector": {
                "market": "us",
                "description": "S&P500 섹터별 대표 종목",
                "tickers": [
                    "AAPL",  # IT
                    "UNH",  # Healthcare
                    "JPM",  # Financials
                    "XOM",  # Energy
                    "AMZN",  # Consumer Discretionary
                    "PG",  # Consumer Staples
                    "LIN",  # Materials
                    "CAT",  # Industrials
                    "AMT",  # Real Estate
                    "NEE",  # Utilities
                    "T",  # Communication
                ],
            },
        }

    @classmethod
    def from_preset(cls, preset_name: str) -> StockUniverse:
        """사전 정의된 유니버스 로드"""
        instance = cls(name=preset_name, market="")
        if preset_name not in instance.PRESETS:
            available = ", ".join(instance.PRESETS.keys())
            raise ValueError(
                f"알 수 없는 유니버스: {preset_name}. 사용 가능: {available}"
            )
        preset = instance.PRESETS[preset_name]
        return cls(
            name=preset_name,
            market=preset["market"],
            tickers=preset["tickers"],
            description=preset["description"],
        )

    @classmethod
    def custom(cls, name: str, market: str, tickers: list[str]) -> StockUniverse:
        """커스텀 유니버스 생성"""
        return cls(name=name, market=market, tickers=tickers, description="사용자 정의")

    def get_description_for_llm(self) -> str:
        """LLM 프롬프트용 유니버스 설명 생성"""
        ticker_list = ", ".join(self.tickers[:10])
        if len(self.tickers) > 10:
            ticker_list += f" 외 {len(self.tickers) - 10}종목"

        return (
            f"시장: {'한국' if self.market == 'kr' else '미국'}\n"
            f"유니버스: {self.description}\n"
            f"종목 수: {len(self.tickers)}\n"
            f"대표 종목: {ticker_list}"
        )
