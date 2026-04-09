"""시장 데이터 로더

yfinance(미국/한국 ETF)와 pykrx(한국 개별주)를 통해
OHLCV 데이터를 로드한다.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 데이터 캐시 디렉토리
CACHE_DIR = Path("./data_cache")


class MarketDataLoader:
    """시장 데이터 통합 로더

    한국 시장은 pykrx, 미국 시장은 yfinance를 사용한다.
    로컬 캐시를 활용하여 반복 요청을 최소화한다.
    """

    def __init__(self, market: str = "kr", cache_dir: Path | None = None):
        self.market = market
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """단일 종목 OHLCV 데이터 로드

        Args:
            ticker: 종목 코드 (한국: '005930', 미국: 'AAPL')
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            use_cache: 캐시 사용 여부

        Returns:
            OHLCV DataFrame (컬럼: Date, Open, High, Low, Close, Volume)
        """
        # 캐시 확인
        if use_cache:
            cached = self._load_cache(ticker, start_date, end_date)
            if cached is not None:
                logger.info(f"[캐시] {ticker} 데이터 로드 완료")
                return cached

        # 시장별 로더 분기
        if self.market == "kr":
            df = self._load_kr(ticker, start_date, end_date)
        else:
            df = self._load_us(ticker, start_date, end_date)

        # 컬럼 표준화
        df = self._standardize(df)

        # 캐시 저장
        if use_cache and not df.empty:
            self._save_cache(df, ticker, start_date, end_date)

        logger.info(f"[다운로드] {ticker}: {len(df)}행 로드 완료")
        return df

    def load_multiple(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """복수 종목 데이터 로드"""
        result = {}
        for ticker in tickers:
            try:
                df = self.load(ticker, start_date, end_date)
                if not df.empty:
                    result[ticker] = df
            except Exception as e:
                logger.warning(f"{ticker} 로드 실패: {e}")
        return result

    def _load_kr(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """한국 주식 데이터 로드 (pykrx)"""
        try:
            from pykrx import stock

            # pykrx는 날짜 형식이 YYYYMMDD
            start_fmt = start.replace("-", "")
            end_fmt = end.replace("-", "")

            df = stock.get_market_ohlcv_by_date(start_fmt, end_fmt, ticker)
            if df.empty:
                logger.warning(f"[pykrx] {ticker}: 데이터 없음")
                return pd.DataFrame()

            df = df.reset_index()
            df.columns = ["Date", "Open", "High", "Low", "Close", "Volume",
                          "TradingValue", "PriceChange"]
            return df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        except ImportError:
            logger.warning("pykrx 미설치. yfinance로 대체 시도")
            # 한국 종목도 yfinance로 시도 (코드 뒤에 .KS/.KQ 추가)
            return self._load_us(f"{ticker}.KS", start, end)

    def _load_us(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """미국 주식 데이터 로드 (yfinance)"""
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            df = stock.history(start=start, end=end, auto_adjust=True)

            if df.empty:
                logger.warning(f"[yfinance] {ticker}: 데이터 없음")
                return pd.DataFrame()

            df = df.reset_index()
            return df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        except ImportError:
            raise ImportError("yfinance 설치 필요: pip install yfinance")

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 표준화: 컬럼명 통일, 정렬, 결측 처리"""
        if df.empty:
            return df

        expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in expected_cols:
            if col not in df.columns:
                raise ValueError(f"필수 컬럼 누락: {col}")

        df = df[expected_cols].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # 결측 처리: 전일 종가로 보간
        df[["Open", "High", "Low", "Close"]] = df[
            ["Open", "High", "Low", "Close"]
        ].ffill()
        df["Volume"] = df["Volume"].fillna(0)

        return df

    def _cache_path(self, ticker: str, start: str, end: str) -> Path:
        """캐시 파일 경로 생성"""
        safe_ticker = ticker.replace(".", "_").replace("/", "_")
        return self.cache_dir / f"{safe_ticker}_{start}_{end}.parquet"

    def _load_cache(
        self, ticker: str, start: str, end: str
    ) -> pd.DataFrame | None:
        """캐시에서 데이터 로드"""
        path = self._cache_path(ticker, start, end)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                return None
        return None

    def _save_cache(
        self, df: pd.DataFrame, ticker: str, start: str, end: str
    ) -> None:
        """데이터를 캐시에 저장"""
        path = self._cache_path(ticker, start, end)
        try:
            df.to_parquet(path, index=False)
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {e}")
