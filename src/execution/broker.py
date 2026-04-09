"""브로커 인터페이스

한국투자증권 API(KIS)와 모의투자(Paper Trading)를 지원한다.
실거래 주문 실행, 잔고 조회, 체결 확인을 처리한다.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """주문 객체"""

    ticker: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    price: float | None = None  # 지정가 주문 시
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float = 0.0
    filled_quantity: int = 0
    order_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""


@dataclass
class Position:
    """보유 포지션"""

    ticker: str
    quantity: int
    avg_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl_pct(self) -> float:
        if self.avg_price == 0:
            return 0.0
        return (self.current_price - self.avg_price) / self.avg_price


@dataclass
class AccountInfo:
    """계좌 정보"""

    total_equity: float = 0.0
    cash: float = 0.0
    positions: list[Position] = field(default_factory=list)

    @property
    def invested(self) -> float:
        return sum(p.market_value for p in self.positions)

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions)


class BrokerBase(ABC):
    """브로커 추상 인터페이스"""

    @abstractmethod
    def connect(self) -> bool:
        """브로커 연결"""
        ...

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """계좌 정보 조회"""
        ...

    @abstractmethod
    def get_current_price(self, ticker: str) -> float:
        """현재가 조회"""
        ...

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """주문 제출"""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """주문 상태 조회"""
        ...

    def execute_signals(
        self,
        signals: dict[str, int],
        account: AccountInfo,
        max_position_pct: float = 0.05,
    ) -> list[Order]:
        """시그널을 실제 주문으로 변환

        Args:
            signals: {ticker: signal} (1=매수, -1=매도, 0=관망)
            account: 현재 계좌 정보
            max_position_pct: 종목당 최대 비중

        Returns:
            실행된 Order 리스트
        """
        orders = []
        held_tickers = {p.ticker for p in account.positions}
        per_stock_budget = account.total_equity * max_position_pct

        for ticker, signal in signals.items():
            if signal == 1 and ticker not in held_tickers:
                # 매수 주문
                price = self.get_current_price(ticker)
                if price <= 0:
                    continue
                quantity = int(per_stock_budget / price)
                if quantity > 0:
                    order = Order(
                        ticker=ticker,
                        side=OrderSide.BUY,
                        quantity=quantity,
                    )
                    order = self.place_order(order)
                    orders.append(order)

            elif signal == -1 and ticker in held_tickers:
                # 매도 주문 (전량)
                pos = next(p for p in account.positions if p.ticker == ticker)
                order = Order(
                    ticker=ticker,
                    side=OrderSide.SELL,
                    quantity=pos.quantity,
                )
                order = self.place_order(order)
                orders.append(order)

        return orders


class PaperBroker(BrokerBase):
    """모의투자 브로커

    실제 주문을 내지 않고, 메모리에서 시뮬레이션한다.
    전략 검증 및 개발 단계에서 사용한다.
    """

    def __init__(self, initial_capital: float = 100_000_000):
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.order_history: list[Order] = []
        self._prices: dict[str, float] = {}
        self._order_counter = 0

    def connect(self) -> bool:
        logger.info("[Paper] 모의투자 브로커 연결")
        return True

    def set_prices(self, prices: dict[str, float]) -> None:
        """현재가 설정 (시뮬레이션용)"""
        self._prices = prices

    def get_account(self) -> AccountInfo:
        positions = list(self.positions.values())
        for p in positions:
            p.current_price = self._prices.get(p.ticker, p.avg_price)
            p.unrealized_pnl = (p.current_price - p.avg_price) * p.quantity

        invested = sum(p.market_value for p in positions)
        return AccountInfo(
            total_equity=self.cash + invested,
            cash=self.cash,
            positions=positions,
        )

    def get_current_price(self, ticker: str) -> float:
        return self._prices.get(ticker, 0.0)

    def place_order(self, order: Order) -> Order:
        self._order_counter += 1
        order.order_id = f"PAPER-{self._order_counter:06d}"

        price = self.get_current_price(order.ticker)
        if price <= 0:
            order.status = OrderStatus.REJECTED
            order.message = "현재가 조회 불가"
            return order

        if order.side == OrderSide.BUY:
            cost = price * order.quantity
            if cost > self.cash:
                order.quantity = int(self.cash / price)
                cost = price * order.quantity

            if order.quantity <= 0:
                order.status = OrderStatus.REJECTED
                order.message = "자금 부족"
                return order

            self.cash -= cost
            if order.ticker in self.positions:
                pos = self.positions[order.ticker]
                total_qty = pos.quantity + order.quantity
                pos.avg_price = (
                    (pos.avg_price * pos.quantity + price * order.quantity)
                    / total_qty
                )
                pos.quantity = total_qty
            else:
                self.positions[order.ticker] = Position(
                    ticker=order.ticker,
                    quantity=order.quantity,
                    avg_price=price,
                    current_price=price,
                )

        elif order.side == OrderSide.SELL:
            if order.ticker not in self.positions:
                order.status = OrderStatus.REJECTED
                order.message = "보유 종목 없음"
                return order

            pos = self.positions[order.ticker]
            sell_qty = min(order.quantity, pos.quantity)
            self.cash += price * sell_qty
            pos.quantity -= sell_qty
            if pos.quantity <= 0:
                del self.positions[order.ticker]

        order.status = OrderStatus.FILLED
        order.filled_price = price
        order.filled_quantity = order.quantity
        self.order_history.append(order)

        logger.info(
            f"[Paper] {order.side.value} {order.ticker} "
            f"x{order.filled_quantity} @ {price:,.0f}"
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        return False  # 모의투자는 즉시 체결

    def get_order_status(self, order_id: str) -> OrderStatus:
        for o in self.order_history:
            if o.order_id == order_id:
                return o.status
        return OrderStatus.PENDING


class KISBroker(BrokerBase):
    """한국투자증권 Open API 브로커

    실거래 연동을 위한 KIS Open API 래퍼.
    환경변수에서 인증 정보를 읽는다.
    """

    API_BASE = "https://openapi.koreainvestment.com:9443"
    PAPER_BASE = "https://openapivts.koreainvestment.com:29443"

    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.base_url = self.PAPER_BASE if paper_mode else self.API_BASE
        self.app_key = os.environ.get("KIS_APP_KEY", "")
        self.app_secret = os.environ.get("KIS_APP_SECRET", "")
        self.account_no = os.environ.get("KIS_ACCOUNT_NO", "")
        self._token = ""
        self._token_expires = datetime.min

    def connect(self) -> bool:
        """OAuth 토큰 발급"""
        if not self.app_key or not self.app_secret:
            logger.error("KIS_APP_KEY, KIS_APP_SECRET 환경변수 필요")
            return False

        try:
            import requests

            url = f"{self.base_url}/oauth2/tokenP"
            body = {
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
            }
            resp = requests.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
            self._token = data["access_token"]
            logger.info(
                f"[KIS] {'모의투자' if self.paper_mode else '실거래'} 연결 성공"
            )
            return True

        except Exception as e:
            logger.error(f"[KIS] 연결 실패: {e}")
            return False

    def _headers(self) -> dict:
        """API 요청 헤더"""
        return {
            "authorization": f"Bearer {self._token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "Content-Type": "application/json; charset=utf-8",
        }

    def get_account(self) -> AccountInfo:
        """계좌 잔고 조회"""
        try:
            import requests

            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
            headers = self._headers()
            headers["tr_id"] = "VTTC8434R" if self.paper_mode else "TTTC8434R"

            params = {
                "CANO": self.account_no[:8],
                "ACNT_PRDT_CD": self.account_no[8:],
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "01",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            }

            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            positions = []
            for item in data.get("output1", []):
                if int(item.get("hldg_qty", 0)) > 0:
                    positions.append(Position(
                        ticker=item["pdno"],
                        quantity=int(item["hldg_qty"]),
                        avg_price=float(item["pchs_avg_pric"]),
                        current_price=float(item["prpr"]),
                        unrealized_pnl=float(item.get("evlu_pfls_amt", 0)),
                    ))

            output2 = data.get("output2", [{}])
            summary = output2[0] if output2 else {}

            return AccountInfo(
                total_equity=float(summary.get("tot_evlu_amt", 0)),
                cash=float(summary.get("dnca_tot_amt", 0)),
                positions=positions,
            )

        except Exception as e:
            logger.error(f"[KIS] 잔고 조회 실패: {e}")
            return AccountInfo()

    def get_current_price(self, ticker: str) -> float:
        """현재가 조회"""
        try:
            import requests

            url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
            headers = self._headers()
            headers["tr_id"] = "FHKST01010100"

            params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker}
            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            return float(data["output"]["stck_prpr"])

        except Exception as e:
            logger.error(f"[KIS] 현재가 조회 실패 ({ticker}): {e}")
            return 0.0

    def place_order(self, order: Order) -> Order:
        """주문 제출"""
        try:
            import requests

            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
            headers = self._headers()

            if order.side == OrderSide.BUY:
                headers["tr_id"] = "VTTC0802U" if self.paper_mode else "TTTC0802U"
            else:
                headers["tr_id"] = "VTTC0801U" if self.paper_mode else "TTTC0801U"

            body = {
                "CANO": self.account_no[:8],
                "ACNT_PRDT_CD": self.account_no[8:],
                "PDNO": order.ticker,
                "ORD_DVSN": "01" if order.order_type == OrderType.MARKET else "00",
                "ORD_QTY": str(order.quantity),
                "ORD_UNPR": str(int(order.price)) if order.price else "0",
            }

            resp = requests.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

            if data.get("rt_cd") == "0":
                order.order_id = data["output"]["ODNO"]
                order.status = OrderStatus.PENDING
                logger.info(
                    f"[KIS] 주문 제출: {order.side.value} {order.ticker} "
                    f"x{order.quantity}"
                )
            else:
                order.status = OrderStatus.REJECTED
                order.message = data.get("msg1", "주문 실패")

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.message = str(e)
            logger.error(f"[KIS] 주문 실패: {e}")

        return order

    def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        try:
            import requests

            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-rvsecncl"
            headers = self._headers()
            headers["tr_id"] = "VTTC0803U" if self.paper_mode else "TTTC0803U"

            body = {
                "CANO": self.account_no[:8],
                "ACNT_PRDT_CD": self.account_no[8:],
                "KRX_FWDG_ORD_ORGNO": "",
                "ORGN_ODNO": order_id,
                "ORD_DVSN": "01",
                "RVSE_CNCL_DVSN_CD": "02",  # 취소
                "ORD_QTY": "0",
                "ORD_UNPR": "0",
                "QTY_ALL_ORD_YN": "Y",
            }

            resp = requests.post(url, headers=headers, json=body)
            return resp.json().get("rt_cd") == "0"

        except Exception as e:
            logger.error(f"[KIS] 주문 취소 실패: {e}")
            return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        # 간소화: 체결 조회 API 호출 필요
        return OrderStatus.PENDING
