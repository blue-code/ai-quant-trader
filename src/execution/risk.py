"""리스크 관리 모듈

실거래 전 리스크 체크, 포지션 한도 관리, 서킷 브레이커를 제공한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from .broker import AccountInfo, Order, OrderSide

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """리스크 한도 설정"""

    max_position_pct: float = 0.05  # 종목당 최대 비중
    max_positions: int = 20  # 최대 보유 종목 수
    max_daily_loss_pct: float = 0.03  # 일일 최대 손실
    max_total_loss_pct: float = 0.10  # 누적 최대 손실
    max_order_value: float = 10_000_000  # 건당 최대 주문 금액
    max_daily_orders: int = 50  # 일일 최대 주문 수
    max_leverage: float = 1.0  # 최대 레버리지


class RiskManager:
    """리스크 관리자

    모든 주문을 사전 검증하고, 한도 초과 시 차단한다.
    서킷 브레이커: 일일/누적 손실 한도 초과 시 거래를 중단한다.
    """

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self.daily_orders: list[Order] = []
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.initial_equity: float = 0.0
        self.circuit_breaker_active: bool = False
        self._last_reset_date: str = ""

    def initialize(self, account: AccountInfo) -> None:
        """초기 자산 기록"""
        self.initial_equity = account.total_equity
        logger.info(f"[리스크] 초기 자산: {self.initial_equity:,.0f}")

    def check_order(
        self, order: Order, account: AccountInfo
    ) -> tuple[bool, str]:
        """주문 사전 검증

        Returns:
            (승인여부, 사유)
        """
        # 서킷 브레이커 확인
        if self.circuit_breaker_active:
            return False, "서킷 브레이커 작동 중 - 거래 중단"

        # 일일 리셋
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._last_reset_date:
            self.daily_orders = []
            self.daily_pnl = 0.0
            self._last_reset_date = today

        # 일일 주문 횟수
        if len(self.daily_orders) >= self.limits.max_daily_orders:
            return False, f"일일 주문 한도 초과 ({self.limits.max_daily_orders}회)"

        # 보유 종목 수 제한 (매수 시)
        if order.side == OrderSide.BUY:
            current_positions = len(account.positions)
            if current_positions >= self.limits.max_positions:
                return False, f"최대 보유 종목 수 초과 ({self.limits.max_positions})"

        # 건당 주문 금액
        if order.price:
            order_value = order.price * order.quantity
        else:
            # 시장가면 대략적 추정
            order_value = order.quantity * 100_000  # 보수적 추정
        if order_value > self.limits.max_order_value:
            return False, f"건당 주문 한도 초과 ({self.limits.max_order_value:,.0f})"

        # 종목당 비중 제한 (매수 시)
        if order.side == OrderSide.BUY and account.total_equity > 0:
            position_pct = order_value / account.total_equity
            if position_pct > self.limits.max_position_pct:
                return False, (
                    f"종목 비중 한도 초과: "
                    f"{position_pct:.1%} > {self.limits.max_position_pct:.1%}"
                )

        return True, "승인"

    def update_pnl(self, account: AccountInfo) -> None:
        """손익 업데이트 및 서킷 브레이커 체크"""
        if self.initial_equity <= 0:
            return

        current_pnl = (account.total_equity / self.initial_equity) - 1
        self.total_pnl = current_pnl

        # 일일 손실 한도
        if self.daily_pnl < -self.limits.max_daily_loss_pct:
            self.circuit_breaker_active = True
            logger.warning(
                f"[리스크] 서킷 브레이커 작동! "
                f"일일 손실 {self.daily_pnl:.2%} < "
                f"-{self.limits.max_daily_loss_pct:.2%}"
            )

        # 누적 손실 한도
        if self.total_pnl < -self.limits.max_total_loss_pct:
            self.circuit_breaker_active = True
            logger.warning(
                f"[리스크] 서킷 브레이커 작동! "
                f"누적 손실 {self.total_pnl:.2%} < "
                f"-{self.limits.max_total_loss_pct:.2%}"
            )

    def record_order(self, order: Order) -> None:
        """체결된 주문 기록"""
        self.daily_orders.append(order)

    def reset_circuit_breaker(self) -> None:
        """서킷 브레이커 수동 해제"""
        self.circuit_breaker_active = False
        logger.info("[리스크] 서킷 브레이커 수동 해제")

    def status_report(self, account: AccountInfo) -> str:
        """리스크 현황 보고"""
        return (
            f"=== 리스크 현황 ===\n"
            f"총 자산: {account.total_equity:,.0f}\n"
            f"현금: {account.cash:,.0f}\n"
            f"보유 종목: {len(account.positions)}개 / {self.limits.max_positions}\n"
            f"일일 주문: {len(self.daily_orders)}회 / {self.limits.max_daily_orders}\n"
            f"누적 손익: {self.total_pnl:.2%}\n"
            f"서킷 브레이커: {'작동 중' if self.circuit_breaker_active else '정상'}"
        )
