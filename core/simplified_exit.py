"""
Simplified Exit Manager - R-Based Exit Logic

DESIGN PRINCIPLES:
1. ATR is the ONLY loss authority
2. R = ATR × multiplier (default 2.0) defines risk unit
3. Stop never widens, only tightens
4. No trailing stop (cuts winners)
5. Clear phases with no overlap

PHASES:
- Entry → +2R: ATR stop only (at -1R from entry)
- +2R → +3R: Stop moves to -0.25R (fixed, not trailing)
- > +3R: Optional partial exit (40-50%), stop stays at -0.25R

Author: Claude Code
Date: 2026-01-04
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import pytz


@dataclass
class RBasedPosition:
    """
    Track position state using R-multiples.

    R = ATR × multiplier = the risk unit for this trade.
    All stops and targets are expressed in R-multiples.
    """
    symbol: str
    entry_price: float
    entry_time: datetime
    quantity: int
    atr_at_entry: float
    atr_multiplier: float = 2.0

    # Calculated from ATR
    r_value: float = field(init=False)  # Dollar value of 1R
    stop_price: float = field(init=False)  # Initial stop at -1R

    # State tracking
    profit_floor_active: bool = False
    partial_exit_executed: bool = False
    partial_exit_qty: int = 0
    peak_r_multiple: float = 0.0  # Track peak profit in R

    def __post_init__(self):
        """Calculate R value and initial stop."""
        self.r_value = self.atr_at_entry * self.atr_multiplier
        self.stop_price = self.entry_price - self.r_value  # -1R stop


class SimplifiedExitManager:
    """
    Simplified exit manager using R-multiples.

    Key difference from ExitManager:
    - NO percentage-based stops
    - NO trailing stop (causes early exits)
    - ATR is the ONLY loss authority
    - Clear phases with no overlap
    """

    # Exit reasons for logging
    REASON_ATR_STOP = 'atr_stop'
    REASON_PROFIT_FLOOR = 'profit_floor'
    REASON_PARTIAL_EXIT = 'partial_exit'

    def __init__(self, settings: Dict = None):
        """
        Initialize with R-based settings.

        Args:
            settings: Dict with optional keys:
                - atr_multiplier: Multiplier for ATR to get R (default 2.0)
                - profit_floor_r: R-multiple to activate floor (default 2.0)
                - floor_stop_r: R-multiple for stop when floor active (default -0.25)
                - partial_exit_r: R-multiple for partial exit (default 3.0)
                - partial_exit_pct: Fraction to exit at partial (default 0.50)
        """
        self.logger = logging.getLogger(__name__)
        settings = settings or {}

        # R-based settings
        self.atr_multiplier = settings.get('atr_multiplier', 2.0)
        self.profit_floor_r = settings.get('profit_floor_r', 2.0)
        self.floor_stop_r = settings.get('floor_stop_r', -0.25)
        self.partial_exit_r = settings.get('partial_exit_r', 3.0)
        self.partial_exit_pct = settings.get('partial_exit_pct', 0.50)

        # Active positions
        self.positions: Dict[str, RBasedPosition] = {}

        self.logger.info(
            f"SimplifiedExitManager initialized: "
            f"R=ATR×{self.atr_multiplier}, "
            f"Floor at +{self.profit_floor_r}R → stop at {self.floor_stop_r}R, "
            f"Partial at +{self.partial_exit_r}R ({self.partial_exit_pct*100:.0f}%)"
        )

    def register_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        atr: float,
        entry_time: datetime = None
    ) -> RBasedPosition:
        """
        Register a new position for R-based exit management.

        Args:
            symbol: Stock symbol
            entry_price: Entry price per share
            quantity: Number of shares
            atr: ATR(14) value at entry time
            entry_time: Entry timestamp (defaults to now)

        Returns:
            RBasedPosition object

        Raises:
            ValueError: If ATR is zero, negative, or NaN
        """
        # Validate ATR
        if atr is None or pd.isna(atr) or atr <= 0:
            raise ValueError(f"ATR must be positive, got: {atr}")

        if entry_time is None:
            entry_time = datetime.now(pytz.UTC)

        pos = RBasedPosition(
            symbol=symbol,
            entry_price=entry_price,
            entry_time=entry_time,
            quantity=quantity,
            atr_at_entry=atr,
            atr_multiplier=self.atr_multiplier
        )

        self.positions[symbol] = pos

        self.logger.info(
            f"SIMPLIFIED_EXIT | {symbol} | REGISTERED | "
            f"Entry: ${entry_price:.2f}, ATR: ${atr:.2f}, "
            f"R: ${pos.r_value:.2f}, Stop: ${pos.stop_price:.2f} (-1R)"
        )

        return pos

    def get_position(self, symbol: str) -> Optional[RBasedPosition]:
        """Get position state for a symbol."""
        return self.positions.get(symbol)

    def update_quantity(self, symbol: str, new_qty: int) -> bool:
        """Update quantity after partial exit."""
        if symbol in self.positions:
            old_qty = self.positions[symbol].quantity
            self.positions[symbol].quantity = new_qty
            self.logger.info(f"SIMPLIFIED_EXIT | {symbol} | QTY_UPDATE | {old_qty} -> {new_qty}")
            return True
        return False

    def evaluate_exit(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[dict]:
        """
        Evaluate exit conditions for a position.

        This is the CORE METHOD called on every price update.

        Args:
            symbol: Stock symbol
            current_price: Current market price

        Returns:
            None if no action, or dict with:
            - action: 'full_exit' or 'partial_exit'
            - reason: Exit reason string
            - qty: Shares to close
            - r_multiple: Current R-multiple
            - stop_price: Stop price that triggered exit
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]

        # Calculate current R-multiple
        r_multiple = (current_price - pos.entry_price) / pos.r_value

        # Update peak R-multiple
        if r_multiple > pos.peak_r_multiple:
            pos.peak_r_multiple = r_multiple

        # ================================================================
        # PHASE 2: PROFIT FLOOR (activates at +2R)
        # Move stop to -0.25R from entry (NOT trailing)
        # ================================================================
        if not pos.profit_floor_active and r_multiple >= self.profit_floor_r:
            pos.profit_floor_active = True
            # Move stop to floor level (-0.25R from entry)
            new_stop = pos.entry_price + (self.floor_stop_r * pos.r_value)
            pos.stop_price = new_stop

            self.logger.info(
                f"SIMPLIFIED_EXIT | {symbol} | PROFIT_FLOOR_ACTIVATED | "
                f"R: {r_multiple:+.2f}, Stop moved to ${new_stop:.2f} ({self.floor_stop_r}R)"
            )

        # ================================================================
        # PHASE 1: STOP CHECK (always active)
        # Stop is at -1R initially, or -0.25R if floor active
        # ================================================================
        if current_price <= pos.stop_price:
            reason = self.REASON_PROFIT_FLOOR if pos.profit_floor_active else self.REASON_ATR_STOP

            self.logger.info(
                f"SIMPLIFIED_EXIT | {symbol} | STOP_TRIGGERED | "
                f"Reason: {reason}, Price: ${current_price:.2f}, "
                f"Stop: ${pos.stop_price:.2f}, R: {r_multiple:+.2f}"
            )
            return {
                'action': 'full_exit',
                'reason': reason,
                'qty': pos.quantity,
                'r_multiple': round(r_multiple, 2),
                'stop_price': pos.stop_price
            }

        # ================================================================
        # PHASE 3: PARTIAL EXIT (at +3R, one-time only)
        # ================================================================
        if not pos.partial_exit_executed and r_multiple >= self.partial_exit_r:
            partial_qty = int(pos.quantity * self.partial_exit_pct)

            if partial_qty > 0:
                pos.partial_exit_executed = True
                pos.partial_exit_qty = partial_qty

                self.logger.info(
                    f"SIMPLIFIED_EXIT | {symbol} | PARTIAL_EXIT_TRIGGERED | "
                    f"R: {r_multiple:+.2f}, Qty: {partial_qty}/{pos.quantity}"
                )

                return {
                    'action': 'partial_exit',
                    'reason': self.REASON_PARTIAL_EXIT,
                    'qty': partial_qty,
                    'r_multiple': round(r_multiple, 2),
                    'stop_price': pos.stop_price
                }

        # No exit triggered
        return None

    def unregister_position(self, symbol: str) -> bool:
        """Remove position from tracking."""
        if symbol in self.positions:
            del self.positions[symbol]
            self.logger.info(f"SIMPLIFIED_EXIT | {symbol} | UNREGISTERED")
            return True
        return False

    def get_all_positions(self) -> Dict[str, RBasedPosition]:
        """Get copy of all tracked positions."""
        return self.positions.copy()
