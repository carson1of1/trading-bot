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
