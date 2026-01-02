import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import logging
import pytz

class RiskManager:
    """Manage trading risks and position sizing"""

    def __init__(self, bot_settings=None):
        # Risk parameters - Can be overridden by bot_settings (Nov 3, 2025)
        # CRITICAL FIX (Nov 5, 2025): Load stop_loss_pct and take_profit_pct from settings
        # Store bot_settings for later use (Dec 8, 2025)
        self.settings = bot_settings if bot_settings else {}

        if bot_settings:
            self.max_risk_per_trade = bot_settings.get('max_risk_per_trade', 0.015)
            self.max_daily_loss = bot_settings.get('daily_loss_limit', 0.04)
            # BUG FIX (Dec 4, 2025): Changed default from 0.15 to 0.25 to match config.yaml
            self.max_position_size = bot_settings.get('max_position_size', 0.25)
            self.max_positions = bot_settings.get('max_positions', 9999)  # DISABLED (Dec 8, 2025): No limit
            self.enable_position_scaling = bot_settings.get('enable_position_scaling', False)
            self.scaling_profit_threshold = bot_settings.get('scaling_profit_threshold', 1.5)

            # NEW (Nov 5, 2025): Read configured stop/profit percentages
            self.stop_loss_pct = bot_settings.get('stop_loss_pct', 2.0) / 100  # Convert to decimal
            self.take_profit_pct = bot_settings.get('take_profit_pct', 3.2) / 100  # Convert to decimal
            self.risk_reward_ratio = bot_settings.get('risk_reward_ratio', 1.6)

            # BUG FIX (Dec 2025): Load leveraged ETF risk settings from config
            self.leveraged_etf_stop_loss_pct = bot_settings.get('leveraged_etf_stop_loss_pct', 1.0) / 100
            self.leveraged_etf_take_profit_pct = bot_settings.get('leveraged_etf_take_profit_pct', 2.0) / 100
            self.leveraged_etf_max_position_size = bot_settings.get('leveraged_etf_max_position_size', 0.20)
            self.max_leveraged_positions = bot_settings.get('max_leveraged_positions', 4)
        else:
            # Default values
            self.max_risk_per_trade = 0.015  # 1.5% of portfolio per trade
            self.max_daily_loss = 0.04      # 4% maximum daily loss
            # BUG FIX (Dec 4, 2025): Changed default from 0.15 to 0.25 to match config.yaml
            self.max_position_size = 0.25   # 25% maximum position size
            self.max_positions = 9999  # DISABLED (Dec 8, 2025): No limit
            self.enable_position_scaling = False
            self.scaling_profit_threshold = 1.5

            # BUG FIX (Dec 4, 2025): Made defaults CONSISTENT with config.yaml
            # Previously: without settings used 1.5%/2.5%, with settings used 2.0%/3.2%
            # This inconsistency caused different behavior depending on initialization path
            # Now both paths use config.yaml values: 2.0% stop loss, 4.0% take profit
            self.stop_loss_pct = 0.02   # 2.0% (matches config.yaml stop_loss_pct: 2.0)
            self.take_profit_pct = 0.04  # 4.0% (matches config.yaml take_profit_pct: 4.0)
            self.risk_reward_ratio = 2.0  # 2:1 ratio (4% profit / 2% loss)

            # BUG FIX (Dec 2025): Leveraged ETF defaults
            self.leveraged_etf_stop_loss_pct = 0.01  # 1.0%
            self.leveraged_etf_take_profit_pct = 0.02  # 2.0%
            self.leveraged_etf_max_position_size = 0.20  # 20%
            self.max_leveraged_positions = 4

        # Per-symbol take-profit optimization - DISABLED (ML not included)
        self.symbol_take_profit_pct: dict = {}  # Per-symbol optimal take-profit percentages
        self.use_per_symbol_tp = False  # Always disabled - no ML
        self._load_symbol_take_profits()

        # BUG FIX (Dec 2025): Define leveraged ETF symbols
        # These symbols get special risk treatment (tighter stops, smaller positions)
        # BUG FIX (Dec 10, 2025): Added missing leveraged ETFs (SDOW, YANG, YINN, etc.)
        self.leveraged_etf_symbols = {
            # 3x Bull ETFs
            'SPXL', 'UPRO', 'TNA', 'TQQQ', 'SOXL', 'TECL', 'FNGU', 'UDOW',
            'LABU', 'NAIL', 'DPST', 'DUSL', 'MIDU', 'RETL', 'WANT', 'WEBL', 'YINN',
            # 2x Bull ETFs
            'QLD', 'SSO', 'UWM', 'UYG', 'ROM', 'UGE', 'USD',
            # 3x Bear ETFs
            'SPXS', 'SRTY', 'SQQQ', 'SOXS', 'TECS', 'LABD', 'FAZ', 'ERY', 'SPXU',
            'SDOW', 'TMV', 'TZA', 'YANG', 'WEBS', 'DRIP',
            # 2x Bear ETFs
            'QID', 'SDS', 'SRS', 'TWM', 'SKF', 'DXD'
        }

        # UNIFIED STOP LOSS (Nov 20, 2025): All strategies use 1.5% stop loss
        # Simplified risk management - consistent stops across all strategies
        self.strategy_stop_loss_pct = {
            'Mean Reversion': 0.015,      # 1.5% - unified stop loss
            'Momentum': 0.015,            # 1.5% - unified stop loss
            'Trend Continuation': 0.015,  # 1.5% - unified stop loss
            'Breakout': 0.015,            # 1.5% - unified stop loss
            'VWAP': 0.015,               # 1.5% - unified stop loss
            'Volatility Spike': 0.015,    # 1.5% - unified stop loss
            'RSI Divergence': 0.015       # 1.5% - unified stop loss
        }

        # EXPERT UPGRADE (Nov 10, 2025): Strategy concentration limits
        # DISABLED (Dec 8, 2025): User requested removal of all limits
        self.max_positions_per_strategy = {}  # Empty - no limits
        self.enable_strategy_concentration_limits = False  # DISABLED

        self.max_portfolio_risk = 0.08  # 8% maximum total portfolio risk

        # Risk tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.current_risk = 0.0
        self.positions_risk = {}

        # Trading limits
        self.max_trades_per_day = 20
        self.min_liquidity_threshold = 1000000  # $1M average daily volume

        # Volatility controls
        self.max_volatility = 0.5  # 50% annualized volatility
        self.volatility_lookback = 20  # days

        # Setup logging
        self.logger = logging.getLogger(__name__)

    # ==================== NEW HELPER METHODS (Dec 3, 2025) ====================

    def is_leveraged_etf(self, symbol):
        """
        Check if a symbol is a leveraged ETF.

        BUG FIX (Dec 2025): Added leveraged ETF detection
        Leveraged ETFs require special risk treatment (tighter stops, smaller positions)

        Args:
            symbol: Stock ticker symbol

        Returns:
            bool: True if symbol is a leveraged ETF
        """
        return symbol.upper() in self.leveraged_etf_symbols

    def _load_symbol_take_profits(self):
        """
        Load per-symbol take-profit percentages from ML optimizer

        MINIMAL EXTRACTION: ML not included - this method returns early
        """
        # ML not included in minimal extraction - always return early
        return

    def round_price_for_broker(self, price):
        """
        Round price to valid increment for broker submission.

        CRITICAL FIX (Dec 3, 2025): Alpaca rejects sub-penny prices.
        - Stocks >= $1: Round to 2 decimal places (cents)
        - Stocks < $1: Round to 4 decimal places (sub-penny allowed for penny stocks)

        Args:
            price: Raw price to round

        Returns:
            float: Properly rounded price for broker
        """
        if price >= 1.0:
            return round(price, 2)
        else:
            return round(price, 4)

    def check_buying_power(self, available_buying_power, position_value, entry_price, position_size):
        """
        Validate sufficient buying power before order submission.

        CRITICAL FIX (Dec 3, 2025): Check BP BEFORE submitting order.
        Bug: Orders were being rejected with "insufficient day trading buying power"
        because the bot calculated position size without checking available BP.

        Args:
            available_buying_power: Current available buying power from account
            position_value: Total value of the proposed position (price * size)
            entry_price: Entry price per share
            position_size: Number of shares to buy

        Returns:
            dict: {
                'approved': bool,
                'available_bp': float,
                'required_bp': float,
                'reason': str (if not approved)
            }
        """
        # Use 90% of BP to leave margin for price movement
        usable_bp = available_buying_power * 0.9

        if position_value <= usable_bp:
            return {
                'approved': True,
                'available_bp': available_buying_power,
                'required_bp': position_value,
                'usable_bp': usable_bp
            }
        else:
            return {
                'approved': False,
                'available_bp': available_buying_power,
                'required_bp': position_value,
                'usable_bp': usable_bp,
                'reason': f"Insufficient buying power: need ${position_value:.2f}, have ${usable_bp:.2f} usable (90% of ${available_buying_power:.2f})"
            }

    def adjust_position_for_buying_power(self, requested_size, entry_price, available_buying_power, safety_margin=0.9):
        """
        Adjust position size to fit within available buying power.

        CRITICAL FIX (Dec 3, 2025): Prevent order rejection by sizing to BP.

        Args:
            requested_size: Originally calculated position size
            entry_price: Price per share
            available_buying_power: Current available BP
            safety_margin: Fraction of BP to use (default 0.9 = 90%)

        Returns:
            int: Adjusted position size that fits within buying power
        """
        if available_buying_power <= 0 or entry_price <= 0:
            return 0

        max_affordable = int((available_buying_power * safety_margin) / entry_price)
        adjusted = min(requested_size, max_affordable)

        if adjusted < requested_size:
            self.logger.warning(
                f"Position size reduced from {requested_size} to {adjusted} shares "
                f"due to buying power limit (${available_buying_power:.2f} BP, "
                f"${entry_price:.2f}/share)"
            )

        return max(0, adjusted)

    def ensure_utc_aware(self, dt):
        """
        Ensure datetime is UTC-aware.

        CRITICAL FIX (Dec 3, 2025): Prevent "can't subtract offset-naive
        and offset-aware datetimes" errors.

        Args:
            dt: datetime object (naive or aware)

        Returns:
            datetime: UTC-aware datetime
        """
        if dt is None:
            return datetime.now(pytz.UTC)

        if dt.tzinfo is None:
            # Naive datetime - assume it's UTC and localize
            return pytz.UTC.localize(dt)
        elif dt.tzinfo != pytz.UTC:
            # Aware but not UTC - convert to UTC
            return dt.astimezone(pytz.UTC)
        else:
            # Already UTC-aware
            return dt

    def calculate_hold_time_minutes(self, entry_time, current_time=None):
        """
        Calculate hold time in minutes with timezone-safe handling.

        CRITICAL FIX (Dec 3, 2025): Handle mixed timezone datetimes.

        Args:
            entry_time: Position entry time (naive or aware)
            current_time: Current time (naive or aware), defaults to now

        Returns:
            float: Hold time in minutes
        """
        if current_time is None:
            current_time = datetime.now(pytz.UTC)

        # Ensure both are UTC-aware
        entry_utc = self.ensure_utc_aware(entry_time)
        current_utc = self.ensure_utc_aware(current_time)

        delta = current_utc - entry_utc
        return delta.total_seconds() / 60

    # ==================== END NEW HELPER METHODS ====================

    def update_settings(self, bot_settings):
        """Update risk manager settings dynamically - NEW Nov 3, 2025, ENHANCED Nov 5, 2025"""
        if bot_settings:
            self.max_risk_per_trade = bot_settings.get('max_risk_per_trade', self.max_risk_per_trade)
            self.max_daily_loss = bot_settings.get('daily_loss_limit', self.max_daily_loss)
            self.max_position_size = bot_settings.get('max_position_size', self.max_position_size)
            self.max_positions = bot_settings.get('max_positions', self.max_positions)
            self.enable_position_scaling = bot_settings.get('enable_position_scaling', self.enable_position_scaling)
            self.scaling_profit_threshold = bot_settings.get('scaling_profit_threshold', self.scaling_profit_threshold)

            # CRITICAL FIX (Nov 5, 2025): Update stop/profit percentages
            if 'stop_loss_pct' in bot_settings:
                self.stop_loss_pct = bot_settings['stop_loss_pct'] / 100  # Convert to decimal
            if 'take_profit_pct' in bot_settings:
                self.take_profit_pct = bot_settings['take_profit_pct'] / 100  # Convert to decimal
            if 'risk_reward_ratio' in bot_settings:
                self.risk_reward_ratio = bot_settings['risk_reward_ratio']

            self.logger.info(f"RiskManager settings updated: max_positions={self.max_positions}, "
                           f"max_position_size={self.max_position_size*100:.0f}%, "
                           f"max_risk={self.max_risk_per_trade*100:.1f}%, "
                           f"stop_loss={self.stop_loss_pct*100:.1f}%, "
                           f"take_profit={self.take_profit_pct*100:.1f}%")

    def calculate_position_size(self, portfolio_value, entry_price, stop_loss_price, volatility=None, vix_level=None):
        """
        Calculate optimal position size based on risk management rules

        CRITICAL FIX (Nov 18, 2025): Added hard cap to prevent oversized positions
        Bug: HUM trade was 517 shares @ $283 = $146K (should have been max 52-235 shares)
        Root cause: Portfolio value calculation included unrealized P&L
        Solution: Multiple layers of position size caps

        NEW (Dec 8, 2025): VIX-based position sizing - reduce size during high volatility
        """
        try:
            # BUG FIX (Dec 17, 2025): Validate inputs are not NaN
            # NaN comparisons (e.g., NaN <= 0) return False, bypassing validation
            # This caused "cannot convert float NaN to integer" errors in intraday backtests
            if pd.isna(entry_price) or pd.isna(stop_loss_price) or pd.isna(portfolio_value):
                self.logger.debug(f"Skipping position size calc: NaN input (entry={entry_price}, stop={stop_loss_price}, portfolio={portfolio_value})")
                return 0

            if entry_price <= 0 or stop_loss_price <= 0:
                return 0

            # BUG FIX (Dec 2025): Validate stop-loss direction makes sense
            # For LONG positions, stop must be BELOW entry (stop_loss < entry)
            # For SHORT positions, stop must be ABOVE entry (stop_loss > entry)
            # If validation fails and it's clearly wrong, return 0
            # Note: abs() is used for risk_per_share so position sizing still works

            # CRITICAL FIX (Nov 18, 2025): Validate portfolio value is reasonable
            # Prevent using inflated portfolio values from unrealized P&L
            MAX_REASONABLE_PORTFOLIO = 500000  # $500K max (safety check)
            if portfolio_value > MAX_REASONABLE_PORTFOLIO:
                self.logger.warning(f"Portfolio value ${portfolio_value:,.0f} exceeds ${MAX_REASONABLE_PORTFOLIO:,.0f} - capping for safety")
                portfolio_value = MAX_REASONABLE_PORTFOLIO

            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss_price)

            # BUG FIX (Dec 4, 2025): Use tolerance for floating point comparison
            # Using == 0 with floats is unreliable due to precision issues
            if risk_per_share < 0.0001:  # Less than 0.01 cents
                return 0

            # Calculate maximum dollar risk based on portfolio percentage
            max_dollar_risk = portfolio_value * self.max_risk_per_trade

            # Calculate position size based on dollar risk
            position_size_by_risk = int(max_dollar_risk / risk_per_share)

            # Calculate maximum position size by portfolio percentage
            max_position_value = portfolio_value * self.max_position_size
            position_size_by_value = int(max_position_value / entry_price)

            # Take the smaller of the two
            position_size = min(position_size_by_risk, position_size_by_value)

            # CRITICAL FIX (Nov 18, 2025): Hard cap on position value
            # Prevents catastrophic losses like HUM (-$12K on single trade)
            # BUG FIX (Dec 10, 2025): Load max_position_dollars from config instead of hardcoding
            # This allows runtime configuration changes without code modification
            # Default to $10K if not specified in settings
            MAX_POSITION_VALUE_DOLLARS = self.settings.get('max_position_dollars', 10000)
            max_shares_by_dollars = int(MAX_POSITION_VALUE_DOLLARS / entry_price)

            if position_size > max_shares_by_dollars:
                self.logger.warning(f"Position size capped at ${MAX_POSITION_VALUE_DOLLARS:,} "
                                  f"({max_shares_by_dollars} shares @ ${entry_price:.2f}, "
                                  f"reduced from {position_size} shares)")
                position_size = max_shares_by_dollars

            # Adjust for volatility if provided
            # BUG FIX (Dec 4, 2025): Added pd.notna() check to handle NaN volatility values
            # Without this check, NaN volatility would pass the > 0 check and cause division issues
            # BUG FIX (Dec 4, 2025): Also check for inf - pd.notna(inf) returns True!
            # Infinite volatility would cause volatility_adjustment = 0, making position_size = 0
            if volatility is not None and pd.notna(volatility) and not np.isinf(volatility) and volatility > 0:
                volatility_adjustment = min(1.0, 0.3 / volatility)  # Reduce size for high volatility
                position_size = int(position_size * volatility_adjustment)

            # NEW (Dec 8, 2025): VIX-based position sizing adjustment
            # Reduce position size during high market volatility (VIX levels)
            if vix_level is not None and pd.notna(vix_level) and vix_level > 0:
                vix_adjustment = self.adjust_position_for_volatility(position_size, vix_level)
                if vix_adjustment < position_size:
                    self.logger.info(f"VIX-based size reduction: {position_size} -> {vix_adjustment} shares (VIX: {vix_level:.1f})")
                    position_size = vix_adjustment

            # Final validation
            position_value = position_size * entry_price
            if position_value > MAX_POSITION_VALUE_DOLLARS:
                self.logger.error(f"SAFETY CHECK FAILED: Position ${position_value:,.0f} > ${MAX_POSITION_VALUE_DOLLARS:,}")
                position_size = max_shares_by_dollars

            return max(0, position_size)

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0

    def check_daily_loss_limit(self, current_pnl, portfolio_value):
        """
        Check if daily loss limit has been reached

        BUG FIX (Dec 2025): Fixed division by zero and added proper validation
        """
        # Validate inputs first
        if portfolio_value is None or portfolio_value <= 0:
            self.logger.error("Cannot check daily loss limit: invalid portfolio value")
            return True  # Block trading if we can't validate

        if current_pnl >= 0:
            return False  # No loss, all good

        # Calculate loss percentage safely
        loss_percentage = abs(current_pnl) / portfolio_value

        if loss_percentage >= self.max_daily_loss:
            self.logger.warning(f"Daily loss limit reached: {loss_percentage:.2%} (${current_pnl:,.2f} on ${portfolio_value:,.2f})")
            return True

        return False

    def check_portfolio_risk(self, positions, portfolio_value=None):
        """
        Check total portfolio risk exposure.

        BUG FIX (Dec 4, 2025): Added portfolio_value parameter.
        Previously used positions.get('portfolio_value', 1) which always returned 1
        because positions is a dict of {symbol: position_data}, not a dict containing
        'portfolio_value' as a key. This made the risk check useless.

        Args:
            positions: Dict of {symbol: position_data}
            portfolio_value: Current portfolio value in dollars (required for accurate check)

        Returns:
            bool: True if risk is acceptable, False if exceeded
        """
        total_risk = 0.0

        for symbol, position in positions.items():
            if isinstance(position, dict) and 'risk_amount' in position:
                total_risk += position['risk_amount']

        # BUG FIX: Use explicit portfolio_value parameter, not positions.get()
        # BUG FIX (Dec 2025): Return False when portfolio_value invalid - don't allow risky trades
        if portfolio_value is None or portfolio_value <= 0:
            self.logger.error("Portfolio value not provided or zero, cannot calculate risk percentage - BLOCKING trade")
            return False  # BLOCK trade - can't validate risk without portfolio value

        risk_percentage = total_risk / portfolio_value

        if risk_percentage > self.max_portfolio_risk:
            self.logger.warning(f"Portfolio risk limit exceeded: {risk_percentage:.2%}")
            return False

        return True

    def check_position_limits(self, current_positions):
        """Check if position limits are exceeded"""
        if len(current_positions) >= self.max_positions:
            self.logger.warning(f"Maximum positions limit reached: {len(current_positions)}")
            return False

        return True

    def check_trade_frequency(self, trades_today):
        """Check if daily trade frequency limits are exceeded"""
        if trades_today >= self.max_trades_per_day:
            self.logger.warning(f"Daily trade limit reached: {trades_today}")
            return False

        return True

    def calculate_stop_loss(self, entry_price, side, atr=None, volatility=None, strategy=None, symbol=None):
        """
        Calculate stop loss price based on various methods
        CRITICAL FIX (Nov 5, 2025): Use configured stop_loss_pct as MINIMUM floor
        EXPERT UPGRADE (Nov 10, 2025): Use strategy-specific stop loss percentages
        BUG FIX (Dec 2025): Use leveraged ETF rules for 3x/2x ETFs
        ENHANCED (Dec 8, 2025): ATR-based dynamic stops with configurable multiplier and bounds
        """
        # Get ATR stop settings from config (Dec 8, 2025 - Proposal #6)
        use_atr_stops = self.settings.get('use_atr_stops', False)
        atr_multiplier = self.settings.get('atr_multiplier', 2.0)
        min_stop_pct = self.settings.get('min_stop_pct', 0.5) / 100  # Convert to decimal
        max_stop_pct = self.settings.get('max_stop_pct', 5.0) / 100  # Convert to decimal

        # BUG FIX (Dec 2025): Use leveraged ETF stop-loss if applicable
        if symbol and self.is_leveraged_etf(symbol):
            stop_loss_pct = self.leveraged_etf_stop_loss_pct
        # EXPERT FIX: Use strategy-specific stop loss if provided
        elif strategy and strategy in self.strategy_stop_loss_pct:
            stop_loss_pct = self.strategy_stop_loss_pct[strategy]
        else:
            stop_loss_pct = self.stop_loss_pct

        # ================================================================
        # ATR-BASED DYNAMIC STOPS (Dec 8, 2025) - Proposal #6
        # ================================================================
        if use_atr_stops and atr is not None and not pd.isna(atr) and atr > 0:
            # Calculate ATR-based stop percentage
            atr_stop_pct = (atr * atr_multiplier) / entry_price

            # Clamp to min/max bounds (per guidelines: 0.5% - 5.0%)
            atr_stop_pct = max(min_stop_pct, min(max_stop_pct, atr_stop_pct))

            self.logger.debug(f"ATR Dynamic Stop: ATR=${atr:.2f}, multiplier={atr_multiplier}, "
                            f"raw_pct={atr_stop_pct*100:.2f}%, clamped to {min_stop_pct*100:.1f}%-{max_stop_pct*100:.1f}%")

            # Use ATR-based stop instead of fixed percentage
            stop_loss_pct = atr_stop_pct

        # Calculate stop price
        if side.lower() == 'buy':
            stop_loss = entry_price * (1 - stop_loss_pct)
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)

        # FALLBACK: If ATR stops disabled, try legacy ATR/volatility calculation
        if not use_atr_stops:
            alternative_stop = None

            if side.lower() == 'buy':
                # BUG FIX #9 (Dec 9, 2025): ATR validation - check for None, NaN, and positive value
                if atr is not None and not pd.isna(atr) and atr > 0:
                    alternative_stop = entry_price - (1.2 * atr)
                elif volatility is not None:
                    alternative_stop = entry_price * (1 - min(0.03, volatility * 1.5))
            else:
                # BUG FIX #9 (Dec 9, 2025): ATR validation - check for None, NaN, and positive value
                if atr is not None and not pd.isna(atr) and atr > 0:
                    alternative_stop = entry_price + (1.2 * atr)
                elif volatility is not None:
                    alternative_stop = entry_price * (1 + min(0.03, volatility * 1.5))

            # Use wider stop if alternative exists
            if alternative_stop is not None:
                if side.lower() == 'buy':
                    stop_loss = min(stop_loss, alternative_stop)
                else:
                    stop_loss = max(stop_loss, alternative_stop)

        # CRITICAL FIX (Dec 3, 2025): Use broker-safe rounding to avoid sub-penny rejections
        return self.round_price_for_broker(stop_loss)

    def calculate_take_profit(self, entry_price, stop_loss_price, side, risk_reward_ratio=None, symbol=None):
        """
        Calculate take profit price based on configured percentage
        CRITICAL FIX (Nov 5, 2025): Use configured take_profit_pct directly
        BUG FIX (Dec 2025): Use leveraged ETF take-profit for 3x/2x ETFs
        Bug: Old method used risk/reward ratio which gave tiny profits (0.13-0.59%)
        Solution: Use configured 3.2% take-profit percentage (or 2.0% for leveraged)
        """
        # Priority 1: Use leveraged ETF take-profit if applicable
        if symbol and self.is_leveraged_etf(symbol):
            take_profit_pct = self.leveraged_etf_take_profit_pct
        # Priority 2: Use global configured take-profit
        else:
            take_profit_pct = self.take_profit_pct

        # Use configured percentage directly (Nov 5, 2025)
        if side.lower() == 'buy':
            take_profit = entry_price * (1 + take_profit_pct)
        else:
            take_profit = entry_price * (1 - take_profit_pct)

        # CRITICAL FIX (Dec 3, 2025): Use broker-safe rounding to avoid sub-penny rejections
        return self.round_price_for_broker(take_profit)

    def assess_market_conditions(self, market_data):
        """Assess current market conditions for risk adjustment"""
        try:
            # BUG FIX (Dec 4, 2025): Check for empty or insufficient data
            if market_data is None or len(market_data) < 2:
                self.logger.warning("Insufficient market data for condition assessment")
                return {'risk_level': 'normal', 'volatility': 0.2, 'volume_ratio': 1.0, 'trend_strength': 0}

            # Calculate market volatility
            # BUG FIX (Dec 4, 2025): Add fill_method=None to prevent FutureWarning in pandas
            returns = market_data['close'].pct_change(fill_method=None).dropna()

            # BUG FIX (Dec 4, 2025): Handle empty returns or NaN std
            if len(returns) < 2:
                volatility = 0.2  # Default volatility
            else:
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                # Handle NaN volatility
                if pd.isna(volatility):
                    volatility = 0.2

            # Calculate average volume
            avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
            current_volume = market_data['volume'].iloc[-1]

            # BUG FIX (Dec 4, 2025): Handle NaN values in volume ratio calculation
            # If avg_volume or current_volume is NaN/0, default to 1.0
            if pd.notna(avg_volume) and pd.notna(current_volume) and avg_volume > 0:
                volume_ratio = current_volume / avg_volume
            else:
                volume_ratio = 1.0

            # Assess market regime
            conditions = {
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'trend_strength': self._calculate_trend_strength(market_data),
                'risk_level': 'normal'
            }

            # Determine risk level
            if volatility > 0.4:  # High volatility
                conditions['risk_level'] = 'high'
            elif volatility < 0.15:  # Low volatility
                conditions['risk_level'] = 'low'

            return conditions

        except Exception as e:
            self.logger.error(f"Error assessing market conditions: {e}")
            return {'risk_level': 'normal', 'volatility': 0.2, 'volume_ratio': 1.0, 'trend_strength': 0}

    def _calculate_trend_strength(self, data):
        """Calculate trend strength indicator"""
        try:
            # BUG FIX (Dec 4, 2025): Check for sufficient data
            if data is None or len(data) < 5:
                return 0  # No trend with insufficient data

            # Use multiple moving averages to determine trend strength
            sma_5 = data['close'].rolling(5).mean()
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()

            # Count trend alignment
            latest_price = data['close'].iloc[-1]
            latest_sma5 = sma_5.iloc[-1]
            latest_sma20 = sma_20.iloc[-1]
            latest_sma50 = sma_50.iloc[-1]

            # BUG FIX (Dec 4, 2025): Handle NaN values from insufficient data for longer SMAs
            # If any SMA is NaN, use only the available SMAs for trend calculation
            if pd.isna(latest_sma50):
                # Not enough data for 50-SMA, use simplified calculation
                if pd.isna(latest_sma20):
                    # Not enough data for 20-SMA either
                    if pd.isna(latest_sma5):
                        return 0  # No trend data available
                    # Only have 5-SMA
                    if latest_price > latest_sma5:
                        return 0.2
                    elif latest_price < latest_sma5:
                        return -0.2
                    return 0
                # Have 5 and 20 SMA
                if latest_price > latest_sma5 > latest_sma20:
                    return 0.5
                elif latest_price < latest_sma5 < latest_sma20:
                    return -0.5
                elif latest_price > latest_sma20:
                    return 0.3
                elif latest_price < latest_sma20:
                    return -0.3
                return 0

            # Full calculation with all SMAs available
            # Strong uptrend
            if latest_price > latest_sma5 > latest_sma20 > latest_sma50:
                return 0.8
            # Strong downtrend
            elif latest_price < latest_sma5 < latest_sma20 < latest_sma50:
                return -0.8
            # Weak trends
            elif latest_price > latest_sma20:
                return 0.3
            elif latest_price < latest_sma20:
                return -0.3
            else:
                return 0  # Sideways

        except Exception as e:
            return 0

    def validate_trade(self, symbol, side, quantity, entry_price, portfolio_value, current_positions, strategy=None, internal_positions=None):
        """
        Validate if a trade meets all risk management criteria
        CRITICAL FIX (Nov 5, 2025): Add HARD position limit guard
        EXPERT UPGRADE (Nov 10, 2025): Add strategy concentration limits
        CRITICAL FIX (Nov 10, 2025): Use internal_positions for strategy tracking
        BUG FIX (Dec 2025): Add leveraged ETF position limits and size checks
        Bug: Bot opened 41 positions when max was 20
        Solution: Absolute block if position count >= max_positions
        """
        try:
            validation_results = {
                'approved': True,
                'reasons': [],
                'warnings': []
            }

            # BUG FIX (Dec 10, 2025): Re-enabled position limit check as SAFETY GUARD
            # Even if user requested removal, we need a hard cap to prevent runaway position count
            # The max_positions is set to 9999 by default (effectively unlimited) but can be overridden
            if self.max_positions < 9999 and current_positions is not None:
                if side.lower() == 'buy' and len(current_positions) >= self.max_positions:
                    validation_results['approved'] = False
                    validation_results['reasons'].append(
                        f"HARD LIMIT: Max positions ({self.max_positions}) reached. "
                        f"Current: {len(current_positions)}"
                    )
                    return validation_results

            # Calculate position value for all checks
            position_value = quantity * entry_price

            # BUG FIX (Dec 10, 2025): Add warning for duplicate positions
            # Allows trade but warns user about adding to existing position
            if current_positions is not None and symbol in current_positions:
                validation_results['warnings'].append(f"Already holding {symbol} - this will add to existing position")

            # BUG FIX (Dec 2025): Increased minimum from $100 to $500
            # $100 positions lose money to slippage/spread - need larger positions
            # Minimum position value check (keep this to avoid tiny useless trades)
            if position_value < 500:  # Minimum $500 position
                validation_results['approved'] = False
                validation_results['reasons'].append(f"Position value too small: ${position_value:.2f} < $500 minimum")

            # BUG FIX (Dec 10, 2025): Enforce max position size for ALL stocks
            # Previously only leveraged ETFs had this check, but oversized positions are risky for all stocks
            if portfolio_value > 0:
                max_position_value = portfolio_value * self.max_position_size
                max_position_dollars = self.settings.get('max_position_dollars', 10000)
                # Use the smaller of percentage-based limit and dollar limit
                effective_max = min(max_position_value, max_position_dollars)
                if position_value > effective_max:
                    validation_results['approved'] = False
                    validation_results['reasons'].append(
                        f"Position too large: ${position_value:.2f} > ${effective_max:.2f} "
                        f"(max {self.max_position_size*100:.0f}% of portfolio or ${max_position_dollars:,})"
                    )

            # BUG FIX (Dec 2025): Enforce leveraged ETF position limits
            if self.is_leveraged_etf(symbol):
                # Check max position size for leveraged ETFs
                max_leveraged_value = portfolio_value * self.leveraged_etf_max_position_size
                if position_value > max_leveraged_value:
                    validation_results['approved'] = False
                    validation_results['reasons'].append(
                        f"Leveraged ETF position too large: ${position_value:.2f} > "
                        f"${max_leveraged_value:.2f} ({self.leveraged_etf_max_position_size*100:.0f}% max for leveraged)"
                    )

                # Count current leveraged positions
                if current_positions:
                    leveraged_count = sum(1 for sym in current_positions.keys() if self.is_leveraged_etf(sym))
                    if leveraged_count >= self.max_leveraged_positions:
                        validation_results['approved'] = False
                        validation_results['reasons'].append(
                            f"Max leveraged positions reached: {leveraged_count}/{self.max_leveraged_positions}"
                        )

            return validation_results

        except Exception as e:
            self.logger.error(f"Error validating trade: {e}")
            return {'approved': False, 'reasons': ['Validation error']}

    def update_risk_metrics(self, trade_data):
        """Update risk tracking metrics after a trade"""
        try:
            self.daily_trades += 1

            if 'pnl' in trade_data:
                self.daily_pnl += trade_data['pnl']

            # Update position risk if it's a new position
            if trade_data.get('side') == 'buy':
                symbol = trade_data.get('symbol')
                risk_amount = trade_data.get('risk_amount', 0)
                self.positions_risk[symbol] = risk_amount

            # Remove position risk if closing
            elif trade_data.get('side') == 'sell':
                symbol = trade_data.get('symbol')
                if symbol in self.positions_risk:
                    del self.positions_risk[symbol]

        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")

    def get_risk_summary(self):
        """Get current risk summary"""
        return {
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'current_positions': len(self.positions_risk),
            'total_risk': sum(self.positions_risk.values()),
            'max_trades_remaining': max(0, self.max_trades_per_day - self.daily_trades),
            'max_positions_remaining': max(0, self.max_positions - len(self.positions_risk))
        }

    def adjust_position_for_volatility(self, position_size, vix_level):
        """
        Adjust position size based on VIX level (market volatility)

        NEW (Dec 8, 2025): VIX-based position sizing
        - VIX < 15: Normal position size (100%)
        - VIX 15-25: Reduced position size (80%)
        - VIX > 25: Heavily reduced position size (50%)

        Args:
            position_size: Originally calculated position size
            vix_level: Current VIX level (using VIXY as proxy)

        Returns:
            int: Adjusted position size
        """
        if vix_level < 15:
            # Low volatility - normal size
            adjustment_factor = 1.0
        elif vix_level < 25:
            # Moderate volatility - reduce to 80%
            adjustment_factor = 0.8
        else:
            # High volatility - reduce to 50%
            adjustment_factor = 0.5

        adjusted_size = int(position_size * adjustment_factor)
        return max(0, adjusted_size)

    def reset_daily_metrics(self):
        """Reset daily tracking metrics (call at market open)"""
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.positions_risk.clear()
        self.logger.info("Daily risk metrics reset")


# =============================================================================
# EXIT MANAGER - Tiered Exit Logic System (Dec 18, 2025, TIGHTENED Dec 29, 2025)
# =============================================================================
#
# TIERED EXIT LOGIC (TIGHTENED Dec 29, 2025):
# 1. Unrealized PnL < +0.40%: No trailing, only hard stop-loss (-0.50%)
# 2. Unrealized PnL >= +0.40%: PROFIT FLOOR ACTIVE - locks in +0.15% minimum
# 3. Unrealized PnL >= +0.60%: ATR TRAILING ACTIVE - trails by ATR(14) x 1.5
# 4. Unrealized PnL >= +1.00%: PARTIAL TAKE PROFIT - close 50% of position
#
# KEY FEATURES:
# - Profit protection floor (non-decreasing hard stop at +0.15% once +0.40% reached)
# - ATR-based trailing stops (volatility-aware, not fixed percentage)
# - Partial profit taking at +1.0%
# - All exits are programmatic, logged, and backtestable
# =============================================================================


@dataclass
class PositionExitState:
    """
    Track exit state for a single position.

    This state persists across price updates and ensures:
    - Profit floor only tightens (never loosens)
    - Trailing stop only tightens (never loosens)
    - Partial take profit executes exactly once
    """
    symbol: str
    entry_price: float
    entry_time: datetime
    quantity: int

    # Exit thresholds (percentages as decimals, e.g., 0.0125 for 1.25%)
    profit_floor_activation_pct: float = 0.0125  # +1.25%
    profit_floor_lock_pct: float = 0.005         # +0.50%
    trailing_activation_pct: float = 0.0175      # +1.75%
    partial_tp_pct: float = 0.02                 # +2.00%
    partial_tp_size: float = 0.50                # Close 50% at partial TP
    hard_stop_pct: float = 0.005                 # -0.50% hard stop (from entry)

    # State tracking
    profit_floor_active: bool = False
    profit_floor_price: float = 0.0              # Locked floor price (only goes UP)
    trailing_active: bool = False
    trailing_stop_price: float = 0.0             # Trailing stop (only goes UP)
    partial_tp_executed: bool = False
    partial_tp_qty: int = 0                      # Shares closed at partial TP

    # Peak tracking for trailing
    peak_price: float = 0.0
    peak_unrealized_pnl_pct: float = 0.0

    # ATR for volatility-aware trailing
    current_atr: float = 0.0
    atr_multiplier: float = 2.0

    # Minimum hold time (Dec 29, 2025)
    min_hold_bars: int = 12                    # Minimum bars before non-stop exits
    min_hold_bypass_loss_pct: float = 0.003    # Allow early exit if loss > 0.30%
    bars_held: int = 0                         # Current bars held

    # Logging
    last_update_time: datetime = field(default_factory=lambda: datetime.now(pytz.UTC))

    def __post_init__(self):
        """Initialize peak price to entry price."""
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price


class ExitManager:
    """
    Manages tiered exit logic for all positions.

    USAGE:
        exit_mgr = ExitManager(bot_settings)

        # On each price update:
        action = exit_mgr.evaluate_exit(symbol, current_price, current_atr)

        # action is one of:
        # - None: No action needed
        # - {'action': 'partial_tp', 'qty': 50, 'reason': '...'}
        # - {'action': 'full_exit', 'reason': '...'}
    """

    # Exit reasons (for logging and backtesting)
    REASON_PROFIT_FLOOR = 'profit_floor'
    REASON_ATR_TRAILING = 'atr_trailing'
    REASON_PARTIAL_TP = 'partial_tp'
    REASON_HARD_STOP = 'hard_stop'

    def __init__(self, bot_settings: dict = None):
        """
        Initialize exit manager with configuration.

        Args:
            bot_settings: Bot configuration dict with 'risk' section
        """
        self.logger = logging.getLogger(__name__)
        self.positions: Dict[str, PositionExitState] = {}

        # Load settings from config
        risk_settings = (bot_settings or {}).get('risk', {})

        # Exit thresholds (convert from percentage to decimal)
        self.profit_floor_activation_pct = risk_settings.get('profit_floor_activation_pct', 1.25) / 100
        self.profit_floor_lock_pct = risk_settings.get('trailing_stop_min_profit_floor', 0.50) / 100
        self.trailing_activation_pct = risk_settings.get('trailing_activation_pct', 1.75) / 100
        self.partial_tp_pct = risk_settings.get('partial_tp_pct', 2.0) / 100
        self.partial_tp_size = risk_settings.get('partial_tp_size', 0.50)  # 50% default
        self.hard_stop_pct = risk_settings.get('hard_stop_pct', 0.50) / 100

        # ATR settings
        self.atr_multiplier = risk_settings.get('atr_trailing_multiplier', 2.0)

        # Minimum hold time settings (Dec 29, 2025)
        # Analysis showed short trades (<=10 bars) lose money, long trades (>30 bars) profit
        self.min_hold_bars = risk_settings.get('min_hold_bars', 12)
        self.min_hold_bypass_loss_pct = risk_settings.get('min_hold_bypass_loss_pct', 0.30) / 100  # 0.30%

        self.logger.info(
            f"ExitManager initialized: "
            f"profit_floor={self.profit_floor_activation_pct*100:.2f}%->{self.profit_floor_lock_pct*100:.2f}%, "
            f"trailing_activation={self.trailing_activation_pct*100:.2f}%, "
            f"partial_tp={self.partial_tp_pct*100:.2f}%@{self.partial_tp_size*100:.0f}%, "
            f"hard_stop=-{self.hard_stop_pct*100:.2f}%, "
            f"ATR_mult={self.atr_multiplier}, "
            f"min_hold={self.min_hold_bars} bars"
        )

    def register_position(self, symbol: str, entry_price: float, quantity: int,
                         entry_time: datetime = None) -> PositionExitState:
        """
        Register a new position for exit management.

        Args:
            symbol: Stock symbol
            entry_price: Entry price per share
            quantity: Number of shares
            entry_time: Entry timestamp (defaults to now)

        Returns:
            PositionExitState object for the position
        """
        if entry_time is None:
            entry_time = datetime.now(pytz.UTC)

        state = PositionExitState(
            symbol=symbol,
            entry_price=entry_price,
            entry_time=entry_time,
            quantity=quantity,
            profit_floor_activation_pct=self.profit_floor_activation_pct,
            profit_floor_lock_pct=self.profit_floor_lock_pct,
            trailing_activation_pct=self.trailing_activation_pct,
            partial_tp_pct=self.partial_tp_pct,
            partial_tp_size=self.partial_tp_size,
            hard_stop_pct=self.hard_stop_pct,
            atr_multiplier=self.atr_multiplier,
            peak_price=entry_price,
            min_hold_bars=self.min_hold_bars,
            min_hold_bypass_loss_pct=self.min_hold_bypass_loss_pct,
            bars_held=0
        )

        self.positions[symbol] = state

        self.logger.info(
            f"EXIT_MGR | {symbol} | REGISTERED | "
            f"Entry: ${entry_price:.2f}, Qty: {quantity}, "
            f"Hard Stop: ${entry_price * (1 - self.hard_stop_pct):.2f} (-{self.hard_stop_pct*100:.2f}%), "
            f"Min Hold: {self.min_hold_bars} bars"
        )

        return state

    def unregister_position(self, symbol: str) -> bool:
        """
        Remove a position from exit management.

        Args:
            symbol: Stock symbol

        Returns:
            True if position was removed, False if not found
        """
        if symbol in self.positions:
            del self.positions[symbol]
            self.logger.info(f"EXIT_MGR | {symbol} | UNREGISTERED")
            return True
        return False

    def update_quantity(self, symbol: str, new_quantity: int) -> bool:
        """
        Update quantity after partial close.

        Args:
            symbol: Stock symbol
            new_quantity: New total quantity

        Returns:
            True if updated, False if position not found
        """
        if symbol in self.positions:
            old_qty = self.positions[symbol].quantity
            self.positions[symbol].quantity = new_quantity
            self.logger.info(f"EXIT_MGR | {symbol} | QTY_UPDATE | {old_qty} -> {new_quantity}")
            return True
        return False

    def increment_bars_held(self, symbol: str) -> int:
        """
        Increment the bars held counter for a position.

        Called once per bar to track hold time.

        Args:
            symbol: Stock symbol

        Returns:
            New bars_held count, or -1 if position not found
        """
        if symbol in self.positions:
            self.positions[symbol].bars_held += 1
            return self.positions[symbol].bars_held
        return -1

    def evaluate_exit(self, symbol: str, current_price: float,
                     current_atr: float = None) -> Optional[dict]:
        """
        Evaluate exit conditions for a position.

        This is the CORE METHOD called on every price update.

        Args:
            symbol: Stock symbol
            current_price: Current market price
            current_atr: Current ATR(14) value (optional but recommended)

        Returns:
            None if no action needed, or dict with:
            - action: 'partial_tp' or 'full_exit'
            - qty: Shares to close (for partial_tp)
            - reason: Exit reason (for logging)
            - stop_price: Stop price that triggered exit
            - unrealized_pnl_pct: P&L percentage at trigger
        """
        if symbol not in self.positions:
            return None

        state = self.positions[symbol]
        entry_price = state.entry_price

        # Calculate current unrealized P&L
        unrealized_pnl_pct = (current_price - entry_price) / entry_price

        # Update ATR if provided
        if current_atr is not None and current_atr > 0:
            state.current_atr = current_atr

        # Update peak tracking
        if current_price > state.peak_price:
            state.peak_price = current_price
            state.peak_unrealized_pnl_pct = unrealized_pnl_pct

        state.last_update_time = datetime.now(pytz.UTC)

        # ================================================================
        # TIER 0: HARD STOP (-0.50% from entry) - Always active
        # ================================================================
        hard_stop_price = entry_price * (1 - state.hard_stop_pct)
        if current_price <= hard_stop_price:
            self._log_exit_trigger(
                symbol, self.REASON_HARD_STOP,
                current_price, hard_stop_price, unrealized_pnl_pct
            )
            return {
                'action': 'full_exit',
                'reason': self.REASON_HARD_STOP,
                'stop_price': hard_stop_price,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'qty': state.quantity
            }

        # ================================================================
        # MINIMUM HOLD TIME CHECK (Dec 29, 2025)
        # Analysis showed short trades (<=10 bars) lose money, long trades profit
        # Block non-stop exits until min_hold_bars is reached
        # Exception: allow early exit if loss exceeds bypass threshold
        # ================================================================
        if state.bars_held < state.min_hold_bars:
            # Check if we should bypass minimum hold due to significant loss
            bypass_for_loss = unrealized_pnl_pct < -state.min_hold_bypass_loss_pct

            if not bypass_for_loss:
                # Not enough bars held and no bypass condition - skip all exits
                # (Hard stop already checked above, so we're safe from catastrophic loss)
                return None

            # Log bypass
            self.logger.debug(
                f"EXIT_MGR | {symbol} | MIN_HOLD_BYPASS | "
                f"Bars: {state.bars_held}/{state.min_hold_bars}, "
                f"Loss: {unrealized_pnl_pct*100:.2f}% < -{state.min_hold_bypass_loss_pct*100:.2f}%"
            )

        # ================================================================
        # TIER 1: PROFIT FLOOR CHECK (activates at +0.40%)
        # Once active, enforces minimum +0.15% profit (non-decreasing floor)
        # ================================================================
        if not state.profit_floor_active:
            # Check if we should activate profit floor
            if unrealized_pnl_pct >= state.profit_floor_activation_pct:
                state.profit_floor_active = True
                # Set initial floor at locked profit level
                state.profit_floor_price = entry_price * (1 + state.profit_floor_lock_pct)

                self.logger.info(
                    f"EXIT_MGR | {symbol} | PROFIT_FLOOR_ACTIVATED | "
                    f"Current: +{unrealized_pnl_pct*100:.2f}%, "
                    f"Floor locked at: ${state.profit_floor_price:.2f} (+{state.profit_floor_lock_pct*100:.2f}%)"
                )

        if state.profit_floor_active:
            # Floor only moves UP (tightens), never loosens
            # As price rises, we can raise the floor (but never lower it)
            new_floor = entry_price * (1 + state.profit_floor_lock_pct)
            if new_floor > state.profit_floor_price:
                state.profit_floor_price = new_floor

            # Check if price hit the floor
            if current_price <= state.profit_floor_price:
                self._log_exit_trigger(
                    symbol, self.REASON_PROFIT_FLOOR,
                    current_price, state.profit_floor_price, unrealized_pnl_pct
                )
                return {
                    'action': 'full_exit',
                    'reason': self.REASON_PROFIT_FLOOR,
                    'stop_price': state.profit_floor_price,
                    'unrealized_pnl_pct': unrealized_pnl_pct,
                    'qty': state.quantity
                }

        # ================================================================
        # TIER 2: ATR TRAILING STOP (activates at +1.75%)
        # Uses ATR(14) x 2.0 for volatility-aware trailing
        # ================================================================
        if not state.trailing_active:
            # Check if we should activate trailing stop
            if unrealized_pnl_pct >= state.trailing_activation_pct:
                state.trailing_active = True

                # Calculate initial trailing stop using ATR
                if state.current_atr > 0:
                    trail_distance = state.current_atr * state.atr_multiplier
                else:
                    # Fallback: use 1% if no ATR available
                    trail_distance = current_price * 0.01
                    self.logger.warning(
                        f"EXIT_MGR | {symbol} | NO_ATR | Using 1% fallback for trailing"
                    )

                state.trailing_stop_price = current_price - trail_distance

                # Ensure trailing stop is at least at profit floor level
                if state.profit_floor_active:
                    state.trailing_stop_price = max(
                        state.trailing_stop_price,
                        state.profit_floor_price
                    )

                self.logger.info(
                    f"EXIT_MGR | {symbol} | TRAILING_ACTIVATED | "
                    f"Current: ${current_price:.2f} (+{unrealized_pnl_pct*100:.2f}%), "
                    f"Trail: ${state.trailing_stop_price:.2f}, "
                    f"ATR: ${state.current_atr:.2f}, Distance: ${trail_distance:.2f}"
                )

        if state.trailing_active:
            # Update trailing stop (only tightens, never loosens)
            if state.current_atr > 0:
                trail_distance = state.current_atr * state.atr_multiplier
            else:
                trail_distance = current_price * 0.01

            new_trail = current_price - trail_distance

            # Trailing stop only moves UP
            if new_trail > state.trailing_stop_price:
                old_trail = state.trailing_stop_price
                state.trailing_stop_price = new_trail
                self.logger.debug(
                    f"EXIT_MGR | {symbol} | TRAIL_TIGHTENED | "
                    f"${old_trail:.2f} -> ${new_trail:.2f}"
                )

            # Ensure trailing is never below profit floor
            if state.profit_floor_active:
                state.trailing_stop_price = max(
                    state.trailing_stop_price,
                    state.profit_floor_price
                )

            # Check if price hit trailing stop
            if current_price <= state.trailing_stop_price:
                self._log_exit_trigger(
                    symbol, self.REASON_ATR_TRAILING,
                    current_price, state.trailing_stop_price, unrealized_pnl_pct
                )
                return {
                    'action': 'full_exit',
                    'reason': self.REASON_ATR_TRAILING,
                    'stop_price': state.trailing_stop_price,
                    'unrealized_pnl_pct': unrealized_pnl_pct,
                    'qty': state.quantity
                }

        # ================================================================
        # TIER 3: PARTIAL TAKE PROFIT (at +2.00%)
        # Close 50% of position, move stop to breakeven on remainder
        # ================================================================
        if not state.partial_tp_executed:
            if unrealized_pnl_pct >= state.partial_tp_pct:
                # Calculate shares to close (50% of remaining)
                partial_qty = int(state.quantity * state.partial_tp_size)

                if partial_qty > 0:
                    state.partial_tp_executed = True
                    state.partial_tp_qty = partial_qty

                    # After partial TP, raise profit floor to at least breakeven
                    # This ensures the remaining position doesn't go negative
                    state.profit_floor_price = max(
                        state.profit_floor_price,
                        entry_price  # Breakeven
                    )

                    self.logger.info(
                        f"EXIT_MGR | {symbol} | PARTIAL_TP_TRIGGERED | "
                        f"Current: ${current_price:.2f} (+{unrealized_pnl_pct*100:.2f}%), "
                        f"Closing: {partial_qty} shares, "
                        f"Remaining: {state.quantity - partial_qty} shares, "
                        f"New floor: ${state.profit_floor_price:.2f}"
                    )

                    return {
                        'action': 'partial_tp',
                        'reason': self.REASON_PARTIAL_TP,
                        'stop_price': current_price,  # Exit at current price
                        'unrealized_pnl_pct': unrealized_pnl_pct,
                        'qty': partial_qty
                    }

        # No exit triggered
        return None

    def _log_exit_trigger(self, symbol: str, reason: str,
                         current_price: float, stop_price: float,
                         unrealized_pnl_pct: float):
        """
        Log exit trigger for accountability and backtesting.

        REQUIRED LOGGING (per spec):
        - Symbol
        - Exit reason
        - Unrealized PnL at trigger
        - Stop price
        """
        state = self.positions.get(symbol)
        entry_price = state.entry_price if state else 0

        self.logger.info(
            f"EXIT_MGR | {symbol} | EXIT_TRIGGERED | "
            f"Reason: {reason}, "
            f"Entry: ${entry_price:.2f}, "
            f"Current: ${current_price:.2f}, "
            f"Stop: ${stop_price:.2f}, "
            f"PnL: {unrealized_pnl_pct*100:+.2f}%"
        )

    def get_position_state(self, symbol: str) -> Optional[PositionExitState]:
        """Get the current exit state for a position."""
        return self.positions.get(symbol)

    def get_all_states(self) -> Dict[str, PositionExitState]:
        """Get all position exit states."""
        return self.positions.copy()

    def get_status_summary(self, symbol: str) -> Optional[dict]:
        """
        Get a human-readable status summary for a position.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with status information or None if position not found
        """
        state = self.positions.get(symbol)
        if not state:
            return None

        return {
            'symbol': symbol,
            'entry_price': state.entry_price,
            'quantity': state.quantity,
            'peak_price': state.peak_price,
            'profit_floor_active': state.profit_floor_active,
            'profit_floor_price': state.profit_floor_price,
            'trailing_active': state.trailing_active,
            'trailing_stop_price': state.trailing_stop_price,
            'partial_tp_executed': state.partial_tp_executed,
            'partial_tp_qty': state.partial_tp_qty,
            'current_atr': state.current_atr,
            'hard_stop_price': state.entry_price * (1 - state.hard_stop_pct)
        }


def create_exit_manager(bot_settings: dict = None) -> ExitManager:
    """
    Factory function to create ExitManager with proper configuration.

    Args:
        bot_settings: Bot configuration dict

    Returns:
        Configured ExitManager instance
    """
    return ExitManager(bot_settings)
