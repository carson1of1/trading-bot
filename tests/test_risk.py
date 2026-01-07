"""Tests for RiskManager module - Position sizing, stop-loss, take-profit, and trade validation"""
import pytest
import os
import sys
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.risk import RiskManager


class TestRiskManagerInitialization:
    """Test RiskManager initialization with default and custom settings"""

    def test_default_initialization(self):
        """Should initialize with default values when no settings provided"""
        rm = RiskManager()
        assert rm.max_risk_per_trade == 0.015  # 1.5%
        assert rm.max_daily_loss == 0.04  # 4%
        assert rm.max_position_size == 0.03  # 3% (conservative default)
        assert rm.max_positions == 5  # Conservative default
        assert rm.stop_loss_pct == 0.02  # 2%
        assert rm.take_profit_pct == 0.04  # 4%
        assert rm.risk_reward_ratio == 2.0

    def test_custom_settings_initialization(self):
        """Should use custom settings when provided"""
        settings = {
            'max_risk_per_trade': 0.02,
            'daily_loss_limit': 0.05,
            'max_position_size': 0.30,
            'max_positions': 10,
            'stop_loss_pct': 3.0,  # Percentage, converted to decimal
            'take_profit_pct': 5.0,  # Percentage, converted to decimal
            'risk_reward_ratio': 2.5
        }
        rm = RiskManager(bot_settings=settings)
        assert rm.max_risk_per_trade == 0.02
        assert rm.max_daily_loss == 0.05
        assert rm.max_position_size == 0.30
        assert rm.max_positions == 10
        assert rm.stop_loss_pct == 0.03  # 3% as decimal
        assert rm.take_profit_pct == 0.05  # 5% as decimal

    def test_leveraged_etf_settings(self):
        """Should have leveraged ETF specific settings"""
        rm = RiskManager()
        assert rm.leveraged_etf_stop_loss_pct == 0.01  # 1%
        assert rm.leveraged_etf_take_profit_pct == 0.02  # 2%
        assert rm.leveraged_etf_max_position_size == 0.20  # 20%
        assert rm.max_leveraged_positions == 4

    def test_leveraged_etf_symbols_defined(self):
        """Should have a set of leveraged ETF symbols"""
        rm = RiskManager()
        # Check some common leveraged ETFs are in the set
        assert 'TQQQ' in rm.leveraged_etf_symbols
        assert 'SQQQ' in rm.leveraged_etf_symbols
        assert 'SPXL' in rm.leveraged_etf_symbols
        assert 'SPXS' in rm.leveraged_etf_symbols
        # Regular stocks should not be in the set
        assert 'AAPL' not in rm.leveraged_etf_symbols
        assert 'NVDA' not in rm.leveraged_etf_symbols

    def test_strategy_stop_loss_pct_defined(self):
        """Should have strategy-specific stop loss percentages"""
        rm = RiskManager()
        assert 'Momentum' in rm.strategy_stop_loss_pct
        assert 'Mean Reversion' in rm.strategy_stop_loss_pct
        assert 'Breakout' in rm.strategy_stop_loss_pct
        # All strategies should use 1.5% stop loss
        for strategy, pct in rm.strategy_stop_loss_pct.items():
            assert pct == 0.015


class TestIsLeveragedETF:
    """Test leveraged ETF detection"""

    def test_detects_3x_bull_etfs(self):
        """Should detect 3x bull ETFs"""
        rm = RiskManager()
        assert rm.is_leveraged_etf('TQQQ') is True
        assert rm.is_leveraged_etf('SPXL') is True
        assert rm.is_leveraged_etf('SOXL') is True

    def test_detects_3x_bear_etfs(self):
        """Should detect 3x bear ETFs"""
        rm = RiskManager()
        assert rm.is_leveraged_etf('SQQQ') is True
        assert rm.is_leveraged_etf('SPXS') is True
        assert rm.is_leveraged_etf('SOXS') is True

    def test_detects_2x_etfs(self):
        """Should detect 2x ETFs"""
        rm = RiskManager()
        assert rm.is_leveraged_etf('QLD') is True
        assert rm.is_leveraged_etf('SSO') is True
        assert rm.is_leveraged_etf('QID') is True

    def test_rejects_regular_stocks(self):
        """Should not flag regular stocks as leveraged ETFs"""
        rm = RiskManager()
        assert rm.is_leveraged_etf('AAPL') is False
        assert rm.is_leveraged_etf('NVDA') is False
        assert rm.is_leveraged_etf('SPY') is False
        assert rm.is_leveraged_etf('QQQ') is False

    def test_case_insensitive(self):
        """Should be case insensitive"""
        rm = RiskManager()
        assert rm.is_leveraged_etf('tqqq') is True
        assert rm.is_leveraged_etf('Tqqq') is True
        assert rm.is_leveraged_etf('TQQQ') is True


class TestRoundPriceForBroker:
    """Test broker-safe price rounding"""

    def test_rounds_to_cents_for_stocks_above_dollar(self):
        """Should round to 2 decimal places for stocks >= $1"""
        rm = RiskManager()
        assert rm.round_price_for_broker(150.123) == 150.12
        assert rm.round_price_for_broker(150.126) == 150.13
        assert rm.round_price_for_broker(1.005) == 1.0  # Note: floating point

    def test_rounds_to_4_decimals_for_penny_stocks(self):
        """Should round to 4 decimal places for stocks < $1"""
        rm = RiskManager()
        assert rm.round_price_for_broker(0.12345) == 0.1235
        assert rm.round_price_for_broker(0.99999) == 1.0


class TestCheckBuyingPower:
    """Test buying power validation"""

    def test_approves_when_sufficient_buying_power(self):
        """Should approve when position fits within 90% of buying power"""
        rm = RiskManager()
        result = rm.check_buying_power(
            available_buying_power=10000,
            position_value=5000,
            entry_price=100,
            position_size=50
        )
        assert result['approved'] is True
        assert result['available_bp'] == 10000
        assert result['required_bp'] == 5000

    def test_rejects_when_insufficient_buying_power(self):
        """Should reject when position exceeds 90% of buying power"""
        rm = RiskManager()
        result = rm.check_buying_power(
            available_buying_power=10000,
            position_value=9500,  # More than 90% of $10K
            entry_price=100,
            position_size=95
        )
        assert result['approved'] is False
        assert 'reason' in result
        assert 'Insufficient buying power' in result['reason']

    def test_boundary_at_90_percent(self):
        """Should approve at exactly 90% of buying power"""
        rm = RiskManager()
        result = rm.check_buying_power(
            available_buying_power=10000,
            position_value=9000,  # Exactly 90%
            entry_price=100,
            position_size=90
        )
        assert result['approved'] is True


class TestAdjustPositionForBuyingPower:
    """Test position size adjustment for buying power"""

    def test_returns_original_size_when_affordable(self):
        """Should return original size when it fits within buying power"""
        rm = RiskManager()
        adjusted = rm.adjust_position_for_buying_power(
            requested_size=50,
            entry_price=100,
            available_buying_power=10000
        )
        assert adjusted == 50

    def test_reduces_size_when_exceeds_buying_power(self):
        """Should reduce size to fit within buying power"""
        rm = RiskManager()
        adjusted = rm.adjust_position_for_buying_power(
            requested_size=200,  # Would be $20K at $100/share
            entry_price=100,
            available_buying_power=10000  # Only have $10K
        )
        assert adjusted == 90  # 90% of $10K / $100 = 90 shares

    def test_returns_zero_for_zero_buying_power(self):
        """Should return 0 when no buying power available"""
        rm = RiskManager()
        adjusted = rm.adjust_position_for_buying_power(
            requested_size=50,
            entry_price=100,
            available_buying_power=0
        )
        assert adjusted == 0

    def test_returns_zero_for_zero_price(self):
        """Should return 0 when entry price is zero"""
        rm = RiskManager()
        adjusted = rm.adjust_position_for_buying_power(
            requested_size=50,
            entry_price=0,
            available_buying_power=10000
        )
        assert adjusted == 0


class TestEnsureUTCAware:
    """Test UTC timezone handling"""

    def test_converts_naive_datetime_to_utc(self):
        """Should convert naive datetime to UTC"""
        rm = RiskManager()
        naive_dt = datetime(2025, 12, 20, 10, 0, 0)
        utc_dt = rm.ensure_utc_aware(naive_dt)
        assert utc_dt.tzinfo == pytz.UTC

    def test_converts_other_timezone_to_utc(self):
        """Should convert other timezone to UTC"""
        rm = RiskManager()
        et = pytz.timezone('America/New_York')
        et_dt = et.localize(datetime(2025, 12, 20, 10, 0, 0))
        utc_dt = rm.ensure_utc_aware(et_dt)
        assert utc_dt.tzinfo == pytz.UTC
        # 10:00 AM ET should be 15:00 UTC
        assert utc_dt.hour == 15

    def test_keeps_utc_unchanged(self):
        """Should not modify already UTC datetime"""
        rm = RiskManager()
        utc_dt = datetime(2025, 12, 20, 10, 0, 0, tzinfo=pytz.UTC)
        result = rm.ensure_utc_aware(utc_dt)
        assert result == utc_dt

    def test_handles_none(self):
        """Should return current UTC time for None"""
        rm = RiskManager()
        result = rm.ensure_utc_aware(None)
        assert result.tzinfo == pytz.UTC


class TestCalculateHoldTimeMinutes:
    """Test hold time calculation"""

    def test_calculates_hold_time_correctly(self):
        """Should calculate hold time in minutes"""
        rm = RiskManager()
        entry = datetime(2025, 12, 20, 10, 0, 0, tzinfo=pytz.UTC)
        current = datetime(2025, 12, 20, 10, 30, 0, tzinfo=pytz.UTC)
        hold_time = rm.calculate_hold_time_minutes(entry, current)
        assert hold_time == 30.0

    def test_handles_mixed_timezones(self):
        """Should handle entry and current in different timezones"""
        rm = RiskManager()
        et = pytz.timezone('America/New_York')
        entry = et.localize(datetime(2025, 12, 20, 10, 0, 0))  # 10 AM ET
        current = datetime(2025, 12, 20, 15, 30, 0, tzinfo=pytz.UTC)  # 3:30 PM UTC (10:30 AM ET)
        hold_time = rm.calculate_hold_time_minutes(entry, current)
        assert hold_time == 30.0

    def test_handles_naive_datetimes(self):
        """Should handle naive datetimes by assuming UTC"""
        rm = RiskManager()
        entry = datetime(2025, 12, 20, 10, 0, 0)
        current = datetime(2025, 12, 20, 11, 0, 0)
        hold_time = rm.calculate_hold_time_minutes(entry, current)
        assert hold_time == 60.0


class TestCalculatePositionSize:
    """Test position size calculation"""

    def test_calculates_position_size_by_risk(self):
        """Should calculate position size based on risk per trade"""
        rm = RiskManager()
        # Portfolio: $100K, risk 1.5% = $1500 max risk
        # Entry $100, stop $98, risk per share = $2
        # Position size = $1500 / $2 = 750 shares
        # But capped by 3% position size = $3K / $100 = 30 shares
        # With 95% buffer (Jan 7, 2026 fix): 30 * 0.95 = 28 shares
        size = rm.calculate_position_size(
            portfolio_value=100000,
            entry_price=100,
            stop_loss_price=98
        )
        assert size == 28  # Capped at 3% of portfolio with 95% buffer

    def test_respects_max_position_dollars(self):
        """Should respect max_position_dollars setting"""
        # Set max_position_size high enough that max_position_dollars becomes the limit
        rm = RiskManager(bot_settings={'max_position_dollars': 5000, 'max_position_size': 0.25})
        size = rm.calculate_position_size(
            portfolio_value=100000,
            entry_price=100,
            stop_loss_price=98
        )
        assert size == 50  # $5000 / $100 = 50 shares

    def test_returns_zero_for_nan_inputs(self):
        """Should return 0 for NaN inputs"""
        rm = RiskManager()
        assert rm.calculate_position_size(100000, float('nan'), 98) == 0
        assert rm.calculate_position_size(100000, 100, float('nan')) == 0
        assert rm.calculate_position_size(float('nan'), 100, 98) == 0

    def test_returns_zero_for_zero_prices(self):
        """Should return 0 for zero or negative prices"""
        rm = RiskManager()
        assert rm.calculate_position_size(100000, 0, 98) == 0
        assert rm.calculate_position_size(100000, 100, 0) == 0
        assert rm.calculate_position_size(100000, -10, 98) == 0

    def test_caps_portfolio_value(self):
        """Should cap portfolio value at $500K for safety"""
        rm = RiskManager()
        size_normal = rm.calculate_position_size(500000, 100, 98)
        size_inflated = rm.calculate_position_size(1000000, 100, 98)
        # Both should return same because inflated is capped to 500K
        assert size_normal == size_inflated

    def test_adjusts_for_volatility(self):
        """Should reduce position size for high volatility"""
        rm = RiskManager()
        size_no_vol = rm.calculate_position_size(100000, 100, 98)
        size_high_vol = rm.calculate_position_size(100000, 100, 98, volatility=0.6)
        assert size_high_vol < size_no_vol

    def test_adjusts_for_vix(self):
        """Should reduce position size for high VIX"""
        rm = RiskManager()
        size_low_vix = rm.calculate_position_size(100000, 100, 98, vix_level=10)
        size_high_vix = rm.calculate_position_size(100000, 100, 98, vix_level=30)
        assert size_high_vix < size_low_vix


class TestCheckDailyLossLimit:
    """Test daily loss limit checking"""

    def test_returns_false_for_no_loss(self):
        """Should return False when no loss"""
        rm = RiskManager()
        assert rm.check_daily_loss_limit(current_pnl=100, portfolio_value=100000) is False

    def test_returns_false_for_loss_under_limit(self):
        """Should return False when loss is under 4% limit"""
        rm = RiskManager()
        assert rm.check_daily_loss_limit(current_pnl=-3000, portfolio_value=100000) is False  # 3%

    def test_returns_true_for_loss_at_limit(self):
        """Should return True when loss reaches 4% limit"""
        rm = RiskManager()
        assert rm.check_daily_loss_limit(current_pnl=-4000, portfolio_value=100000) is True

    def test_returns_true_for_invalid_portfolio_value(self):
        """Should return True (block trading) for invalid portfolio value"""
        rm = RiskManager()
        assert rm.check_daily_loss_limit(current_pnl=-100, portfolio_value=0) is True
        assert rm.check_daily_loss_limit(current_pnl=-100, portfolio_value=None) is True


class TestCheckPortfolioRisk:
    """Test portfolio risk checking"""

    def test_approves_acceptable_risk(self):
        """Should approve when total risk is under 8% limit"""
        rm = RiskManager()
        positions = {
            'AAPL': {'risk_amount': 2000},
            'NVDA': {'risk_amount': 3000}
        }
        assert rm.check_portfolio_risk(positions, portfolio_value=100000) is True

    def test_rejects_excessive_risk(self):
        """Should reject when total risk exceeds 8% limit"""
        rm = RiskManager()
        positions = {
            'AAPL': {'risk_amount': 5000},
            'NVDA': {'risk_amount': 5000}
        }
        assert rm.check_portfolio_risk(positions, portfolio_value=100000) is False

    def test_rejects_when_no_portfolio_value(self):
        """Should reject when portfolio value is not provided"""
        rm = RiskManager()
        positions = {'AAPL': {'risk_amount': 1000}}
        assert rm.check_portfolio_risk(positions, portfolio_value=None) is False
        assert rm.check_portfolio_risk(positions, portfolio_value=0) is False


class TestCheckPositionLimits:
    """Test position limit checking"""

    def test_allows_under_limit(self):
        """Should allow when under position limit"""
        rm = RiskManager(bot_settings={'max_positions': 10})
        assert rm.check_position_limits({'AAPL': {}, 'NVDA': {}}) is True

    def test_blocks_at_limit(self):
        """Should block when at position limit"""
        rm = RiskManager(bot_settings={'max_positions': 2})
        assert rm.check_position_limits({'AAPL': {}, 'NVDA': {}}) is False


class TestCheckTradeFrequency:
    """Test trade frequency checking"""

    def test_allows_under_limit(self):
        """Should allow when under daily trade limit"""
        rm = RiskManager()
        assert rm.check_trade_frequency(trades_today=10) is True

    def test_blocks_at_limit(self):
        """Should block when at daily trade limit (20)"""
        rm = RiskManager()
        assert rm.check_trade_frequency(trades_today=20) is False


class TestCalculateStopLoss:
    """Test stop loss calculation"""

    def test_calculates_stop_loss_for_buy(self):
        """Should calculate stop loss below entry for buy orders"""
        rm = RiskManager()
        stop = rm.calculate_stop_loss(entry_price=100, side='buy')
        assert stop < 100
        assert stop == 98.0  # 2% below

    def test_calculates_stop_loss_for_sell(self):
        """Should calculate stop loss above entry for sell orders"""
        rm = RiskManager()
        stop = rm.calculate_stop_loss(entry_price=100, side='sell')
        assert stop > 100
        assert stop == 102.0  # 2% above

    def test_uses_strategy_specific_stop(self):
        """Should use strategy-specific stop loss percentage"""
        rm = RiskManager()
        stop = rm.calculate_stop_loss(entry_price=100, side='buy', strategy='Momentum')
        # Momentum uses 1.5% stop loss
        assert stop == 98.5

    def test_uses_leveraged_etf_stop(self):
        """Should use tighter stop for leveraged ETFs"""
        rm = RiskManager()
        stop = rm.calculate_stop_loss(entry_price=100, side='buy', symbol='TQQQ')
        # Leveraged ETFs use 1% stop loss
        assert stop == 99.0

    def test_uses_atr_based_stop_when_enabled(self):
        """Should use ATR-based stop when enabled"""
        rm = RiskManager(bot_settings={'use_atr_stops': True, 'atr_multiplier': 2.0})
        stop = rm.calculate_stop_loss(entry_price=100, side='buy', atr=1.5)
        # ATR-based: 1.5 * 2.0 = 3.0 = 3% stop
        # Clamped between 0.5% and 5%
        assert stop == 97.0  # 100 * (1 - 0.03) = 97

    def test_rounds_for_broker(self):
        """Should round price for broker compatibility"""
        rm = RiskManager()
        stop = rm.calculate_stop_loss(entry_price=100.123, side='buy')
        assert stop == round(stop, 2)


class TestCalculateTakeProfit:
    """Test take profit calculation"""

    def test_calculates_take_profit_for_buy(self):
        """Should calculate take profit above entry for buy orders"""
        rm = RiskManager()
        tp = rm.calculate_take_profit(entry_price=100, stop_loss_price=98, side='buy')
        assert tp > 100
        assert tp == 104.0  # 4% above

    def test_calculates_take_profit_for_sell(self):
        """Should calculate take profit below entry for sell orders"""
        rm = RiskManager()
        tp = rm.calculate_take_profit(entry_price=100, stop_loss_price=102, side='sell')
        assert tp < 100
        assert tp == 96.0  # 4% below

    def test_uses_leveraged_etf_take_profit(self):
        """Should use tighter take profit for leveraged ETFs"""
        rm = RiskManager()
        tp = rm.calculate_take_profit(entry_price=100, stop_loss_price=99, side='buy', symbol='TQQQ')
        # Leveraged ETFs use 2% take profit
        assert tp == 102.0

    def test_rounds_for_broker(self):
        """Should round price for broker compatibility"""
        rm = RiskManager()
        tp = rm.calculate_take_profit(entry_price=100.123, stop_loss_price=98.123, side='buy')
        assert tp == round(tp, 2)


class TestValidateTrade:
    """Test trade validation"""

    def test_approves_valid_trade(self):
        """Should approve a valid trade"""
        rm = RiskManager()
        result = rm.validate_trade(
            symbol='AAPL',
            side='buy',
            quantity=10,
            entry_price=150,
            portfolio_value=100000,
            current_positions={}
        )
        assert result['approved'] is True
        assert len(result['reasons']) == 0

    def test_rejects_small_position(self):
        """Should reject positions under $500"""
        rm = RiskManager()
        result = rm.validate_trade(
            symbol='AAPL',
            side='buy',
            quantity=3,
            entry_price=100,  # $300 total
            portfolio_value=100000,
            current_positions={}
        )
        assert result['approved'] is False
        assert any('too small' in r for r in result['reasons'])

    def test_rejects_oversized_position(self):
        """Should reject positions exceeding max size"""
        rm = RiskManager(bot_settings={'max_position_dollars': 10000})
        result = rm.validate_trade(
            symbol='AAPL',
            side='buy',
            quantity=200,
            entry_price=100,  # $20K total
            portfolio_value=100000,
            current_positions={}
        )
        assert result['approved'] is False
        assert any('too large' in r for r in result['reasons'])

    def test_rejects_when_max_positions_reached(self):
        """Should reject when max positions reached"""
        rm = RiskManager(bot_settings={'max_positions': 2})
        result = rm.validate_trade(
            symbol='TSLA',
            side='buy',
            quantity=10,
            entry_price=200,
            portfolio_value=100000,
            current_positions={'AAPL': {}, 'NVDA': {}}
        )
        assert result['approved'] is False
        assert any('Max positions' in r for r in result['reasons'])

    def test_warns_on_duplicate_position(self):
        """Should warn when adding to existing position"""
        rm = RiskManager()
        result = rm.validate_trade(
            symbol='AAPL',
            side='buy',
            quantity=10,
            entry_price=150,
            portfolio_value=100000,
            current_positions={'AAPL': {}}
        )
        assert result['approved'] is True
        assert any('Already holding' in w for w in result['warnings'])

    def test_rejects_oversized_leveraged_etf(self):
        """Should reject oversized leveraged ETF positions"""
        rm = RiskManager()
        result = rm.validate_trade(
            symbol='TQQQ',
            side='buy',
            quantity=500,
            entry_price=50,  # $25K = 25% of portfolio
            portfolio_value=100000,
            current_positions={}
        )
        assert result['approved'] is False
        assert any('Leveraged ETF' in r for r in result['reasons'])

    def test_rejects_when_max_leveraged_positions_reached(self):
        """Should reject when max leveraged positions reached"""
        rm = RiskManager()
        result = rm.validate_trade(
            symbol='SQQQ',
            side='buy',
            quantity=10,
            entry_price=20,
            portfolio_value=100000,
            current_positions={'TQQQ': {}, 'SPXL': {}, 'SOXL': {}, 'LABU': {}}  # 4 leveraged
        )
        assert result['approved'] is False
        assert any('Max leveraged positions' in r for r in result['reasons'])


class TestUpdateRiskMetrics:
    """Test risk metrics tracking"""

    def test_increments_daily_trades(self):
        """Should increment daily trade count"""
        rm = RiskManager()
        assert rm.daily_trades == 0
        rm.update_risk_metrics({'symbol': 'AAPL', 'side': 'buy'})
        assert rm.daily_trades == 1

    def test_updates_daily_pnl(self):
        """Should update daily P&L"""
        rm = RiskManager()
        rm.update_risk_metrics({'symbol': 'AAPL', 'side': 'sell', 'pnl': 100})
        assert rm.daily_pnl == 100
        rm.update_risk_metrics({'symbol': 'NVDA', 'side': 'sell', 'pnl': -50})
        assert rm.daily_pnl == 50

    def test_tracks_position_risk(self):
        """Should track position risk on buy"""
        rm = RiskManager()
        rm.update_risk_metrics({'symbol': 'AAPL', 'side': 'buy', 'risk_amount': 500})
        assert 'AAPL' in rm.positions_risk
        assert rm.positions_risk['AAPL'] == 500

    def test_removes_position_risk_on_sell(self):
        """Should remove position risk on sell"""
        rm = RiskManager()
        rm.positions_risk['AAPL'] = 500
        rm.update_risk_metrics({'symbol': 'AAPL', 'side': 'sell'})
        assert 'AAPL' not in rm.positions_risk


class TestGetRiskSummary:
    """Test risk summary generation"""

    def test_returns_correct_summary(self):
        """Should return accurate risk summary"""
        rm = RiskManager()
        rm.daily_trades = 5
        rm.daily_pnl = 200.50
        rm.positions_risk = {'AAPL': 500, 'NVDA': 300}

        summary = rm.get_risk_summary()
        assert summary['daily_trades'] == 5
        assert summary['daily_pnl'] == 200.50
        assert summary['current_positions'] == 2
        assert summary['total_risk'] == 800
        assert summary['max_trades_remaining'] == 15  # 20 - 5


class TestAdjustPositionForVolatility:
    """Test VIX-based position sizing"""

    def test_no_adjustment_for_low_vix(self):
        """Should not adjust for VIX < 15"""
        rm = RiskManager()
        adjusted = rm.adjust_position_for_volatility(100, vix_level=10)
        assert adjusted == 100

    def test_reduces_to_80_for_moderate_vix(self):
        """Should reduce to 80% for VIX 15-25"""
        rm = RiskManager()
        adjusted = rm.adjust_position_for_volatility(100, vix_level=20)
        assert adjusted == 80

    def test_reduces_to_50_for_high_vix(self):
        """Should reduce to 50% for VIX > 25"""
        rm = RiskManager()
        adjusted = rm.adjust_position_for_volatility(100, vix_level=30)
        assert adjusted == 50


class TestResetDailyMetrics:
    """Test daily metrics reset"""

    def test_resets_all_metrics(self):
        """Should reset all daily tracking metrics"""
        rm = RiskManager()
        rm.daily_trades = 10
        rm.daily_pnl = 500.0
        rm.positions_risk = {'AAPL': 500}

        rm.reset_daily_metrics()

        assert rm.daily_trades == 0
        assert rm.daily_pnl == 0.0
        assert len(rm.positions_risk) == 0


class TestUpdateSettings:
    """Test dynamic settings update"""

    def test_updates_settings(self):
        """Should update settings dynamically"""
        rm = RiskManager()
        initial_max_positions = rm.max_positions

        rm.update_settings({
            'max_positions': 5,
            'max_position_size': 0.20,
            'stop_loss_pct': 3.0,
            'take_profit_pct': 6.0
        })

        assert rm.max_positions == 5
        assert rm.max_position_size == 0.20
        assert rm.stop_loss_pct == 0.03
        assert rm.take_profit_pct == 0.06


class TestAssessMarketConditions:
    """Test market condition assessment"""

    def test_returns_default_for_insufficient_data(self):
        """Should return default conditions for insufficient data"""
        rm = RiskManager()
        result = rm.assess_market_conditions(None)
        assert result['risk_level'] == 'normal'
        assert result['volatility'] == 0.2

    def test_assesses_high_volatility(self):
        """Should detect high volatility market"""
        rm = RiskManager()
        # Create data with high volatility
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        # Large price swings = high volatility
        prices = [100 + i * 10 * (-1 if i % 2 else 1) for i in range(100)]
        data = pd.DataFrame({
            'close': prices,
            'volume': [1000000] * 100
        }, index=dates)
        result = rm.assess_market_conditions(data)
        assert result['risk_level'] == 'high'

    def test_assesses_low_volatility(self):
        """Should detect low volatility market"""
        rm = RiskManager()
        # Create data with low volatility (steady prices)
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        prices = [100 + i * 0.01 for i in range(100)]  # Very small changes
        data = pd.DataFrame({
            'close': prices,
            'volume': [1000000] * 100
        }, index=dates)
        result = rm.assess_market_conditions(data)
        assert result['risk_level'] == 'low'


class TestLoadSymbolTakeProfits:
    """Test that ML loading is stubbed out"""

    def test_ml_load_does_nothing(self):
        """Should not load any ML data (feature disabled)"""
        rm = RiskManager()
        # Per-symbol TP should always be empty in minimal extraction
        assert rm.symbol_take_profit_pct == {}
        assert rm.use_per_symbol_tp is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
