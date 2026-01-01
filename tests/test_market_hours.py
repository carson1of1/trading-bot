"""Tests for MarketHours module"""
import pytest
from datetime import datetime, date, time
from unittest.mock import patch, MagicMock
import pytz

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.market_hours import MarketHours


class TestIsTradingDay:
    """Test is_trading_day method"""

    def test_weekday_is_trading_day(self):
        """Monday-Friday should be trading days (unless holiday)"""
        mh = MarketHours()
        # A regular Wednesday
        assert mh.is_trading_day(date(2025, 12, 3)) is True

    def test_saturday_not_trading_day(self):
        """Saturday is not a trading day"""
        mh = MarketHours()
        assert mh.is_trading_day(date(2025, 12, 6)) is False

    def test_sunday_not_trading_day(self):
        """Sunday is not a trading day"""
        mh = MarketHours()
        assert mh.is_trading_day(date(2025, 12, 7)) is False

    def test_holiday_not_trading_day(self):
        """Market holidays are not trading days"""
        mh = MarketHours()
        # New Year's Day 2025
        assert mh.is_trading_day(date(2025, 1, 1)) is False
        # Thanksgiving 2025
        assert mh.is_trading_day(date(2025, 11, 27)) is False


class TestIsMarketOpen:
    """Test is_market_open method with mocked time"""

    def test_market_open_during_trading_hours(self):
        """Market should be open during 9:30 AM - 4:00 PM ET on trading days"""
        mh = MarketHours()
        # Mock time to 11:00 AM ET on a Wednesday
        mock_time = datetime(2025, 12, 3, 11, 0, 0, tzinfo=pytz.timezone('America/New_York'))
        with patch.object(mh, 'get_market_time', return_value=mock_time):
            assert mh.is_market_open() is True

    def test_market_closed_before_open(self):
        """Market should be closed before 9:30 AM ET"""
        mh = MarketHours()
        # Mock time to 9:00 AM ET on a Wednesday
        mock_time = datetime(2025, 12, 3, 9, 0, 0, tzinfo=pytz.timezone('America/New_York'))
        with patch.object(mh, 'get_market_time', return_value=mock_time):
            assert mh.is_market_open() is False

    def test_market_closed_after_close(self):
        """Market should be closed after 4:00 PM ET"""
        mh = MarketHours()
        # Mock time to 5:00 PM ET on a Wednesday
        mock_time = datetime(2025, 12, 3, 17, 0, 0, tzinfo=pytz.timezone('America/New_York'))
        with patch.object(mh, 'get_market_time', return_value=mock_time):
            assert mh.is_market_open() is False

    def test_market_closed_on_weekend(self):
        """Market should be closed on weekends even during trading hours"""
        mh = MarketHours()
        # Mock time to 11:00 AM ET on a Saturday
        mock_time = datetime(2025, 12, 6, 11, 0, 0, tzinfo=pytz.timezone('America/New_York'))
        with patch.object(mh, 'get_market_time', return_value=mock_time):
            assert mh.is_market_open() is False


class TestMinutesUntilClose:
    """Test time_until_market_close method"""

    def test_minutes_until_close_mid_day(self):
        """Should return correct minutes until 4 PM"""
        mh = MarketHours()
        # Mock time to 2:00 PM ET (120 minutes before close)
        # Use localize() instead of tzinfo= to avoid pytz LMT offset issues
        eastern = pytz.timezone('America/New_York')
        mock_time = eastern.localize(datetime(2025, 12, 3, 14, 0, 0))
        with patch.object(mh, 'get_market_time', return_value=mock_time):
            with patch.object(mh, 'is_market_open', return_value=True):
                assert mh.time_until_market_close() == 120

    def test_minutes_until_close_when_closed(self):
        """Should return 0 when market is closed"""
        mh = MarketHours()
        # Mock time to 5:00 PM ET (market closed)
        mock_time = datetime(2025, 12, 3, 17, 0, 0, tzinfo=pytz.timezone('America/New_York'))
        with patch.object(mh, 'get_market_time', return_value=mock_time):
            # is_market_open will return False naturally
            assert mh.time_until_market_close() == 0


class TestEarlyCloseDay:
    """Test early close day detection"""

    def test_christmas_eve_is_early_close(self):
        """Christmas Eve should be an early close day"""
        mh = MarketHours()
        assert mh.is_early_close_day(date(2025, 12, 24)) is True

    def test_regular_day_not_early_close(self):
        """Regular trading days should not be early close"""
        mh = MarketHours()
        assert mh.is_early_close_day(date(2025, 12, 3)) is False


class TestShouldStartTrading:
    """Test should_start_trading logic"""

    def test_should_not_trade_when_market_closed(self):
        """Should not start trading when market is closed"""
        mh = MarketHours()
        with patch.object(mh, 'is_market_open', return_value=False):
            can_trade, reason = mh.should_start_trading()
            assert can_trade is False
            assert "closed" in reason.lower()

    def test_should_not_trade_near_close(self):
        """Should not start trading within 30 minutes of close"""
        mh = MarketHours()
        with patch.object(mh, 'is_market_open', return_value=True):
            with patch.object(mh, 'minutes_until_close', return_value=15):
                can_trade, reason = mh.should_start_trading()
                assert can_trade is False
                assert "close" in reason.lower()


class TestShouldStopTrading:
    """Test should_stop_trading logic"""

    def test_should_stop_when_market_closed(self):
        """Should stop trading when market is closed"""
        mh = MarketHours()
        with patch.object(mh, 'is_market_open', return_value=False):
            should_stop, reason = mh.should_stop_trading()
            assert should_stop is True

    def test_should_stop_near_close(self):
        """Should stop trading within 15 minutes of close"""
        mh = MarketHours()
        with patch.object(mh, 'is_market_open', return_value=True):
            with patch.object(mh, 'minutes_until_close', return_value=10):
                should_stop, reason = mh.should_stop_trading()
                assert should_stop is True


class TestGetMarketTime:
    """Test get_market_time returns timezone-aware datetime"""

    def test_returns_timezone_aware(self):
        """get_market_time should return timezone-aware datetime"""
        mh = MarketHours()
        market_time = mh.get_market_time()
        assert market_time.tzinfo is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
