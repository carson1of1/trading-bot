import pytz
from datetime import datetime, time, timedelta
import logging


class MarketHours:
    """Handle NYSE market hours validation and calculations"""

    def __init__(self):
        # Market timezone (Eastern Time)
        self.market_tz = pytz.timezone('America/New_York')

        # Regular trading hours (9:30 AM - 4:00 PM ET)
        self.market_open_time = time(9, 30)  # 9:30 AM
        self.market_close_time = time(16, 0)  # 4:00 PM

        # Pre-market and after-hours (for reference)
        self.premarket_start = time(4, 0)   # 4:00 AM
        self.afterhours_end = time(20, 0)   # 8:00 PM

        # Market holidays for 2025 (NYSE observed holidays)
        self.market_holidays_2025 = [
            '2025-01-01',  # New Year's Day
            '2025-01-20',  # Martin Luther King Jr. Day
            '2025-02-17',  # Presidents' Day
            '2025-04-18',  # Good Friday
            '2025-05-26',  # Memorial Day
            '2025-06-19',  # Juneteenth
            '2025-07-04',  # Independence Day
            '2025-09-01',  # Labor Day
            '2025-11-27',  # Thanksgiving Day
            '2025-12-25',  # Christmas Day
        ]

        # Convert to datetime objects for easier comparison
        self.holidays = [datetime.strptime(date, '%Y-%m-%d').date()
                        for date in self.market_holidays_2025]

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def get_market_time(self):
        """Get current time in market timezone

        BUG FIX (Dec 5, 2025): Changed from datetime.utcnow() to datetime.now(pytz.UTC)
        datetime.utcnow() is deprecated in Python 3.12+ and returns naive datetime.
        datetime.now(pytz.UTC) returns a timezone-aware datetime directly.
        """
        utc_now = datetime.now(pytz.UTC)
        return utc_now.astimezone(self.market_tz)

    def is_trading_day(self, date=None):
        """Check if a given date is a trading day (weekday, not holiday)"""
        if date is None:
            date = self.get_market_time().date()

        # Check if it's a weekday (Monday=0, Sunday=6)
        if date.weekday() >= 5:  # Saturday or Sunday
            return False

        # Check if it's a market holiday
        if date in self.holidays:
            return False

        return True

    def is_market_open(self):
        """Check if the market is currently open for regular trading"""
        current_time = self.get_market_time()
        current_date = current_time.date()
        current_time_only = current_time.time()

        # Check if it's a trading day
        if not self.is_trading_day(current_date):
            return False

        # Check if current time is within market hours
        if self.market_open_time <= current_time_only <= self.market_close_time:
            return True

        return False

    def is_premarket_hours(self):
        """Check if current time is in pre-market hours"""
        current_time = self.get_market_time()
        current_date = current_time.date()
        current_time_only = current_time.time()

        if not self.is_trading_day(current_date):
            return False

        return self.premarket_start <= current_time_only < self.market_open_time

    def is_afterhours(self):
        """Check if current time is in after-hours trading"""
        current_time = self.get_market_time()
        current_date = current_time.date()
        current_time_only = current_time.time()

        if not self.is_trading_day(current_date):
            return False

        return self.market_close_time < current_time_only <= self.afterhours_end

    def time_until_market_open(self):
        """Get time until market opens (in minutes)"""
        current_time = self.get_market_time()
        current_date = current_time.date()

        if self.is_market_open():
            return 0  # Market is already open

        # Find next trading day
        next_trading_day = current_date
        while not self.is_trading_day(next_trading_day):
            next_trading_day += timedelta(days=1)

        # Calculate time until market opens
        market_open_datetime = datetime.combine(next_trading_day, self.market_open_time)
        market_open_datetime = self.market_tz.localize(market_open_datetime)

        time_diff = market_open_datetime - current_time
        return int(time_diff.total_seconds() / 60)  # Return minutes

    def time_until_market_close(self):
        """Get time until market closes (in minutes)"""
        if not self.is_market_open():
            return 0  # Market is not open

        current_time = self.get_market_time()
        current_date = current_time.date()

        market_close_datetime = datetime.combine(current_date, self.market_close_time)
        market_close_datetime = self.market_tz.localize(market_close_datetime)

        time_diff = market_close_datetime - current_time
        return int(time_diff.total_seconds() / 60)  # Return minutes

    def minutes_until_close(self):
        """Alias for time_until_market_close for compatibility"""
        return self.time_until_market_close()

    def get_next_market_open(self):
        """Get the next market open datetime"""
        current_time = self.get_market_time()
        current_date = current_time.date()

        if self.is_market_open():
            # Market is currently open, next open is tomorrow (or next trading day)
            next_date = current_date + timedelta(days=1)
        else:
            # Market is closed, could be today or next trading day
            if (current_time.time() < self.market_open_time and
                self.is_trading_day(current_date)):
                next_date = current_date
            else:
                next_date = current_date + timedelta(days=1)

        # Find next trading day
        while not self.is_trading_day(next_date):
            next_date += timedelta(days=1)

        market_open_datetime = datetime.combine(next_date, self.market_open_time)
        return self.market_tz.localize(market_open_datetime)

    def get_next_market_close(self):
        """Get the next market close datetime"""
        current_time = self.get_market_time()
        current_date = current_time.date()

        if self.is_market_open():
            # Market is open, close is today
            market_close_datetime = datetime.combine(current_date, self.market_close_time)
            return self.market_tz.localize(market_close_datetime)
        else:
            # Market is closed, find next trading day
            next_date = current_date
            while not self.is_trading_day(next_date):
                next_date += timedelta(days=1)

            market_close_datetime = datetime.combine(next_date, self.market_close_time)
            return self.market_tz.localize(market_close_datetime)

    def get_trading_session_info(self):
        """Get comprehensive trading session information"""
        current_time = self.get_market_time()

        info = {
            'current_time': current_time,
            'is_trading_day': self.is_trading_day(),
            'market_open': self.is_market_open(),
            'premarket': self.is_premarket_hours(),
            'afterhours': self.is_afterhours(),
            'next_open': self.get_next_market_open(),
            'next_close': self.get_next_market_close() if not self.is_market_open() else None,
            'minutes_until_open': self.time_until_market_open(),
            'minutes_until_close': self.time_until_market_close() if self.is_market_open() else None
        }

        # Determine current session status
        if info['market_open']:
            info['session_status'] = 'OPEN'
        elif info['premarket']:
            info['session_status'] = 'PREMARKET'
        elif info['afterhours']:
            info['session_status'] = 'AFTERHOURS'
        else:
            info['session_status'] = 'CLOSED'

        return info

    def is_early_close_day(self, date=None):
        """Check if it's an early close day (e.g., day before holiday)"""
        if date is None:
            date = self.get_market_time().date()

        # Days with early close (1:00 PM ET) - simplified list
        early_close_days = [
            '2025-07-03',  # Day before Independence Day
            '2025-11-26',  # Day before Thanksgiving
            '2025-12-24',  # Christmas Eve
        ]

        early_close_dates = [datetime.strptime(d, '%Y-%m-%d').date()
                           for d in early_close_days]

        return date in early_close_dates

    def get_market_schedule(self, days_ahead=5):
        """Get market schedule for the next N days"""
        schedule = []
        current_date = self.get_market_time().date()

        for i in range(days_ahead):
            date = current_date + timedelta(days=i)

            schedule_info = {
                'date': date,
                'is_trading_day': self.is_trading_day(date),
                'is_holiday': date in self.holidays,
                'is_weekend': date.weekday() >= 5,
                'is_early_close': self.is_early_close_day(date)
            }

            if schedule_info['is_trading_day']:
                schedule_info['market_open'] = datetime.combine(date, self.market_open_time)
                if schedule_info['is_early_close']:
                    schedule_info['market_close'] = datetime.combine(date, time(13, 0))  # 1:00 PM
                else:
                    schedule_info['market_close'] = datetime.combine(date, self.market_close_time)

            schedule.append(schedule_info)

        return schedule

    def log_market_status(self):
        """Log current market status"""
        info = self.get_trading_session_info()

        status_msg = f"Market Status: {info['session_status']}"

        if info['market_open']:
            status_msg += f" - {info['minutes_until_close']} minutes until close"
        elif info['minutes_until_open'] > 0:
            status_msg += f" - {info['minutes_until_open']} minutes until open"

        self.logger.info(status_msg)
        return status_msg

    def should_start_trading(self):
        """Determine if trading should start based on market conditions"""
        if not self.is_market_open():
            return False, "Market is closed"

        # Don't start trading too close to market close
        if self.minutes_until_close() < 30:
            return False, "Too close to market close"

        # Don't start trading right at market open (let things settle)
        current_time = self.get_market_time().time()
        start_buffer = time(9, 35)  # Start 5 minutes after open

        if current_time < start_buffer:
            return False, "Waiting for market to settle after open"

        return True, "Ready to trade"

    def should_stop_trading(self):
        """Determine if trading should stop based on market conditions"""
        if not self.is_market_open():
            return True, "Market is closed"

        # Stop trading 15 minutes before close
        if self.minutes_until_close() <= 15:
            return True, "Too close to market close"

        return False, "Continue trading"
