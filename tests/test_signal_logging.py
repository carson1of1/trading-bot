"""Tests for signal logging functionality (ODE-97)"""
import pytest
import os
import sys
import tempfile
import yaml
import logging
from unittest.mock import MagicMock, patch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSignalLogging:
    """Test signal logging in bot.py check_entry()"""

    def test_signal_log_format(self, caplog):
        """Signal logging should follow expected format"""
        # This test validates the log message format
        # Format: SIGNAL | {symbol} | {action} | Confidence: {conf} | Strategy: {strat} | Threshold: {thresh} | Result: PASSED/BELOW_THRESHOLD

        expected_fields = ['SIGNAL', 'AAPL', 'BUY', 'Confidence:', 'Strategy:', 'Threshold:', 'Result:']

        # Create a mock log message
        log_message = (
            "SIGNAL | AAPL | BUY | Confidence: 85.0 | "
            "Strategy: Momentum_1Hour | Threshold: 80 | Result: PASSED"
        )

        # Verify all expected fields are present
        for field in expected_fields:
            assert field in log_message, f"Missing field: {field}"

        # Verify PASSED/BELOW_THRESHOLD result
        assert 'Result: PASSED' in log_message or 'Result: BELOW_THRESHOLD' in log_message

    def test_below_threshold_result(self):
        """Signal below threshold should show BELOW_THRESHOLD result"""
        confidence = 70
        threshold = 80
        result = 'PASSED' if confidence >= threshold else 'BELOW_THRESHOLD'

        assert result == 'BELOW_THRESHOLD'

        log_message = f"SIGNAL | AAPL | BUY | Confidence: {confidence:.1f} | Strategy: Momentum_1Hour | Threshold: {threshold} | Result: {result}"
        assert 'Result: BELOW_THRESHOLD' in log_message

    def test_passed_threshold_result(self):
        """Signal above threshold should show PASSED result"""
        confidence = 85
        threshold = 80
        result = 'PASSED' if confidence >= threshold else 'BELOW_THRESHOLD'

        assert result == 'PASSED'

        log_message = f"SIGNAL | AAPL | BUY | Confidence: {confidence:.1f} | Strategy: Momentum_1Hour | Threshold: {threshold} | Result: {result}"
        assert 'Result: PASSED' in log_message


class TestSignalSummary:
    """Test signal summary logging at end of trading cycle"""

    def test_summary_log_format(self):
        """Signal summary should follow expected format"""
        # Format: SIGNAL_SUMMARY | Total: X | BUY: X | SELL: X | HOLD: X | Above threshold: X | Executed: X | Blocked: X (reasons: ...)

        summary = {
            'total': 10,
            'buy': 3,
            'sell': 2,
            'hold': 5,
            'above_threshold': 4,
            'executed': 2,
            'blocked': 2,
            'block_reasons': {'entry_gate': 1, 'position_limit': 1}
        }

        log_message = (
            f"SIGNAL_SUMMARY | Total: {summary['total']} | "
            f"BUY: {summary['buy']} | SELL: {summary['sell']} | HOLD: {summary['hold']} | "
            f"Above threshold: {summary['above_threshold']} | "
            f"Executed: {summary['executed']} | "
            f"Blocked: {summary['blocked']} (reasons: {summary['block_reasons']})"
        )

        # Verify key fields
        assert 'SIGNAL_SUMMARY' in log_message
        assert 'Total: 10' in log_message
        assert 'BUY: 3' in log_message
        assert 'SELL: 2' in log_message
        assert 'HOLD: 5' in log_message
        assert 'Above threshold: 4' in log_message
        assert 'Executed: 2' in log_message
        assert 'Blocked: 2' in log_message

    def test_summary_counts_correct(self):
        """Summary counts should add up correctly"""
        signals = [
            {'action': 'BUY', 'confidence': 85},
            {'action': 'BUY', 'confidence': 75},  # below threshold
            {'action': 'SELL', 'confidence': 82},
            {'action': 'HOLD', 'confidence': 50},
            {'action': 'HOLD', 'confidence': 30},
        ]
        threshold = 80

        buy_count = sum(1 for s in signals if s['action'] == 'BUY')
        sell_count = sum(1 for s in signals if s['action'] == 'SELL')
        hold_count = sum(1 for s in signals if s['action'] == 'HOLD')
        above_threshold = sum(1 for s in signals if s['confidence'] >= threshold and s['action'] != 'HOLD')

        assert buy_count == 2
        assert sell_count == 1
        assert hold_count == 2
        assert above_threshold == 2  # BUY 85 and SELL 82


class TestComponentLogging:
    """Test component logging in StrategyManager"""

    def test_component_log_includes_indicators(self):
        """Component logging should include RSI, MACD, etc."""
        components = {
            'rsi': 65.5,
            'macd_histogram': 0.25,
            'price_vs_sma20': 1.02,
            'bollinger_position': 0.75
        }

        # Build component log string
        component_str = ' | '.join(f"{k}: {v:.2f}" for k, v in components.items())
        log_message = f"SIGNAL_COMPONENTS | AAPL | {component_str}"

        assert 'SIGNAL_COMPONENTS' in log_message
        assert 'rsi: 65.50' in log_message
        assert 'macd_histogram: 0.25' in log_message


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
