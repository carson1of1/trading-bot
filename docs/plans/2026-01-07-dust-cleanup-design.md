# Dust Cleanup Implementation Plan (ODE-123)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Auto-close small "dust" positions remaining after partial fills to prevent holding cost drift and position clutter.

**Architecture:** Add immediate cleanup in `execute_exit()` after partial fill handling. If remaining position value < configurable threshold, submit market order to close and clean up all tracking.

**Tech Stack:** Python, pytest, existing TradingBot infrastructure

---

## Task 1: Add Configuration Parameter

**Files:**
- Modify: `config.yaml:14-22`

**Step 1: Add min_position_value to config**

Add under `risk_management` section:

```yaml
risk_management:
  max_position_size_pct: 10
  max_portfolio_risk_pct: 30.0
  stop_loss_pct: 5.0
  take_profit_pct: 5.0
  max_daily_loss_pct: 4.0
  emergency_stop_pct: 8.0
  max_open_positions: 3
  max_position_dollars: 999999
  min_position_value: 50  # Force-close dust positions under this dollar value
```

**Step 2: Commit**

```bash
git add config.yaml
git commit -m "config: Add min_position_value for dust cleanup (ODE-123)"
```

---

## Task 2: Write Failing Tests for Dust Cleanup

**Files:**
- Modify: `tests/test_partial_fills.py`

**Step 1: Add test class for dust cleanup**

Add at end of `tests/test_partial_fills.py`:

```python
class TestDustCleanup:
    """Test automatic cleanup of small 'dust' positions after partial fills."""

    @pytest.fixture
    def bot_with_dust_config(self, tmp_path):
        """Create a bot with dust cleanup enabled."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
risk_management:
  min_position_value: 50  # $50 threshold
logging:
  database: "logs/trades.db"
exit_manager:
  enabled: true
"""
        universe = """
proven_symbols:
  - AAPL
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config)
        universe_path = tmp_path / "universe.yaml"
        universe_path.write_text(universe)

        with patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger') as mock_logger, \
             patch('bot.YFinanceDataFetcher'):
            mock_broker_instance = MagicMock()
            mock_broker.return_value = mock_broker_instance
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance
            bot = TradingBot(config_path=str(config_path))
            bot.portfolio_value = 100000.0

            # Set up an existing position: 100 shares @ $150
            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'entry_price': 150.0,
                    'direction': 'LONG',
                    'strategy': 'Momentum',
                    'entry_time': datetime.now(),
                }
            }
            bot.highest_prices['AAPL'] = 150.0
            bot.lowest_prices['AAPL'] = 150.0
            bot.trailing_stops['AAPL'] = {'activated': False, 'price': 0.0}

            return bot

    def test_dust_cleanup_triggers_on_small_remaining(self, bot_with_dust_config, caplog):
        """Partial fill leaving < $50 should trigger dust cleanup."""
        bot = bot_with_dust_config

        # First order: partial fill leaves 2 shares (2 * $25 = $50, just at threshold)
        # We need remaining value < 50, so use exit_price that makes 2 shares < $50
        mock_order_partial = MagicMock()
        mock_order_partial.status = 'filled'
        mock_order_partial.id = 'order_partial'
        mock_order_partial.filled_avg_price = 24.0  # 2 shares * $24 = $48 < $50
        mock_order_partial.filled_qty = 98  # 98 of 100 filled, 2 remaining

        # Second order: dust cleanup order
        mock_order_dust = MagicMock()
        mock_order_dust.status = 'filled'
        mock_order_dust.id = 'order_dust'
        mock_order_dust.filled_avg_price = 24.0
        mock_order_dust.filled_qty = 2

        bot.broker.submit_order.side_effect = [mock_order_partial, mock_order_dust]

        exit_signal = {'exit': True, 'reason': 'stop_loss', 'price': 24.0, 'qty': 100}
        result = bot.execute_exit('AAPL', exit_signal)

        assert result['filled'] is True
        # Position should be fully cleaned up (not left with 2 shares)
        assert 'AAPL' not in bot.open_positions
        # Should have logged dust cleanup
        assert 'DUST_CLEANUP' in caplog.text

    def test_dust_cleanup_skipped_when_above_threshold(self, bot_with_dust_config):
        """Partial fill leaving >= $50 should NOT trigger dust cleanup."""
        bot = bot_with_dust_config

        # Partial fill leaves 10 shares at $150 = $1500, well above threshold
        mock_order = MagicMock()
        mock_order.status = 'filled'
        mock_order.id = 'order456'
        mock_order.filled_avg_price = 150.0
        mock_order.filled_qty = 90  # 90 of 100 filled, 10 remaining
        bot.broker.submit_order.return_value = mock_order

        exit_signal = {'exit': True, 'reason': 'stop_loss', 'price': 150.0, 'qty': 100}
        bot.execute_exit('AAPL', exit_signal)

        # Position should still exist with 10 shares
        assert 'AAPL' in bot.open_positions
        assert bot.open_positions['AAPL']['qty'] == 10
        # Only one order should have been submitted
        assert bot.broker.submit_order.call_count == 1

    def test_dust_cleanup_disabled_when_threshold_zero(self, tmp_path):
        """Dust cleanup should not trigger when min_position_value is 0 or missing."""
        config = """
mode: PAPER
timeframe: 1Hour
trading:
  watchlist_file: "universe.yaml"
risk_management:
  min_position_value: 0  # Disabled
logging:
  database: "logs/trades.db"
"""
        universe = """
proven_symbols:
  - AAPL
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config)
        universe_path = tmp_path / "universe.yaml"
        universe_path.write_text(universe)

        with patch('bot.create_broker') as mock_broker, \
             patch('bot.TradeLogger'), \
             patch('bot.YFinanceDataFetcher'):
            mock_broker_instance = MagicMock()
            mock_broker.return_value = mock_broker_instance
            bot = TradingBot(config_path=str(config_path))
            bot.portfolio_value = 100000.0

            bot.open_positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'entry_price': 150.0,
                    'direction': 'LONG',
                    'strategy': 'Momentum',
                    'entry_time': datetime.now(),
                }
            }
            bot.highest_prices['AAPL'] = 150.0
            bot.lowest_prices['AAPL'] = 150.0
            bot.trailing_stops['AAPL'] = {'activated': False, 'price': 0.0}

            # Partial fill leaving tiny amount
            mock_order = MagicMock()
            mock_order.status = 'filled'
            mock_order.id = 'order456'
            mock_order.filled_avg_price = 10.0
            mock_order.filled_qty = 98  # Leaves 2 shares @ $10 = $20
            bot.broker.submit_order.return_value = mock_order

            exit_signal = {'exit': True, 'reason': 'stop_loss', 'price': 10.0, 'qty': 100}
            bot.execute_exit('AAPL', exit_signal)

            # Position should still exist (no cleanup)
            assert 'AAPL' in bot.open_positions
            assert bot.open_positions['AAPL']['qty'] == 2

    def test_dust_cleanup_logs_pnl_correctly(self, bot_with_dust_config):
        """Dust cleanup should log P&L for the dust portion."""
        bot = bot_with_dust_config

        mock_order_partial = MagicMock()
        mock_order_partial.status = 'filled'
        mock_order_partial.id = 'order_partial'
        mock_order_partial.filled_avg_price = 24.0
        mock_order_partial.filled_qty = 98

        mock_order_dust = MagicMock()
        mock_order_dust.status = 'filled'
        mock_order_dust.id = 'order_dust'
        mock_order_dust.filled_avg_price = 24.0
        mock_order_dust.filled_qty = 2

        bot.broker.submit_order.side_effect = [mock_order_partial, mock_order_dust]

        exit_signal = {'exit': True, 'reason': 'stop_loss', 'price': 24.0, 'qty': 100}
        bot.execute_exit('AAPL', exit_signal)

        # Trade logger should have been called twice (main exit + dust)
        assert bot.trade_logger.log_trade.call_count == 2

        # Check dust trade was logged with correct P&L
        dust_call = bot.trade_logger.log_trade.call_args_list[1]
        dust_trade = dust_call[0][0]
        assert dust_trade['exit_reason'] == 'dust_cleanup'
        assert dust_trade['quantity'] == 2
        # P&L: (24 - 150) * 2 = -252
        assert dust_trade['pnl'] == pytest.approx(-252.0)

    def test_dust_cleanup_handles_failed_order(self, bot_with_dust_config, caplog):
        """Dust cleanup should handle broker errors gracefully."""
        bot = bot_with_dust_config

        mock_order_partial = MagicMock()
        mock_order_partial.status = 'filled'
        mock_order_partial.id = 'order_partial'
        mock_order_partial.filled_avg_price = 24.0
        mock_order_partial.filled_qty = 98

        # Dust cleanup order fails
        bot.broker.submit_order.side_effect = [mock_order_partial, Exception("Broker error")]

        exit_signal = {'exit': True, 'reason': 'stop_loss', 'price': 24.0, 'qty': 100}
        result = bot.execute_exit('AAPL', exit_signal)

        # Original exit should still succeed
        assert result['filled'] is True
        # Error should be logged
        assert 'DUST_CLEANUP' in caplog.text and 'Failed' in caplog.text
        # Position should be cleaned up anyway to avoid stuck state
        assert 'AAPL' not in bot.open_positions
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_partial_fills.py::TestDustCleanup -v
```

Expected: 5 FAILED (methods not implemented)

**Step 3: Commit failing tests**

```bash
git add tests/test_partial_fills.py
git commit -m "test: Add failing tests for dust cleanup (ODE-123)"
```

---

## Task 3: Implement Dust Cleanup Method

**Files:**
- Modify: `bot.py:1095-1115`

**Step 1: Add _cleanup_dust_position method**

Add new method to TradingBot class (after `_cleanup_position` method):

```python
def _cleanup_dust_position(self, symbol: str, qty: int, price: float, position: dict) -> None:
    """
    Force-close a dust position that's below minimum value threshold.

    Args:
        symbol: Stock symbol
        qty: Remaining quantity to close
        price: Current price for P&L calculation
        position: Position dict with entry_price, direction, strategy
    """
    direction = position.get('direction', 'LONG')
    side = 'sell' if direction == 'LONG' else 'buy'
    entry_price = position['entry_price']

    logger.info(f"DUST_CLEANUP | {symbol} | Closing {qty} shares (${qty * price:.2f} < min threshold)")

    try:
        order = self.broker.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )

        if order and hasattr(order, 'filled_qty') and order.filled_qty:
            fill_price = float(order.filled_avg_price) if order.filled_avg_price else price
            filled_qty = int(order.filled_qty)

            # Calculate P&L for dust portion
            if direction == 'LONG':
                pnl = (fill_price - entry_price) * filled_qty
            else:
                pnl = (entry_price - fill_price) * filled_qty

            # Log the dust trade
            self.trade_logger.log_trade({
                'symbol': symbol,
                'action': 'SELL' if direction == 'LONG' else 'BUY',
                'quantity': filled_qty,
                'price': fill_price,
                'strategy': position.get('strategy', 'Unknown'),
                'pnl': pnl,
                'exit_reason': 'dust_cleanup'
            })

            # Record P&L in guards
            if pnl < 0 and self.entry_gate:
                self.entry_gate.record_loss(datetime.now())
            if self.drawdown_guard.enabled:
                self.drawdown_guard.record_realized_pnl(pnl)
            if self.losing_streak_guard.enabled:
                risk_amount = position.get('risk_amount', abs(pnl))
                scaled_risk = risk_amount * (filled_qty / position.get('qty', filled_qty))
                self.losing_streak_guard.record_trade(
                    symbol=symbol,
                    realized_pnl=pnl,
                    risk_amount=scaled_risk,
                    close_time=datetime.now()
                )

            logger.info(f"DUST_CLEANUP | {symbol} | Closed {filled_qty} @ ${fill_price:.2f} | P&L: ${pnl:+.2f}")

    except Exception as e:
        logger.error(f"DUST_CLEANUP | {symbol} | Failed: {e}")

    # Full cleanup regardless of order result to avoid stuck state
    if self.use_tiered_exits and self.exit_manager:
        self.exit_manager.unregister_position(symbol)
    self._cleanup_position(symbol)
    if symbol in self.open_positions:
        del self.open_positions[symbol]
```

**Step 2: Modify execute_exit to call dust cleanup**

In `execute_exit()`, after line 1103 (after `self.exit_manager.update_quantity(symbol, remaining_qty)`), add:

```python
                    # ODE-123: Dust cleanup - force close if remaining value is too small
                    min_position_value = self.config.get('risk_management', {}).get('min_position_value', 0)
                    if min_position_value > 0:
                        remaining_value = remaining_qty * exit_price
                        if remaining_value < min_position_value:
                            self._cleanup_dust_position(symbol, remaining_qty, exit_price, position)
```

**Step 3: Run tests**

```bash
python3 -m pytest tests/test_partial_fills.py::TestDustCleanup -v
```

Expected: 5 PASSED

**Step 4: Run full test suite**

```bash
python3 -m pytest tests/test_partial_fills.py -v
```

Expected: All tests pass

**Step 5: Commit implementation**

```bash
git add bot.py
git commit -m "feat: Add dust cleanup for partial fill positions (ODE-123)

After partial fill, if remaining position value < min_position_value
threshold, automatically submit market order to close the dust.

- New _cleanup_dust_position() method handles order, P&L, logging
- Configurable via risk_management.min_position_value (default: 0)
- Logs with DUST_CLEANUP prefix and exit_reason: 'dust_cleanup'"
```

---

## Task 4: Run Full Test Suite and Verify

**Step 1: Run all tests**

```bash
python3 -m pytest -v
```

Expected: All tests pass

**Step 2: Verify no regressions in partial fill handling**

```bash
python3 -m pytest tests/test_partial_fills.py -v
```

Expected: All 14+ tests pass

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add config parameter | `config.yaml` |
| 2 | Write failing tests | `tests/test_partial_fills.py` |
| 3 | Implement dust cleanup | `bot.py` |
| 4 | Verify full test suite | - |

Total: 4 tasks, ~5 tests added
