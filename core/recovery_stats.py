"""
Recovery Statistics Calculator

Calculates the probability of a trade recovering from a drawdown,
hitting take profit, and provides EV-based recommendations.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class RecoveryStats:
    """Statistics about recovery probability for a position."""

    # Core probabilities
    recovery_probability: float  # Odds of recovering to breakeven (only meaningful if underwater)
    take_profit_probability: float  # Odds of hitting TP
    stop_loss_probability: float  # Odds of hitting stop

    # Drawdown stats
    avg_max_drawdown: float  # Avg max DD when current DD level is hit
    avg_bars_to_recovery: float  # Avg bars to recover

    # Gap context (if applicable)
    is_gap_entry: bool
    gap_percentage: Optional[float]
    gap_win_rate: Optional[float]

    # Current position context
    current_drawdown_pct: float  # Current P&L % (negative if underwater, positive if profitable)
    distance_to_stop_pct: float  # How far from stop (always positive)
    distance_to_tp_pct: float  # How far from TP (always positive)

    # Expected values
    ev_cut_now: float
    ev_hold_to_breakeven: float
    ev_hold_for_tp: float

    # Recommendation
    recommendation: str  # "CUT", "HOLD_TO_BE", "HOLD_FOR_TP"
    recommendation_reason: str
    risk_warning: Optional[str]


def calculate_recovery_stats(
    symbol: str,
    entry_price: float,
    current_price: float,
    stop_price: float,
    take_profit_pct: float = 5.0,
    daily_loss_pct: float = 0.0,
    daily_loss_limit_pct: float = 4.0,
    position_size_pct: float = 10.0,
) -> RecoveryStats:
    """
    Calculate recovery statistics for a position.

    Args:
        symbol: Stock ticker
        entry_price: Entry price of the position
        current_price: Current market price
        stop_price: Stop loss price
        take_profit_pct: Take profit target as % from entry
        daily_loss_pct: Current daily loss %
        daily_loss_limit_pct: Daily loss limit %
        position_size_pct: Position size as % of portfolio

    Returns:
        RecoveryStats with all calculated metrics
    """
    # Calculate current position metrics
    current_pnl_pct = (current_price - entry_price) / entry_price * 100
    stop_pnl_pct = (stop_price - entry_price) / entry_price * 100
    tp_price = entry_price * (1 + take_profit_pct / 100)

    # Distance calculations (always positive - "how far away")
    # Distance to stop = current P&L - stop P&L (how much more we can lose)
    distance_to_stop = current_pnl_pct - stop_pnl_pct

    # Distance to TP = TP level - current P&L (how much more we need to gain)
    distance_to_tp = take_profit_pct - current_pnl_pct

    # Determine if position is profitable or underwater
    is_profitable = current_pnl_pct >= 0

    # Fetch historical data
    ticker = yf.Ticker(symbol)
    hourly_data = ticker.history(period="1y", interval="1h")
    daily_data = ticker.history(period="2y", interval="1d")

    if len(hourly_data) < 50:
        raise ValueError(f"Insufficient data for {symbol}")

    # Calculate stats based on whether we're profitable or underwater
    if is_profitable:
        # Position is GREEN - use continuation analysis
        stats = _analyze_profitable_position(
            hourly_data,
            current_profit_pct=current_pnl_pct,
            stop_loss_pct=abs(stop_pnl_pct),
            take_profit_pct=take_profit_pct
        )
    else:
        # Position is RED - use recovery analysis
        stats = _analyze_drawdown_recovery(
            hourly_data,
            current_drawdown_pct=abs(current_pnl_pct),
            stop_loss_pct=abs(stop_pnl_pct),
            take_profit_pct=take_profit_pct
        )

    # Check for gap entry
    gap_stats = _analyze_gap_context(daily_data, entry_price)

    # Calculate expected values based on position state
    if is_profitable:
        # Already profitable - different EV calculation
        # EV Cut Now = lock in current profit
        ev_cut_now = current_pnl_pct

        # EV Hold to BE = risk of giving back profits
        # Outcomes: stay above BE, fall to stop
        # "Recovery" for profitable position means "staying above water"
        p_stay_above_be = stats['recovery_prob'] / 100
        p_hit_stop = stats['stop_prob'] / 100

        # If we hold and stay above BE, we keep some profit (avg maybe half)
        # If we hold and hit stop, we lose distance_to_stop
        ev_hold_to_be = (
            p_stay_above_be * (current_pnl_pct * 0.5) -  # Partial profit retention
            p_hit_stop * distance_to_stop
        )

        # EV Hold for TP
        p_hit_tp = stats['tp_prob'] / 100
        gain_to_tp = distance_to_tp  # Additional gain needed

        ev_hold_for_tp = (
            p_hit_tp * take_profit_pct +  # Full TP
            (p_stay_above_be - p_hit_tp) * current_pnl_pct -  # Stay profitable but miss TP
            p_hit_stop * distance_to_stop  # Hit stop
        )
    else:
        # Underwater - original EV calculation
        gain_to_be = abs(current_pnl_pct)  # Getting back to 0%
        gain_to_tp = take_profit_pct - current_pnl_pct  # Getting to TP from current
        loss_to_stop = distance_to_stop  # Additional loss to stop

        # P(BE but not TP) = P(recovery) - P(TP)
        p_be_only = max(0, stats['recovery_prob'] - stats['tp_prob'])

        ev_cut_now = current_pnl_pct  # Lock in current loss

        ev_hold_to_be = (
            stats['recovery_prob'] / 100 * gain_to_be -
            stats['stop_prob'] / 100 * loss_to_stop
        )

        ev_hold_for_tp = (
            stats['tp_prob'] / 100 * gain_to_tp +
            p_be_only / 100 * gain_to_be -  # Partial credit for hitting BE
            stats['stop_prob'] / 100 * loss_to_stop
        )

    # Apply risk adjustments
    risk_multiplier = _calculate_risk_multiplier(
        daily_loss_pct, daily_loss_limit_pct, position_size_pct
    )

    ev_hold_to_be_adjusted = ev_hold_to_be * risk_multiplier
    ev_hold_for_tp_adjusted = ev_hold_for_tp * risk_multiplier

    # Determine recommendation
    recommendation, reason = _get_recommendation(
        ev_cut_now, ev_hold_to_be_adjusted, ev_hold_for_tp_adjusted,
        stats['recovery_prob'], stats['tp_prob'],
        risk_multiplier, is_profitable, current_pnl_pct
    )

    # Risk warning
    risk_warning = None
    if daily_loss_pct > daily_loss_limit_pct * 0.75:
        risk_warning = f"Near daily loss limit ({daily_loss_pct:.1f}% of {daily_loss_limit_pct:.1f}%)"
    elif position_size_pct > 20:
        risk_warning = f"Large position size ({position_size_pct:.1f}% of portfolio)"
    elif not is_profitable and stats['avg_max_dd'] < current_pnl_pct * 2:
        risk_warning = f"Historically drops to {stats['avg_max_dd']:.1f}% before recovering"

    return RecoveryStats(
        recovery_probability=stats['recovery_prob'],
        take_profit_probability=stats['tp_prob'],
        stop_loss_probability=stats['stop_prob'],
        avg_max_drawdown=stats['avg_max_dd'],
        avg_bars_to_recovery=stats['avg_bars'],
        is_gap_entry=gap_stats['is_gap'],
        gap_percentage=gap_stats.get('gap_pct'),
        gap_win_rate=gap_stats.get('gap_win_rate'),
        current_drawdown_pct=current_pnl_pct,  # Now correctly signed
        distance_to_stop_pct=distance_to_stop,
        distance_to_tp_pct=distance_to_tp,
        ev_cut_now=ev_cut_now,
        ev_hold_to_breakeven=ev_hold_to_be,
        ev_hold_for_tp=ev_hold_for_tp,
        recommendation=recommendation,
        recommendation_reason=reason,
        risk_warning=risk_warning
    )


def _analyze_profitable_position(
    data,
    current_profit_pct: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    lookforward_bars: int = 50
) -> dict:
    """
    Analyze what happens when a position is already profitable.

    Questions answered:
    - What % of time does a +X% profit continue to TP?
    - What % of time does it give back gains and hit stop?
    - What % stays profitable but doesn't hit either?

    Outcomes are MUTUALLY EXCLUSIVE:
    - hit_tp: reached TP target
    - hit_stop: fell all the way to stop loss
    - neither: ended somewhere in between (may have given back some gains)
    """
    closes = data['Close'].values

    results = []

    for i in range(len(closes) - lookforward_bars):
        entry = closes[i]
        future = closes[i+1:i+lookforward_bars+1]

        # Find when price first reaches current_profit_pct level
        crossed_profit_threshold = False
        threshold_bar = None

        for j, price in enumerate(future):
            pnl = (price - entry) / entry * 100

            if pnl >= current_profit_pct and not crossed_profit_threshold:
                crossed_profit_threshold = True
                threshold_bar = j
                break

        if not crossed_profit_threshold:
            continue

        # Now track what happens AFTER reaching current profit level
        remaining_future = future[threshold_bar:]
        hit_tp = False
        hit_stop = False
        ended_profitable = True  # Did it end above BE?
        bars_held = 0

        for k, price in enumerate(remaining_future):
            pnl = (price - entry) / entry * 100

            if pnl >= take_profit_pct:
                hit_tp = True
                ended_profitable = True
                break

            if pnl <= -stop_loss_pct:
                hit_stop = True
                ended_profitable = False
                break

            bars_held = k

        # If we didn't hit TP or stop, check where we ended
        if not hit_tp and not hit_stop and len(remaining_future) > 0:
            final_pnl = (remaining_future[-1] - entry) / entry * 100
            ended_profitable = final_pnl >= 0

        results.append({
            'hit_tp': hit_tp,
            'hit_stop': hit_stop,
            'ended_profitable': ended_profitable,
            'bars_held': bars_held
        })

    if not results:
        return {
            'recovery_prob': 60.0,  # "Stays above BE" probability
            'tp_prob': 30.0,
            'stop_prob': 25.0,
            'avg_max_dd': 0,
            'avg_bars': 15
        }

    n = len(results)
    tp_count = sum(1 for r in results if r['hit_tp'])
    stop_count = sum(1 for r in results if r['hit_stop'])
    # "Stays above BE" = hit TP OR ended profitable without hitting stop
    stayed_above_be_count = sum(1 for r in results if r['ended_profitable'])

    avg_bars = np.mean([r['bars_held'] for r in results])

    return {
        'recovery_prob': stayed_above_be_count / n * 100,  # Stays above BE (not double counted)
        'tp_prob': tp_count / n * 100,
        'stop_prob': stop_count / n * 100,
        'avg_max_dd': 0,  # Not applicable for profitable positions
        'avg_bars': avg_bars,
        'sample_size': n
    }


def _analyze_drawdown_recovery(
    data,
    current_drawdown_pct: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    lookforward_bars: int = 50
) -> dict:
    """
    Analyze historical drawdowns similar to current position.

    Returns dict with recovery_prob, tp_prob, stop_prob, avg_max_dd, avg_bars
    """
    closes = data['Close'].values

    results = []

    for i in range(len(closes) - lookforward_bars):
        entry = closes[i]
        future = closes[i+1:i+lookforward_bars+1]

        # Track outcomes
        hit_recovery = False
        hit_tp = False
        hit_stop = False
        max_dd = 0
        bars_to_recovery = None

        # First check if we ever hit current_drawdown_pct level
        crossed_dd_threshold = False
        threshold_bar = None

        for j, price in enumerate(future):
            pnl = (price - entry) / entry * 100

            if pnl <= -current_drawdown_pct and not crossed_dd_threshold:
                crossed_dd_threshold = True
                threshold_bar = j
                max_dd = pnl

            if crossed_dd_threshold:
                max_dd = min(max_dd, pnl)

                if pnl >= 0 and not hit_recovery:
                    hit_recovery = True
                    bars_to_recovery = j - threshold_bar

                if pnl >= take_profit_pct:
                    hit_tp = True
                    break

                if pnl <= -stop_loss_pct:
                    hit_stop = True
                    break

        if crossed_dd_threshold:
            results.append({
                'hit_recovery': hit_recovery,
                'hit_tp': hit_tp,
                'hit_stop': hit_stop,
                'max_dd': max_dd,
                'bars_to_recovery': bars_to_recovery
            })

    if not results:
        # Fallback to general stats
        return {
            'recovery_prob': 50.0,
            'tp_prob': 25.0,
            'stop_prob': 50.0,
            'avg_max_dd': -current_drawdown_pct * 2,
            'avg_bars': 20
        }

    n = len(results)
    recovery_count = sum(1 for r in results if r['hit_recovery'])
    tp_count = sum(1 for r in results if r['hit_tp'])
    stop_count = sum(1 for r in results if r['hit_stop'])

    avg_max_dd = np.mean([r['max_dd'] for r in results])

    recovery_bars = [r['bars_to_recovery'] for r in results if r['bars_to_recovery'] is not None]
    avg_bars = np.mean(recovery_bars) if recovery_bars else 20

    return {
        'recovery_prob': recovery_count / n * 100,
        'tp_prob': tp_count / n * 100,
        'stop_prob': stop_count / n * 100,
        'avg_max_dd': avg_max_dd,
        'avg_bars': avg_bars,
        'sample_size': n
    }


def _analyze_gap_context(daily_data, entry_price: float, gap_threshold: float = 3.0) -> dict:
    """Check if entry was after a gap and return gap-specific stats."""
    if len(daily_data) < 5:
        return {'is_gap': False}

    closes = daily_data['Close'].values
    opens = daily_data['Open'].values

    # Find gaps > threshold
    gap_results = []
    for i in range(1, len(daily_data) - 5):
        prev_close = closes[i-1]
        open_price = opens[i]
        gap_pct = (open_price - prev_close) / prev_close * 100

        if abs(gap_pct) > gap_threshold:
            # Track what happened after gap
            future_closes = closes[i:i+5]
            end_pnl = (future_closes[-1] - open_price) / open_price * 100
            gap_results.append({
                'gap_pct': gap_pct,
                'ended_positive': end_pnl > 0
            })

    if not gap_results:
        return {'is_gap': False}

    # Check if current entry is near a gap
    latest_close = closes[-2] if len(closes) >= 2 else closes[-1]
    current_open = opens[-1]
    current_gap = (current_open - latest_close) / latest_close * 100

    if abs(current_gap) > gap_threshold:
        gap_ups = [r for r in gap_results if r['gap_pct'] > gap_threshold]
        gap_win_rate = sum(1 for r in gap_ups if r['ended_positive']) / len(gap_ups) * 100 if gap_ups else 50

        return {
            'is_gap': True,
            'gap_pct': current_gap,
            'gap_win_rate': gap_win_rate
        }

    return {'is_gap': False}


def _calculate_risk_multiplier(
    daily_loss_pct: float,
    daily_loss_limit_pct: float,
    position_size_pct: float
) -> float:
    """Calculate risk adjustment multiplier based on current risk exposure."""
    multiplier = 1.0

    # Daily loss proximity
    if daily_loss_pct > 0:
        loss_ratio = daily_loss_pct / daily_loss_limit_pct
        if loss_ratio > 0.75:
            multiplier *= 0.3
        elif loss_ratio > 0.5:
            multiplier *= 0.6
        elif loss_ratio > 0.25:
            multiplier *= 0.8

    # Position size
    if position_size_pct > 25:
        multiplier *= 0.5
    elif position_size_pct > 20:
        multiplier *= 0.7
    elif position_size_pct > 15:
        multiplier *= 0.85

    return multiplier


def _get_recommendation(
    ev_cut: float,
    ev_be: float,
    ev_tp: float,
    recovery_prob: float,
    tp_prob: float,
    risk_mult: float,
    is_profitable: bool,
    current_pnl_pct: float
) -> tuple[str, str]:
    """Determine the recommended action based on EVs."""

    # Find best EV
    evs = {
        'CUT': ev_cut,
        'HOLD_TO_BE': ev_be,
        'HOLD_FOR_TP': ev_tp
    }
    best_action = max(evs, key=evs.get)
    best_ev = evs[best_action]

    # Generate reason based on position state
    if is_profitable:
        # Different messaging for profitable positions
        if best_action == 'CUT':
            if tp_prob < 25:
                reason = f"Low TP probability ({tp_prob:.0f}%) - lock in +{current_pnl_pct:.1f}% profit"
            elif risk_mult < 0.5:
                reason = f"Risk limits suggest taking profit at +{current_pnl_pct:.1f}%"
            else:
                reason = f"Best EV is to take profit at +{current_pnl_pct:.1f}%"
        elif best_action == 'HOLD_TO_BE':
            reason = f"Good odds of staying profitable ({recovery_prob:.0f}%)"
        else:  # HOLD_FOR_TP
            if tp_prob > 40:
                reason = f"Strong TP probability ({tp_prob:.0f}%) - hold for full target"
            else:
                reason = f"Positive EV (+{best_ev:.2f}%) from holding for TP"
    else:
        # Original messaging for underwater positions
        if best_action == 'CUT':
            if recovery_prob < 35:
                reason = f"Low recovery odds ({recovery_prob:.0f}%) - cut losses"
            elif risk_mult < 0.5:
                reason = "Risk limits suggest protecting capital"
            else:
                reason = "Negative EV on holding - minimize losses"

        elif best_action == 'HOLD_TO_BE':
            reason = f"Good recovery odds ({recovery_prob:.0f}%) but TP unlikely ({tp_prob:.0f}%)"

        else:  # HOLD_FOR_TP
            if tp_prob > 40:
                reason = f"Strong TP probability ({tp_prob:.0f}%) justifies holding"
            else:
                reason = f"Higher EV (+{best_ev:.2f}%) from larger upside"

    return best_action, reason
