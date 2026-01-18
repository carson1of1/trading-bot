#!/usr/bin/env python3
"""
Research Hypotheses Testing Framework

Tests 3 new alpha hypotheses with walk-forward validation:
1. Volatility Compression Breakout
2. Liquidity Sweep Reversal
3. Session Range Failure

Research Rules:
- No optimization loops
- One change per experiment
- Walk-forward validation mandatory
- Binary decisions: PROCEED or ARCHIVE
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_symbols(limit=50):
    """Load symbols from universe - use high volatility + tech for best signal frequency"""
    universe_path = Path(__file__).parent / 'universe_original.yaml'
    with open(universe_path, 'r') as f:
        universe = yaml.safe_load(f)

    scanner = universe.get('scanner_universe', {})

    # Prioritize high volatility and tech for more signal frequency
    priority_categories = ['high_volatility', 'tech_growth', 'sp500']
    symbols = []

    for cat in priority_categories:
        if cat in scanner and isinstance(scanner[cat], list):
            symbols.extend(scanner[cat])

    # Remove duplicates, crypto (different data format), and blacklisted
    blacklist = universe.get('blacklist', [])
    symbols = [s for s in symbols if '/' not in s and s not in blacklist]
    symbols = list(dict.fromkeys(symbols))  # Preserve order, remove dupes

    return symbols[:limit]


def fetch_data(symbols, start_date, end_date, interval='1h'):
    """Fetch OHLCV data for all symbols"""
    print(f"  Fetching {interval} data for {len(symbols)} symbols...")

    all_data = {}
    successful = 0

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            if df is not None and len(df) >= 100:  # Need enough bars for indicators
                df = df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                })
                all_data[symbol] = df
                successful += 1
        except Exception:
            pass

    print(f"  Successfully loaded {successful}/{len(symbols)} symbols")
    return all_data


# =============================================================================
# HYPOTHESIS 1: VOLATILITY COMPRESSION BREAKOUT
# =============================================================================

def h1_volatility_compression(df,
                               atr_lookback=14,
                               compression_threshold=0.75,
                               expansion_threshold=1.5,
                               recency_window=10):
    """
    Hypothesis 1: Volatility Compression Breakout (RELAXED)

    Economic Rationale:
    Markets cycle between compression (low volatility) and expansion.
    After prolonged compression, the first decisive expansion move often continues
    due to institutional accumulation/distribution resolving.

    Entry Logic:
    - ATR contracts below threshold of recent average (compression)
    - ATR then expands significantly (expansion trigger)
    - Enter in direction of expansion bar

    Parameters RELAXED from original (was generating 0 signals):
    - atr_lookback: 14 (standard ATR period)
    - compression_threshold: 0.75 (ATR < 75% of average = "compressed")
    - expansion_threshold: 1.5 (ATR > 150% of prior bar = "expanding")
    - recency_window: 10 (must have been compressed within last 10 bars)
    """
    df = df.copy()

    # Calculate ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(atr_lookback).mean()
    df['atr_avg'] = df['atr'].rolling(atr_lookback * 2).mean()  # Longer lookback for average

    # Compression: ATR < threshold of recent average
    df['compression_ratio'] = df['atr'] / df['atr_avg']
    df['is_compressed'] = df['compression_ratio'] < compression_threshold

    # Track if recently compressed (within last N bars)
    df['was_compressed_recently'] = df['is_compressed'].rolling(recency_window).max().fillna(0).astype(bool)

    # Expansion: Current bar's range is significantly larger than ATR
    current_range = df['high'] - df['low']
    df['is_expanding'] = current_range > df['atr'] * expansion_threshold

    # Direction of expansion bar (must be decisive - close near high/low)
    bar_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / bar_range
    df['bullish_bar'] = (df['close'] > df['open']) & (close_position > 0.6)
    df['bearish_bar'] = (df['close'] < df['open']) & (close_position < 0.4)

    # Entry signals
    entry_condition = df['was_compressed_recently'] & df['is_expanding']

    # Long entries: expansion bar is bullish
    df['long_entry'] = entry_condition & df['bullish_bar']

    # Short entries: expansion bar is bearish
    df['short_entry'] = entry_condition & df['bearish_bar']

    return df


# =============================================================================
# HYPOTHESIS 2: LIQUIDITY SWEEP REVERSAL
# =============================================================================

def h2_liquidity_sweep(df,
                        swing_lookback=20,
                        sweep_threshold=0.001,
                        reversal_bars=3):
    """
    Hypothesis 2: Liquidity Sweep Reversal

    Economic Rationale:
    Large players need liquidity to fill orders. They push price beyond obvious
    levels (prior swing highs/lows) to trigger stops and create liquidity.
    After sweeping, price often reverses. This is market microstructure, not pattern-fitting.

    Entry Logic:
    - Identify prior swing high/low (N-bar lookback)
    - Price breaks beyond it briefly (sweep)
    - Price reverses back inside the prior range within M bars
    - Enter in direction of reversal

    Parameters are FIXED:
    - swing_lookback: 20 (identify 20-bar swing points)
    - sweep_threshold: 0.1% (must break by at least this much)
    - reversal_bars: 3 (must reverse within 3 bars)
    """
    df = df.copy()

    # Identify swing highs and lows
    df['swing_high'] = df['high'].rolling(swing_lookback).max().shift(1)
    df['swing_low'] = df['low'].rolling(swing_lookback).min().shift(1)

    # Detect sweep of swing high (potential short setup)
    df['broke_swing_high'] = df['high'] > df['swing_high'] * (1 + sweep_threshold)

    # Detect sweep of swing low (potential long setup)
    df['broke_swing_low'] = df['low'] < df['swing_low'] * (1 - sweep_threshold)

    # Check for reversal: close back inside range
    df['closed_below_swing_high'] = df['close'] < df['swing_high']
    df['closed_above_swing_low'] = df['close'] > df['swing_low']

    # Long entry: swept low then reversed (closed back above)
    # Must have broken low recently and now closing above
    df['recent_low_sweep'] = df['broke_swing_low'].rolling(reversal_bars).max().fillna(0).astype(bool)
    df['long_entry'] = df['recent_low_sweep'] & df['closed_above_swing_low'] & ~df['broke_swing_low']

    # Short entry: swept high then reversed (closed back below)
    df['recent_high_sweep'] = df['broke_swing_high'].rolling(reversal_bars).max().fillna(0).astype(bool)
    df['short_entry'] = df['recent_high_sweep'] & df['closed_below_swing_high'] & ~df['broke_swing_high']

    return df


# =============================================================================
# HYPOTHESIS 3: SESSION RANGE FAILURE
# =============================================================================

def h3_session_range_failure(df,
                              range_bars=6,
                              failure_threshold=0.5):
    """
    Hypothesis 3: Session/Opening Range Failure

    Economic Rationale:
    The opening range represents initial price discovery where institutional
    orders establish boundaries. A FAILURE to hold a breakout (price breaks out
    then reverses back inside) indicates trapped traders and higher probability
    of mean reversion toward the opposite boundary.

    For hourly data: Use rolling 6-bar range as proxy for "session range"

    Entry Logic:
    - Define range (high/low of last N bars)
    - Price breaks range
    - Price fails and closes back inside range (traps breakout traders)
    - Enter in opposite direction of failed breakout

    Parameters are FIXED:
    - range_bars: 6 (6-hour rolling range)
    - failure_threshold: 0.5 (must close at least 50% back into range)
    """
    df = df.copy()

    # Define rolling range
    df['range_high'] = df['high'].rolling(range_bars).max().shift(1)
    df['range_low'] = df['low'].rolling(range_bars).min().shift(1)
    df['range_size'] = df['range_high'] - df['range_low']

    # Detect breakout attempts
    df['broke_above'] = df['high'] > df['range_high']
    df['broke_below'] = df['low'] < df['range_low']

    # Detect failure: broke out but closed back inside
    # For upside failure (short signal): broke above but closed below range_high
    df['upside_failure'] = df['broke_above'] & (df['close'] < df['range_high'])

    # For downside failure (long signal): broke below but closed above range_low
    df['downside_failure'] = df['broke_below'] & (df['close'] > df['range_low'])

    # Additional filter: close should be well back inside range (not just barely)
    df['range_mid'] = (df['range_high'] + df['range_low']) / 2

    # Long entry: downside failure with close above midpoint
    df['long_entry'] = df['downside_failure'] & (df['close'] > df['range_mid'] * (1 - (1 - failure_threshold) * 0.5))

    # Short entry: upside failure with close below midpoint
    df['short_entry'] = df['upside_failure'] & (df['close'] < df['range_mid'] * (1 + (1 - failure_threshold) * 0.5))

    return df


# =============================================================================
# HYPOTHESIS 4: EXTREME MOVE REVERSAL
# =============================================================================

def h4_extreme_move_reversal(df,
                              lookback=20,
                              extreme_threshold=2.5,
                              confirmation_bars=1):
    """
    Hypothesis 4: Extreme Move Reversal

    Economic Rationale:
    Large single-bar moves (measured in ATR multiples) often overshoot fair value
    due to panic/euphoria and tend to partially revert. This exploits short-term
    mean reversion after emotional extremes.

    Entry Logic:
    - Bar move exceeds X standard deviations of recent moves
    - Enter counter-trend expecting partial reversion
    - Requires confirmation (next bar doesn't continue extreme move)

    Parameters FIXED:
    - lookback: 20 bars for baseline
    - extreme_threshold: 2.5 ATR move = "extreme"
    - confirmation_bars: 1 (wait 1 bar for confirmation)
    """
    df = df.copy()

    # Calculate ATR for normalizing moves
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # Calculate bar move in ATR terms
    df['bar_move'] = df['close'] - df['open']
    df['bar_move_atr'] = df['bar_move'].abs() / df['atr']

    # Identify extreme moves
    df['extreme_up'] = df['bar_move_atr'] > extreme_threshold
    df['extreme_down'] = df['bar_move_atr'] > extreme_threshold

    # Direction of extreme move
    df['was_extreme_up'] = (df['bar_move'] > 0) & df['extreme_up']
    df['was_extreme_down'] = (df['bar_move'] < 0) & df['extreme_down']

    # Shift for confirmation - enter after extreme bar
    df['extreme_up_prev'] = df['was_extreme_up'].shift(confirmation_bars)
    df['extreme_down_prev'] = df['was_extreme_down'].shift(confirmation_bars)

    # Confirmation: current bar didn't continue the extreme move
    df['not_continuing_up'] = df['close'] <= df['open']  # Not another strong up bar
    df['not_continuing_down'] = df['close'] >= df['open']  # Not another strong down bar

    # Long entry: after extreme down move (reversal up expected)
    df['long_entry'] = df['extreme_down_prev'].fillna(False) & df['not_continuing_down']

    # Short entry: after extreme up move (reversal down expected)
    df['short_entry'] = df['extreme_up_prev'].fillna(False) & df['not_continuing_up']

    return df


# =============================================================================
# HYPOTHESIS 5: VOLUME-PRICE DIVERGENCE
# =============================================================================

def h5_volume_divergence(df,
                          lookback=10,
                          price_threshold=0.02,
                          volume_decline=0.7):
    """
    Hypothesis 5: Volume-Price Divergence

    Economic Rationale:
    When price makes new highs/lows but volume is declining, it signals
    exhaustion - fewer participants are driving the move. This divergence
    often precedes reversals as the move lacks conviction.

    Entry Logic:
    - Price makes new N-bar high/low
    - Volume is below average (declining participation)
    - Enter counter-trend expecting reversal

    Parameters FIXED:
    - lookback: 10 bars for high/low
    - price_threshold: 2% above prior high to qualify
    - volume_decline: volume < 70% of average
    """
    df = df.copy()

    # Calculate volume average
    df['volume_avg'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_avg']

    # Identify new highs and lows
    df['highest_high'] = df['high'].rolling(lookback).max().shift(1)
    df['lowest_low'] = df['low'].rolling(lookback).min().shift(1)

    # New high/low with threshold
    df['new_high'] = df['high'] > df['highest_high'] * (1 + price_threshold)
    df['new_low'] = df['low'] < df['lowest_low'] * (1 - price_threshold)

    # Volume divergence (low volume on breakout)
    df['low_volume'] = df['volume_ratio'] < volume_decline

    # Long entry: new low on declining volume (exhaustion, expect bounce)
    df['long_entry'] = df['new_low'] & df['low_volume']

    # Short entry: new high on declining volume (exhaustion, expect drop)
    df['short_entry'] = df['new_high'] & df['low_volume']

    return df


# =============================================================================
# HYPOTHESIS 6: TREND PULLBACK CONTINUATION
# =============================================================================

def h6_trend_pullback(df,
                       trend_period=50,
                       pullback_threshold=0.03,
                       rsi_oversold=35,
                       rsi_overbought=65):
    """
    Hypothesis 6: Trend Pullback Continuation

    Economic Rationale:
    Strong trends tend to continue. Pullbacks within a trend offer better
    risk/reward entries than chasing breakouts. Enter when price pulls back
    to support in an uptrend (or resistance in downtrend) with RSI confirmation.

    Entry Logic:
    - Establish trend direction (price vs SMA)
    - Wait for pullback (X% from recent high/low)
    - RSI confirms oversold/overbought condition
    - Enter in trend direction

    Parameters FIXED:
    - trend_period: 50 bar SMA for trend
    - pullback_threshold: 3% pullback from recent extreme
    - rsi_oversold: 35 (for long entries)
    - rsi_overbought: 65 (for short entries)
    """
    df = df.copy()

    # Trend determination
    df['sma_50'] = df['close'].rolling(trend_period).mean()
    df['uptrend'] = df['close'] > df['sma_50']
    df['downtrend'] = df['close'] < df['sma_50']

    # RSI calculation
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Recent highs and lows for pullback measurement
    df['recent_high'] = df['high'].rolling(20).max()
    df['recent_low'] = df['low'].rolling(20).min()

    # Pullback from recent high (in uptrend)
    df['pullback_from_high'] = (df['recent_high'] - df['close']) / df['recent_high']
    df['has_pulled_back_up'] = df['pullback_from_high'] > pullback_threshold

    # Pullback from recent low (in downtrend)
    df['pullback_from_low'] = (df['close'] - df['recent_low']) / df['recent_low']
    df['has_pulled_back_down'] = df['pullback_from_low'] > pullback_threshold

    # Long entry: uptrend + pullback + RSI oversold
    df['long_entry'] = df['uptrend'] & df['has_pulled_back_up'] & (df['rsi'] < rsi_oversold)

    # Short entry: downtrend + bounce + RSI overbought
    df['short_entry'] = df['downtrend'] & df['has_pulled_back_down'] & (df['rsi'] > rsi_overbought)

    return df


# =============================================================================
# HYPOTHESIS 7: MOMENTUM ACCELERATION
# =============================================================================

def h7_momentum_acceleration(df,
                              short_period=5,
                              long_period=20,
                              acceleration_threshold=1.5):
    """
    Hypothesis 7: Momentum Acceleration

    Economic Rationale:
    When short-term momentum significantly exceeds long-term momentum,
    it signals acceleration - new information is being priced in rapidly.
    This often continues as other participants catch up.

    Entry Logic:
    - Calculate short and long-term rate of change
    - When short-term ROC > long-term ROC by threshold, momentum is accelerating
    - Enter in direction of acceleration

    Parameters FIXED:
    - short_period: 5 bars
    - long_period: 20 bars
    - acceleration_threshold: short ROC > 1.5x long ROC
    """
    df = df.copy()

    # Rate of change calculations
    df['roc_short'] = (df['close'] - df['close'].shift(short_period)) / df['close'].shift(short_period)
    df['roc_long'] = (df['close'] - df['close'].shift(long_period)) / df['close'].shift(long_period)

    # Acceleration: short-term momentum exceeding long-term
    df['roc_ratio'] = df['roc_short'].abs() / df['roc_long'].abs().replace(0, np.nan)

    # Direction alignment (both pointing same way)
    df['both_positive'] = (df['roc_short'] > 0) & (df['roc_long'] > 0)
    df['both_negative'] = (df['roc_short'] < 0) & (df['roc_long'] < 0)

    # Acceleration detected
    df['accelerating'] = df['roc_ratio'] > acceleration_threshold

    # Long entry: positive acceleration
    df['long_entry'] = df['accelerating'] & df['both_positive']

    # Short entry: negative acceleration
    df['short_entry'] = df['accelerating'] & df['both_negative']

    return df


# =============================================================================
# HYPOTHESIS 8: GAP FADE
# =============================================================================

def h8_gap_fade(df,
                 gap_threshold=0.015,
                 max_gap=0.05):
    """
    Hypothesis 8: Gap Fade

    Economic Rationale:
    Overnight gaps often represent overreaction to news. Gaps tend to fill
    as the market digests information and finds fair value. Small-to-medium
    gaps have higher fill rates than extreme gaps.

    Entry Logic:
    - Detect gap (open vs prior close)
    - Gap must be significant but not extreme
    - Fade the gap (enter counter-direction)

    Parameters FIXED:
    - gap_threshold: 1.5% minimum gap
    - max_gap: 5% maximum (larger gaps may be legitimate)
    """
    df = df.copy()

    # Calculate gap
    df['prior_close'] = df['close'].shift(1)
    df['gap_pct'] = (df['open'] - df['prior_close']) / df['prior_close']
    df['gap_abs'] = df['gap_pct'].abs()

    # Gap qualifies (significant but not extreme)
    df['gap_qualifies'] = (df['gap_abs'] > gap_threshold) & (df['gap_abs'] < max_gap)

    # Gap direction
    df['gap_up'] = df['gap_pct'] > 0
    df['gap_down'] = df['gap_pct'] < 0

    # Long entry: gap down (fade by going long)
    df['long_entry'] = df['gap_qualifies'] & df['gap_down']

    # Short entry: gap up (fade by going short)
    df['short_entry'] = df['gap_qualifies'] & df['gap_up']

    return df


# =============================================================================
# HYPOTHESIS 9: RSI DIVERGENCE
# =============================================================================

def h9_rsi_divergence(df,
                       lookback=14,
                       rsi_period=14,
                       divergence_bars=10):
    """
    Hypothesis 9: RSI Divergence

    Economic Rationale:
    When price makes a new high but RSI makes a lower high, momentum is waning.
    This divergence often precedes reversals as buying/selling pressure exhausts.

    Entry Logic:
    - Price makes new N-bar high/low
    - RSI fails to confirm (makes lower high / higher low)
    - Enter counter-trend

    Parameters FIXED:
    - lookback: 14 bars for price extremes
    - rsi_period: 14 (standard)
    - divergence_bars: 10 (window to detect divergence)
    """
    df = df.copy()

    # RSI calculation
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Price extremes
    df['price_high'] = df['high'].rolling(lookback).max()
    df['price_low'] = df['low'].rolling(lookback).min()
    df['new_high'] = df['high'] >= df['price_high']
    df['new_low'] = df['low'] <= df['price_low']

    # RSI extremes
    df['rsi_high'] = df['rsi'].rolling(divergence_bars).max()
    df['rsi_low'] = df['rsi'].rolling(divergence_bars).min()

    # Bearish divergence: new price high but RSI below recent RSI high
    df['rsi_prev_high'] = df['rsi'].shift(divergence_bars).rolling(lookback).max()
    df['bearish_div'] = df['new_high'] & (df['rsi'] < df['rsi_prev_high'] * 0.95)

    # Bullish divergence: new price low but RSI above recent RSI low
    df['rsi_prev_low'] = df['rsi'].shift(divergence_bars).rolling(lookback).min()
    df['bullish_div'] = df['new_low'] & (df['rsi'] > df['rsi_prev_low'] * 1.05)

    # Entries
    df['long_entry'] = df['bullish_div']
    df['short_entry'] = df['bearish_div']

    return df


# =============================================================================
# HYPOTHESIS 10: INSIDE BAR BREAKOUT
# =============================================================================

def h10_inside_bar(df,
                    min_inside_bars=1):
    """
    Hypothesis 10: Inside Bar Breakout

    Economic Rationale:
    Inside bars (where high < prior high AND low > prior low) represent
    consolidation and indecision. A breakout from this compression often
    leads to directional movement as the market resolves uncertainty.

    Entry Logic:
    - Identify inside bar (range contained within prior bar)
    - Enter on breakout of inside bar's range
    - Direction determined by which side breaks first

    Parameters FIXED:
    - min_inside_bars: 1 (at least 1 inside bar)
    """
    df = df.copy()

    # Inside bar detection
    df['is_inside'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))

    # Track if we had inside bar recently
    df['had_inside'] = df['is_inside'].shift(1)

    # Mother bar (the bar before inside bar) range
    df['mother_high'] = df['high'].shift(2)
    df['mother_low'] = df['low'].shift(2)

    # Breakout detection (current bar breaks mother bar range after inside bar)
    df['breakout_up'] = df['had_inside'] & (df['close'] > df['mother_high'])
    df['breakout_down'] = df['had_inside'] & (df['close'] < df['mother_low'])

    df['long_entry'] = df['breakout_up']
    df['short_entry'] = df['breakout_down']

    return df


# =============================================================================
# HYPOTHESIS 11: HIGHER HIGH / HIGHER LOW CONTINUATION
# =============================================================================

def h11_trend_structure(df,
                         swing_period=10):
    """
    Hypothesis 11: Trend Structure Continuation

    Economic Rationale:
    Markets in trends make higher highs and higher lows (uptrend) or
    lower highs and lower lows (downtrend). Enter on pullbacks within
    established trend structure.

    Entry Logic:
    - Identify trend structure (HH/HL or LH/LL)
    - Wait for pullback that maintains structure
    - Enter in trend direction

    Parameters FIXED:
    - swing_period: 10 bars for swing detection
    """
    df = df.copy()

    # Swing highs and lows
    df['swing_high'] = df['high'].rolling(swing_period, center=True).max()
    df['swing_low'] = df['low'].rolling(swing_period, center=True).min()

    df['is_swing_high'] = df['high'] == df['swing_high']
    df['is_swing_low'] = df['low'] == df['swing_low']

    # Track recent swing points
    df['prev_swing_high'] = df['high'].where(df['is_swing_high']).ffill().shift(1)
    df['prev_swing_low'] = df['low'].where(df['is_swing_low']).ffill().shift(1)

    # Higher high / higher low structure
    df['higher_high'] = df['high'] > df['prev_swing_high']
    df['higher_low'] = df['low'] > df['prev_swing_low']
    df['uptrend_structure'] = df['higher_high'].rolling(swing_period).sum() > 0

    # Lower high / lower low structure
    df['lower_high'] = df['high'] < df['prev_swing_high']
    df['lower_low'] = df['low'] < df['prev_swing_low']
    df['downtrend_structure'] = df['lower_low'].rolling(swing_period).sum() > 0

    # Pullback detection (price drops but maintains structure)
    df['pullback_in_uptrend'] = df['uptrend_structure'] & (df['close'] < df['close'].rolling(5).mean())
    df['bounce_in_downtrend'] = df['downtrend_structure'] & (df['close'] > df['close'].rolling(5).mean())

    df['long_entry'] = df['pullback_in_uptrend'] & (df['close'] > df['open'])  # Bullish bar on pullback
    df['short_entry'] = df['bounce_in_downtrend'] & (df['close'] < df['open'])  # Bearish bar on bounce

    return df


# =============================================================================
# HYPOTHESIS 12: MEAN REVERSION WITH TREND FILTER
# =============================================================================

def h12_filtered_mean_reversion(df,
                                  trend_period=100,
                                  oversold_rsi=30,
                                  overbought_rsi=70,
                                  bb_period=20):
    """
    Hypothesis 12: Mean Reversion with Strong Trend Filter

    Economic Rationale:
    Mean reversion works best in established trends where pullbacks are
    buying opportunities (uptrend) or shorting opportunities (downtrend).
    Combining trend filter with extreme RSI/BB improves odds.

    Entry Logic:
    - Strong trend filter (price above/below long SMA)
    - RSI at extreme
    - Price at Bollinger Band extreme
    - Enter counter to short-term move but with trend

    Parameters FIXED:
    - trend_period: 100 (long-term trend)
    - oversold_rsi: 30
    - overbought_rsi: 70
    - bb_period: 20
    """
    df = df.copy()

    # Trend filter
    df['sma_100'] = df['close'].rolling(trend_period).mean()
    df['strong_uptrend'] = df['close'] > df['sma_100']
    df['strong_downtrend'] = df['close'] < df['sma_100']

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['oversold'] = df['rsi'] < oversold_rsi
    df['overbought'] = df['rsi'] > overbought_rsi

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(bb_period).mean()
    df['bb_std'] = df['close'].rolling(bb_period).std()
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']

    df['at_lower_bb'] = df['close'] <= df['bb_lower']
    df['at_upper_bb'] = df['close'] >= df['bb_upper']

    # Long: strong uptrend + oversold + at lower BB
    df['long_entry'] = df['strong_uptrend'] & df['oversold'] & df['at_lower_bb']

    # Short: strong downtrend + overbought + at upper BB
    df['short_entry'] = df['strong_downtrend'] & df['overbought'] & df['at_upper_bb']

    return df


# =============================================================================
# HYPOTHESIS 13: OPENING RANGE BREAKOUT (for daily data)
# =============================================================================

def h13_range_breakout(df,
                        range_period=5,
                        breakout_threshold=1.01):
    """
    Hypothesis 13: N-Day Range Breakout

    Economic Rationale:
    After consolidation (narrow range), a breakout often leads to
    continuation as new trends begin. Works well on daily timeframe.

    Entry Logic:
    - Identify N-day range
    - Enter on breakout above/below range
    - Requires close beyond range (not just wick)

    Parameters FIXED:
    - range_period: 5 days
    - breakout_threshold: 1% beyond range
    """
    df = df.copy()

    df['range_high'] = df['high'].rolling(range_period).max().shift(1)
    df['range_low'] = df['low'].rolling(range_period).min().shift(1)

    # Breakout with threshold
    df['long_entry'] = df['close'] > df['range_high'] * breakout_threshold
    df['short_entry'] = df['close'] < df['range_low'] * (2 - breakout_threshold)

    return df


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(all_data, signal_func, hypothesis_name,
                 initial_capital=100000,
                 position_size_pct=0.10,
                 stop_loss_pct=0.05,
                 take_profit_pct=0.05,
                 max_hold_bars=48):
    """
    Run backtest for a hypothesis across all symbols.

    Uses simple simulation to match research requirements:
    - Fixed stop loss and take profit
    - Max hold period
    - No position limits (each symbol independent for cleaner analysis)
    """
    all_trades = []
    signal_counts = {'long': 0, 'short': 0}

    for symbol, df in all_data.items():
        if len(df) < 100:
            continue

        # Apply signal function
        df = signal_func(df)

        # Count signals
        signal_counts['long'] += df['long_entry'].sum()
        signal_counts['short'] += df['short_entry'].sum()

        # Simulate trades
        position = None

        for i in range(50, len(df)):  # Skip warmup
            row = df.iloc[i]

            # Check exits first
            if position is not None:
                bars_held = i - position['entry_bar']
                exit_triggered = False
                exit_price = row['close']
                exit_reason = None

                if position['direction'] == 'long':
                    # Stop loss
                    if row['low'] <= position['stop']:
                        exit_triggered = True
                        exit_price = position['stop']
                        exit_reason = 'stop_loss'
                    # Take profit
                    elif row['high'] >= position['take']:
                        exit_triggered = True
                        exit_price = position['take']
                        exit_reason = 'take_profit'
                else:  # short
                    # Stop loss (price goes up)
                    if row['high'] >= position['stop']:
                        exit_triggered = True
                        exit_price = position['stop']
                        exit_reason = 'stop_loss'
                    # Take profit (price goes down)
                    elif row['low'] <= position['take']:
                        exit_triggered = True
                        exit_price = position['take']
                        exit_reason = 'take_profit'

                # Max hold
                if not exit_triggered and bars_held >= max_hold_bars:
                    exit_triggered = True
                    exit_reason = 'max_hold'

                if exit_triggered:
                    if position['direction'] == 'long':
                        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

                    all_trades.append({
                        'symbol': symbol,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'bars_held': bars_held,
                        'entry_time': df.index[position['entry_bar']],
                        'exit_time': df.index[i]
                    })
                    position = None

            # Check entries (only if no position)
            if position is None:
                entry_price = row['close']

                if row['long_entry']:
                    position = {
                        'direction': 'long',
                        'entry_price': entry_price,
                        'stop': entry_price * (1 - stop_loss_pct),
                        'take': entry_price * (1 + take_profit_pct),
                        'entry_bar': i
                    }
                elif row['short_entry']:
                    position = {
                        'direction': 'short',
                        'entry_price': entry_price,
                        'stop': entry_price * (1 + stop_loss_pct),
                        'take': entry_price * (1 - take_profit_pct),
                        'entry_bar': i
                    }

    return all_trades, signal_counts


def calculate_metrics(trades, hypothesis_name):
    """Calculate performance metrics from trade list"""
    if not trades:
        return {
            'hypothesis': hypothesis_name,
            'total_trades': 0,
            'win_rate': 0,
            'avg_pnl_pct': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'long_trades': 0,
            'short_trades': 0,
            'verdict': 'INSUFFICIENT_DATA'
        }

    df = pd.DataFrame(trades)

    winners = df[df['pnl_pct'] > 0]
    losers = df[df['pnl_pct'] <= 0]

    total_trades = len(df)
    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
    avg_pnl = df['pnl_pct'].mean() * 100

    gross_profit = winners['pnl_pct'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['pnl_pct'].sum()) if len(losers) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = winners['pnl_pct'].mean() if len(winners) > 0 else 0
    avg_loss = losers['pnl_pct'].mean() if len(losers) > 0 else 0
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    # Exit reason breakdown
    exit_reasons = df['exit_reason'].value_counts().to_dict()

    long_trades = len(df[df['direction'] == 'long'])
    short_trades = len(df[df['direction'] == 'short'])

    return {
        'hypothesis': hypothesis_name,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_pnl_pct': avg_pnl,
        'profit_factor': profit_factor,
        'expectancy': expectancy * 100,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'exit_reasons': exit_reasons,
        'avg_bars_held': df['bars_held'].mean()
    }


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def run_walk_forward(all_data, signal_func, hypothesis_name,
                      window_months=1, n_windows=6):
    """
    Run walk-forward validation.

    Splits data into N consecutive windows and tests each independently.
    No parameter changes between windows.
    """
    # Get date range from data
    all_dates = []
    for symbol, df in all_data.items():
        all_dates.extend(df.index.tolist())

    min_date = min(all_dates)
    max_date = max(all_dates)

    # Create windows
    total_days = (max_date - min_date).days
    window_days = total_days // n_windows

    window_results = []

    for i in range(n_windows):
        window_start = min_date + timedelta(days=i * window_days)
        window_end = min_date + timedelta(days=(i + 1) * window_days)

        # Filter data for this window
        window_data = {}
        for symbol, df in all_data.items():
            mask = (df.index >= window_start) & (df.index < window_end)
            window_df = df[mask]
            if len(window_df) >= 50:
                window_data[symbol] = window_df

        if not window_data:
            continue

        # Run backtest for this window
        trades, _ = run_backtest(window_data, signal_func, hypothesis_name)
        metrics = calculate_metrics(trades, hypothesis_name)

        metrics['window'] = i + 1
        metrics['start'] = window_start.strftime('%Y-%m-%d')
        metrics['end'] = window_end.strftime('%Y-%m-%d')

        window_results.append(metrics)

    return window_results


def evaluate_walk_forward(window_results, hypothesis_name):
    """
    Evaluate walk-forward results against pass/fail criteria.

    Pass Criteria:
    - Positive expectancy in >= 4 of 6 months
    - No single month contributes > 50% of total P&L
    - Win rate between 35-65%
    - Positive overall profit factor

    Archival Triggers:
    - Negative expectancy in 3+ consecutive months
    - < 10 trades per month average
    """
    if not window_results:
        return {
            'hypothesis': hypothesis_name,
            'verdict': 'ARCHIVE',
            'reason': 'No walk-forward data'
        }

    n_windows = len(window_results)
    positive_windows = sum(1 for w in window_results if w['expectancy'] > 0)
    total_trades = sum(w['total_trades'] for w in window_results)
    avg_trades_per_window = total_trades / n_windows if n_windows > 0 else 0

    # Check for consecutive negative months
    max_consecutive_negative = 0
    current_consecutive = 0
    for w in window_results:
        if w['expectancy'] <= 0:
            current_consecutive += 1
            max_consecutive_negative = max(max_consecutive_negative, current_consecutive)
        else:
            current_consecutive = 0

    # Calculate aggregate metrics
    all_win_rates = [w['win_rate'] for w in window_results if w['total_trades'] > 0]
    avg_win_rate = np.mean(all_win_rates) if all_win_rates else 0

    all_pf = [w['profit_factor'] for w in window_results if w['total_trades'] > 0 and w['profit_factor'] != float('inf')]
    avg_pf = np.mean(all_pf) if all_pf else 0

    # Check if any single window dominates
    window_pnls = [w['avg_pnl_pct'] * w['total_trades'] for w in window_results]
    total_pnl = sum(window_pnls)
    max_window_contribution = max(abs(p) for p in window_pnls) / abs(total_pnl) if total_pnl != 0 else 1

    # Determine verdict
    archival_reasons = []
    proceed_reasons = []

    # Archival triggers
    if max_consecutive_negative >= 3:
        archival_reasons.append(f"{max_consecutive_negative} consecutive negative windows")

    if avg_trades_per_window < 10:
        archival_reasons.append(f"Insufficient trades ({avg_trades_per_window:.1f}/window)")

    # Pass criteria
    if positive_windows >= n_windows * 0.6:
        proceed_reasons.append(f"{positive_windows}/{n_windows} windows positive")
    else:
        archival_reasons.append(f"Only {positive_windows}/{n_windows} windows positive")

    if 35 <= avg_win_rate <= 65:
        proceed_reasons.append(f"Win rate {avg_win_rate:.1f}% in healthy range")
    elif avg_win_rate > 65:
        archival_reasons.append(f"Win rate {avg_win_rate:.1f}% suspiciously high (curve-fit risk)")
    else:
        archival_reasons.append(f"Win rate {avg_win_rate:.1f}% too low")

    if avg_pf > 1.0:
        proceed_reasons.append(f"Profit factor {avg_pf:.2f} > 1")
    else:
        archival_reasons.append(f"Profit factor {avg_pf:.2f} <= 1")

    if max_window_contribution <= 0.5:
        proceed_reasons.append("No single window dominates")
    else:
        archival_reasons.append(f"Single window contributes {max_window_contribution*100:.0f}% of P&L")

    # Final verdict
    if archival_reasons:
        verdict = 'ARCHIVE'
        reason = '; '.join(archival_reasons)
    else:
        verdict = 'PROCEED'
        reason = '; '.join(proceed_reasons)

    return {
        'hypothesis': hypothesis_name,
        'verdict': verdict,
        'reason': reason,
        'positive_windows': positive_windows,
        'total_windows': n_windows,
        'avg_trades_per_window': avg_trades_per_window,
        'avg_win_rate': avg_win_rate,
        'avg_profit_factor': avg_pf,
        'max_consecutive_negative': max_consecutive_negative,
        'window_results': window_results
    }


# =============================================================================
# MAIN RESEARCH EXECUTION
# =============================================================================

def print_results(metrics, signal_counts):
    """Pretty print results"""
    print(f"\n  {'='*50}")
    print(f"  {metrics['hypothesis']}")
    print(f"  {'='*50}")
    print(f"  Signals Generated: {signal_counts['long']} long, {signal_counts['short']} short")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.1f}%")
    print(f"  Avg P&L: {metrics['avg_pnl_pct']:.2f}%")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Expectancy: {metrics['expectancy']:.2f}%")
    print(f"  Avg Bars Held: {metrics.get('avg_bars_held', 0):.1f}")

    if 'exit_reasons' in metrics:
        print(f"  Exit Breakdown: {metrics['exit_reasons']}")


def print_walk_forward_results(evaluation):
    """Pretty print walk-forward evaluation"""
    print(f"\n  {'='*60}")
    print(f"  WALK-FORWARD VALIDATION: {evaluation['hypothesis']}")
    print(f"  {'='*60}")

    print(f"\n  Window Results:")
    for w in evaluation.get('window_results', []):
        status = "+" if w['expectancy'] > 0 else "-"
        print(f"    [{status}] Window {w['window']}: {w['start']} to {w['end']}")
        print(f"        Trades: {w['total_trades']}, WR: {w['win_rate']:.1f}%, Exp: {w['expectancy']:.2f}%")

    print(f"\n  Summary:")
    print(f"    Positive Windows: {evaluation['positive_windows']}/{evaluation['total_windows']}")
    print(f"    Avg Trades/Window: {evaluation['avg_trades_per_window']:.1f}")
    print(f"    Avg Win Rate: {evaluation['avg_win_rate']:.1f}%")
    print(f"    Avg Profit Factor: {evaluation['avg_profit_factor']:.2f}")
    print(f"    Max Consecutive Negative: {evaluation['max_consecutive_negative']}")

    print(f"\n  {'='*60}")
    verdict_color = "" if evaluation['verdict'] == 'ARCHIVE' else ""
    print(f"  VERDICT: {evaluation['verdict']}")
    print(f"  REASON: {evaluation['reason']}")
    print(f"  {'='*60}")


def run_parameter_sweep(all_data, signal_func, hypothesis_name, long_only=False):
    """
    Test a hypothesis with different exit parameters.
    NOT optimization - just testing robustness across reasonable parameter ranges.
    """
    configs = [
        # (SL%, TP%, max_hold, description)
        (0.03, 0.03, 24, "Tight 3/3, 24h"),
        (0.05, 0.05, 48, "Standard 5/5, 48h"),
        (0.05, 0.10, 72, "Asymmetric 5/10, 72h"),  # Let winners run
        (0.03, 0.09, 72, "Trend-follow 3/9, 72h"),  # 3:1 R:R
        (0.07, 0.07, 48, "Wider 7/7, 48h"),
    ]

    results = []
    for sl, tp, max_hold, desc in configs:
        trades, _ = run_backtest(
            all_data, signal_func, hypothesis_name,
            stop_loss_pct=sl, take_profit_pct=tp, max_hold_bars=max_hold
        )

        # Filter to long-only if specified
        if long_only and trades:
            trades = [t for t in trades if t['direction'] == 'long']

        metrics = calculate_metrics(trades, f"{hypothesis_name} ({desc})")
        metrics['config'] = desc
        metrics['sl'] = sl
        metrics['tp'] = tp
        metrics['max_hold'] = max_hold
        results.append(metrics)

    return results


def test_all_hypotheses(all_data, hypotheses, sl=0.05, tp=0.05, max_hold=48):
    """Test all hypotheses with given parameters and return sorted results."""
    results = []

    for name, signal_func in hypotheses:
        trades, signal_counts = run_backtest(
            all_data, signal_func, name,
            stop_loss_pct=sl, take_profit_pct=tp, max_hold_bars=max_hold
        )
        metrics = calculate_metrics(trades, name)
        metrics['signals'] = signal_counts['long'] + signal_counts['short']
        results.append(metrics)

    # Sort by profit factor (descending)
    results.sort(key=lambda x: x['profit_factor'] if x['profit_factor'] != float('inf') else 0, reverse=True)
    return results


def main():
    print("\n" + "="*70)
    print("  HYPOTHESIS RESEARCH FRAMEWORK - BATCH 3")
    print("  Testing All Hypotheses + Daily Timeframe")
    print("="*70)

    # Configuration
    symbols = load_symbols(limit=50)

    # Use 6 months of data for walk-forward (6 x 1-month windows)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f"\n  Configuration:")
    print(f"  - Symbols: {len(symbols)}")
    print(f"  - Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Fetch data
    print(f"\n  Loading data...")
    all_data = fetch_data(symbols, start_date, end_date, interval='1h')

    if not all_data:
        print("  ERROR: No data loaded!")
        return

    # ALL HYPOTHESES
    all_hypotheses = [
        ('H1: Vol Compression', h1_volatility_compression),
        ('H2: Liquidity Sweep', h2_liquidity_sweep),
        ('H3: Range Failure', h3_session_range_failure),
        ('H4: Extreme Reversal', h4_extreme_move_reversal),
        ('H5: Volume Divergence', h5_volume_divergence),
        ('H6: Trend Pullback', h6_trend_pullback),
        ('H7: Mom Acceleration', h7_momentum_acceleration),
        ('H8: Gap Fade', h8_gap_fade),
        ('H9: RSI Divergence', h9_rsi_divergence),
        ('H10: Inside Bar', h10_inside_bar),
        ('H11: Trend Structure', h11_trend_structure),
        ('H12: Filtered MR', h12_filtered_mean_reversion),
        ('H13: Range Breakout', h13_range_breakout),
    ]

    # =========================================================================
    # TEST 1: HOURLY DATA - Standard 5/5 exits
    # =========================================================================
    print(f"\n\n{'='*70}")
    print("  TEST 1: HOURLY DATA - Standard Exits (5% SL / 5% TP)")
    print(f"{'='*70}")

    results_hourly = test_all_hypotheses(all_data, all_hypotheses, sl=0.05, tp=0.05, max_hold=48)

    print(f"\n  {'Hypothesis':<25} {'Signals':>8} {'Trades':>8} {'WR%':>8} {'PF':>8} {'Exp%':>8}")
    print(f"  {'-'*75}")

    for r in results_hourly:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        marker = "*" if r['profit_factor'] > 1.1 and r['total_trades'] >= 50 else " "
        print(f" {marker}{r['hypothesis']:<25} {r['signals']:>8} {r['total_trades']:>8} {r['win_rate']:>7.1f}% {pf_str:>8} {r['expectancy']:>7.2f}%")

    # =========================================================================
    # TEST 2: HOURLY DATA - Asymmetric 3:1 R:R
    # =========================================================================
    print(f"\n\n{'='*70}")
    print("  TEST 2: HOURLY DATA - Asymmetric R:R (3% SL / 9% TP)")
    print(f"{'='*70}")

    results_asymm = test_all_hypotheses(all_data, all_hypotheses, sl=0.03, tp=0.09, max_hold=72)

    print(f"\n  {'Hypothesis':<25} {'Signals':>8} {'Trades':>8} {'WR%':>8} {'PF':>8} {'Exp%':>8}")
    print(f"  {'-'*75}")

    for r in results_asymm:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        marker = "*" if r['profit_factor'] > 1.1 and r['total_trades'] >= 50 else " "
        print(f" {marker}{r['hypothesis']:<25} {r['signals']:>8} {r['total_trades']:>8} {r['win_rate']:>7.1f}% {pf_str:>8} {r['expectancy']:>7.2f}%")

    # =========================================================================
    # TEST 3: DAILY DATA
    # =========================================================================
    print(f"\n\n{'='*70}")
    print("  TEST 3: DAILY DATA")
    print(f"{'='*70}")

    print(f"\n  Fetching daily data...")
    # Longer period for daily data
    daily_start = end_date - timedelta(days=365)
    daily_data = fetch_data(symbols, daily_start, end_date, interval='1d')

    if daily_data:
        results_daily = test_all_hypotheses(daily_data, all_hypotheses, sl=0.05, tp=0.10, max_hold=20)

        print(f"\n  {'Hypothesis':<25} {'Signals':>8} {'Trades':>8} {'WR%':>8} {'PF':>8} {'Exp%':>8}")
        print(f"  {'-'*75}")

        for r in results_daily:
            pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
            marker = "*" if r['profit_factor'] > 1.1 and r['total_trades'] >= 50 else " "
            print(f" {marker}{r['hypothesis']:<25} {r['signals']:>8} {r['total_trades']:>8} {r['win_rate']:>7.1f}% {pf_str:>8} {r['expectancy']:>7.2f}%")

    # =========================================================================
    # IDENTIFY TOP PERFORMERS FOR WALK-FORWARD
    # =========================================================================
    print(f"\n\n{'='*70}")
    print("  TOP PERFORMERS - Running Walk-Forward Validation")
    print(f"{'='*70}")

    # Find candidates with PF > 1.1 and sufficient trades
    candidates = []

    for r in results_hourly:
        if r['profit_factor'] > 1.1 and r['total_trades'] >= 50:
            candidates.append(('hourly_std', r['hypothesis'], 0.05, 0.05, 48, all_data))

    for r in results_asymm:
        if r['profit_factor'] > 1.1 and r['total_trades'] >= 50:
            candidates.append(('hourly_3:1', r['hypothesis'], 0.03, 0.09, 72, all_data))

    if daily_data:
        for r in results_daily:
            if r['profit_factor'] > 1.1 and r['total_trades'] >= 50:
                candidates.append(('daily', r['hypothesis'], 0.05, 0.10, 20, daily_data))

    all_best = []

    for config_name, hypo_name, sl, tp, mh, data in candidates:
        # Find the signal function
        signal_func = None
        for name, func in all_hypotheses:
            if name == hypo_name:
                signal_func = func
                break

        if not signal_func:
            continue

        print(f"\n  Walk-forward: {hypo_name} ({config_name})")

        # Run walk-forward
        all_dates = []
        for symbol, df in data.items():
            all_dates.extend(df.index.tolist())
        min_date = min(all_dates)
        max_date = max(all_dates)
        total_days = (max_date - min_date).days
        window_days = total_days // 6

        window_results = []
        for i in range(6):
            window_start = min_date + timedelta(days=i * window_days)
            window_end = min_date + timedelta(days=(i + 1) * window_days)

            window_data = {}
            min_bars = 20 if config_name == 'daily' else 50
            for symbol, df in data.items():
                mask = (df.index >= window_start) & (df.index < window_end)
                window_df = df[mask]
                if len(window_df) >= min_bars:
                    window_data[symbol] = window_df

            if window_data:
                trades, _ = run_backtest(
                    window_data, signal_func, hypo_name,
                    stop_loss_pct=sl, take_profit_pct=tp, max_hold_bars=mh
                )
                metrics = calculate_metrics(trades, hypo_name)
                metrics['window'] = i + 1
                window_results.append(metrics)

        positive_windows = sum(1 for w in window_results if w['expectancy'] > 0)
        total_exp = sum(w['expectancy'] for w in window_results)
        avg_pf = np.mean([w['profit_factor'] for w in window_results if w['profit_factor'] != float('inf') and w['total_trades'] > 0])

        for w in window_results:
            status = "+" if w['expectancy'] > 0 else "-"
            print(f"    [{status}] W{w['window']}: {w['total_trades']:>3} trades, WR: {w['win_rate']:>5.1f}%, Exp: {w['expectancy']:>6.2f}%")

        print(f"    => Positive: {positive_windows}/6, Avg PF: {avg_pf:.2f}")

        all_best.append({
            'name': f"{hypo_name} ({config_name})",
            'positive_windows': positive_windows,
            'avg_pf': avg_pf,
            'total_exp': total_exp
        })

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)

    if all_best:
        all_best.sort(key=lambda x: (x['positive_windows'], x['avg_pf']), reverse=True)

        for b in all_best:
            verdict = "PROCEED" if b['positive_windows'] >= 4 and b['avg_pf'] > 1.05 else "ARCHIVE"
            print(f"\n  [{verdict}] {b['name']}")
            print(f"           Positive Windows: {b['positive_windows']}/6, Avg PF: {b['avg_pf']:.2f}, Total Exp: {b['total_exp']:.2f}%")

        proceed_count = sum(1 for b in all_best if b['positive_windows'] >= 4 and b['avg_pf'] > 1.05)

        print("\n" + "-"*70)
        if proceed_count > 0:
            print(f"  RECOMMENDATION: {proceed_count} strategy(s) warrant extended validation")
        else:
            print("  RECOMMENDATION: No strategies passed walk-forward. Continue research.")
        print("-"*70 + "\n")
    else:
        print("\n  No candidates met initial PF > 1.1 threshold")
        print("  RECOMMENDATION: Continue hypothesis development")
        print("-"*70 + "\n")


if __name__ == '__main__':
    main()
