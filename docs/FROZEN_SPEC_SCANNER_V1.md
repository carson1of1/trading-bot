# Frozen Specification - Baseline v1

> **WARNING:** This configuration is FROZEN. No parameter changes allowed during OOS test period.
> Any modification invalidates all test results.

**Created:** 2026-01-02T22:04:28.202527
**Checksum:** `5c989be6b5b23a88387a02b75fea4378`

---

## 1. Universe Source

| Parameter | Value |
|-----------|-------|
| Source File | `universe.yaml:scanner_universe` |
| Symbol Count | 400 |
| Symbols Hash | `af0c0103b3bbcef7` |
| Categories | sp500, tech_growth, high_volatility, biotech, finance, energy, clean_energy, consumer, industrial, reits, etfs, cannabis, spacs_ipos |

## 2. Scanner Formula

| Parameter | Value |
|-----------|-------|
| Enabled | True |
| Top N | 10 |
| Min Price | $5 |
| Max Price | $1000 |
| Min Volume | 500,000 |
| Lookback Days | 14 |

**Volatility Score Weights:**
- ATR %: 50%
- Daily Range %: 30%
- Volume Ratio: 20%

## 3. Strategy Set

| Strategy | Enabled | Weight |
|----------|---------|--------|
| momentum | Y | 35% |
| mean_reversion | Y | 25% |
| breakout | Y | 20% |
| ml_ensemble | N | - |
| vwap | N | - |

## 4. Risk Parameters

| Parameter | Value |
|-----------|-------|
| Max Position Size | 3.0% |
| Max Portfolio Risk | 15.0% |
| Stop Loss | 5.0% |
| Take Profit | 8.0% |
| Max Daily Loss | 3.0% |
| Max Open Positions | 5 |

## 5. Exit Policy

| Parameter | Value |
|-----------|-------|
| Hard Stop | -5.0% |
| Profit Floor | 2.0% |
| ATR Trailing Activation | 3.0% |
| Partial Take Profit | 4.0% |
| Max Hold Hours | 168 |
| EOD Close | False |

## 6. Trailing Stop

| Parameter | Value |
|-----------|-------|
| Enabled | True |
| Activation | 0.25% |
| Trail Distance | 0.25% |
| Move to Breakeven | True |

## 7. Execution Assumptions

| Parameter | Value |
|-----------|-------|
| Slippage | 5 bps |
| Half Spread | 2 bps |
| Commission | 0.0 |

## 8. Shorting Constraints

| Parameter | Value |
|-----------|-------|
| Shorting Enabled | True |
| Shorts Only Mode | False |

## 9. Entry Gate

| Parameter | Value |
|-----------|-------|
| Confidence Threshold | 60 |
| Max Trades/Symbol/Day | 3 |
| Min Time Between Trades | 60 min |

---

## Validation Command

```bash
python -c "from validation.frozen_spec import FrozenSpec; s=FrozenSpec.from_config_files(); print('Checksum:', s.checksum); assert s.checksum == '5c989be6b5b23a88387a02b75fea4378', 'CONFIG CHANGED!'"
```

## Git Tag

```bash
git tag -a baseline_frozen_v1 -m "Frozen spec: 5c989be6b5b23a88"
```
