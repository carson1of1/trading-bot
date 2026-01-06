# Trading Bot UI Design

## Overview

A professional, futuristic dashboard for monitoring and controlling the trading bot. Features a morphic glass aesthetic with a floating dock navigation system.

## Visual Design

### Theme
- **Primary background:** `#0a0a0a` (near-black)
- **Glass surfaces:** `rgba(255,255,255,0.05)` with `backdrop-blur-xl`
- **Accent color:** `#10b981` (emerald green) - highlights, active states, positive P&L
- **Negative/loss:** `#ef4444` (red)
- **Text:** White primary, `#a1a1aa` (zinc-400) secondary

### Glass Morphism Style
- Cards float with subtle `border border-white/10`
- Soft glow effects on hover (`shadow-lg shadow-emerald-500/10`)
- Frosted glass blur behind all panels
- Subtle rounded corners (`rounded-2xl`)

### Typography
- Font: Inter or Geist
- Large bold numbers for key metrics
- No decorative elements, no emojis

## Navigation

### Floating Dock (Bottom Center)
- Fixed to bottom center of screen
- Glass container with 5 category icons
- Hover reveals category name + submenu slides up
- Active page icon glows green
- Auto-hides on scroll down, reappears on scroll up

### Dock Structure

| Icon | Category | Pages |
|------|----------|-------|
| Home | Home | Dashboard |
| Chart | Portfolio | Positions, Trade History |
| Search | Markets | Scanner, Market Overview |
| TrendingUp | Insights | Analytics, Backtesting, Strategies |
| Settings | System | Risk Monitor, Activity Feed, Settings |

## Pages

### 1. Dashboard (Home)

**Top Row - Key Metrics (4 cards):**
- Account Balance: Large dollar amount, +/- percentage below
- Today's P&L: Dollar + percentage, green/red based on value
- Open Positions: Count with symbol list underneath
- Win Rate: Percentage with "X/Y trades" subtitle

**Middle Row:**
- Bot Status Card (1/3 width):
  - Pulsing dot: green=running, amber=paused, red=error
  - Mode badge: PAPER / LIVE / DRY_RUN
  - Last action timestamp
  - Start/Stop button
- Equity Curve Card (2/3 width):
  - Sparkline of portfolio value (30 days)
  - Hover for exact values
  - Toggle: 7D / 30D / 90D / ALL

**Bottom Row:**
- Mini positions table (5 rows max): Symbol, Side, P&L%

### 2. Positions (Portfolio)

- Header: "Open Positions" with count badge
- Table columns:
  - Symbol (clickable for detail modal)
  - Side (LONG/SHORT badge)
  - Entry Price
  - Current Price
  - P&L $ and %
  - Stop Loss / Take Profit
  - Hold Time
  - Strategy
- Row styling: Subtle green/red left border based on P&L
- Empty state: Glass card with "No open positions"

### 3. Trade History (Portfolio)

- Filters bar (glass, sticky):
  - Date range picker
  - Symbol search
  - Side filter (All / Long / Short)
  - Strategy filter
  - Win/Loss toggle
- Table columns:
  - Date/Time
  - Symbol
  - Side
  - Entry → Exit price
  - P&L $ and %
  - Hold Duration
  - Strategy
- Infinite scroll pagination
- CSV export button

### 4. Scanner (Markets)

- Header: "Volatility Scanner" with last refresh timestamp
- Auto-refresh every minute with pulse animation
- Table (top 10):
  - Rank
  - Symbol
  - Price
  - ATR %
  - Daily Range %
  - Volume vs Average
  - Composite Score
- Row click expands for mini chart + watchlist status
- Manual refresh button

### 5. Market Overview (Markets)

**Top - Index Cards (3):**
- SPY: Price, % change, sparkline
- QQQ: Same
- VIX: Same (red tint when elevated)

**Middle - Sector Heat Map:**
- Grid of sector boxes
- Color intensity = performance
- Click to see sector movers

**Bottom - Two columns:**
- Left: Top Movers from watchlist
- Right: Active Signals (e.g., "BUY AAPL - Momentum 78%")

### 6. Analytics (Insights)

**Three stacked charts:**

1. **Equity Curve (largest):**
   - Line chart with green fill
   - Time toggles: 7D / 30D / 90D / YTD / ALL
   - Hover tooltip

2. **Trade Distribution by Hour:**
   - 24 bars (one per hour)
   - Height = trade count
   - Color = win rate (red to green gradient)

3. **Drawdown Chart:**
   - Area chart showing % below peak
   - Red fill
   - Max drawdown label

### 7. Backtesting (Insights)

**Config Panel (left, 1/3):**
- Symbol multi-select
- Date range picker
- Longs only / Shorts only / Both
- "Run Backtest" button

**Results Panel (right, 2/3):**
- Summary cards: Total P&L, Win Rate, Profit Factor, Max Drawdown
- Trade list table
- Equity curve for backtest period

### 8. Strategies (Insights)

**Card per strategy:**
- Enable/disable toggle
- Weight slider
- Stats: Win rate, P&L, trade count
- Last signal info

### 9. Risk Monitor (System)

**Top Row - Risk Cards (4):**
- Daily Loss: Progress toward limit
- Open Risk: Portfolio exposure %
- Losing Trades Today: Count toward lockout
- Largest Position: Symbol + % of portfolio

**Middle - Position Sizing Visualizer:**
- Horizontal bar chart of position sizes
- Max position line overlay
- Color-coded by limit proximity

**Bottom - Drawdown Gauge:**
- Circular gauge
- Zones: Green (0-2%), Amber (2-5%), Red (5%+)

### 10. Activity Feed (System)

- Real-time scrolling log
- Color-coded entries:
  - Green: Trade entries
  - Red: Stop losses, exits
  - Amber: Skipped trades
  - Gray: System messages
- Filter buttons: All / Trades / Skipped / Errors
- Search box

### 11. Settings (System)

**Sections:**
- Mode: PAPER / LIVE / DRY_RUN selector
- Risk Parameters: Editable fields
- API Keys: Masked input fields
- Notifications: Toggle switches

## Technical Architecture

### Project Structure
```
trading-bot/
├── bot.py              # Existing bot
├── api/                # FastAPI layer
│   ├── main.py
│   ├── routes/
│   └── websocket.py
├── frontend/           # Next.js app
│   ├── app/
│   ├── components/
│   └── lib/
```

### Tech Stack
- **Frontend:** Next.js + React + Tailwind CSS + Shadcn/ui
- **Backend API:** FastAPI (Python)
- **Real-time:** WebSocket
- **Database:** Existing SQLite (logs/trades_1hour.db)

### API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | /positions | Current open positions |
| GET | /trades | Trade history with filters |
| GET | /stats | P&L, win rate, account balance |
| GET | /scanner | Current scanner results |
| GET | /strategies | Strategy status and stats |
| POST | /bot/start | Start the bot |
| POST | /bot/stop | Stop the bot |
| WS | /feed | Real-time activity stream |

### Real-time Strategy
- WebSocket for Activity Feed + Position updates
- Poll every 30s for stats, scanner data
