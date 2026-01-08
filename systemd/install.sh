#!/bin/bash
# Install trading bot systemd units

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing systemd units..."

# Bot service and timer (auto-start at market open)
sudo cp "$SCRIPT_DIR/trading-bot.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/trading-bot.timer" /etc/systemd/system/

# API service (web dashboard - always running)
sudo cp "$SCRIPT_DIR/trading-bot-api.service" /etc/systemd/system/

# Watchdog service (monitoring - always running)
sudo cp "$SCRIPT_DIR/trading-bot-watchdog.service" /etc/systemd/system/

echo "Reloading systemd..."
sudo systemctl daemon-reload

echo "Enabling services..."
sudo systemctl enable trading-bot.timer
sudo systemctl enable trading-bot-api.service
sudo systemctl enable trading-bot-watchdog.service

echo ""
echo "Installation complete!"
echo ""
echo "Starting API and Watchdog now..."
sudo systemctl start trading-bot-api.service
sudo systemctl start trading-bot-watchdog.service

echo ""
echo "Status:"
echo "======="
sudo systemctl status trading-bot-api.service --no-pager -l || true
echo ""
sudo systemctl status trading-bot-watchdog.service --no-pager -l || true
echo ""
sudo systemctl status trading-bot.timer --no-pager -l || true

echo ""
echo "Commands:"
echo "  sudo systemctl start trading-bot.timer      # Enable auto-start at 9:25 AM ET"
echo "  sudo systemctl stop trading-bot.timer       # Disable auto-start"
echo "  sudo systemctl restart trading-bot-api      # Restart API"
echo "  sudo systemctl restart trading-bot-watchdog # Restart watchdog"
echo "  sudo journalctl -u trading-bot -f           # View bot logs"
echo "  sudo journalctl -u trading-bot-api -f       # View API logs"
echo "  sudo journalctl -u trading-bot-watchdog -f  # View watchdog logs"
