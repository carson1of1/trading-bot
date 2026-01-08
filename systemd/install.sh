#!/bin/bash
# Install trading bot systemd units

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing systemd units..."
sudo cp "$SCRIPT_DIR/trading-bot.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/trading-bot.timer" /etc/systemd/system/

echo "Reloading systemd..."
sudo systemctl daemon-reload

echo "Enabling timer..."
sudo systemctl enable trading-bot.timer

echo ""
echo "Installation complete!"
echo ""
echo "Commands:"
echo "  sudo systemctl start trading-bot.timer   # Start the timer"
echo "  sudo systemctl status trading-bot.timer  # Check timer status"
echo "  sudo systemctl list-timers               # List all timers"
echo "  sudo journalctl -u trading-bot           # View bot logs"
