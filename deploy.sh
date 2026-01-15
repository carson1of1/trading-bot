#!/bin/bash
# Deploy trading bot to AWS
# Usage: ./deploy.sh

set -e

AWS_HOST="ubuntu@13.58.138.4"
KEY="$HOME/Downloads/trading-bot-key.pem"
REMOTE_DIR="~/trading-bot"

echo "Deploying to AWS..."

# Sync files (excluding logs, cache, git, pycache)
rsync -avz --delete \
  -e "ssh -i $KEY" \
  --exclude 'logs/' \
  --exclude 'data/cache/' \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'venv/' \
  --exclude 'frontend/node_modules/' \
  --exclude 'frontend/.next/' \
  /home/carsonodell/projects/trading-bot/ $AWS_HOST:$REMOTE_DIR/

echo "Restarting services..."

# Restart both services
ssh -i $KEY $AWS_HOST "sudo systemctl restart trading-bot trading-api"

echo "Checking status..."
ssh -i $KEY $AWS_HOST "sudo systemctl status trading-bot --no-pager -l | head -15"

echo ""
echo "Deploy complete!"
