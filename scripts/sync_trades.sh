#!/bin/bash
# scripts/sync_trades.sh
# ──────────────────────
# Syncs the remote trades log from the GCP VM to your local machine.
set -e

PROJECT_ID="atr-trader-lgbm-v1"
ZONE="asia-east1-a"
INSTANCE_NAME="atr-trader-vm"
LOCAL_CSV="data/trades_log.csv"

echo "Fetching remote trades log files from VM..."
gcloud compute scp "$INSTANCE_NAME:~/atr_trader/data/trades_log_*.csv" "data/" \
    --zone=$ZONE \
    --project=$PROJECT_ID

echo "Sync completed successfully!"
echo "Local files updated in data/:"
ls -lh data/trades_log_*.csv
