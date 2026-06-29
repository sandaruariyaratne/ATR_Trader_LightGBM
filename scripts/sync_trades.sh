#!/bin/bash
# scripts/sync_trades.sh
# ──────────────────────
# Syncs the remote trades log from the GCP VM to your local machine.
set -e

PROJECT_ID="atr-trader-lgbm-v1"
ZONE="asia-east1-a"
INSTANCE_NAME="atr-trader-vm"
LOCAL_CSV="data/trades_log.csv"

echo "Fetching remote trades_log.csv from VM..."
gcloud compute ssh $INSTANCE_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="cat ~/atr_trader/data/trades_log.csv" > "$LOCAL_CSV"

echo "Sync completed successfully!"
echo "Local file updated: $LOCAL_CSV"
echo "Total rows: $(wc -l < "$LOCAL_CSV" | tr -d ' ')"
