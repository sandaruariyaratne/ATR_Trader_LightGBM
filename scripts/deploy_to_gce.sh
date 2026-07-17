#!/bin/bash
set -e

PROJECT_ID="trader-42f"
ZONE="asia-east1-a"
INSTANCE_NAME="atr-trader-vm"

echo "=== 1. Provisioning GCP VM Instance ($INSTANCE_NAME) ==="
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID > /dev/null 2>&1; then
    echo "Instance $INSTANCE_NAME already exists, skipping creation."
else
    echo "Creating instance $INSTANCE_NAME in zone $ZONE..."
    gcloud compute instances create $INSTANCE_NAME \
        --project=$PROJECT_ID \
        --zone=$ZONE \
        --machine-type=e2-medium \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=25GB
fi

echo "Waiting for SSH server on $INSTANCE_NAME to boot up..."
for i in {1..15}; do
    if gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command="echo SSH_READY" > /dev/null 2>&1; then
        echo "SSH connection established successfully!"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "Error: SSH connection timed out."
        exit 1
    fi
    echo "SSH not ready yet, retrying in 5 seconds ($i/15)..."
    sleep 5
done

echo "=== 2. Packaging project files ==="
tar -czf atr_trader.tar.gz \
    --exclude='.venv' \
    --exclude='.git' \
    --exclude='*.csv' \
    --exclude='atr_trader.tar.gz' \
    --exclude='algo_trader.tar.gz' \
    --exclude='logs/*.log' \
    --exclude='pip_cache' \
    --exclude='tmp' \
    --exclude='scratch' \
    .

echo "=== 3. Uploading code archive to VM ==="
gcloud compute scp atr_trader.tar.gz $INSTANCE_NAME:~/ \
    --zone=$ZONE \
    --project=$PROJECT_ID

echo "=== 4. Setting up Docker and running trading bot on VM ==="
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command="
    sudo apt-get update && \
    sudo apt-get install -y docker.io docker-compose && \
    sudo systemctl enable docker && \
    sudo systemctl start docker && \
    sudo usermod -aG docker \$USER && \
    mkdir -p ~/atr_trader && \
    tar -xzf ~/atr_trader.tar.gz -C ~/atr_trader && \
    rm ~/atr_trader.tar.gz && \
    cd ~/atr_trader && \
    sudo sg docker -c 'docker-compose down || true' && \
    sudo sg docker -c 'docker-compose up -d --build'
"

rm -f atr_trader.tar.gz

echo "=== Deployment Completed Successfully! ==="
echo "Monitor live logs with:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command=\"sudo docker logs -f atr-trader-lgbm\""
