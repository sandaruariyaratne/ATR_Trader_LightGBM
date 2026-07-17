import os
import tarfile
import subprocess
import time

PROJECT_ID = "trader-42f"
ZONE = "asia-east1-a"
INSTANCE_NAME = "atr-trader-vm"

# 1. Provision / check VM
print("=== 1. Checking VM Instance ===")
check_cmd = ["gcloud", "compute", "instances", "describe", INSTANCE_NAME, f"--zone={ZONE}", f"--project={PROJECT_ID}"]
res = subprocess.run(check_cmd, capture_output=True)
if res.returncode == 0:
    print(f"Instance {INSTANCE_NAME} already exists.")
else:
    print(f"Creating instance {INSTANCE_NAME} ...")
    create_cmd = [
        "gcloud", "compute", "instances", "create", INSTANCE_NAME,
        f"--project={PROJECT_ID}", f"--zone={ZONE}",
        "--machine-type=e2-medium", "--image-family=ubuntu-2204-lts",
        "--image-project=ubuntu-os-cloud", "--boot-disk-size=25GB"
    ]
    subprocess.run(create_cmd, check=True)

# Wait for SSH
print("Waiting for SSH server to boot up...")
for i in range(1, 16):
    ssh_check = ["gcloud", "compute", "ssh", INSTANCE_NAME, f"--zone={ZONE}", f"--project={PROJECT_ID}", "--command=echo SSH_READY"]
    res = subprocess.run(ssh_check, capture_output=True)
    if b"SSH_READY" in res.stdout:
        print("SSH connection established successfully!")
        break
    print(f"SSH not ready yet, retrying in 5 seconds ({i}/15)...")
    time.sleep(5)

# 2. Package files
print("=== 2. Packaging project files ===")
tar_path = "atr_trader.tar.gz"

def tar_filter(tarinfo):
    name = tarinfo.name
    # Remove leading './' if present
    if name.startswith("./"):
        name = name[2:]
        
    # Exclude patterns
    excludes = [
        ".venv", ".git", "atr_trader.tar.gz", "algo_trader.tar.gz",
        "pip_cache", "tmp", "scratch"
    ]
    # Check if name contains any excludes
    for ex in excludes:
        if name == ex or name.startswith(ex + "/"):
            return None
    # Exclude all CSV files
    if name.endswith(".csv"):
        return None
    # Exclude log files
    if name.startswith("logs/") and name.endswith(".log"):
        return None
    return tarinfo

with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(".", filter=tar_filter)

print("Package created successfully.")

# 3. Upload code archive
print("=== 3. Uploading code archive to VM ===")
scp_cmd = [
    "gcloud", "compute", "scp", tar_path, f"{INSTANCE_NAME}:~/",
    f"--zone={ZONE}", f"--project={PROJECT_ID}"
]
subprocess.run(scp_cmd, check=True)

# 4. Set up Docker and run on VM
print("=== 4. Setting up Docker and running trading bot on VM ===")
remote_cmd = """
    sudo apt-get update && \
    sudo apt-get install -y docker.io docker-compose && \
    sudo systemctl enable docker && \
    sudo systemctl start docker && \
    sudo usermod -aG docker $USER && \
    mkdir -p ~/atr_trader && \
    tar -xzf ~/atr_trader.tar.gz -C ~/atr_trader && \
    rm ~/atr_trader.tar.gz && \
    cd ~/atr_trader && \
    sudo sg docker -c 'docker-compose down || true' && \
    sudo sg docker -c 'docker-compose up -d --build'
"""
ssh_cmd = [
    "gcloud", "compute", "ssh", INSTANCE_NAME, f"--zone={ZONE}", f"--project={PROJECT_ID}",
    f"--command={remote_cmd}"
]
subprocess.run(ssh_cmd, check=True)

# Clean up
if os.path.exists(tar_path):
    os.remove(tar_path)

print("=== Deployment Completed Successfully! ===")
print("Monitor live logs with:")
print(f"  gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --project={PROJECT_ID} --command=\"sudo docker logs -f atr-trader-avax\"")
