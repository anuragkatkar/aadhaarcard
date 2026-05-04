#!/bin/bash
set -e

echo "=============================="
echo "STEP 1: Remove old Docker versions"
echo "=============================="

sudo apt remove -y docker docker-engine docker.io containerd runc || true

echo "=============================="
echo "STEP 2: Install dependencies"
echo "=============================="

sudo apt update
sudo apt install -y ca-certificates curl gnupg

echo "=============================="
echo "STEP 3: Add Docker GPG key"
echo "=============================="

sudo install -m 0755 -d /etc/apt/keyrings

sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc

sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "=============================="
echo "STEP 4: Add Docker repository"
echo "=============================="

sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

echo "=============================="
echo "STEP 5: Install Docker"
echo "=============================="

sudo apt update

sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "=============================="
echo "STEP 6: Start Docker"
echo "=============================="

sudo systemctl enable docker
sudo systemctl start docker

docker --version

echo "=============================="
echo "STEP 7: Test Docker"
echo "=============================="

sudo docker run hello-world

echo "=============================="
echo "STEP 8: Install NVIDIA Container Toolkit"
echo "=============================="

sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
 | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
 | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
 | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

echo "=============================="
echo "STEP 9: Configure Docker GPU runtime"
echo "=============================="

sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker

echo "=============================="
echo "STEP 10: Test GPU inside Docker"
echo "=============================="

docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

echo "=============================="
echo "SUCCESS: Docker + GPU setup complete"
echo "=============================="