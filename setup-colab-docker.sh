#!/bin/bash
set -euo pipefail

echo "=============================="
echo "STEP 1: Refresh apt metadata"
echo "=============================="

apt-get update

echo "=============================="
echo "STEP 2: Install Docker packages"
echo "=============================="

DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates \
  curl \
  docker.io \
  docker-compose-v2

echo "=============================="
echo "STEP 3: Start Docker daemon"
echo "=============================="

if docker info >/dev/null 2>&1; then
  echo "Docker daemon is already running."
else
  rm -f /var/run/docker.pid
  nohup dockerd > /tmp/dockerd.log 2>&1 &

  for attempt in $(seq 1 30); do
    if docker info >/dev/null 2>&1; then
      echo "Docker daemon is ready."
      break
    fi
    sleep 2
  done

  if ! docker info >/dev/null 2>&1; then
    echo "Docker failed to start. Dumping /tmp/dockerd.log"
    cat /tmp/dockerd.log
    exit 1
  fi
fi

echo "=============================="
echo "STEP 4: Show Docker versions"
echo "=============================="

docker --version
docker compose version

echo "=============================="
echo "STEP 5: Test Docker"
echo "=============================="

docker run --rm hello-world

echo "=============================="
echo "STEP 6: Notes for CPU-only OCR"
echo "=============================="

echo "Google Colab does not need NVIDIA Container Toolkit for this setup."
echo "Use OCR_DEVICE=cpu in .env when running docker compose."
echo "Recommended CPU-safe Paddle flags:"
echo "  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"
echo "  FLAGS_use_mkldnn=0"
echo "  FLAGS_enable_pir_api=0"
echo "  FLAGS_enable_pir_in_executor=0"

echo "=============================="
echo "SUCCESS: Docker setup complete for Google Colab CPU runtime"
echo "=============================="