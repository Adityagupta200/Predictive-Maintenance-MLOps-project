#!/bin/bash
set -e

apt-get update
apt-get install -y docker.io
systemctl enable docker
systemctl start docker

DOCKER_IMAGE="${DOCKER_IMAGE}"  # ← Change docker_image to DOCKER_IMAGE (uppercase)

# Pull the Docker image
docker pull "${DOCKER_IMAGE}"

# Run the Docker container
docker run -d \
  --name predictive-maintenance-api \
  -p 80:8000 \
  "${DOCKER_IMAGE}"
