#!/bin/bash
set -e

# Update package list and install Docker
apt-get update
apt-get install -y docker.io
systemctl enable docker
systemctl start docker

# Define Docker image
DOCKER_IMAGE="adityagupta20/predictive-maintenance-api:latest"

# Pull the Docker image
docker pull "${DOCKER_IMAGE}"

# Run the Docker container
docker run -d \
  --name predictive-maintenance-api \
  -p 80:8000 \
  "${DOCKER_IMAGE}"