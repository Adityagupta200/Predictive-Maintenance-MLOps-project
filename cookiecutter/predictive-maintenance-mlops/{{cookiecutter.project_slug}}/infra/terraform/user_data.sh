#!/bin/bash
set -e

# Update package list and install Docker
apt-get update
apt-get install -y docker.io
systemctl enable docker
systemctl start docker

# Docker image injected by Terraform templatefile (repo:tag), e.g.
# adityagupta20/predictive-maintenance-api:<commit-sha>
DOCKER_IMAGE="${docker_image}"

# Pull the Docker image
docker pull "${DOCKER_IMAGE}"

# Run the Docker container
docker run -d \
  --name predictive-maintenance-api \
  -p 80:8000 \
  "${DOCKER_IMAGE}"
