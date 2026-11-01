#!/bin/bash
# EC2 user-data for Amazon Linux 2023: installs Docker and the Compose plugin
# on first boot. Passed to `aws ec2 run-instances --user-data`; see the
# "Deploying to AWS" section of README.md. Progress is logged to
# /var/log/user-data.log; the box is ready when /var/lib/cloud/instance/boot-finished exists.
set -euo pipefail
exec > >(tee -a /var/log/user-data.log) 2>&1

dnf install -y docker
systemctl enable --now docker
usermod -aG docker ec2-user

# AL2023 doesn't package the Compose v2 plugin; install the pinned release binary.
COMPOSE_VERSION=v2.29.7
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

docker --version
docker compose version
