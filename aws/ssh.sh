#!/usr/bin/env bash
# SSH into running HippoRAG EC2 instance
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source config.env

if [[ ! -f .instance-ip ]]; then
    echo "No instance IP found. Is an instance running? Try: ./launch.sh"
    exit 1
fi

IP=$(cat .instance-ip)
echo "Connecting to $IP..."

ssh -i ~/.ssh/"$KEY_NAME".pem \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -t \
    ubuntu@"$IP" "$@"
