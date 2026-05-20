#!/usr/bin/env bash
# terminate the running HippoRAG EC2 instance
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source config.env
export AWS_DEFAULT_REGION="$AWS_REGION"

if [[ ! -f .instance-id ]]; then
    echo "No instance ID found (.instance-id missing)"
    exit 1
fi

INSTANCE_ID=$(cat .instance-id)
echo "Terminating instance: $INSTANCE_ID"

aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"
echo "Waiting for termination..."
aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"

rm -f .instance-id .instance-ip
echo "Done."
