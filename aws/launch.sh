#!/usr/bin/env bash
# packages code, uploads to S3, launches EC2 instance with bootstrap userdata
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f config.env ]]; then
    echo "ERROR: config.env not found. Copy config.env.example and fill in values."
    exit 1
fi
source config.env

for var in S3_BUCKET AWS_REGION INSTANCE_TYPE KEY_NAME SECURITY_GROUP_NAME INSTANCE_PROFILE_NAME OPENAI_API_KEY; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: $var is not set in config.env"
        exit 1
    fi
done

export AWS_DEFAULT_REGION="$AWS_REGION"

RUN_ID=$(date +%Y%m%d-%H%M%S)
echo "$RUN_ID" > .last-run-id
echo "Run ID: $RUN_ID"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Repo root: $REPO_ROOT"

# tar up code, excluding non-essentials
echo "Packaging code..."
TARBALL="/tmp/hipporag-code.tar.gz"
tar -czf "$TARBALL" \
    -C "$REPO_ROOT" \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='.env' \
    --exclude='aws' \
    --exclude='*.pyc' \
    --exclude='.git' \
    .

TARBALL_SIZE=$(du -h "$TARBALL" | cut -f1)
echo "Tarball size: $TARBALL_SIZE"

echo "Uploading code to S3..."
aws s3 cp "$TARBALL" "s3://$S3_BUCKET/code/hipporag-code.tar.gz"

echo "Uploading bootstrap scripts..."
aws s3 sync "$SCRIPT_DIR/remote/" "s3://$S3_BUCKET/kit/" --delete

echo "Uploading config..."
aws s3 cp "$SCRIPT_DIR/config.env" "s3://$S3_BUCKET/kit/config.env"

# latest Ubuntu 22.04 AMI (no GPU needed, all inference via OpenAI API)
echo "Finding latest Ubuntu 22.04 AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
              "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [[ "$AMI_ID" == "None" || -z "$AMI_ID" ]]; then
    echo "ERROR: Could not find Ubuntu 22.04 AMI"
    exit 1
fi
echo "Using AMI: $AMI_ID"

USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -euxo pipefail

exec > /var/log/hipporag-user-data.log 2>&1
echo "User-data script started at $(date)"

apt-get update -qq
apt-get install -y -qq awscli

export S3_BUCKET="${S3_BUCKET}"
export AWS_REGION="${AWS_REGION}"
export AUTO_TERMINATE="${AUTO_TERMINATE}"
export RUN_ID="${RUN_ID}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"

aws s3 cp "s3://${S3_BUCKET}/kit/bootstrap.sh" /tmp/bootstrap.sh
chmod +x /tmp/bootstrap.sh

sudo -u ubuntu --preserve-env=S3_BUCKET,AWS_REGION,AUTO_TERMINATE,RUN_ID,OPENAI_API_KEY \
    bash /tmp/bootstrap.sh

echo "User-data script completed at $(date)"
USERDATA
)

# substitute env vars into userdata
USER_DATA="${USER_DATA//\$\{S3_BUCKET\}/$S3_BUCKET}"
USER_DATA="${USER_DATA//\$\{AWS_REGION\}/$AWS_REGION}"
USER_DATA="${USER_DATA//\$\{AUTO_TERMINATE\}/${AUTO_TERMINATE:-true}}"
USER_DATA="${USER_DATA//\$\{RUN_ID\}/$RUN_ID}"
USER_DATA="${USER_DATA//\$\{OPENAI_API_KEY\}/$OPENAI_API_KEY}"

LAUNCH_ARGS=(
    --image-id "$AMI_ID"
    --instance-type "$INSTANCE_TYPE"
    --key-name "$KEY_NAME"
    --security-groups "$SECURITY_GROUP_NAME"
    --iam-instance-profile "Name=$INSTANCE_PROFILE_NAME"
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${EBS_SIZE_GB:-100},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"
    --user-data "$USER_DATA"
    --metadata-options "HttpTokens=required,HttpEndpoint=enabled"
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=hipporag-$RUN_ID}]"
)

if [[ "${USE_SPOT:-false}" == "true" ]]; then
    echo "Using spot instance (max price: \$${SPOT_MAX_PRICE:-1.50}/hr)"
    LAUNCH_ARGS+=(
        --instance-market-options "MarketType=spot,SpotOptions={MaxPrice=${SPOT_MAX_PRICE:-1.50},SpotInstanceType=one-time}"
    )
fi

echo "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances "${LAUNCH_ARGS[@]}" --query 'Instances[0].InstanceId' --output text)

echo "Instance ID: $INSTANCE_ID"
echo "$INSTANCE_ID" > .instance-id

echo "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "$PUBLIC_IP" > .instance-ip
echo "Instance running at: $PUBLIC_IP"

echo ""
echo "============================================"
echo "Launch complete!"
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP:   $PUBLIC_IP"
echo "  Run ID:      $RUN_ID"
echo ""
echo "SSH: ./ssh.sh"
echo "Terminate: ./terminate.sh"
echo "Download results: ./download_report.sh"
echo "============================================"
