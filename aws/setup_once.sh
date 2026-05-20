#!/usr/bin/env bash
# one-time AWS setup: S3 bucket, SSH keypair, security group, IAM role + instance profile
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f config.env ]]; then
    echo "ERROR: config.env not found. Copy config.env.example first."
    exit 1
fi
source config.env

export AWS_DEFAULT_REGION="$AWS_REGION"

echo "Setting up AWS infra for HippoRAG..."
echo "Region: $AWS_REGION"
echo ""

# S3 bucket
echo "1. Creating S3 bucket: $S3_BUCKET"
if aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
    echo "   Already exists."
else
    aws s3api create-bucket \
        --bucket "$S3_BUCKET" \
        --region "$AWS_REGION" \
        --create-bucket-configuration LocationConstraint="$AWS_REGION"
    echo "   Created."
fi

# SSH keypair
echo ""
echo "2. Creating SSH keypair: $KEY_NAME"
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" 2>/dev/null; then
    echo "   Already exists."
else
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --query 'KeyMaterial' \
        --output text > ~/.ssh/"$KEY_NAME".pem
    chmod 600 ~/.ssh/"$KEY_NAME".pem
    echo "   Saved to ~/.ssh/$KEY_NAME.pem"
fi

# security group
echo ""
echo "3. Creating security group: $SECURITY_GROUP_NAME"
SG_ID=$(aws ec2 describe-security-groups \
    --group-names "$SECURITY_GROUP_NAME" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null || echo "")

if [[ -n "$SG_ID" && "$SG_ID" != "None" ]]; then
    echo "   Already exists: $SG_ID"
else
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP_NAME" \
        --description "HippoRAG EC2 instances" \
        --query 'GroupId' \
        --output text)

    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0

    echo "   Created: $SG_ID (SSH open to 0.0.0.0/0)"
fi

# IAM role
echo ""
echo "4. Creating IAM role: $IAM_ROLE_NAME"

TRUST_POLICY='{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "ec2.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}'

PERMISSIONS_POLICY='{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["s3:*"],
            "Resource": [
                "arn:aws:s3:::'"$S3_BUCKET"'",
                "arn:aws:s3:::'"$S3_BUCKET"'/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": ["ec2:TerminateInstances"],
            "Resource": "*",
            "Condition": {
                "StringLike": {
                    "ec2:ResourceTag/Name": "hipporag-*"
                }
            }
        }
    ]
}'

if aws iam get-role --role-name "$IAM_ROLE_NAME" 2>/dev/null; then
    echo "   Already exists."
else
    aws iam create-role \
        --role-name "$IAM_ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY"

    aws iam put-role-policy \
        --role-name "$IAM_ROLE_NAME" \
        --policy-name "${IAM_ROLE_NAME}-policy" \
        --policy-document "$PERMISSIONS_POLICY"

    echo "   Created with S3 + self-terminate permissions."
fi

# instance profile
echo ""
echo "5. Creating instance profile: $INSTANCE_PROFILE_NAME"
if aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" 2>/dev/null; then
    echo "   Already exists."
else
    aws iam create-instance-profile \
        --instance-profile-name "$INSTANCE_PROFILE_NAME"

    aws iam add-role-to-instance-profile \
        --instance-profile-name "$INSTANCE_PROFILE_NAME" \
        --role-name "$IAM_ROLE_NAME"

    echo "   Created. Waiting 10s for IAM propagation..."
    sleep 10
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Next:"
echo "  1. Set OPENAI_API_KEY in config.env"
echo "  2. Run: ./launch.sh"
echo "============================================"
