#!/usr/bin/env bash
# upload benchmark datasets to S3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source config.env
export AWS_DEFAULT_REGION="$AWS_REGION"

DATA_DIR="$SCRIPT_DIR/../data"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

echo "Uploading benchmark data to s3://$S3_BUCKET/data/"
echo ""

if [[ -f "$DATA_DIR/musique.json" && -f "$DATA_DIR/musique_corpus.json" ]]; then
    echo "Uploading MuSiQue..."
    aws s3 cp "$DATA_DIR/musique.json" "s3://$S3_BUCKET/data/musique.json"
    aws s3 cp "$DATA_DIR/musique_corpus.json" "s3://$S3_BUCKET/data/musique_corpus.json"
else
    echo "  MuSiQue files not found, skipping."
fi

if [[ -f "$DATA_DIR/hotpotqa.json" && -f "$DATA_DIR/hotpotqa_corpus.json" ]]; then
    echo "Uploading HotpotQA..."
    aws s3 cp "$DATA_DIR/hotpotqa.json" "s3://$S3_BUCKET/data/hotpotqa.json"
    aws s3 cp "$DATA_DIR/hotpotqa_corpus.json" "s3://$S3_BUCKET/data/hotpotqa_corpus.json"
else
    echo "  HotpotQA files not found, skipping."
fi

echo ""
echo "============================================"
echo "Upload complete!"
echo ""
echo "Files in s3://$S3_BUCKET/data/:"
aws s3 ls "s3://$S3_BUCKET/data/" --human-readable
echo "============================================"
echo ""
echo "Set in config.env:"
echo "  CORPUS_PATH=data/musique_corpus.json"
echo "  QUERIES_PATH=data/musique.json"
echo "or"
echo "  CORPUS_PATH=data/hotpotqa_corpus.json"
echo "  QUERIES_PATH=data/hotpotqa.json"
