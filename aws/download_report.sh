#!/usr/bin/env bash
# pull results, figures, and logs from S3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source config.env
export AWS_DEFAULT_REGION="$AWS_REGION"

usage() {
    echo "Usage: $0 [RUN_ID | --list | --latest]"
    echo ""
    echo "  RUN_ID    Download from specific run"
    echo "  --list    List available runs"
    echo "  --latest  Download from latest mirror (default)"
    echo ""
    exit 1
}

RUN_ID=""
LIST_ONLY=false

if [[ $# -eq 0 ]]; then
    if [[ -f .last-run-id ]]; then
        RUN_ID=$(cat .last-run-id)
    fi
elif [[ "$1" == "--list" ]]; then
    LIST_ONLY=true
elif [[ "$1" == "--latest" ]]; then
    RUN_ID=""
elif [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
else
    RUN_ID="$1"
fi

if [[ "$LIST_ONLY" == "true" ]]; then
    echo "Available runs in s3://$S3_BUCKET/runs/:"
    aws s3 ls "s3://$S3_BUCKET/runs/" | awk '{print $2}' | sed 's|/$||' | sort -r
    exit 0
fi

if [[ -n "$RUN_ID" ]]; then
    BASE_S3="s3://$S3_BUCKET/runs/$RUN_ID"
    BASE_LOCAL="./downloads/$RUN_ID"
    echo "Downloading run: $RUN_ID"
else
    BASE_S3="s3://$S3_BUCKET"
    BASE_LOCAL="./downloads/latest"
    echo "Downloading from latest mirror"
fi

mkdir -p "$BASE_LOCAL/results" "$BASE_LOCAL/figures" "$BASE_LOCAL/logs"

echo ""
echo "Downloading results..."
aws s3 sync "$BASE_S3/results/" "$BASE_LOCAL/results/" 2>/dev/null || echo "  (none found)"

echo "Downloading figures..."
aws s3 sync "$BASE_S3/figures/" "$BASE_LOCAL/figures/" 2>/dev/null || echo "  (none found)"

echo "Downloading logs..."
aws s3 sync "$BASE_S3/logs/" "$BASE_LOCAL/logs/" 2>/dev/null || echo "  (none found)"

echo ""
echo "============================================"
echo "Downloaded to: $BASE_LOCAL"
echo "============================================"
echo ""

if [[ -f "$BASE_LOCAL/results/benchmark_results.json" ]]; then
    echo "Benchmark results:"
    cat "$BASE_LOCAL/results/benchmark_results.json"
    echo ""
fi

if [[ -d "$BASE_LOCAL/figures" ]]; then
    FIGURE_COUNT=$(find "$BASE_LOCAL/figures" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
    echo "Figures: $FIGURE_COUNT PNG files"
    ls "$BASE_LOCAL/figures/"*.png 2>/dev/null || true
fi

echo ""
echo "Results directory:"
ls -la "$BASE_LOCAL/results/" 2>/dev/null || echo "  (empty)"

if [[ -f "$BASE_LOCAL/results/benchmark_report.md" ]]; then
    echo ""
    echo "Report: $BASE_LOCAL/results/benchmark_report.md"
fi
