#!/usr/bin/env bash
# runs on EC2 boot: pulls code from S3, installs deps, runs benchmark, syncs results back
set -euo pipefail

export HOME=/home/ubuntu
cd "$HOME"

sudo mkdir -p /var/log/hipporag
sudo chown ubuntu:ubuntu /var/log/hipporag

exec > >(tee /var/log/hipporag/bootstrap.log) 2>&1
echo "Bootstrap started at $(date)"
echo "RUN_ID: ${RUN_ID:-unset}"
echo "S3_BUCKET: ${S3_BUCKET:-unset}"

aws s3 cp "s3://$S3_BUCKET/kit/config.env" /tmp/config.env
source /tmp/config.env

cat > "$HOME/.env" << EOF
OPENAI_API_KEY=$OPENAI_API_KEY
RATE_LIMIT_DELAY=${RATE_LIMIT_DELAY:-0.1}
EOF

echo "Downloading code from S3..."
aws s3 cp "s3://$S3_BUCKET/code/hipporag-code.tar.gz" /tmp/hipporag-code.tar.gz
mkdir -p "$HOME/hipporag"
tar -xzf /tmp/hipporag-code.tar.gz -C "$HOME/hipporag"

cd "$HOME/hipporag"
cp "$HOME/.env" .env
mkdir -p results figures cache data

echo "Installing Python deps..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

if [[ -f requirements.txt ]]; then
    pip install -r requirements.txt
else
    pip install openai python-dotenv pydantic networkx numpy scikit-learn matplotlib
fi

echo "Downloading benchmark data from S3..."
aws s3 sync "s3://$S3_BUCKET/data/" data/ || true

echo "Downloading cached data from S3..."
aws s3 sync "s3://$S3_BUCKET/cache/" cache/ || true

echo "Data files available:"
ls -la data/

# background sync: push results/figures/cache/logs to S3 every 60s
echo "Starting background S3 sync..."
nohup bash -c '
while true; do
    aws s3 sync results/ "s3://'"$S3_BUCKET"'/runs/'"$RUN_ID"'/results/" 2>/dev/null || true
    aws s3 sync results/ "s3://'"$S3_BUCKET"'/results/" --delete 2>/dev/null || true
    aws s3 sync figures/ "s3://'"$S3_BUCKET"'/runs/'"$RUN_ID"'/figures/" 2>/dev/null || true
    aws s3 sync figures/ "s3://'"$S3_BUCKET"'/figures/" --delete 2>/dev/null || true
    aws s3 sync cache/ "s3://'"$S3_BUCKET"'/cache/" 2>/dev/null || true
    aws s3 sync /var/log/hipporag/ "s3://'"$S3_BUCKET"'/runs/'"$RUN_ID"'/logs/" 2>/dev/null || true
    sleep 60
done
' > /var/log/hipporag/sync.log 2>&1 &
SYNC_PID=$!
echo "Sync daemon PID: $SYNC_PID"

# final sync on exit
cleanup() {
    echo "Final sync..."
    aws s3 sync results/ "s3://$S3_BUCKET/runs/$RUN_ID/results/" || true
    aws s3 sync results/ "s3://$S3_BUCKET/results/" --delete || true
    aws s3 sync figures/ "s3://$S3_BUCKET/runs/$RUN_ID/figures/" || true
    aws s3 sync figures/ "s3://$S3_BUCKET/figures/" --delete || true
    aws s3 sync cache/ "s3://$S3_BUCKET/cache/" || true
    aws s3 sync /var/log/hipporag/ "s3://$S3_BUCKET/runs/$RUN_ID/logs/" || true
    kill $SYNC_PID 2>/dev/null || true
}
trap cleanup EXIT

RUN_MODE="${RUN_MODE:-benchmark}"

echo "============================================"
echo "Running HippoRAG ($RUN_MODE mode) at $(date)"
echo "============================================"

set +e

if [[ "$RUN_MODE" == "demo" ]]; then
    python main.py 2>&1 | tee results/run_output.txt
    EXIT_CODE=${PIPESTATUS[0]}

elif [[ "$RUN_MODE" == "benchmark" ]]; then
    BENCHMARK_ARGS=(
        --corpus "${CORPUS_PATH:-data/corpus.json}"
        --queries "${QUERIES_PATH:-data/queries.json}"
        --limit "${QUERY_LIMIT:-100}"
        --top-k "${TOP_K:-5}"
        --output-dir results
        --cache-dir cache
    )
    case "${BENCHMARK_VERSION:-both}" in
        v1) BENCHMARK_ARGS+=(--v1-only) ;;
        v2) BENCHMARK_ARGS+=(--v2-only) ;;
        bm25) BENCHMARK_ARGS+=(--bm25-only) ;;
    esac

    if [[ ! -f "${CORPUS_PATH:-data/corpus.json}" ]]; then
        echo "ERROR: Corpus file not found. Upload benchmark data to S3 first."
        echo "Expected: ${CORPUS_PATH:-data/corpus.json}"
        EXIT_CODE=1
    else
        echo "Running benchmark with args: ${BENCHMARK_ARGS[*]}"
        python benchmark.py "${BENCHMARK_ARGS[@]}" 2>&1 | tee results/benchmark_output.txt
        EXIT_CODE=${PIPESTATUS[0]}

        if [[ $EXIT_CODE -eq 0 && -f results/benchmark_results.json ]]; then
            echo ""
            echo "Generating plots..."
            python visualize.py --results results/benchmark_results.json --output-dir figures 2>&1 | tee -a results/benchmark_output.txt
        fi
    fi
fi

set -e

cat > results/run_metadata.json << EOF
{
    "run_id": "$RUN_ID",
    "mode": "$RUN_MODE",
    "exit_code": $EXIT_CODE,
    "completed_at": "$(date -Iseconds)",
    "instance_type": "${INSTANCE_TYPE:-unknown}",
    "query_limit": "${QUERY_LIMIT:-100}",
    "top_k": "${TOP_K:-5}"
}
EOF

echo "============================================"
echo "HippoRAG completed with exit code: $EXIT_CODE"
echo "============================================"

# self-terminate if configured
if [[ "${AUTO_TERMINATE:-true}" == "true" ]]; then
    echo "Auto-terminating instance..."
    TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
    INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"
fi

echo "Bootstrap completed at $(date)"
