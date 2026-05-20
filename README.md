## Setup

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```
# quick demo with hardcoded passages
python main.py

# benchmark against MuSiQue/HotpotQA
python benchmark.py --corpus data/corpus.json --queries data/queries.json

# generate plots from results
python visualize.py --results results/benchmark_results.json
```

## AWS

Scripts in `aws/` launch an EC2 instance, run benchmarks, sync results to S3, and self-terminate.

```
cd aws
cp config.env.example config.env  # fill in S3_BUCKET, OPENAI_API_KEY
./setup_once.sh                   # create bucket, keypair, security group, IAM role
./upload_data.sh                  # push benchmark data to S3
./launch.sh                       # launch instance
./ssh.sh "tail -f /var/log/hipporag-user-data.log"  # examine status of live run
./download_report.sh              # pull results when done
```

## Results

### MuSiQue @5
| Method      | Our Result | Paper Result |
| ----------- | ----------: | -----------: |
| BM25        |      0.3746 |         41.2 |
| HippoRAG v1 |      0.4783 |         52.1 |
| HippoRAG v2 |      0.5346 |         74.7 | 

### HotpotQA @5
| Method      | Our Result | Paper Result |
| ----------- | ----------: | -----------: |
| BM25        |      0.6900 |         72.2 |
| HippoRAG v1 |      0.7235 |         76.2 |
| HippoRAG v2 |      0.7915 |         96.3 |

The minimal jump from v1 to v2 likely due to unoptimized synonym threshold hyperparameter

## Future improvements
- Hyperparam sweep either dataset to determine best threshold (<80) for 4o-mini
- Wire in classes for swapping in / out embedding models and LLMs