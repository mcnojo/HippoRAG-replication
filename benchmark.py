import os
import json
import pickle
import hashlib
import numpy as np
from tqdm import tqdm
import bm25s
import Stemmer
import argparse
from main import Hippo, emb, emb_batch, client, _call_with_retry
from graph_metrics import evaluate_retrieval, evaluate_qa


class EntityCache:
    # caches entity extractions to disk so reruns skip the LLM call

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, "entity_cache.pkl")
        self.cache = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        return {}

    def save(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> list[str] | None:
        return self.cache.get(self._key(text))

    def set(self, text: str, entities: list[str]):
        self.cache[self._key(text)] = entities


class QueryEmbeddingCache:
    """Caches whole-query embeddings for v2 query-to-triple retrieval."""

    def __init__(self, cache_dir: str = "cache"):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, "query_emb_cache.pkl")
        self.cache: dict = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        return {}

    def save(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        return self.cache.get(self._key(text))

    def set(self, text: str, vec: list[float]):
        self.cache[self._key(text)] = vec


class BenchmarkHippo(Hippo):
    # extends Hippo with graph persistence, entity caching, and LLM reranking (v2)

    def __init__(self, passages: list[str], version: int = 1, cache_dir: str = "cache"):
        super().__init__(version=version)
        self.passages = passages
        self.p = len(passages)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.entity_cache = EntityCache(cache_dir)
        self.query_emb_cache = QueryEmbeddingCache(cache_dir)

    def _corpus_hash(self) -> str:
        return hashlib.md5(json.dumps(self.passages, sort_keys=True).encode()).hexdigest()[:12]

    CHECKPOINT_EVERY = 50  # save partial progress every N passages

    def _cache_path(self) -> str:
        suffix = "" if self.version == 1 else "_v2"
        return os.path.join(self.cache_dir, f"hippo_{self._corpus_hash()}{suffix}.pkl")

    def _partial_cache_path(self) -> str:
        suffix = "" if self.version == 1 else "_v2"
        return os.path.join(self.cache_dir, f"hippo_{self._corpus_hash()}{suffix}_partial.pkl")

    def save(self):
        path = self._cache_path()
        payload = {
            "graph": self.graph,
            "adjacency": self.adjacency,
            "adj_entity_indices": self.adj_entity_indices,
            "version": self.version,
            "graph_stats": getattr(self, '_graph_stats', {}),
        }
        if self.version == 2:
            payload["triple_texts"] = self.triple_texts
            payload["triple_embeddings"] = self.triple_embeddings
            payload["triple_head_nodes"] = self.triple_head_nodes
            payload["triple_tail_nodes"] = self.triple_tail_nodes
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved index to {path}")
        partial = self._partial_cache_path()
        if os.path.exists(partial):
            os.remove(partial)
            print("Removed partial checkpoint")

    def _save_partial(self, last_index: int):
        path = self._partial_cache_path()
        with open(path, "wb") as f:
            pickle.dump({
                "graph": self.graph,
                "last_index": last_index,
            }, f)
        print(f"Checkpoint saved at passage {last_index} -> {path}")

    def _load_partial(self) -> int | None:
        # returns the next passage index to resume from, or None
        path = self._partial_cache_path()
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.graph = state["graph"]
        resume_from = state["last_index"] + 1
        print(f"Resuming from passage {resume_from} (loaded partial checkpoint)")
        return resume_from

    def load(self) -> bool:
        path = self._cache_path()
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            state = pickle.load(f)
        if state.get("version") != self.version:
            print(f"Cache version mismatch (got {state.get('version')}, want {self.version}), rebuilding")
            return False
        self.graph = state["graph"]
        self.adjacency = state["adjacency"]
        self.adj_entity_indices = state["adj_entity_indices"]
        self._graph_stats = state.get("graph_stats", {})
        if self.version == 2:
            self.triple_texts = state.get("triple_texts", [])
            self.triple_embeddings = state.get("triple_embeddings")
            self.triple_head_nodes = state.get("triple_head_nodes", [])
            self.triple_tail_nodes = state.get("triple_tail_nodes", [])
            n_triples = len(self.triple_texts)
            print(f"Loaded v2 index: {n_triples} triples in query-to-triple index")
            if n_triples == 0:
                print("WARNING: v2 cache has no triple index — delete cache and rebuild")
                return False
        if self._graph_stats:
            print(f"Graph stats: {self._graph_stats}")
        print(f"Loaded index from {path}")
        return True

    def _on_passage_done(self, index: int):
        if (index + 1) % self.CHECKPOINT_EVERY == 0:
            self._save_partial(index)

    def createGraph(self):
        if self.load():
            return
        start_index = self._load_partial() or 0
        super().createGraph(start_index=start_index)
        self.save()

    def onlineIE(self, query):
        # check entity cache before calling the LLM
        cached = self.entity_cache.get(query)
        if cached is not None:
            class CachedEntities:
                def __init__(self, entities):
                    self.named_entities = entities
            return CachedEntities(cached)

        result = super().onlineIE(query)

        self.entity_cache.set(query, result.named_entities)
        self.entity_cache.save()

        return result

    def _get_query_embedding(self, query: str) -> list[float]:
        """Return (and cache) the embedding for a full query string (v2)."""
        cached = self.query_emb_cache.get(query)
        if cached is not None:
            return cached
        vec = emb(query)
        self.query_emb_cache.set(query, vec)
        self.query_emb_cache.save()
        return vec

    def queryHippoWithRetrieval(self, query: str, top_k: int = 5) -> tuple[str, list[str], list[str]]:
        """
        Returns (answer, retrieved_passages, debug_info).
        v1: NER entities -> entity embeddings -> onlineMatch (PPR + adjacency)
        v2: whole-query embedding -> onlineMatchV2WithQuery (query-to-triple -> filter -> PPR)
        """
        if self.version == 1:
            entities = self.onlineIE(query)
            entity_embeddings = emb_batch(entities.named_entities)
            retrieved = self.onlineMatch(top_k, entity_embeddings)
            debug_info = entities.named_entities
        else:
            q_emb = self._get_query_embedding(query)
            retrieved = self.onlineMatchV2WithQuery(top_k, query, q_emb)
            debug_info = []

        prompt = (
            "I have a question: " + query +
            "\n\nPlease make use of the following context to answer my question:\n\n" +
            "\n\n".join(retrieved)
        )
        response = _call_with_retry(lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=100,
            temperature=0,
        ))
        answer = response.choices[0].message.content.strip()
        return answer, retrieved, debug_info


class BM25Baseline:
    # BM25S with stemming + stopword removal to match the paper's baseline

    def __init__(self, passages: list[str]):
        self.passages = passages
        self.p = len(passages)
        self.stemmer = Stemmer.Stemmer("english")
        corpus_tokens = bm25s.tokenize(passages, stopwords="en", stemmer=self.stemmer)
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens)
        print(f"BM25S index built: {self.p} passages")

    def queryWithRetrieval(self, query: str, top_k: int = 5) -> tuple[str, list[str], list[str]]:
        query_tokens = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer)
        results, scores = self.bm25.retrieve(query_tokens, k=top_k)
        retrieved = [self.passages[i] for i in results[0]]

        prompt = (
            "I have a question: " + query +
            "\n\nPlease make use of the following context to answer my question:\n\n" +
            "\n\n".join(retrieved)
        )
        response = _call_with_retry(lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=100,
            temperature=0,
        ))
        answer = response.choices[0].message.content.strip()
        return answer, retrieved, []


def load_benchmark(corpus_path: str, queries_path: str) -> tuple[list[str], list[dict]]:
    # returns (passages, questions) where questions have: question, answers, gold_passages
    with open(corpus_path) as f:
        corpus = json.load(f)
    with open(queries_path) as f:
        queries = json.load(f)

    # handle different corpus formats
    if isinstance(corpus, list) and len(corpus) > 0:
        if isinstance(corpus[0], dict):
            if "text" in corpus[0]:
                passages = [item["text"] for item in corpus]
            elif "paragraph_text" in corpus[0]:
                passages = [item["paragraph_text"] for item in corpus]
            elif "passage" in corpus[0]:
                passages = [item["passage"] for item in corpus]
            else:
                raise ValueError(f"Unknown corpus dict format, keys: {list(corpus[0].keys())}")
        elif isinstance(corpus[0], str):
            passages = corpus
        else:
            raise ValueError(f"Unknown corpus format: {type(corpus[0])}")
    else:
        raise ValueError(f"Unexpected corpus type: {type(corpus)}")

    questions = []
    for q in queries:
        gold = []

        if "paragraphs" in q:
            # MuSiQue format
            for p in q["paragraphs"]:
                if p.get("is_supporting"):
                    text = p.get("text") or p.get("paragraph_text", "")
                    if text:
                        gold.append(text)

        elif "context" in q and "supporting_facts" in q:
            # HotpotQA format: context = [[title, [sentences]], ...], supporting_facts = [[title, idx], ...]
            context_lookup = {item[0]: " ".join(item[1]) for item in q["context"]}
            seen_titles = set()
            for title, sent_idx in q["supporting_facts"]:
                if title not in seen_titles and title in context_lookup:
                    gold.append(context_lookup[title])
                    seen_titles.add(title)

        elif "gold_passages" in q:
            gold = q["gold_passages"]

        answers = q.get("answer", q.get("answers", []))
        if isinstance(answers, str):
            answers = [answers]

        questions.append({
            "question": q.get("question", q.get("query", "")),
            "answers": answers,
            "gold_passages": gold,
        })

    return passages, questions


def run_benchmark(
    retriever,
    questions: list[dict],
    top_k: int = 5,
    label: str = "hippo",
    output_dir: str = "results",
    checkpoint_every: int = 10,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # resume from checkpoint if one exists
    checkpoint_path = os.path.join(output_dir, f"{label}_checkpoint.json")
    start_idx = 0
    detailed_results = []

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        detailed_results = checkpoint.get("results", [])
        start_idx = len(detailed_results)
        print(f"Resuming from checkpoint: {start_idx}/{len(questions)} already completed")

    pbar = tqdm(enumerate(questions[start_idx:], start=start_idx),
                total=len(questions), initial=start_idx, desc=f"Queries ({label})")
    for i, q in pbar:
        try:
            if isinstance(retriever, BM25Baseline):
                answer, retrieved, entity_names = retriever.queryWithRetrieval(
                    q["question"], top_k=top_k
                )
            else:
                answer, retrieved, entity_names = retriever.queryHippoWithRetrieval(
                    q["question"], top_k=top_k
                )

            detailed_results.append({
                "question": q["question"],
                "entities_extracted": entity_names,
                "retrieved_passages": retrieved,
                "gold_passages": q["gold_passages"],
                "prediction": answer,
                "gold_answers": q["answers"],
            })

        except Exception as e:
            tqdm.write(f"  Error on Q{i+1}: {e}")
            detailed_results.append({
                "question": q["question"],
                "gold_passages": q["gold_passages"],
                "gold_answers": q["answers"],
                "error": str(e),
            })

        if (i + 1) % checkpoint_every == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({"results": detailed_results}, f)

    # compute metrics
    retrieval_results = []
    qa_results = []
    for r in detailed_results:
        retrieval_results.append({
            "retrieved": r.get("retrieved_passages", []),
            "gold": r.get("gold_passages", []),
        })
        qa_results.append({
            "prediction": r.get("prediction", ""),
            "answers": r.get("gold_answers", []),
        })

    retrieval_metrics = evaluate_retrieval(retrieval_results)
    qa_metrics = evaluate_qa(qa_results)

    graph_stats = getattr(retriever, '_graph_stats', {})
    metrics = {
        "label": label,
        "num_questions": len(questions),
        "num_passages": retriever.p,
        "top_k": top_k,
        **retrieval_metrics,
        **qa_metrics,
        "kg": graph_stats,
    }

    detailed_path = os.path.join(output_dir, f"{label}_detailed.json")
    with open(detailed_path, "w") as f:
        json.dump(detailed_results, f, indent=2)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\n{'='*45}")
    print(f" {label}")
    print(f"{'='*45}")
    print(f" R@2: {metrics['recall@2']:.4f}  R@5: {metrics['recall@5']:.4f}")
    print(f" F1:  {metrics['answer_f1']:.4f}  EM:  {metrics['exact_match']:.4f}")
    if graph_stats:
        syn_pairs = graph_stats.get('synonymy_pairs', graph_stats.get('synonymy_edges', '?'))
        syn_edges = graph_stats.get('synonymy_edges', '?')
        print(f" KG:  {graph_stats.get('entity_nodes', '?')} nodes, "
              f"{graph_stats.get('total_edges', '?')} edges, "
              f"{syn_pairs} synonymy pairs ({syn_edges} directed edges)")
    print(f"{'='*45}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run HippoRAG benchmarks")
    parser.add_argument("--corpus", type=str, default="data/corpus.json")
    parser.add_argument("--queries", type=str, default="data/queries.json")
    parser.add_argument("--limit", type=int, default=100, help="max questions (0 = all)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--v1-only", action="store_true")
    parser.add_argument("--v2-only", action="store_true")
    parser.add_argument("--bm25-only", action="store_true")
    parser.add_argument("--no-bm25", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        print(f"ERROR: Corpus file not found: {args.corpus}")
        print("Download benchmark data to data/. See: https://huggingface.co/datasets/osunlp/HippoRAG")
        return

    if not os.path.exists(args.queries):
        print(f"ERROR: Queries file not found: {args.queries}")
        return

    print(f"Loading corpus from {args.corpus}")
    print(f"Loading queries from {args.queries}")
    passages, questions = load_benchmark(args.corpus, args.queries)
    print(f"Loaded {len(passages)} passages and {len(questions)} questions")

    if args.limit > 0:
        questions = questions[:args.limit]
        print(f"Limited to {len(questions)} questions")

    os.makedirs(args.output_dir, exist_ok=True)
    all_metrics = []

    if not args.no_bm25:
        print("\n" + "="*60)
        print("Running BM25 baseline")
        print("="*60)
        bm25 = BM25Baseline(passages)
        bm25_metrics = run_benchmark(
            bm25, questions,
            top_k=args.top_k,
            label="bm25",
            output_dir=args.output_dir
        )
        all_metrics.append(bm25_metrics)
        del bm25

    if not args.v2_only and not args.bm25_only:
        print("\n" + "="*60)
        print("Running HippoRAG v1")
        print("="*60)
        hippo_v1 = BenchmarkHippo(passages, version=1, cache_dir=args.cache_dir)
        hippo_v1.createGraph()
        v1_metrics = run_benchmark(
            hippo_v1, questions,
            top_k=args.top_k,
            label="hipporag_v1",
            output_dir=args.output_dir
        )
        all_metrics.append(v1_metrics)
        del hippo_v1

    if not args.v1_only and not args.bm25_only:
        print("\n" + "="*60)
        print("Running HippoRAG v2")
        print("="*60)
        hippo_v2 = BenchmarkHippo(passages, version=2, cache_dir=args.cache_dir)
        hippo_v2.createGraph()
        v2_metrics = run_benchmark(
            hippo_v2, questions,
            top_k=args.top_k,
            label="hipporag_v2",
            output_dir=args.output_dir
        )
        all_metrics.append(v2_metrics)

    config = {
        "dataset": os.path.splitext(os.path.basename(args.corpus))[0],
        "openie_model": "gpt-4o-mini",
        "qa_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "synonymy_threshold": 0.8,
        "ppr_alpha": 0.5,
        "passage_weight_lambda": 0.05,
        "top_k_triples": 5,
        "corpus_size": len(passages),
        "query_count": len(questions),
        "top_k": args.top_k,
        "instance_type": os.environ.get("INSTANCE_TYPE", "local"),
    }
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump({"config": config, "metrics": all_metrics}, f, indent=2)
    print(f"\nSaved results to {results_path}")

    if len(all_metrics) > 1:
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        labels = [m["label"] for m in all_metrics]
        header = f"{'Metric':<15}" + "".join(f"{l:>12}" for l in labels)
        print(header)
        print("-" * len(header))
        for metric in ["recall@2", "recall@5", "answer_f1", "exact_match"]:
            row = f"{metric:<15}"
            for m in all_metrics:
                row += f"{m.get(metric, 0):>12.4f}"
            print(row)


if __name__ == "__main__":
    main()
