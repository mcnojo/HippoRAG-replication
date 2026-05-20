import numpy as np
import string
from collections import Counter


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def recall_at_k(retrieved_passages: list[str], gold_passages: list[str], k: int) -> float:
    # proportion of gold passages found in top-k retrieved
    retrieved_k = set(_normalize_ws(p) for p in retrieved_passages[:k])
    hits = sum(1 for g in gold_passages if _normalize_ws(g) in retrieved_k)
    return hits / len(gold_passages) if gold_passages else 0.0


def _normalize(text: str) -> str:
    # lowercase, strip punctuation, collapse whitespace
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gold_tokens) if gold_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize(prediction) == _normalize(ground_truth))


def evaluate_retrieval(results: list[dict]) -> dict:
    # results: list of {"retrieved": [...], "gold": [...]}
    r2 = np.mean([recall_at_k(r["retrieved"], r["gold"], k=2) for r in results])
    r5 = np.mean([recall_at_k(r["retrieved"], r["gold"], k=5) for r in results])
    return {"recall@2": round(r2, 4), "recall@5": round(r5, 4)}


def evaluate_qa(results: list[dict]) -> dict:
    # results: list of {"prediction": "...", "answers": ["...", ...]}
    # takes best score across all acceptable gold answers
    f1_scores = []
    em_scores = []
    for r in results:
        pred = r["prediction"]
        answers = r["answers"]
        if not answers:
            f1_scores.append(0.0)
            em_scores.append(0.0)
            continue
        f1_scores.append(max(token_f1(pred, a) for a in answers))
        em_scores.append(max(exact_match(pred, a) for a in answers))
    return {
        "answer_f1": round(np.mean(f1_scores), 4),
        "exact_match": round(np.mean(em_scores), 4)
    }
