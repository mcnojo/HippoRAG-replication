import json
import logging
import time
from collections import defaultdict
from dotenv import load_dotenv
import os
import numpy as np
import networkx as nx
from scipy import sparse
from tqdm import tqdm
from static import passages, offline_ie_connections, offline_ie_entities, online_ie
from pydantic import BaseModel
from openai import OpenAI


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

RATE_LIMIT_DELAY = float(os.environ.get("RATE_LIMIT_DELAY", "0.1"))
_last_api_call = 0.0

def _rate_limit():
    global _last_api_call
    elapsed = time.time() - _last_api_call
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    _last_api_call = time.time()


class EntityRecognition(BaseModel):
    named_entities: list[str]

class OfflineRDF(BaseModel):
    triples: list[list[str]]

def _call_with_retry(api_func, max_retries=5, base_delay=1.0):
    # exponential backoff on rate limit errors
    import openai
    for attempt in range(max_retries):
        try:
            _rate_limit()
            return api_func()
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)


def generate(input_content, secondary_input, kind):
    match kind:
        case "offline_entity":
            final_content = offline_ie_entities.substitute(_offline_passage=input_content)
            response = _call_with_retry(lambda: client.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": final_content}],
                max_completion_tokens=500,
                temperature=0,
                response_format=EntityRecognition,
            ))
            parsed: EntityRecognition = response.choices[0].message.parsed

        case "offline_relation":
            final_content = offline_ie_connections.substitute(_paragraph_for_entities=secondary_input, _entity_list=input_content)
            response = _call_with_retry(lambda: client.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": final_content}],
                max_completion_tokens=3000,
                temperature=0,
                response_format=OfflineRDF,
            ))
            parsed: OfflineRDF = response.choices[0].message.parsed

        case "online_entity":
            final_content = online_ie.substitute(_online_question=input_content)
            response = _call_with_retry(lambda: client.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": final_content}],
                max_completion_tokens=500,
                temperature=0,
                response_format=EntityRecognition,
            ))
            parsed: EntityRecognition = response.choices[0].message.parsed

        case _:
            raise ValueError(f"Unknown generate kind: {kind!r}")

    return parsed

def emb(content):
    # embed a single string
    def _call():
        return client.embeddings.create(
            model="text-embedding-3-small",
            input=content
        )
    embed = _call_with_retry(_call)
    return embed.data[0].embedding


def emb_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple strings, chunked at OpenAI's 2048-per-request limit."""
    if not texts:
        return []
    MAX_PER_CALL = 2048
    results = []
    for i in range(0, len(texts), MAX_PER_CALL):
        chunk = texts[i:i + MAX_PER_CALL]
        def _call(c=chunk):
            return client.embeddings.create(model="text-embedding-3-small", input=c)
        resp = _call_with_retry(_call)
        results.extend([d.embedding for d in resp.data])
    return results


def triple_to_text(head: str, rel: str, tail: str) -> str:
    """Canonical triple serialisation — must match at index time and query time."""
    return f"{head} | {rel} | {tail}"


class Hippo:
    # version=1: standard HippoRAG (entity seeds -> PPR -> adjacency matrix ranking)
    # version=2: adds passage nodes + appears_in edges for entity->passage->entity PPR paths

    def __init__(self, version: int = 1):
        self.version = version
        self.graph = nx.MultiDiGraph()
        self.adjacency = None          # sparse CSR, entities x passages
        self.adj_entity_indices = {}

        # v2-only: parallel arrays for query-to-triple cosine lookup
        self.triple_texts: list[str] = []
        self.triple_embeddings: np.ndarray | None = None
        self.triple_head_nodes: list[str] = []
        self.triple_tail_nodes: list[str] = []

        self.passages = passages
        self.p = len(self.passages)

    def createGraph(self, start_index: int = 0):
        self._skipped_passages = 0

        pbar = tqdm(enumerate(self.passages), total=len(self.passages),
                    desc="Indexing passages", initial=start_index)
        for index, p in pbar:
            if index < start_index:
                continue

            triples = self.offlineIE(p)

            # collect new entities that need embeddings
            new_entities = []
            for triple in triples.triples:
                if len(triple) != 3:
                    continue
                head, _, tail = triple
                if not head.strip() or not tail.strip():
                    continue
                if head not in self.graph and head not in new_entities:
                    new_entities.append(head)
                if tail not in self.graph and tail not in new_entities:
                    new_entities.append(tail)

            # batch embed new entities (and passage text for v2)
            texts_to_embed = list(new_entities)
            if self.version == 2:
                texts_to_embed.append(p)
            embeddings = emb_batch(texts_to_embed) if texts_to_embed else []
            emb_lookup = {text: emb_vec for text, emb_vec in zip(texts_to_embed, embeddings)}

            if self.version == 2:
                passage_node = f"__passage_{index}__"
                self.graph.add_node(passage_node, embedding=emb_lookup[p],
                                    ordering={index}, is_passage=True)

            for triple in triples.triples:
                if len(triple) != 3:
                    continue

                head, rel, tail = triple
                if not head.strip() or not tail.strip():
                    continue

                if head in self.graph:
                    self.graph.nodes[head]["ordering"].add(index)
                else:
                    self.graph.add_node(
                        head,
                        embedding=emb_lookup[head],
                        ordering={index},
                        is_passage=False
                    )

                if tail in self.graph:
                    self.graph.nodes[tail]["ordering"].add(index)
                else:
                    self.graph.add_node(
                        tail,
                        embedding=emb_lookup[tail],
                        ordering={index},
                        is_passage=False
                    )

                self.graph.add_edge(head, tail, relation=rel)
                self.graph.add_edge(tail, head, relation=f"inv_{rel}")

                if self.version == 2:
                    self.graph.add_edge(head, passage_node, relation="appears_in")
                    self.graph.add_edge(passage_node, head, relation="contains")
                    self.graph.add_edge(tail, passage_node, relation="appears_in")
                    self.graph.add_edge(passage_node, tail, relation="contains")

            self._on_passage_done(index)

        # build sparse adjacency matrix from entity nodes only
        entity_nodes = [n for n in self.graph.nodes()
                        if not self.graph.nodes[n].get("is_passage")]
        rows, cols = [], []
        for index, node in enumerate(entity_nodes):
            indices = list(self.graph.nodes[node]["ordering"])
            rows.extend([index] * len(indices))
            cols.extend(indices)
            self.adj_entity_indices[node] = index
        data = np.ones(len(rows), dtype=np.int8)
        self.adjacency = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(entity_nodes), self.p)
        )
        print(f"Adjacency matrix: {self.adjacency.shape}, nnz={self.adjacency.nnz}")

        # add synonymy edges between entities with cosine sim >= threshold
        threshold = 0.8
        X = np.vstack([self.graph.nodes[n]["embedding"] for n in entity_nodes]).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        N = X.shape[0]
        BATCH = 1024
        synonymy_pairs = 0
        for start in tqdm(range(0, N, BATCH), desc="Computing synonymy edges"):
            end = min(start + BATCH, N)
            S_chunk = X[start:end] @ X.T
            i_chunk, j_chunk = np.where(S_chunk >= threshold)
            i_chunk += start
            mask = i_chunk < j_chunk
            for a, b in zip(i_chunk[mask], j_chunk[mask]):
                self.graph.add_edge(entity_nodes[a], entity_nodes[b], relation="synonymy edge")
                self.graph.add_edge(entity_nodes[b], entity_nodes[a], relation="synonymy edge")
                synonymy_pairs += 1

        # build triple index for query-to-triple retrieval (v2 only)
        # Reconstruct triples from the graph rather than tracking a local accumulator,
        # because checkpoint-resume resets local state and would lose earlier triples.
        if self.version == 2:
            _STRUCTURAL_RELATIONS = {"synonymy edge", "appears_in", "contains"}
            raw_triples_for_index = [
                (u, data["relation"], v)
                for u, v, data in self.graph.edges(data=True)
                if data.get("relation") not in _STRUCTURAL_RELATIONS
                and not data["relation"].startswith("inv_")
            ]
            self._build_triple_index(raw_triples_for_index)

        self._graph_stats = {
            "entity_nodes": len(entity_nodes),
            "total_edges": self.graph.number_of_edges(),
            "synonymy_pairs": synonymy_pairs,
            "synonymy_edges": synonymy_pairs * 2,
            "skipped_passages": self._skipped_passages,
            "triple_index_size": len(self.triple_texts) if self.version == 2 else 0,
        }
        print(f"Graph built: {self._graph_stats}")

    def _on_passage_done(self, index: int):
        pass

    def _build_triple_index(self, raw_triples: list[tuple[str, str, str]]):
        """Deduplicate triples, batch-embed, L2-normalise. Called once at end of createGraph (v2)."""
        seen: set[tuple[str, str, str]] = set()
        unique: list[tuple[str, str, str]] = []
        for t in raw_triples:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        if not unique:
            return

        texts = [triple_to_text(h, r, t) for h, r, t in unique]
        print(f"Embedding {len(texts)} unique triples for query-to-triple index...")
        vecs = emb_batch(texts)

        E = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        E /= norms

        self.triple_texts = texts
        self.triple_embeddings = E
        self.triple_head_nodes = [h for h, _, _ in unique]
        self.triple_tail_nodes = [t for _, _, t in unique]
        print(f"Triple index ready: {len(texts)} entries, embedding shape {E.shape}")

    # -----------------------------------------------------------------------
    # v2 online retrieval
    # -----------------------------------------------------------------------

    def _query_to_triple(self, query_vec: np.ndarray, top_k: int = 5) -> list[int]:
        """Return indices into self.triple_* for the top_k most similar triples."""
        if self.triple_embeddings is None or len(self.triple_texts) == 0:
            return []
        sims = self.triple_embeddings @ query_vec
        k = min(top_k, len(sims))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return top_idx.tolist()

    # Recognition memory prompt from HippoRAG 2
    _RECOGNITION_MEMORY_PROMPT = """You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, ensuring that the most crucial information is presented to these stakeholders. The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information.
You must select up to 4 relevant facts from the provided candidate list that have a strong connection to the query, aiding in reasoning and providing an accurate answer.
The output should be in JSON format, e.g., {{"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}}, and if no facts are relevant, return an empty list, {{"fact": []}}.
The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use facts from the candidate list and not generate new facts. The future of critical decision-making relies on your ability to accurately filter and present relevant information.

Question: Are Imperial River (Florida) and Amaradia (Dolj) both located in the same country?
Fact Before Filter: {{"fact": [["imperial river", "is located in", "florida"], ["imperial river", "is a river in", "united states"], ["imperial river", "may refer to", "south america"], ["amaradia", "flows through", "ro ia de amaradia"], ["imperial river", "may refer to", "united states"]]}}
Fact After Filter: {{"fact": [["imperial river", "is located in", "florida"], ["imperial river", "is a river in", "united states"], ["amaradia", "flows through", "ro ia de amaradia"]]}}

Question: When is the director of film The Ancestor 's birthday?
Fact Before Filter: {{"fact": [["jean jacques annaud", "born on", "1 october 1943"], ["tsui hark", "born on", "15 february 1950"], ["pablo trapero", "born on", "4 october 1971"], ["the ancestor", "directed by", "guido brignone"], ["benh zeitlin", "born on", "october 14 1982"]]}}
Fact After Filter: {{"fact": [["the ancestor", "directed by", "guido brignone"]]}}

Question: In what geographic region is the country where Teafuone is located?
Fact Before Filter: {{"fact": [["teafuaniua", "is on the", "east"], ["motuloa", "lies between", "teafuaniua"], ["motuloa", "lies between", "teafuanonu"], ["teafuone", "is", "islet"], ["teafuone", "located in", "nukufetau"]]}}
Fact After Filter: {{"fact": [["teafuone", "is", "islet"], ["teafuone", "located in", "nukufetau"]]}}

Question: {query}
Fact Before Filter: {fact_before}
Fact After Filter: """

    def _recognition_memory_filter(
        self,
        query: str,
        top_triple_indices: list[int],
    ) -> list[int]:
        """
        LLM-based triple relevance filter (recognition memory of v2 paper)
        Returns a subset of top_triple_indices. On parse failure, returns all indices.
        On empty LLM response, returns [] so caller can fall back.
        """
        if not top_triple_indices:
            return []

        # Build fact list in the paper;s format
        fact_triples = []
        for idx in top_triple_indices:
            h, t = self.triple_head_nodes[idx], self.triple_tail_nodes[idx]
            # Extract relation from triple_texts: "head | rel | tail"
            parts = self.triple_texts[idx].split(" | ", 2)
            rel = parts[1] if len(parts) == 3 else ""
            fact_triples.append([h, rel, t])

        fact_before = json.dumps({"fact": fact_triples})

        # Build lookup: normalised triple text -> position in top_triple_indices
        def _norm(s: str) -> str:
            return " ".join(s.lower().split())

        triple_lookup: dict[str, int] = {}
        for pos, (idx, triple) in enumerate(zip(top_triple_indices, fact_triples)):
            key = _norm(f"{triple[0]}|{triple[1]}|{triple[2]}")
            triple_lookup[key] = idx

        prompt = self._RECOGNITION_MEMORY_PROMPT.format(
            query=query, fact_before=fact_before)

        try:
            response = _call_with_retry(lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=300,
                temperature=0,
            ))
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
            kept_triples: list[list[str]] = parsed.get("fact", [])

            # Match returned triples back to indices via normalised text lookup
            filtered = []
            for triple in kept_triples:
                if len(triple) == 3:
                    key = _norm(f"{triple[0]}|{triple[1]}|{triple[2]}")
                    if key in triple_lookup:
                        filtered.append(triple_lookup[key])

            return filtered

        except Exception as e:
            logger.warning(f"Recognition memory filter failed ({e}), using all retrieved triples")
            return top_triple_indices

    def _fallback_passage_ranking(
        self,
        q_vec: np.ndarray,
        passage_nodes: list[str],
        topK: int,
    ) -> list[str]:
        """Direct passage embedding ranking — used if triple filter returns nothing."""
        if not passage_nodes:
            return []

        P_embs = np.vstack([
            np.asarray(self.graph.nodes[pn]["embedding"], dtype=np.float32)
            for pn in passage_nodes
        ])
        P_embs /= np.linalg.norm(P_embs, axis=1, keepdims=True)
        sims = P_embs @ q_vec
        p_scores = np.zeros(self.p)

        for pn, sim in zip(passage_nodes, sims):
            idx = int(pn.split("__passage_")[1].rstrip("_"))
            p_scores[idx] = float(sim)

        return self._top_passages(p_scores, topK)

    def _top_passages(self, p_scores: np.ndarray, topK: int) -> list[str]:
        topK = min(topK, len(p_scores))
        indices = np.argpartition(p_scores, -topK)[-topK:]
        values = p_scores[indices]
        order = np.argsort(-values)
        return [self.passages[i] for i in indices[order]]

    def onlineMatchV2WithQuery(
        self,
        topK: int,
        query: str,
        query_embedding,
    ) -> list[str]:
        """
        Full v2 retrieval: query-to-triple -> recognition memory filter -> PPR
        with phrase + passage seeds -> passage node ranking.
        """
        PASSAGE_WEIGHT = 0.05
        TOP_K_TRIPLES = 5
        MAX_PHRASE_SEEDS = 10

        q_vec = np.asarray(query_embedding, dtype=np.float32)
        q_vec /= max(np.linalg.norm(q_vec), 1e-9)

        passage_nodes = [n for n in self.graph.nodes()
                         if self.graph.nodes[n].get("is_passage")]

        # query-to-triple retrieval
        top_triple_idxs = self._query_to_triple(q_vec, top_k=TOP_K_TRIPLES)

        # recognition memory filter
        filtered_idxs = self._recognition_memory_filter(query, top_triple_idxs)

        # fall back if nothing survives the filter
        if not filtered_idxs:
            logger.debug("Recognition memory filtered all triples, falling back to passage embeddings")
            return self._fallback_passage_ranking(q_vec, passage_nodes, topK)

        # phrase seeds from filtered triple head/tail nodes (average ranking score per v2 section 3.5 )
        node_score_lists: dict[str, list[float]] = defaultdict(list)
        for rank, tidx in enumerate(filtered_idxs):
            score = (len(filtered_idxs) - rank) / len(filtered_idxs)
            for node in (self.triple_head_nodes[tidx], self.triple_tail_nodes[tidx]):
                if node in self.graph:
                    node_score_lists[node].append(score)
        node_rank_scores = {n: sum(s) / len(s) for n, s in node_score_lists.items()}

        phrase_seeds = sorted(node_rank_scores, key=node_rank_scores.get, reverse=True)
        phrase_seeds = phrase_seeds[:MAX_PHRASE_SEEDS]

        # build personalization dict
        personalization: dict[str, float] = {}

        # phrase node weights
        phrase_weight_sum = sum(node_rank_scores[n] for n in phrase_seeds)
        if phrase_weight_sum > 0:
            for node in phrase_seeds:
                personalization[node] = node_rank_scores[node] / phrase_weight_sum

        # passage node weights: embedding similarity × λ
        if passage_nodes:
            P_embs = np.vstack([
                np.asarray(self.graph.nodes[pn]["embedding"], dtype=np.float32)
                for pn in passage_nodes
            ])
            P_embs /= np.linalg.norm(P_embs, axis=1, keepdims=True)
            passage_sims = np.clip(P_embs @ q_vec, 0, None)
            sim_sum = passage_sims.sum()
            if sim_sum > 0:
                for pn, sim in zip(passage_nodes, passage_sims):
                    personalization[pn] = float(sim / sim_sum) * PASSAGE_WEIGHT

        # normalise so values sum to 1
        total = sum(personalization.values())
        if total <= 0:
            return self._fallback_passage_ranking(q_vec, passage_nodes, topK)
        personalization = {k: v / total for k, v in personalization.items()}

        # PPR
        n_dict_ppr = nx.pagerank(self.graph, personalization=personalization, alpha=0.5)

        # rank passages by passage-node PPR score
        p_scores = np.zeros(self.p)
        for pn in passage_nodes:
            idx = int(pn.split("__passage_")[1].rstrip("_"))
            p_scores[idx] = n_dict_ppr.get(pn, 0.0)

        return self._top_passages(p_scores, topK)

    def offlineIE(self, passage):
        import openai
        try:
            entities = generate(passage, "", "offline_entity")
            triples = generate(entities.model_dump_json(), passage, "offline_relation")
            return triples
        except (openai.LengthFinishReasonError, openai.ContentFilterFinishReasonError) as e:
            self._skipped_passages += 1
            tqdm.write(f"! Skipping passage: {type(e).__name__}")
            return OfflineRDF(triples=[])

    def onlineIE(self, query):
        entities = generate(query, "", "online_entity")
        return entities

    def onlineMatch(self, topK, query_embeddings):
        """v1: NER entity embeddings -> best-matching graph nodes -> PPR -> adjacency ranking."""
        entity_nodes = [n for n in self.graph.nodes()
                        if not self.graph.nodes[n].get("is_passage")]
        if not entity_nodes or not query_embeddings:
            return []

        X = np.vstack([self.graph.nodes[n]["embedding"] for n in entity_nodes]).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        Q = np.vstack([np.asarray(q, dtype=np.float32) for q in query_embeddings])
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)

        sims_mat = Q @ X.T
        best_node_indices = np.argmax(sims_mat, axis=1)

        seen = set()
        unique_indices = []
        for idx in best_node_indices.tolist():
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        n = len(unique_indices)
        personalization = {}
        for idx in unique_indices:
            node = entity_nodes[idx]
            specificity = 1 / len(self.graph.nodes[node]["ordering"])
            personalization[node] = (1 / n) * specificity

        n_dict_ppr = nx.pagerank(self.graph, personalization=personalization, alpha=0.5)

        n_prime = np.zeros(len(entity_nodes), dtype=float)
        for key, val in n_dict_ppr.items():
            if key in self.adj_entity_indices:
                n_prime[self.adj_entity_indices[key]] = val

        p = self.adjacency.T @ n_prime
        return self._top_passages(p, topK)

    def queryHippo(self, query):
        if self.version == 1:
            entities = self.onlineIE(query)
            entity_embeddings = [emb(e) for e in entities.named_entities]
            retrieved = self.onlineMatch(3, entity_embeddings)
        else:
            q_emb = emb(query)
            retrieved = self.onlineMatchV2WithQuery(3, query, q_emb)

        prompt = (
            "I have a question: " + query +
            "\n\nPlease make use of the following context to answer my question:\n\n" +
            "\n\n".join(retrieved)
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=100,
            temperature=0,
        )
        return response.choices[0].message.content.strip()



def main():
    hippo = Hippo()
    hippo.createGraph()
    question = "In which district was Alhandra born?"
    answer = hippo.queryHippo(question)
    print(f'Asnwer: {answer}\n')
    return

# algo:
# 1. offline: extract KG triples for all passages, NER to de-dupe
# 2. connect entities by synonymy (high cosine sim)
# 3. build adjacency matrix (|entities| x |passages|) marking entity presence
# 4. extract entities from query
# 5. match query entities to graph nodes by embedding similarity
# 6. weight matched nodes evenly, scale by specificity
# 7. run PPR with those as initial probabilities
# 8. multiply entity prob vector by adj matrix transpose
# 9. top-k passages from resulting scores
# 10. answer question

if __name__ == "__main__":
    main()
