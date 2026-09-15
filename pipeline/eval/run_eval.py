"""
run_eval.py

Phase 7 of the Thirukural RAG project: baseline retrieval evaluation.

Runs every scenario in eval/gold_set.json against the Qdrant index and
reports recall@K -- did the expected kural(s) show up in the top K
results? Measures three retrieval modes separately so we can see which
vector type actually earns its keep:

  1. scenario_tag only -- search using only the scenario_tag vectors
  2. couplet only       -- search using only the couplet+explanation vectors
  3. combined           -- search across both vector types together

This comparison is the direct evidence for whether the whole scenario-
tag enrichment effort (Phases 3-4) actually improved retrieval over
naive couplet-only search, which is the central hypothesis of this
project.

Requires: pip install sentence-transformers qdrant-client
"""

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

GOLD_SET_PATH = Path("eval/gold_set.json")
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "thirukural"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
TOP_K = 5
SEARCH_LIMIT_RAW = 20  # over-fetch raw points before dedup by kural_number,
                        # since multiple scenario_tag points can share a kural


def load_gold_set() -> list[dict]:
    with GOLD_SET_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def dedup_by_kural(raw_results, top_k: int) -> list[int]:
    """Qdrant returns individual points (one per scenario tag or couplet).
    Collapse to unique kural_numbers, keeping the best (first, since
    results are score-sorted) occurrence of each, up to top_k."""
    seen = []
    for point in raw_results:
        num = point.payload["kural_number"]
        if num not in seen:
            seen.append(num)
        if len(seen) >= top_k:
            break
    return seen


def search(client, model, query: str, vector_type: str | None, top_k: int) -> list[int]:
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    if vector_type is not None:
        query_filter = Filter(
            must=[FieldCondition(key="vector_type", match=MatchValue(value=vector_type))]
        )
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=SEARCH_LIMIT_RAW,
        )
        return dedup_by_kural(response.points, top_k)

    # "Combined" mode: search each vector type SEPARATELY and merge by
    # best score, rather than one unified kNN. With 7340 scenario_tag
    # points vs only 1330 couplet points, a single unified top-20 search
    # gets dominated by the far more numerous scenario_tag vectors and
    # couplet results never surface at all -- this keeps both channels
    # genuinely represented.
    tag_filter = Filter(must=[FieldCondition(key="vector_type", match=MatchValue(value="scenario_tag"))])
    couplet_filter = Filter(must=[FieldCondition(key="vector_type", match=MatchValue(value="couplet"))])

    tag_response = client.query_points(
        collection_name=COLLECTION_NAME, query=query_vector,
        query_filter=tag_filter, limit=SEARCH_LIMIT_RAW,
    )
    couplet_response = client.query_points(
        collection_name=COLLECTION_NAME, query=query_vector,
        query_filter=couplet_filter, limit=SEARCH_LIMIT_RAW,
    )

    # Merge by score (higher is better; both are cosine similarity on
    # normalized vectors, so directly comparable), then dedup by kural.
    merged = sorted(
        list(tag_response.points) + list(couplet_response.points),
        key=lambda p: p.score,
        reverse=True,
    )
    return dedup_by_kural(merged, top_k)


def evaluate_mode(client, model, gold_set: list[dict], vector_type: str | None, top_k: int) -> dict:
    hits = 0
    misses = []

    for item in gold_set:
        scenario = item["scenario"]
        expected = set(item["expected_kural_numbers"])
        retrieved = search(client, model, scenario, vector_type, top_k)
        retrieved_at_10 = search(client, model, scenario, vector_type, 10)

        if expected & set(retrieved):
            hits += 1
        else:
            near_miss = bool(expected & set(retrieved_at_10))
            misses.append({
                "scenario": scenario,
                "expected": list(expected),
                "retrieved": retrieved,
                "near_miss_at_10": near_miss,
            })

    recall = hits / len(gold_set)
    return {"recall": recall, "hits": hits, "total": len(gold_set), "misses": misses}


def print_mode_result(label: str, result: dict, show_misses: bool = True):
    print(f"\n=== {label} ===")
    print(f"Recall@{TOP_K}: {result['recall']:.1%} ({result['hits']}/{result['total']})")
    near_misses = sum(1 for m in result["misses"] if m["near_miss_at_10"])
    if result["misses"]:
        print(f"Of the {len(result['misses'])} misses, {near_misses} were near-misses "
              f"(correct kural appeared in top 10 but not top {TOP_K}).")
    if show_misses and result["misses"]:
        print(f"\nMisses ({len(result['misses'])}):")
        for m in result["misses"]:
            tag = " [NEAR MISS - in top 10]" if m["near_miss_at_10"] else " [complete miss]"
            print(f"  Scenario: {m['scenario']}{tag}")
            print(f"    Expected: {m['expected']}, Retrieved: {m['retrieved']}")


def main():
    gold_set = load_gold_set()
    print(f"Loaded {len(gold_set)} gold-set scenarios.")

    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)

    print("\nRunning evaluation across three retrieval modes...")

    couplet_only = evaluate_mode(client, model, gold_set, "couplet", TOP_K)
    scenario_only = evaluate_mode(client, model, gold_set, "scenario_tag", TOP_K)
    combined = evaluate_mode(client, model, gold_set, None, TOP_K)

    print_mode_result("Couplet-only (naive baseline)", couplet_only)
    print_mode_result("Scenario-tag-only (enrichment)", scenario_only)
    print_mode_result("Combined (both vector types)", combined)

    print("\n=== Summary ===")
    print(f"Couplet-only:     {couplet_only['recall']:.1%}")
    print(f"Scenario-tag-only: {scenario_only['recall']:.1%}")
    print(f"Combined:         {combined['recall']:.1%}")


if __name__ == "__main__":
    main()
