"""
batch_diagnose_misses.py

Phase 7 diagnostic: runs the scenario-tag-only eval, finds every
COMPLETE miss (not near-misses), and auto-classifies each one as:

  - "SAME-CHAPTER AMBIGUITY": the top retrieved kural is in the same
    adikaram (chapter) as the expected kural. This usually means the
    gold set picked one valid kural among several closely-related
    ones in a tightly clustered virtue chapter -- a gold-set labeling
    issue, not a retrieval bug. Suggests adding the competitor as an
    additional acceptable answer.

  - "GENUINE GAP": the top retrieved kural is in a different chapter
    entirely, AND the expected kural's own best tag scores meaningfully
    lower than the competitor's. This suggests the expected kural's
    tags genuinely missed the scenario's central concept.

Prints a classification table, then prints ready-to-review suggested
gold_set.json edits for the same-chapter-ambiguity cases.
"""

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, util

GOLD_SET_PATH = Path("eval/gold_set.json")
KURALS_PATH = Path("data/processed/kurals_enriched.json")
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "thirukural"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
TOP_K = 5
SEARCH_LIMIT_RAW = 20
AMBIGUITY_SCORE_GAP_THRESHOLD = 0.15  # if expected kural's best tag is within
    # this much of the top competitor's score, treat as ambiguity even if
    # chapters differ slightly, since it's a close call either way


def dedup_by_kural(raw_results, top_k):
    seen = []
    for point in raw_results:
        num = point.payload["kural_number"]
        if num not in seen:
            seen.append(num)
        if len(seen) >= top_k:
            break
    return seen


def main():
    with GOLD_SET_PATH.open(encoding="utf-8") as f:
        gold_set = json.load(f)
    with KURALS_PATH.open(encoding="utf-8") as f:
        kurals = {k["kural_number"]: k for k in json.load(f)}

    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
    tag_filter = Filter(must=[FieldCondition(key="vector_type", match=MatchValue(value="scenario_tag"))])

    ambiguity_cases = []
    genuine_gaps = []

    for item in gold_set:
        scenario = item["scenario"]
        expected_nums = set(item["expected_kural_numbers"])
        query_embedding = model.encode(scenario, normalize_embeddings=True)

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            query_filter=tag_filter,
            limit=SEARCH_LIMIT_RAW,
        )
        retrieved = dedup_by_kural(response.points, TOP_K)

        if expected_nums & set(retrieved):
            continue  # not a miss, skip

        # It's a miss. Get the expected kural's best tag score.
        expected_num = next(iter(expected_nums))
        expected_kural = kurals.get(expected_num)
        expected_tags = expected_kural.get("scenario_tags", []) if expected_kural else []

        if expected_tags:
            tag_embeddings = model.encode(expected_tags, normalize_embeddings=True)
            expected_best_score = util.cos_sim(query_embedding, tag_embeddings)[0].max().item()
        else:
            expected_best_score = 0.0

        top_competitor_point = response.points[0]
        top_competitor_num = top_competitor_point.payload["kural_number"]
        top_competitor_adikaram = top_competitor_point.payload["adikaram"]
        top_competitor_score = top_competitor_point.score
        expected_adikaram = expected_kural["adikaram"] if expected_kural else "?"

        record = {
            "scenario": scenario,
            "expected_num": expected_num,
            "expected_adikaram": expected_adikaram,
            "expected_best_score": expected_best_score,
            "top_competitor_num": top_competitor_num,
            "top_competitor_adikaram": top_competitor_adikaram,
            "top_competitor_score": top_competitor_score,
        }

        same_chapter = top_competitor_adikaram == expected_adikaram
        close_score = (top_competitor_score - expected_best_score) < AMBIGUITY_SCORE_GAP_THRESHOLD

        if same_chapter or close_score:
            ambiguity_cases.append(record)
        else:
            genuine_gaps.append(record)

    print(f"\n{'='*80}")
    print(f"SAME-CHAPTER / CLOSE-SCORE AMBIGUITY ({len(ambiguity_cases)} cases)")
    print("Likely gold-set labeling issue -- consider adding competitor as an "
          "additional acceptable answer.")
    print(f"{'='*80}")
    for r in ambiguity_cases:
        print(f"\nScenario: {r['scenario']}")
        print(f"  Expected: {r['expected_num']} ({r['expected_adikaram']}), best tag score: {r['expected_best_score']:.3f}")
        print(f"  Top competitor: {r['top_competitor_num']} ({r['top_competitor_adikaram']}), score: {r['top_competitor_score']:.3f}")

    print(f"\n{'='*80}")
    print(f"GENUINE GAPS ({len(genuine_gaps)} cases)")
    print("Different chapter AND meaningfully lower score -- likely a real "
          "tagging gap in the expected kural.")
    print(f"{'='*80}")
    for r in genuine_gaps:
        print(f"\nScenario: {r['scenario']}")
        print(f"  Expected: {r['expected_num']} ({r['expected_adikaram']}), best tag score: {r['expected_best_score']:.3f}")
        print(f"  Top competitor: {r['top_competitor_num']} ({r['top_competitor_adikaram']}), score: {r['top_competitor_score']:.3f}")


if __name__ == "__main__":
    main()
