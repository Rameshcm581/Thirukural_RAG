"""
diagnose_misses.py

Diagnostic helper for Phase 7: for a given gold-set scenario, shows:
  1. The actual scenario_tags stored for the EXPECTED kural, each with
     its cosine similarity to the query -- so we can see whether a
     good matching tag exists but just didn't score high enough, or
     whether none of the tags are actually a good semantic match.
  2. The scenario_tags that WERE retrieved instead (the false positives),
     with their scores, for direct comparison.

This tells us whether a miss is a "tagging problem" (the right tag
doesn't exist or is poorly worded) or a "model problem" (a good tag
exists but the embedding model didn't rank it highly enough).

Usage:
    python3 eval/diagnose_misses.py "My coworker keeps taking credit for my work and I want to snap at them in the meeting" 305
"""

import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, util

KURALS_PATH = Path("data/processed/kurals_enriched.json")
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "thirukural"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


def main():
    if len(sys.argv) < 3:
        print('Usage: python3 eval/diagnose_misses.py "<scenario text>" <expected_kural_number>')
        sys.exit(1)

    scenario = sys.argv[1]
    expected_num = int(sys.argv[2])

    with KURALS_PATH.open(encoding="utf-8") as f:
        kurals = {k["kural_number"]: k for k in json.load(f)}

    expected_kural = kurals.get(expected_num)
    if not expected_kural:
        print(f"Kural {expected_num} not found.")
        sys.exit(1)

    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"\nQuery: {scenario}")
    print(f"Expected kural: {expected_num} ({expected_kural['adikaram']})")

    query_embedding = model.encode(scenario, normalize_embeddings=True)

    print(f"\n--- Stored scenario_tags for kural {expected_num}, ranked by similarity to query ---")
    tags = expected_kural.get("scenario_tags", [])
    if not tags:
        print("  (no scenario tags stored for this kural)")
    else:
        tag_embeddings = model.encode(tags, normalize_embeddings=True)
        scored_tags = sorted(
            zip(tags, util.cos_sim(query_embedding, tag_embeddings)[0].tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        for tag, score in scored_tags:
            print(f"  {score:.3f}  {tag}")

    print(f"\n--- What was actually retrieved instead (top 5, scenario_tag vectors only) ---")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
    tag_filter = Filter(must=[FieldCondition(key="vector_type", match=MatchValue(value="scenario_tag"))])
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        query_filter=tag_filter,
        limit=5,
    )
    for point in response.points:
        print(f"  {point.score:.3f}  Kural {point.payload['kural_number']} "
              f"({point.payload['adikaram']}): \"{point.payload['matched_text']}\"")


if __name__ == "__main__":
    main()
