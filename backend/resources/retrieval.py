"""
resources/retrieval.py

Core retrieval logic -- the single source of truth used by both /search
and /chat. Ported directly from eval/run_eval.py's "combined" mode,
which was validated at 83.3% recall@5 against the 42-scenario gold set.

Any future retrieval improvement (reranking, hybrid search, a different
embedding model) should change ONLY this file.
"""

import logging

from qdrant_client.models import Filter, FieldCondition, MatchValue

from configs import BaseConfig
from resources import get_embedding_model, get_qdrant_client

logger = logging.getLogger(__name__)


def _dedup_by_kural(points, top_k: int) -> list:
    """Qdrant returns individual points (one per scenario tag or couplet).
    Collapse to unique kural_numbers, keeping the best (first, since
    results are score-sorted) occurrence of each, up to top_k."""
    seen_nums = set()
    deduped = []
    for point in points:
        num = point.payload["kural_number"]
        if num not in seen_nums:
            seen_nums.add(num)
            deduped.append(point)
        if len(deduped) >= top_k:
            break
    return deduped


def retrieve_kurals(scenario: str, top_k: int = None) -> list[dict]:
    """
    Embed the scenario, search both scenario_tag and couplet vectors
    separately (validated approach -- a single unified kNN search gets
    dominated by the ~5.5x more numerous scenario_tag points), merge by
    score, dedupe by kural_number, and return the top_k as plain dicts
    ready to serialize into KuralResult.
    """
    top_k = top_k or BaseConfig.TOP_K
    model = get_embedding_model()
    client = get_qdrant_client()

    query_vector = model.encode(scenario, normalize_embeddings=True).tolist()

    tag_filter = Filter(must=[FieldCondition(key="vector_type", match=MatchValue(value="scenario_tag"))])
    couplet_filter = Filter(must=[FieldCondition(key="vector_type", match=MatchValue(value="couplet"))])

    tag_response = client.query_points(
        collection_name=BaseConfig.COLLECTION_NAME,
        query=query_vector,
        query_filter=tag_filter,
        limit=BaseConfig.SEARCH_LIMIT_RAW,
    )
    couplet_response = client.query_points(
        collection_name=BaseConfig.COLLECTION_NAME,
        query=query_vector,
        query_filter=couplet_filter,
        limit=BaseConfig.SEARCH_LIMIT_RAW,
    )

    merged = sorted(
        list(tag_response.points) + list(couplet_response.points),
        key=lambda p: p.score,
        reverse=True,
    )
    top_points = _dedup_by_kural(merged, top_k)

    results = []
    for point in top_points:
        payload = point.payload
        results.append({
            "kural_number": payload["kural_number"],
            "book": payload["book"],
            "adikaram": payload["adikaram"],
            "tamil_couplet": payload["tamil_couplet"],
            "transliteration": payload["transliteration"],
            "english_couplet": payload["english_couplet"],
            "english_explanation": payload["english_explanation"],
            "matched_tag": payload["matched_text"],
            "similarity_score": point.score,
        })

    logger.info("retrieve_kurals | scenario=%r | top_k=%d | results=%s",
                scenario, top_k, [r["kural_number"] for r in results])
    return results
