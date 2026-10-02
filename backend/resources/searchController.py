"""
resources/searchController.py

Follows the reference project's controller pattern exactly:
module-level logger, explicit logging per outcome (INFO success/empty,
WARNING bad input, ERROR unexpected), HTTPException passthrough, and a
catch-all Exception handler with exc_info=True for full tracebacks.
"""

import concurrent.futures
import logging

from fastapi import APIRouter, HTTPException, status

from resources.retrieval import retrieve_kurals
from resources.explain import generate_explanation
from schemas import SearchRequest, SearchResponse, KuralResult

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
def search(request: SearchRequest):
    """Single-turn search: scenario in, top-k kurals + explanations out."""
    try:
        scenario = request.scenario.strip()
        # ── Input validation ─────────────────────────────────────────────
        if not scenario:
            logger.warning("search | 400 | empty scenario")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Scenario cannot be empty",
            )

        # ── Retrieval ────────────────────────────────────────────────────
        raw_results = retrieve_kurals(scenario)

        if not raw_results:
            logger.info("search | 200 | no results | scenario=%r", scenario)
            return SearchResponse(results=[])

        # ── Grounded explanation per result in parallel ──────────────────
        max_workers = min(len(raw_results), 5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            explanations = list(
                executor.map(lambda r: generate_explanation(scenario, r), raw_results)
            )

        results = [
            KuralResult(**r, explanation=exp)
            for r, exp in zip(raw_results, explanations)
        ]

        logger.info(
            "search | 200 | scenario=%r | top_kural=%s | count=%d",
            scenario, results[0].kural_number, len(results),
        )
        return SearchResponse(results=results)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "search | 500 | scenario=%r | error=%s",
            request.scenario, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
