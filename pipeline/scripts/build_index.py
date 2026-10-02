"""
build_index.py

Phase 6 of the Thirukural RAG project: embed and index the enriched
dataset into Qdrant.

Design:
  - Uses a LOCAL multilingual sentence-transformers model (no API, no
    rate limits, no daily quota -- after the Phase 4 saga, a one-time
    local embedding job is much less painful than another hosted API).
  - Two vector "types" are indexed per kural, so Phase 7 can compare
    which one drives better retrieval:
      1. "scenario_tag" -- ONE POINT PER SCENARIO TAG, not one blob per
         kural. A user's query only needs to be close to *one* of a
         kural's 2-12 tag phrasings, which is a much finer-grained
         match than averaging all tags into a single vector.
      2. "couplet" -- one point per kural, embedding the English
         couplet + explanation together, as a fallback/complementary
         signal for direct semantic-literal matches.
  - Every point's payload carries enough metadata (kural_number, book,
    iyal, adikaram, vector_type, source text) to retrieve and display
    the full kural without a second lookup.

Requires: pip install sentence-transformers qdrant-client
Assumes Qdrant is running locally (docker run -p 6333:6333 qdrant/qdrant).
"""

import json
import os
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path(__file__).resolve().parent.parent / "data/processed/kurals_enriched.json"
# Set QDRANT_URL (+ QDRANT_API_KEY) to index into a hosted cluster, e.g. Qdrant
# Cloud; otherwise targets the local docker instance.
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "thirukural")
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-mpnet-base-v2")
BATCH_SIZE = 64


def load_kurals() -> list[dict]:
    with INPUT_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def build_points_metadata(kurals: list[dict]) -> tuple[list[str], list[dict]]:
    """Returns (texts_to_embed, payloads) in matching order. Each entry
    is either a single scenario tag or a couplet+explanation blob."""
    texts = []
    payloads = []

    for k in kurals:
        base_payload = {
            "kural_number": k["kural_number"],
            "book": k["book"],
            "iyal": k["iyal"],
            "adikaram": k["adikaram"],
            "tamil_couplet": k["tamil_couplet"],
            "transliteration": k["transliteration"],
            "english_couplet": k["english_couplet"],
            "english_explanation": k["english_explanation"],
        }

        # One point per scenario tag
        for tag in k.get("scenario_tags", []):
            texts.append(tag)
            payloads.append({
                **base_payload,
                "vector_type": "scenario_tag",
                "matched_text": tag,
            })

        # One point for the couplet + explanation together
        couplet_text = f"{k['english_couplet']} {k['english_explanation']}"
        texts.append(couplet_text)
        payloads.append({
            **base_payload,
            "vector_type": "couplet",
            "matched_text": couplet_text,
        })

    return texts, payloads


def main():
    print(f"Loading kurals from {INPUT_PATH}...")
    kurals = load_kurals()
    print(f"Loaded {len(kurals)} kurals.")

    texts, payloads = build_points_metadata(kurals)
    print(f"Prepared {len(texts)} texts to embed "
          f"({len(texts) - len(kurals)} scenario tags + {len(kurals)} couplets).")

    # Connect BEFORE the (slow) embedding step so a bad URL/key fails in
    # seconds rather than after 15 minutes of CPU work.
    if QDRANT_URL:
        print(f"Connecting to Qdrant at {QDRANT_URL}...")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, check_compatibility=False)
    else:
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
    client.get_collections()  # raises immediately if unreachable or unauthorized
    print("Connected.")

    print(f"Loading embedding model '{MODEL_NAME}' (this may download on first run)...")
    model = SentenceTransformer(MODEL_NAME)
    vector_size = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Vector size: {vector_size}")

    print("Embedding all texts locally (no API calls, no rate limits)...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # so cosine similarity == dot product
    )

    print(f"(Re)creating collection '{COLLECTION_NAME}'...")
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print("Creating payload index for 'vector_type'...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="vector_type",
        field_schema="keyword",
    )

    print("Upserting points into Qdrant...")
    points = [
        PointStruct(id=i, vector=embeddings[i].tolist(), payload=payloads[i])
        for i in range(len(texts))
    ]

    # Upload in chunks to avoid one giant request
    upload_batch_size = 256
    for start in range(0, len(points), upload_batch_size):
        chunk = points[start:start + upload_batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=chunk)
        print(f"  Upserted {min(start + upload_batch_size, len(points))}/{len(points)}")

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {count} points.")
    print(f"  - {len(texts) - len(kurals)} scenario_tag vectors")
    print(f"  - {len(kurals)} couplet vectors")


if __name__ == "__main__":
    main()
