"""
build_index.py

Embed and index the enriched Thirukural dataset into Qdrant.

Supports two embedding providers:
  1. "gemini" (default) -- uses Google's text-embedding-004 API.
     Lightweight, avoids huge PyTorch memory footprint (~1.5GB) on servers,
     and allows backend to run comfortably within Render / free tier limits.
  2. "local" -- uses SentenceTransformer (paraphrase-multilingual-mpnet-base-v2).
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Search and load environment variables from pipeline, backend, or workspace root
for env_candidate in [
    Path(__file__).resolve().parent.parent / ".env",
    Path(__file__).resolve().parent.parent.parent / "backend/.env",
    Path(__file__).resolve().parent.parent.parent / ".env",
]:
    if env_candidate.exists():
        load_dotenv(env_candidate)

INPUT_PATH = Path(__file__).resolve().parent.parent / "data/processed/kurals_enriched.json"

# Qdrant settings
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "thirukural")

# Embedding settings
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "text-embedding-004" if EMBEDDING_PROVIDER == "gemini" else "paraphrase-multilingual-mpnet-base-v2"
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def load_kurals() -> list[dict]:
    with INPUT_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def build_points_metadata(kurals: list[dict]) -> tuple[list[str], list[dict]]:
    """Returns (texts_to_embed, payloads) in matching order."""
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


def embed_texts_gemini(texts: list[str], model_name: str, api_key: str, batch_size: int = 100) -> list[list[float]]:
    from google import genai
    api_key_clean = (api_key or "").strip().strip('"').strip("'")
    if not api_key_clean:
        raise ValueError("GEMINI_API_KEY is required to index with the 'gemini' provider.")

    client = genai.Client(api_key=api_key_clean)
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"Embedding {len(texts)} texts via Gemini API ({model_name}) across {total_batches} batches...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = (i // batch_size) + 1

        for attempt in range(5):
            try:
                response = client.models.embed_content(
                    model=model_name,
                    contents=batch,
                )
                batch_vectors = [emb.values for emb in response.embeddings]
                all_embeddings.extend(batch_vectors)
                print(f"  Batch {batch_num}/{total_batches} ({min(i + batch_size, len(texts))}/{len(texts)}) completed.")
                # Safe rate-limiting delay for free tier (15 RPM limit -> ~4 seconds between batches)
                if i + batch_size < len(texts):
                    time.sleep(4)
                break
            except Exception as e:
                wait_time = (2 ** (attempt + 1)) * 3
                print(f"  Batch {batch_num} attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        else:
            raise RuntimeError(f"Batch {batch_num} failed after 5 retries.")

    return all_embeddings


def embed_texts_local(texts: list[str], model_name: str, batch_size: int = 64) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    print(f"Loading local SentenceTransformer model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("Embedding all texts locally...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def main():
    print(f"Loading kurals from {INPUT_PATH}...")
    kurals = load_kurals()
    print(f"Loaded {len(kurals)} kurals.")

    texts, payloads = build_points_metadata(kurals)
    print(f"Prepared {len(texts)} texts to embed "
          f"({len(texts) - len(kurals)} scenario tags + {len(kurals)} couplets).")

    # Connect to Qdrant early to fail fast before doing embedding work
    if QDRANT_URL:
        print(f"Connecting to Qdrant at {QDRANT_URL}...")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, check_compatibility=False)
    else:
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
    client.get_collections()
    print("Connected to Qdrant.")

    print(f"Embedding Provider: {EMBEDDING_PROVIDER.upper()} | Model: {MODEL_NAME}")
    if EMBEDDING_PROVIDER == "gemini":
        embeddings = embed_texts_gemini(texts, MODEL_NAME, GEMINI_API_KEY)
    else:
        embeddings = embed_texts_local(texts, MODEL_NAME)

    vector_size = len(embeddings[0])
    print(f"Embedding complete. Vector size: {vector_size}")

    print(f"(Re)creating Qdrant collection '{COLLECTION_NAME}'...")
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
        PointStruct(id=i, vector=embeddings[i], payload=payloads[i])
        for i in range(len(texts))
    ]

    upload_batch_size = 256
    for start in range(0, len(points), upload_batch_size):
        chunk = points[start : start + upload_batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=chunk)
        print(f"  Upserted {min(start + upload_batch_size, len(points))}/{len(points)}")

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"\nDone! Collection '{COLLECTION_NAME}' now has {count} points.")
    print(f"  - {len(texts) - len(kurals)} scenario_tag vectors")
    print(f"  - {len(kurals)} couplet vectors")


if __name__ == "__main__":
    main()
