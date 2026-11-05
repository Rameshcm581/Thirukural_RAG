"""
build_index.py

Embed and index the enriched Thirukural dataset into Qdrant.

Supports two embedding providers:
  1. "gemini" (default) -- uses Google's gemini-embedding-001 API (output dim: 768).
     Lightweight, avoids huge PyTorch memory footprint (~1.5GB) on servers,
     and allows backend to run comfortably within Render / free tier limits.
     Includes disk checkpointing (caches every batch) and auto-key rotation.
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data/processed"
INPUT_PATH = DATA_DIR / "kurals_enriched.json"
CACHE_PATH = DATA_DIR / "gemini_embeddings_cache.json"

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
    "gemini-embedding-001" if EMBEDDING_PROVIDER == "gemini" else "paraphrase-multilingual-mpnet-base-v2"
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEYS_RAW = os.getenv("GEMINI_API_KEYS", "")


def get_gemini_keys() -> list[str]:
    """Collect all available Gemini API keys for rotation."""
    keys = []
    if GEMINI_API_KEYS_RAW:
        for k in GEMINI_API_KEYS_RAW.split(","):
            cleaned = k.strip().strip('"').strip("'")
            if cleaned and cleaned not in keys:
                keys.append(cleaned)
    if GEMINI_API_KEY:
        cleaned = GEMINI_API_KEY.strip().strip('"').strip("'")
        if cleaned and cleaned not in keys:
            keys.append(cleaned)
    return keys


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


def load_cached_embeddings() -> list[list[float]]:
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    print(f"Loaded {len(data)} cached embeddings from {CACHE_PATH.name}")
                    return data
        except Exception as e:
            print(f"Warning: Could not read cache {CACHE_PATH}: {e}")
    return []


def save_cached_embeddings(embeddings: list[list[float]]):
    try:
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(embeddings, f)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")


def embed_texts_gemini(texts: list[str], model_name: str, batch_size: int = 100) -> list[list[float]]:
    from google import genai
    from google.genai import types

    keys = get_gemini_keys()
    if not keys:
        raise ValueError("GEMINI_API_KEY (or GEMINI_API_KEYS) is required to index with 'gemini'.")

    active_keys = list(keys)
    clients = {k: genai.Client(api_key=k) for k in active_keys}
    all_embeddings = load_cached_embeddings()

    total_texts = len(texts)
    start_index = len(all_embeddings)
    total_batches = (total_texts + batch_size - 1) // batch_size

    if start_index >= total_texts:
        print(f"All {total_texts} texts are already cached! Proceeding to upload...")
        return all_embeddings[:total_texts]

    start_batch = (start_index // batch_size) + 1
    print(f"Embedding remaining texts ({start_index}/{total_texts}) using {len(active_keys)} keys starting at batch {start_batch}/{total_batches}...")

    key_idx = 0
    for i in range(start_index, total_texts, batch_size):
        batch = texts[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        success = False

        while not success:
            if not active_keys:
                save_cached_embeddings(all_embeddings)
                raise RuntimeError(
                    f"\nAll available Gemini API keys have exhausted their daily quota.\n"
                    f"Progress saved: {len(all_embeddings)}/{total_texts} items are safely cached in {CACHE_PATH.name}.\n"
                    f"Please provide additional keys or resume tomorrow to complete the remaining batches."
                )

            current_key = active_keys[key_idx % len(active_keys)]
            client = clients[current_key]

            try:
                response = client.models.embed_content(
                    model=model_name,
                    contents=batch,
                    config=types.EmbedContentConfig(output_dimensionality=768),
                )
                batch_vectors = [emb.values for emb in response.embeddings]
                all_embeddings.extend(batch_vectors)
                save_cached_embeddings(all_embeddings)

                print(f"  Batch {batch_num}/{total_batches} ({min(i + batch_size, total_texts)}/{total_texts}) completed with Key #{keys.index(current_key) + 1}.")
                key_idx += 1
                success = True
                # Polite 1s pause between batches
                time.sleep(1)

            except Exception as e:
                err_str = str(e)
                if "per day limit exceeded" in err_str or "requests per day" in err_str:
                    print(f"  Key #{keys.index(current_key) + 1} reached its daily quota and is being retired for today.")
                    active_keys.remove(current_key)
                    if active_keys:
                        print(f"  {len(active_keys)} keys remaining. Continuing with next key...")
                elif "per minute limit exceeded" in err_str:
                    print(f"  Key #{keys.index(current_key) + 1} hit per-minute limit. Rotating to next key...")
                    key_idx += 1
                    time.sleep(2)
                else:
                    print(f"  Batch {batch_num} error with Key #{keys.index(current_key) + 1}: {e}")
                    key_idx += 1
                    time.sleep(3)

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
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=60, check_compatibility=False)
    else:
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60, check_compatibility=False)
    client.get_collections()
    print("Connected to Qdrant.")

    print(f"Embedding Provider: {EMBEDDING_PROVIDER.upper()} | Model: {MODEL_NAME}")
    if EMBEDDING_PROVIDER == "gemini":
        embeddings = embed_texts_gemini(texts, MODEL_NAME)
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

    upload_batch_size = 100
    for start in range(0, len(points), upload_batch_size):
        chunk = points[start : start + upload_batch_size]
        for attempt in range(5):
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=chunk)
                break
            except Exception as e:
                print(f"  Upsert retry {attempt + 1}/5: {e}")
                time.sleep(2)
        print(f"  Upserted {min(start + upload_batch_size, len(points))}/{len(points)}")

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"\nDone! Collection '{COLLECTION_NAME}' now has {count} points.")
    print(f"  - {len(texts) - len(kurals)} scenario_tag vectors")
    print(f"  - {len(kurals)} couplet vectors")


if __name__ == "__main__":
    main()
