"""
resources/__init__.py

Manages embedding resources (Gemini embedding API or local SentenceTransformer)
and the Qdrant vector database client.
"""

import logging
from typing import Any

from google import genai
from google.genai import types
from qdrant_client import QdrantClient

from configs import BaseConfig

logger = logging.getLogger(__name__)

_local_embedding_model: Any = None
_gemini_client: genai.Client | None = None
_qdrant_client: QdrantClient | None = None


def get_gemini_client() -> genai.Client | None:
    """Returns an initialized Gemini client, or None if API key is missing."""
    global _gemini_client
    if _gemini_client is None:
        api_key = (BaseConfig.GEMINI_API_KEY or "").strip().strip('"').strip("'")
        if not api_key:
            return None
        try:
            _gemini_client = genai.Client(api_key=api_key)
        except Exception as exc:
            logger.warning("Failed to initialize Gemini client: %s", exc)
            return None
    return _gemini_client


def load_embedding_model():
    """Initializes the configured embedding provider once during app lifespan."""
    provider = BaseConfig.EMBEDDING_PROVIDER
    if provider == "gemini":
        logger.info("Initializing Gemini embedding client (model: %s)...", BaseConfig.EMBEDDING_MODEL_NAME)
        client = get_gemini_client()
        if client is None:
            logger.warning("Gemini client not initialized -- check GEMINI_API_KEY")
        else:
            logger.info("Gemini embedding client ready.")
        return client
    else:
        global _local_embedding_model
        if _local_embedding_model is None:
            logger.info("Loading local embedding model '%s'...", BaseConfig.EMBEDDING_MODEL_NAME)
            from sentence_transformers import SentenceTransformer
            _local_embedding_model = SentenceTransformer(BaseConfig.EMBEDDING_MODEL_NAME)
            logger.info("Local embedding model loaded.")
        return _local_embedding_model


def get_embedding_model():
    """Accessor for local model if local provider is used."""
    if BaseConfig.EMBEDDING_PROVIDER != "gemini" and _local_embedding_model is None:
        raise RuntimeError(
            "Local embedding model not loaded -- load_embedding_model() must be "
            "called during app startup (see main.py lifespan)."
        )
    return _local_embedding_model


def embed_text(text: str) -> list[float]:
    """Embeds a single query text using the configured provider (Gemini API or local)."""
    provider = BaseConfig.EMBEDDING_PROVIDER
    if provider == "gemini":
        client = get_gemini_client()
        if client is None:
            raise RuntimeError("GEMINI_API_KEY not configured or client failed to initialize.")
        response = client.models.embed_content(
            model=BaseConfig.EMBEDDING_MODEL_NAME,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        return response.embeddings[0].values
    else:
        model = get_embedding_model()
        return model.encode(text, normalize_embeddings=True).tolist()


def connect_qdrant() -> QdrantClient:
    """Connect to Qdrant once. Safe to call multiple times."""
    global _qdrant_client
    if _qdrant_client is None:
        if BaseConfig.QDRANT_URL:
            logger.info("Connecting to Qdrant at %s...", BaseConfig.QDRANT_URL)
            _qdrant_client = QdrantClient(
                url=BaseConfig.QDRANT_URL,
                api_key=BaseConfig.QDRANT_API_KEY or None,
                check_compatibility=False,
            )
        else:
            logger.info("Connecting to Qdrant at %s:%s...", BaseConfig.QDRANT_HOST, BaseConfig.QDRANT_PORT)
            _qdrant_client = QdrantClient(
                host=BaseConfig.QDRANT_HOST,
                port=BaseConfig.QDRANT_PORT,
                check_compatibility=False,
            )
        logger.info("Connected to Qdrant.")
    return _qdrant_client


def get_qdrant_client() -> QdrantClient:
    """Dependency-style accessor for use inside request handlers."""
    if _qdrant_client is None:
        raise RuntimeError(
            "Qdrant client not connected -- connect_qdrant() must be "
            "called during app startup (see main.py lifespan)."
        )
    return _qdrant_client


def check_health() -> dict:
    """Returns status dictionary verifying embedding provider and Qdrant connectivity."""
    provider = BaseConfig.EMBEDDING_PROVIDER
    if provider == "gemini":
        model_status = "ready (gemini API)" if _gemini_client is not None else "api_key_missing"
    else:
        model_status = "loaded" if _local_embedding_model is not None else "not_loaded"

    qdrant_status = "not_connected"
    if _qdrant_client is not None:
        try:
            _qdrant_client.get_collections()
            qdrant_status = "connected"
        except Exception as exc:
            qdrant_status = f"unreachable: {exc}"
    is_healthy = ("ready" in model_status or model_status == "loaded") and qdrant_status == "connected"
    return {
        "status": "healthy" if is_healthy else "degraded",
        "embedding_provider": provider,
        "embedding_model": model_status,
        "qdrant": qdrant_status,
    }
