"""
main.py

FastAPI entrypoint for ThirukuralService.

Follows the reference project's main.py pattern: structured logging is
configured before any other import that logs, resources (embedding
model + Qdrant client) are loaded once via `lifespan` rather than
per-request, and routes are registered as a single aggregated router.
"""

import logging
import os
import pathlib
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(pathlib.Path(__file__).parent / ".env")

# ── Logging setup (must run before any other import that logs) ─────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from resources import load_embedding_model, connect_qdrant  # noqa: E402
from routes import router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model and connect to Qdrant once at startup,
    not per-request -- the model is ~1GB and reloading it per request
    would be prohibitively slow."""
    logger.info("ThirukuralService starting -- loading embedding model and connecting to Qdrant")
    load_embedding_model()
    connect_qdrant()
    logger.info("ThirukuralService startup complete")
    yield
    logger.info("ThirukuralService shutting down")


app = FastAPI(
    title="ThirukuralService",
    description="RAG service retrieving Thirukural couplets based on modern life scenarios",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def root():
    """Service info endpoint."""
    return {
        "service": "ThirukuralService",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"service": "ThirukuralService", "status": "healthy"}


app.include_router(router)
