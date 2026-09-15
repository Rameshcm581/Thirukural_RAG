"""
routes/__init__.py

Aggregates every controller's router into one, included once in main.py.
Chat routes will be added here in Phase 9.
"""

from fastapi import APIRouter

from resources.searchController import router as search_router

router = APIRouter()

router.include_router(search_router, tags=["Search"])
