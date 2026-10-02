"""
schemas.py

Pydantic models -- the API contract for every endpoint.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    scenario: str = Field(..., min_length=1, max_length=1000, description="Modern scenario description")


class KuralResult(BaseModel):
    kural_number: int
    book: str
    adikaram: str
    tamil_couplet: list[str]
    transliteration: str
    english_couplet: str
    english_explanation: str
    matched_tag: str
    similarity_score: float
    explanation: Optional[str] = None


class SearchResponse(BaseModel):
    results: list[KuralResult]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    kural_results: Optional[list[KuralResult]] = None
    mode: Literal["answered", "clarifying_question", "multiple_options"]
