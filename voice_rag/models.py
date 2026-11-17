from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class KnowledgePassage(BaseModel):
    passage_id: str
    sku: str
    title: str
    content: str
    embedding: List[float]


class VoiceQueryRequest(BaseModel):
    call_sid: str
    utterance: str
    query_embedding: List[float]
    simulated_webhook_latency_ms: int = Field(default=120, ge=0)


class VoiceQueryResponse(BaseModel):
    call_sid: str
    matched_sku: Optional[str] = None
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    escalate_to_human: bool = False
    timed_out: bool = False
    latency_ms: int = Field(ge=0)
