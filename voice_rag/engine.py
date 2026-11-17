"""Core hybrid voice RAG engine with phonetic SKU resolution and escalation gates."""

import math
import re
from typing import Dict, List, Optional, Tuple

from voice_rag.models import KnowledgePassage, VoiceQueryRequest, VoiceQueryResponse


class EmbeddingDimensionError(ValueError):
    """Raised when an embedding vector does not match the configured dimension."""


_PHONETIC_MAP: Dict[str, str] = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "won": "1",
    "two": "2",
    "too": "2",
    "to": "2",
    "three": "3",
    "tree": "3",
    "four": "4",
    "for": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "ate": "8",
    "nine": "9",
    "niner": "9",
    "ay": "a",
    "bee": "b",
    "see": "c",
    "dee": "d",
    "ee": "e",
    "ef": "f",
    "gee": "g",
    "aitch": "h",
    "jay": "j",
    "kay": "k",
    "el": "l",
    "em": "m",
    "en": "n",
    "pea": "p",
    "cue": "q",
    "queue": "q",
    "are": "r",
    "ess": "s",
    "tee": "t",
    "you": "u",
    "vee": "v",
    "ex": "x",
    "why": "y",
    "zee": "z",
    "zed": "z",
    "dash": "",
    "hyphen": "",
}


def _normalize_sku_text(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    mapped = [_PHONETIC_MAP.get(tok, tok) for tok in tokens]
    return "".join(mapped)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


class VoiceRAGDeskEngine:
    """Low-latency hybrid retrieval engine for voice support desks."""

    def __init__(
        self,
        expected_dim: int = 4,
        min_confidence: float = 0.65,
        webhook_timeout_ms: int = 800,
    ) -> None:
        self.expected_dim = expected_dim
        self.min_confidence = min_confidence
        self.webhook_timeout_ms = webhook_timeout_ms
        self._passages: List[KnowledgePassage] = []

    def _validate_embedding(self, embedding: List[float], label: str) -> None:
        if len(embedding) != self.expected_dim:
            raise EmbeddingDimensionError(
                f"{label} embedding dimension {len(embedding)} != expected {self.expected_dim}"
            )

    def ingest(self, passages: List[KnowledgePassage]) -> None:
        for passage in passages:
            self._validate_embedding(passage.embedding, f"Passage {passage.passage_id}")
        self._passages.extend(passages)

    def _match_sku(self, utterance: str) -> Optional[Tuple[KnowledgePassage, float]]:
        utterance_upper = utterance.upper()
        norm_utterance = _normalize_sku_text(utterance)
        for passage in self._passages:
            if passage.sku.upper() in utterance_upper:
                return passage, 1.0
            norm_sku = _normalize_sku_text(passage.sku)
            if norm_sku and norm_sku in norm_utterance:
                return passage, 0.94
        return None

    def query(self, request: VoiceQueryRequest) -> VoiceQueryResponse:
        if request.simulated_webhook_latency_ms > self.webhook_timeout_ms:
            return VoiceQueryResponse(
                call_sid=request.call_sid,
                matched_sku=None,
                answer=(
                    "Our voice assistant is experiencing a brief delay. "
                    "Transferring you to a human specialist now."
                ),
                confidence=0.0,
                escalate_to_human=True,
                timed_out=True,
                latency_ms=request.simulated_webhook_latency_ms,
            )

        self._validate_embedding(request.query_embedding, "Query")

        sku_hit = self._match_sku(request.utterance)
        best_passage: Optional[KnowledgePassage] = None
        best_score = 0.0

        for passage in self._passages:
            sim = _cosine_similarity(request.query_embedding, passage.embedding)
            if sim > best_score:
                best_score = sim
                best_passage = passage

        if sku_hit is not None:
            hit_passage, sku_score = sku_hit
            vec_score = _cosine_similarity(request.query_embedding, hit_passage.embedding)
            hybrid_score = round(min(1.0, max(sku_score, 0.6 * sku_score + 0.4 * vec_score)), 4)
            best_passage = hit_passage
            best_score = hybrid_score
        else:
            best_score = round(best_score, 4)

        if best_passage is None or best_score < self.min_confidence:
            return VoiceQueryResponse(
                call_sid=request.call_sid,
                matched_sku=None,
                answer=(
                    "I could not verify that product with high confidence. "
                    "Let me connect you with a human support agent."
                ),
                confidence=best_score,
                escalate_to_human=True,
                timed_out=False,
                latency_ms=request.simulated_webhook_latency_ms,
            )

        return VoiceQueryResponse(
            call_sid=request.call_sid,
            matched_sku=best_passage.sku,
            answer=best_passage.content,
            confidence=best_score,
            escalate_to_human=False,
            timed_out=False,
            latency_ms=request.simulated_webhook_latency_ms,
        )
