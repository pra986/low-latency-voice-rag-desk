"""Tests for VoiceRAGDeskEngine."""

import pytest

from voice_rag.engine import EmbeddingDimensionError, VoiceRAGDeskEngine
from voice_rag.models import KnowledgePassage, VoiceQueryRequest


def _seeded_engine() -> VoiceRAGDeskEngine:
    engine = VoiceRAGDeskEngine(expected_dim=4, min_confidence=0.65, webhook_timeout_ms=800)
    engine.ingest(
        [
            KnowledgePassage(
                passage_id="p-101",
                sku="SKU-101",
                title="Pro Noise-Canceling Headset",
                content="SKU-101 Pro Headset supports 30-hour battery life.",
                embedding=[0.9, 0.1, 0.0, 0.0],
            ),
            KnowledgePassage(
                passage_id="p-204",
                sku="SKU-204",
                title="USB-C Docking Hub",
                content="SKU-204 USB-C Hub provides dual 4K HDMI outputs.",
                embedding=[0.0, 0.9, 0.1, 0.0],
            ),
        ]
    )
    return engine


def test_embedding_dimension_mismatch_rejected() -> None:
    engine = VoiceRAGDeskEngine(expected_dim=4)
    bad_passage = KnowledgePassage(
        passage_id="p-bad",
        sku="SKU-999",
        title="Bad Vector",
        content="Mismatched embedding length.",
        embedding=[0.1, 0.2],
    )
    with pytest.raises(EmbeddingDimensionError):
        engine.ingest([bad_passage])


def test_misheard_phonetic_sku_resolution() -> None:
    engine = _seeded_engine()
    req = VoiceQueryRequest(
        call_sid="CA-2001",
        utterance="Can you check ess kay you one oh one battery specs?",
        query_embedding=[0.5, 0.5, 0.0, 0.0],
        simulated_webhook_latency_ms=150,
    )
    resp = engine.query(req)
    assert resp.matched_sku == "SKU-101"
    assert resp.escalate_to_human is False
    assert resp.timed_out is False
    assert resp.confidence >= 0.65


def test_absent_product_human_escalation() -> None:
    engine = _seeded_engine()
    req = VoiceQueryRequest(
        call_sid="CA-2002",
        utterance="Tell me about the quantum flux router",
        query_embedding=[0.0, 0.0, 0.0, 1.0],
        simulated_webhook_latency_ms=120,
    )
    resp = engine.query(req)
    assert resp.matched_sku is None
    assert resp.escalate_to_human is True
    assert resp.timed_out is False


def test_webhook_timeout_fallback() -> None:
    engine = _seeded_engine()
    req = VoiceQueryRequest(
        call_sid="CA-2003",
        utterance="Check SKU-204 warranty",
        query_embedding=[0.0, 0.9, 0.1, 0.0],
        simulated_webhook_latency_ms=1200,
    )
    resp = engine.query(req)
    assert resp.matched_sku is None
    assert resp.timed_out is True
    assert resp.escalate_to_human is True
