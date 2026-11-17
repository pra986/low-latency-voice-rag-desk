"""Runnable CLI demo entrypoint for Low-Latency Voice RAG Desk."""

import json

from voice_rag.engine import VoiceRAGDeskEngine
from voice_rag.models import KnowledgePassage, VoiceQueryRequest


def build_demo_engine() -> VoiceRAGDeskEngine:
    engine = VoiceRAGDeskEngine(expected_dim=4, min_confidence=0.65, webhook_timeout_ms=800)
    passages = [
        KnowledgePassage(
            passage_id="p-101",
            sku="SKU-101",
            title="Pro Noise-Canceling Headset",
            content="SKU-101 Pro Headset supports 30-hour battery life and dual-mic beamforming.",
            embedding=[0.9, 0.1, 0.0, 0.0],
        ),
        KnowledgePassage(
            passage_id="p-204",
            sku="SKU-204",
            title="USB-C Docking Hub",
            content="SKU-204 USB-C Hub provides dual 4K HDMI outputs and 100W pass-through charging.",
            embedding=[0.0, 0.9, 0.1, 0.0],
        ),
    ]
    engine.ingest(passages)
    return engine


def main() -> None:
    engine = build_demo_engine()
    demo_requests = [
        VoiceQueryRequest(
            call_sid="CA-1001",
            utterance="Tell me about ess kay you one oh one battery life",
            query_embedding=[0.85, 0.15, 0.0, 0.0],
            simulated_webhook_latency_ms=140,
        ),
        VoiceQueryRequest(
            call_sid="CA-1002",
            utterance="Do you carry quantum flux routers?",
            query_embedding=[0.0, 0.0, 0.1, 0.9],
            simulated_webhook_latency_ms=110,
        ),
        VoiceQueryRequest(
            call_sid="CA-1003",
            utterance="Check warranty on SKU-204",
            query_embedding=[0.0, 0.9, 0.1, 0.0],
            simulated_webhook_latency_ms=950,
        ),
    ]
    for req in demo_requests:
        resp = engine.query(req)
        print(json.dumps(resp.model_dump(), indent=2))


if __name__ == "__main__":
    main()
