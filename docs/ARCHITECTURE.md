# Architecture Overview

## 1. System Design
The Low-Latency Voice RAG Desk separates offline knowledge ingestion from real-time conversational retrieval so that voice turns stay within a strict `< 450 ms` retrieval budget.

### Components
- **Offline Ingestion (`VoiceRAGDeskEngine.ingest`)**: Enforces embedding dimension parity (`expected_dim`, default 1536) before indexing `KnowledgePassage` records into the Supabase `pgvector` store.
- **Hybrid SKU & Vector Retrieval (`VoiceRAGDeskEngine.query`)**:
  1. **Exact & Phonetic SKU Lookup**: Normalizes spoken or misheard SKUs (e.g., `dash`, `hyphen`, `oh` -> `0`, `one` -> `1`, phonetic NATO/letter mappings) and matches against indexed catalog SKUs.
  2. **Cosine Vector Similarity**: Scores passages by cosine similarity against the query embedding.
- **Confidence & Human Escalation Gate**: If the top score falls below `confidence_threshold` or no matching SKU/product is found, the engine returns `escalate_to_human=True` with a structured handoff message.
- **Webhook Timeout Fallback**: If the downstream ElevenLabs/Twilio webhook exceeds `webhook_timeout_ms`, the engine returns a deterministic low-latency fallback response and flags `timed_out=True`.
