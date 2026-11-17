# Low-Latency Voice RAG Desk

A production-oriented, low-latency Voice Retrieval-Augmented Generation (RAG) desk designed for real-time spoken customer and analyst support over curated knowledge bases backed by PostgreSQL + `pgvector`.

## Architecture

The system separates the real-time audio path from the retrieval and grounding path so that first-token latency stays within strict conversational budgets (< 450 ms p95 target for retrieval + first-chunk synthesis):

```
[Caller Audio] -> [1. VAD & Chunking] -> [2. Streaming ASR]
                                                 |
                                                 v
[3. Query Normalization & Routing] -> [4. Hybrid pgvector + BM25 Retrieval]
                                                 |
                                                 v
[6. Streaming TTS & Barge-In] <- [5. Grounded LLM Synthesis]
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for component-level details and latency budgets.

## 6-Stage Workflow

1. **Audio Ingestion & VAD**: Captures 16 kHz PCM frames, applies Voice Activity Detection (VAD), and emits utterance boundaries with barge-in detection.
2. **Streaming ASR**: Transcribes incremental speech segments with partial hypotheses and finalizes the normalized transcript.
3. **Query Normalization & Intent Routing**: Strips filler tokens, expands domain acronyms, and routes queries to either direct FAQ cache, hybrid RAG, or human escalation.
4. **Hybrid Retrieval (`pgvector` + Lexical)**: Executes cosine-similarity vector search in PostgreSQL (`pgvector` HNSW index) combined with lexical scoring and reciprocal rank fusion (RRF), returning top-k grounded chunks.
5. **Low-Latency Grounded Generation**: Prompts the LLM with strict citation rules, token streaming, and a hard max-sentence constraint optimized for spoken delivery.
6. **Streaming TTS & Barge-In Control**: Streams synthesized audio chunks sentence-by-sentence and immediately halts playback if the caller interrupts.

## Data Contract

All stages exchange strongly-typed Pydantic/dataclass payloads defined in `voice_rag/models.py`:

- **`VoiceQueryRequest`**: `session_id`, `utterance`, `caller_locale`, `max_latency_ms`, `top_k`, `metadata`
- **`RetrievedChunk`**: `chunk_id`, `source_doc`, `content`, `vector_score`, `lexical_score`, `fused_score`
- **`LatencyBreakdown`**: `asr_ms`, `retrieval_ms`, `generation_ms`, `tts_ms`, `total_ms`
- **`VoiceTurnResponse`**: `session_id`, `answer_text`, `citations`, `confidence`, `escalated`, `latency`

## Production Controls

- **Latency SLO & Circuit Breakers**: Enforces per-stage timeout budgets (`max_latency_ms`); falls back to cached/concise responses if retrieval or generation exceeds thresholds.
- **Grounding & Confidence Gate**: Rejects unsupported claims; if top fused score or confidence falls below `min_confidence`, triggers safe fallback or human handoff (`escalated=True`).
- **PII Redaction & Audit Trail**: Masks sensitive identifiers before logging turn telemetry.
- **Evaluation & CI**: Automated regression suite (`evaluation/eval_cases.json`) and unit tests (`tests/test_engine.py`) run on every push via GitHub Actions.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest -v
python -m voice_rag.cli --query "What is the wire transfer cutoff time?"
```

## Attribution

Inspired by and built upon concepts from **Ed Donner's AI Builder course** (LLM Engineering, RAG pipelines, and production AI agents).

## License

MIT License — see [LICENSE](LICENSE).
