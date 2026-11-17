"""Low-latency Voice RAG Support Desk package."""

from voice_rag.engine import EmbeddingDimensionError, VoiceRAGDeskEngine
from voice_rag.models import KnowledgePassage, VoiceQueryRequest, VoiceQueryResponse

__all__ = [
    "EmbeddingDimensionError",
    "KnowledgePassage",
    "VoiceQueryRequest",
    "VoiceQueryResponse",
    "VoiceRAGDeskEngine",
]
