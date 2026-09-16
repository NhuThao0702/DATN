"""Public API for the local traffic-law RAG module."""

from .rag_engine import (
    answer_question,
    get_status,
    initialize_ai_background,
    initialize_ai_system,
    shutdown_ai_system,
)

__all__ = [
    "answer_question",
    "get_status",
    "initialize_ai_background",
    "initialize_ai_system",
    "shutdown_ai_system",
]
