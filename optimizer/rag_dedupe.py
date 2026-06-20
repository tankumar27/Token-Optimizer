from __future__ import annotations

from .rag_compiler import RagContextCompiler, RagChunk


Chunk = RagChunk


def parse_chunks(text: str) -> list[Chunk]:
    return RagContextCompiler().parse_chunks(text)


def retrieval_semantic_chunk_dedupe_backend(text: str) -> tuple[str, list[dict], list[dict]]:
    return RagContextCompiler().compile(text)
