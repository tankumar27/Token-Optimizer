from __future__ import annotations

from difflib import SequenceMatcher
import os
import re


class SemanticValidator:
    _shared: dict | None = None

    def __init__(self) -> None:
        if SemanticValidator._shared is None:
            SemanticValidator._shared = self._load()
        self.embedding_model = SemanticValidator._shared["embedding_model"]
        self.nlp = SemanticValidator._shared["nlp"]
        self.embedding_backend = SemanticValidator._shared["embedding_backend"]
        self.ner_backend = SemanticValidator._shared["ner_backend"]
        self.grammar_backend = SemanticValidator._shared["grammar_backend"]

    def _load(self) -> dict:
        shared = {
            "embedding_model": None,
            "nlp": None,
            "embedding_backend": "lexical",
            "ner_backend": "regex",
            "grammar_backend": "rules",
        }
        if os.getenv("ENABLE_LOCAL_TRANSFORMERS", "0") == "1":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                shared["embedding_model"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                shared["embedding_backend"] = "sentence-transformers/all-MiniLM-L6-v2"
            except Exception:
                try:
                    from transformers import AutoModel, AutoTokenizer  # type: ignore
                    shared["embedding_model"] = _TransformerEmbedder(
                        AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
                        AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
                    )
                    shared["embedding_backend"] = "transformers/all-MiniLM-L6-v2"
                except Exception:
                    pass
            try:
                import spacy  # type: ignore
                try:
                    shared["nlp"] = spacy.load("en_core_web_sm")
                except Exception:
                    shared["nlp"] = spacy.blank("en")
                shared["ner_backend"] = "spacy"
                shared["grammar_backend"] = "spacy_or_rules"
            except Exception:
                pass
        return shared

    def similarity(self, original: str, optimized: str) -> dict:
        if not original.strip() and not optimized.strip():
            score = 1.0
        elif self.embedding_model is not None:
            try:
                import numpy as np  # type: ignore
                vectors = self.embedding_model.encode([original, optimized])
                score = float(np.dot(vectors[0], vectors[1]) / max(1e-9, np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])))
            except Exception:
                score = _lexical_similarity(original, optimized)
        else:
            score = _lexical_similarity(original, optimized)
        return {
            "semantic_similarity": round(score, 3),
            "validator_used": self.embedding_backend,
            "fallback_used": self.embedding_backend == "lexical",
        }

    def entities(self, text: str) -> set[str]:
        if self.nlp is not None and self.ner_backend == "spacy":
            try:
                doc = self.nlp(text)
                found = {ent.text for ent in getattr(doc, "ents", [])}
                if found:
                    return found
            except Exception:
                pass
        return set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)) | set(
            re.findall(r"\bmy name is\s+([a-z][a-z'-]{2,})\b", text, flags=re.IGNORECASE)
        )

    def entity_preservation(self, original: str, optimized: str) -> dict:
        entities = self.entities(original)
        preserved = {entity for entity in entities if entity in optimized}
        score = 1.0 if not entities else len(preserved) / len(entities)
        return {
            "entity_preservation_score": round(score, 3),
            "entities_missing": sorted(entities - preserved),
            "validator_used": self.ner_backend,
            "fallback_used": self.ner_backend == "regex",
        }


def _lexical_similarity(a: str, b: str) -> float:
    aw = set(re.findall(r"[A-Za-z0-9'-]+", a.lower()))
    bw = set(re.findall(r"[A-Za-z0-9'-]+", b.lower()))
    jaccard = len(aw & bw) / max(1, len(aw | bw))
    sequence = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return max(jaccard, sequence * 0.85)


class _TransformerEmbedder:
    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def encode(self, texts: list[str]):
        import torch  # type: ignore

        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**encoded)
        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask).sum(1).numpy() / mask.sum(1).clamp(min=1e-9).numpy()
