"""Extractive, no-model answer generation for Phase 1."""

from __future__ import annotations

from .contracts import Answer, CorpusIndex, FactualClaim, RetrievalHit
from .retrieval import tokenize


class ExtractiveBaseline:
    """Turn retrieved chunks into a transparent answer without calling a model."""

    model_identity = "baseline.extractive.v1"

    def __init__(self, index: CorpusIndex) -> None:
        self._chunks = {chunk.chunk_id: chunk for chunk in index.chunks}

    def answer(
        self,
        query: str,
        hits: list[RetrievalHit],
        answer_version: int = 1,
    ) -> Answer:
        if not hits:
            return Answer(
                answer_text="No supporting evidence was found in the supplied corpus.",
                factual_claims=[],
                uncertainty=(
                    "The baseline found no lexical evidence; a model or broader retrieval strategy "
                    "was not run."
                ),
                answer_version=answer_version,
            )

        claims: list[FactualClaim] = []
        evidence_lines: list[str] = []
        for position, hit in enumerate(hits, start=1):
            chunk = self._chunks[hit.chunk_id]
            claims.append(
                FactualClaim(
                    claim_id=f"claim-{answer_version:02d}-{position:02d}",
                    claim_text=chunk.text,
                    supporting_chunk_ids=[hit.chunk_id],
                )
            )
            evidence_lines.append(f"[{hit.chunk_id}] {chunk.text}")
        query_terms = len(tokenize(query))
        return Answer(
            answer_text=(
                "Extractive baseline evidence from the supplied corpus "
                f"({len(hits)} chunks for {query_terms} query terms):\n"
                + "\n".join(f"- {line}" for line in evidence_lines)
            ),
            factual_claims=claims,
            uncertainty=(
                "This is an extractive, no-model Phase 1 baseline; it does not synthesize, "
                "decompose multiple intents, or refine prior answers."
            ),
            answer_version=answer_version,
        )

