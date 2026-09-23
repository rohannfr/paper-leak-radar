"""
semantic_analyzer.py — Stage 5 of the pipeline.

Computes two levels of semantic similarity between the reference paper
and a candidate document using BGE-small-en embeddings:

  1. Full-document cosine similarity  (broad topic match)
  2. Chunk-level cosine similarity    (localised section match)
     — only on chunks that contain at least one matched keyword

Returns a SemanticResult dataclass with everything downstream needs.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_CHUNKS,
    SEMANTIC_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ── Model singleton ───────────────────────────────────────────────────────────
_embed_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _embed_model


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ChunkMatch:
    """A pair of chunks (one from each doc) with their similarity score."""
    ref_chunk_idx: int
    cand_chunk_idx: int
    similarity: float
    ref_snippet: str        # first 300 chars of the ref chunk
    cand_snippet: str       # first 300 chars of the candidate chunk
    matched_keywords: list[str] = field(default_factory=list)


@dataclass
class SemanticResult:
    """Full semantic analysis output for one candidate document."""
    full_doc_similarity: float
    top_chunk_matches: list[ChunkMatch]
    passes_threshold: bool                       # based on full_doc_similarity
    avg_top_chunk_sim: float = 0.0

    def summary(self) -> dict:
        return {
            "full_doc_sim": round(self.full_doc_similarity, 4),
            "passes_threshold": self.passes_threshold,
            "avg_top_chunk_sim": round(self.avg_top_chunk_sim, 4),
            "top_chunks": [
                {
                    "similarity": round(m.similarity, 4),
                    "ref_snippet": m.ref_snippet[:200],
                    "cand_snippet": m.cand_snippet[:200],
                    "matched_keywords": m.matched_keywords[:10],
                }
                for m in self.top_chunk_matches
            ],
        }


# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Returns a list of chunk strings.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


# ── Cosine similarity ─────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarities between rows of matrix_a and matrix_b.
    Returns an (m × n) similarity matrix.
    """
    # L2-normalise rows
    norms_a = np.linalg.norm(matrix_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(matrix_b, axis=1, keepdims=True)
    norms_a = np.where(norms_a == 0, 1e-8, norms_a)
    norms_b = np.where(norms_b == 0, 1e-8, norms_b)
    a_norm = matrix_a / norms_a
    b_norm = matrix_b / norms_b
    return a_norm @ b_norm.T


# ── Keyword filter for chunks ──────────────────────────────────────────────────

def _chunk_contains_keyword(chunk: str, keywords: set[str]) -> bool:
    """Return True if at least one keyword appears in *chunk* (word-boundary match)."""
    chunk_lower = chunk.lower()
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", chunk_lower):
            return True
    return False


def _keywords_in_chunk(chunk: str, keywords: set[str]) -> list[str]:
    """Return the subset of *keywords* that appear in *chunk*."""
    chunk_lower = chunk.lower()
    return [
        kw for kw in keywords
        if re.search(r"\b" + re.escape(kw) + r"\b", chunk_lower)
    ]


# ── Main analysis function ────────────────────────────────────────────────────

def analyze(
    ref_text: str,
    cand_text: str,
    matched_keywords: set[str],
) -> SemanticResult:
    """
    Run full semantic analysis between reference and candidate texts.

    Args:
        ref_text:          Full text of the reference paper.
        cand_text:         Full text of the candidate document.
        matched_keywords:  Keywords that appeared in BOTH documents
                           (from keyword_matcher.get_matching_keywords).

    Returns:
        SemanticResult with full-doc similarity and top chunk matches.
    """
    model = get_model()

    # ── 1. Full-document similarity ───────────────────────────────────────────
    logger.info("Computing full-document embeddings …")
    # Truncate to 10 000 chars for the full-doc embed to stay within limits
    ref_snippet_full  = ref_text[:10_000]
    cand_snippet_full = cand_text[:10_000]
    ref_emb, cand_emb = model.encode(
        [ref_snippet_full, cand_snippet_full], show_progress_bar=False
    )
    full_doc_sim = cosine_similarity(ref_emb, cand_emb)
    logger.info(f"Full-doc similarity: {full_doc_sim:.4f}")

    passes = full_doc_sim >= SEMANTIC_THRESHOLD

    # ── 2. Chunk-level similarity (only where keywords matched) ───────────────
    ref_chunks  = chunk_text(ref_text)
    cand_chunks = chunk_text(cand_text)

    if not ref_chunks or not cand_chunks or not matched_keywords:
        return SemanticResult(
            full_doc_similarity=full_doc_sim,
            top_chunk_matches=[],
            passes_threshold=passes,
        )

    # Filter to chunks that contain at least one matched keyword
    ref_kw_chunks  = [(i, c) for i, c in enumerate(ref_chunks)
                      if _chunk_contains_keyword(c, matched_keywords)]
    cand_kw_chunks = [(i, c) for i, c in enumerate(cand_chunks)
                      if _chunk_contains_keyword(c, matched_keywords)]

    logger.info(
        f"Keyword-bearing chunks — ref: {len(ref_kw_chunks)}, "
        f"cand: {len(cand_kw_chunks)}"
    )

    if not ref_kw_chunks or not cand_kw_chunks:
        return SemanticResult(
            full_doc_similarity=full_doc_sim,
            top_chunk_matches=[],
            passes_threshold=passes,
        )

    # Embed keyword-bearing chunks
    ref_texts_to_embed  = [c for _, c in ref_kw_chunks]
    cand_texts_to_embed = [c for _, c in cand_kw_chunks]

    logger.info("Embedding keyword-bearing chunks …")
    ref_chunk_embs  = model.encode(ref_texts_to_embed,  show_progress_bar=False)
    cand_chunk_embs = model.encode(cand_texts_to_embed, show_progress_bar=False)

    # Pairwise cosine similarity matrix (ref_kw × cand_kw)
    sim_matrix = batch_cosine_similarity(ref_chunk_embs, cand_chunk_embs)

    # Collect top-N unique chunk pairs
    chunk_matches: list[ChunkMatch] = []

    # Flatten and sort descending
    flat_indices = np.argsort(sim_matrix, axis=None)[::-1]
    seen_ref, seen_cand = set(), set()

    for flat_idx in flat_indices:
        ri, ci = divmod(int(flat_idx), sim_matrix.shape[1])
        real_ref_idx  = ref_kw_chunks[ri][0]
        real_cand_idx = cand_kw_chunks[ci][0]

        # avoid duplicating the same chunk
        if real_ref_idx in seen_ref or real_cand_idx in seen_cand:
            continue

        seen_ref.add(real_ref_idx)
        seen_cand.add(real_cand_idx)

        kws_in_pair = _keywords_in_chunk(
            ref_chunks[real_ref_idx] + " " + cand_chunks[real_cand_idx],
            matched_keywords,
        )

        chunk_matches.append(ChunkMatch(
            ref_chunk_idx=real_ref_idx,
            cand_chunk_idx=real_cand_idx,
            similarity=float(sim_matrix[ri, ci]),
            ref_snippet=ref_chunks[real_ref_idx][:300],
            cand_snippet=cand_chunks[real_cand_idx][:300],
            matched_keywords=kws_in_pair,
        ))

        if len(chunk_matches) >= TOP_CHUNKS:
            break

    avg_top_sim = float(np.mean([m.similarity for m in chunk_matches])) if chunk_matches else 0.0
    logger.info(f"Avg top-{len(chunk_matches)} chunk similarity: {avg_top_sim:.4f}")

    return SemanticResult(
        full_doc_similarity=full_doc_sim,
        top_chunk_matches=chunk_matches,
        passes_threshold=passes,
        avg_top_chunk_sim=avg_top_sim,
    )


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    REF = """
    Faraday's law of electromagnetic induction states that the induced EMF
    in a closed loop equals the negative rate of change of magnetic flux
    through the loop. The solenoid converts electrical energy to magnetic flux.
    Lenz's law ensures conservation of energy by opposing the inducing change.
    Electric generators exploit this phenomenon to produce alternating current.
    """

    CAND = """
    The law of electromagnetic induction, discovered by Faraday, relates the
    induced electromotive force to the change in magnetic flux linkage.
    A solenoid wound with wire produces a magnetic field proportional to the
    current. Lenz's law describes the direction of the induced current.
    This principle underlies the working of every AC generator in power plants.
    """

    from keyword_matcher import extract_keywords, get_matching_keywords
    ref_kw = extract_keywords(REF)
    cand_kw = extract_keywords(CAND)
    matched = get_matching_keywords(ref_kw, cand_kw)
    print("Matched keywords:", matched)

    result = analyze(REF, CAND, matched)
    import json
    print(json.dumps(result.summary(), indent=2))
