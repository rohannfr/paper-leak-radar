"""
llm_judge.py — Stage 6 of the pipeline.

Sends structured evidence (keyword overlap, semantic similarity, chunk snippets)
to a local Ollama LLM and asks it to decide whether the candidate document is:

  • LIKELY_LEAK   — high confidence it's the same paper / leaked source
  • SIMILAR_TOPIC — same subject area, not the same paper
  • UNRELATED     — false positive, ignore

The LLM returns a structured JSON verdict with confidence and reasoning.
Falls back to a rule-based heuristic if Ollama is unavailable.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from config import OLLAMA_MODEL, OLLAMA_HOST, SEMANTIC_THRESHOLD, KEYWORD_THRESHOLD
from semantic_analyzer import SemanticResult

logger = logging.getLogger(__name__)

# Try importing ollama; graceful fallback if not installed
try:
    import ollama as _ollama_lib
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama Python package not installed — using rule-based fallback judge.")


# ── Data structures ───────────────────────────────────────────────────────────

VERDICT_LIKELY_LEAK   = "LIKELY_LEAK"
VERDICT_SIMILAR_TOPIC = "SIMILAR_TOPIC"
VERDICT_UNRELATED     = "UNRELATED"

VERDICT_LABELS = {
    VERDICT_LIKELY_LEAK:   "🚨 Likely Leak",
    VERDICT_SIMILAR_TOPIC: "🟡 Similar Topic",
    VERDICT_UNRELATED:     "✅ Unrelated",
}


@dataclass
class JudgeVerdict:
    verdict: str            # LIKELY_LEAK | SIMILAR_TOPIC | UNRELATED
    confidence: float       # 0.0 – 1.0
    reasoning: str
    used_llm: bool          # True = LLM judged; False = rule-based fallback

    @property
    def label(self) -> str:
        return VERDICT_LABELS.get(self.verdict, self.verdict)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "used_llm": self.used_llm,
        }


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    ref_metadata: dict,
    cand_metadata: dict,
    jaccard_score: float,
    ref_coverage: float,
    semantic_result: SemanticResult,
    matched_keywords: list[str],
) -> str:
    """
    Construct the LLM prompt with all evidence gathered from earlier stages.
    """
    # Build top chunk snippet section
    snippet_lines = []
    for i, chunk in enumerate(semantic_result.top_chunk_matches[:3], start=1):
        snippet_lines.append(
            f"  Chunk pair {i} (similarity={chunk.similarity:.2%}):\n"
            f"    [REF]  {chunk.ref_snippet[:250].strip()}\n"
            f"    [CAND] {chunk.cand_snippet[:250].strip()}"
        )
    snippets_section = "\n".join(snippet_lines) if snippet_lines else "  (no chunk matches)"

    prompt = f"""You are an expert document analyst helping detect exam paper leaks.

## Reference Paper (the paper we are looking for)
- Subject / Exam: {ref_metadata.get('subject', 'Unknown')}
- Year / Session: {ref_metadata.get('year', 'Unknown')}
- Board / Authority: {ref_metadata.get('board', 'Unknown')}

## Candidate Document (found on Telegram)
- Source group: {cand_metadata.get('group_title', 'Unknown')}
- Date of message: {cand_metadata.get('date', 'Unknown')}
- Original filename: {cand_metadata.get('filename', 'Unknown')}

## Evidence from automated analysis

### Keyword Analysis
- Jaccard keyword overlap:  {jaccard_score:.1%}   (threshold: {KEYWORD_THRESHOLD:.0%})
- Reference keyword coverage: {ref_coverage:.1%}
- Matched keywords (sample): {', '.join(matched_keywords[:20])}

### Semantic Similarity
- Full-document cosine similarity: {semantic_result.full_doc_similarity:.1%}  (threshold: {SEMANTIC_THRESHOLD:.0%})
- Avg top-{len(semantic_result.top_chunk_matches)} chunk similarity: {semantic_result.avg_top_chunk_sim:.1%}

### Most Similar Text Chunks (side-by-side)
{snippets_section}

## Your Task
Based ONLY on the evidence above, classify this candidate document as ONE of:
- LIKELY_LEAK   : The candidate is very probably the same exam paper or a pre-release version of it.
- SIMILAR_TOPIC : The candidate covers the same subject area but is NOT the same paper (e.g., study notes, solved papers from a different year).
- UNRELATED     : False positive — the candidate is not related to the reference paper.

Respond with ONLY a valid JSON object (no markdown, no explanation outside JSON):
{{
  "verdict": "<LIKELY_LEAK|SIMILAR_TOPIC|UNRELATED>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<2-4 sentences explaining your decision>"
}}"""
    return prompt


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str) -> Optional[dict]:
    """Call the local Ollama server and parse the JSON response."""
    try:
        client = _ollama_lib.Client(host=OLLAMA_HOST)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},   # low temp for deterministic output
        )
        raw = response["message"]["content"].strip()

        # Extract JSON even if model wrapped it in markdown code fences
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            logger.error(f"LLM returned no JSON:\n{raw}")
            return None

        parsed = json.loads(json_match.group())
        return parsed

    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return None


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _rule_based_verdict(
    jaccard_score: float,
    semantic_result: SemanticResult,
) -> JudgeVerdict:
    """
    Simple heuristic when Ollama is not available.
    High keyword overlap + high semantic similarity → LIKELY_LEAK
    """
    full_sim = semantic_result.full_doc_similarity
    avg_chunk = semantic_result.avg_top_chunk_sim

    # Score combining full-doc sim, chunk sim, and keyword overlap
    combined = 0.4 * full_sim + 0.4 * avg_chunk + 0.2 * jaccard_score

    if combined >= 0.75:
        verdict = VERDICT_LIKELY_LEAK
        conf = min(combined, 0.95)
        reason = (
            f"High keyword overlap ({jaccard_score:.0%}), "
            f"full-doc similarity ({full_sim:.0%}), "
            f"and chunk similarity ({avg_chunk:.0%}) all indicate a likely leak."
        )
    elif combined >= 0.45:
        verdict = VERDICT_SIMILAR_TOPIC
        conf = combined
        reason = (
            f"Moderate similarity scores suggest same subject area "
            f"(keyword overlap {jaccard_score:.0%}, doc sim {full_sim:.0%}) "
            f"but not necessarily the exact leaked paper."
        )
    else:
        verdict = VERDICT_UNRELATED
        conf = 1.0 - combined
        reason = (
            f"Low combined score ({combined:.0%}) — document appears unrelated "
            f"to the reference paper."
        )

    return JudgeVerdict(
        verdict=verdict,
        confidence=round(conf, 3),
        reasoning=reason,
        used_llm=False,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def judge(
    ref_metadata: dict,
    cand_metadata: dict,
    jaccard_score: float,
    ref_coverage: float,
    semantic_result: SemanticResult,
    matched_keywords: set[str],
) -> JudgeVerdict:
    """
    Main entry point. Try LLM first; fall back to rule-based if unavailable.

    Args:
        ref_metadata:     {subject, year, board, ...}
        cand_metadata:    {group_title, date, filename, ...}
        jaccard_score:    From keyword_matcher.jaccard_overlap()
        ref_coverage:     From keyword_matcher.keyword_overlap_ratio()
        semantic_result:  From semantic_analyzer.analyze()
        matched_keywords: From keyword_matcher.get_matching_keywords()

    Returns:
        JudgeVerdict
    """
    if OLLAMA_AVAILABLE:
        prompt = _build_prompt(
            ref_metadata, cand_metadata,
            jaccard_score, ref_coverage,
            semantic_result, sorted(matched_keywords),
        )
        parsed = _call_ollama(prompt)

        if parsed and "verdict" in parsed:
            verdict = parsed.get("verdict", VERDICT_UNRELATED).upper()
            if verdict not in {VERDICT_LIKELY_LEAK, VERDICT_SIMILAR_TOPIC, VERDICT_UNRELATED}:
                verdict = VERDICT_UNRELATED
            return JudgeVerdict(
                verdict=verdict,
                confidence=float(parsed.get("confidence", 0.5)),
                reasoning=str(parsed.get("reasoning", "No reasoning provided.")),
                used_llm=True,
            )
        else:
            logger.warning("LLM response invalid; falling back to rule-based.")

    return _rule_based_verdict(jaccard_score, semantic_result)


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from semantic_analyzer import SemanticResult, ChunkMatch

    mock_result = SemanticResult(
        full_doc_similarity=0.82,
        avg_top_chunk_sim=0.78,
        passes_threshold=True,
        top_chunk_matches=[
            ChunkMatch(
                ref_chunk_idx=0, cand_chunk_idx=0,
                similarity=0.85,
                ref_snippet="The electromagnetic induction experiment demonstrates Faraday's law...",
                cand_snippet="Faraday discovered electromagnetic induction...",
                matched_keywords=["electromagnetic", "induction", "faraday"],
            )
        ],
    )

    verdict = judge(
        ref_metadata={"subject": "Physics (JEE Mains)", "year": "2024", "board": "NTA"},
        cand_metadata={"group_title": "JEE Leaks 2024", "date": "2024-03-15", "filename": "paper.pdf"},
        jaccard_score=0.42,
        ref_coverage=0.55,
        semantic_result=mock_result,
        matched_keywords={"electromagnetic", "induction", "faraday", "solenoid", "flux"},
    )

    print(f"\n{verdict.label}")
    print(f"Confidence: {verdict.confidence:.0%}")
    print(f"Reasoning: {verdict.reasoning}")
    print(f"Used LLM: {verdict.used_llm}")
