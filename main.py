"""
main.py — Pipeline orchestrator for the Paper Leak Detection System.

Usage:
    python main.py --reference <path_to_reference_paper.pdf> \\
                   --groups <groups.txt> \\
                   [--threshold 0.30] \\
                   [--no-dashboard]

The pipeline runs 7 stages:
  1. Scrape Telegram groups for photos + PDFs
  2. ViT filter — keep only document-type files
  3. Text extraction (PyMuPDF + OCR fallback)
  4. Keyword matching — Jaccard threshold
  5. Semantic analysis — full-doc + chunk embeddings
  6. LLM judge — local Ollama verdict
  7. Report dashboard — human-in-loop review UI
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path

import uvicorn

from config import (
    DOWNLOAD_DIR,
    RESULTS_FILE,
    KEYWORD_THRESHOLD,
    SEMANTIC_THRESHOLD,
    REPORT_HOST,
    REPORT_PORT,
    TOP_FLAGGED,
)
from keyword_matcher import (
    extract_keywords,
    jaccard_overlap,
    keyword_overlap_ratio,
    passes_keyword_threshold,
    get_matching_keywords,
)
from llm_judge import judge
from semantic_analyzer import analyze
from telegram_scraper import scrape_groups, load_groups_file
from text_extractor import extract_text
from vit_filter import get_filter

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger("main")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _result_id() -> str:
    return str(uuid.uuid4())[:8]


def _save_results(results: list[dict]):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved → {RESULTS_FILE} ({len(results)} entries)")


# ── Stage runners ─────────────────────────────────────────────────────────────

async def stage1_scrape(groups: list[str], progress_callback=None) -> list:
    logger.info(f"── Stage 1: Scraping {len(groups)} Telegram group(s) ──")
    files = await scrape_groups(
        groups,
        output_dir=DOWNLOAD_DIR,
        progress_callback=progress_callback,
    )
    logger.info(f"Stage 1 complete — {len(files)} files downloaded")
    return files


def stage2_vit_filter(downloaded_files: list) -> list:
    logger.info("── Stage 2: ViT document filter ──")
    vit = get_filter()
    passed = []
    for df in downloaded_files:
        ok, label, conf = vit.is_document(df.file_path)
        if ok:
            passed.append(df)
    logger.info(
        f"Stage 2 complete — {len(passed)}/{len(downloaded_files)} files passed ViT"
    )
    return passed


def stage3_extract_text(filtered_files: list) -> list[tuple]:
    """Returns list of (DownloadedFile, extracted_text)."""
    logger.info("── Stage 3: Text extraction ──")
    results = []
    for df in filtered_files:
        text = extract_text(df.file_path)
        if text.strip():
            results.append((df, text))
        else:
            logger.warning(f"Empty text for {Path(df.file_path).name} — skipping")
    logger.info(f"Stage 3 complete — {len(results)}/{len(filtered_files)} files have text")
    return results


def stage4_keyword_filter(
    file_text_pairs: list[tuple],
    ref_keywords: set[str],
    threshold: float,
) -> list[tuple]:
    """Returns list of (DownloadedFile, text, jaccard, ref_cov, matched_kws)."""
    logger.info(f"── Stage 4: Keyword matching (threshold={threshold:.0%}) ──")
    passed = []
    for df, text in file_text_pairs:
        cand_kw = extract_keywords(text)
        j_score = jaccard_overlap(ref_keywords, cand_kw)
        r_cov   = keyword_overlap_ratio(ref_keywords, cand_kw)
        matched = get_matching_keywords(ref_keywords, cand_kw)

        logger.debug(
            f"  {Path(df.file_path).name}: "
            f"jaccard={j_score:.2%}  ref_cov={r_cov:.2%}"
        )

        if j_score >= threshold:
            passed.append((df, text, j_score, r_cov, matched))

    logger.info(
        f"Stage 4 complete — {len(passed)}/{len(file_text_pairs)} candidates "
        f"cleared keyword threshold"
    )
    return passed


def stage5_semantic(
    candidates: list[tuple],
    ref_text: str,
) -> list[tuple]:
    """Returns list of (DownloadedFile, text, jaccard, ref_cov, matched_kws, SemanticResult)."""
    logger.info("── Stage 5: Semantic analysis ──")
    results = []
    for df, text, j_score, r_cov, matched_kws in candidates:
        logger.info(f"  Analysing: {Path(df.file_path).name}")
        sem_result = analyze(ref_text, text, matched_kws)
        results.append((df, text, j_score, r_cov, matched_kws, sem_result))
    logger.info(f"Stage 5 complete — {len(results)} candidates analysed")
    return results


def stage6_llm_judge(
    analysed: list[tuple],
    ref_metadata: dict,
) -> list[dict]:
    """Returns list of result dicts ready for the report."""
    logger.info("── Stage 6: LLM judge ──")
    flagged = []

    for df, text, j_score, r_cov, matched_kws, sem_result in analysed:
        cand_meta = {
            "group_title":    df.group_title,
            "group_username": df.group_username,
            "date":           df.date,
            "filename":       df.original_filename or Path(df.file_path).name,
        }

        verdict = judge(
            ref_metadata=ref_metadata,
            cand_metadata=cand_meta,
            jaccard_score=j_score,
            ref_coverage=r_cov,
            semantic_result=sem_result,
            matched_keywords=matched_kws,
        )

        result = {
            "result_id":      _result_id(),
            "group_username": df.group_username,
            "group_title":    df.group_title,
            "message_id":     df.message_id,
            "date":           df.date,
            "filename":       df.original_filename or Path(df.file_path).name,
            "file_path":      df.file_path,
            "keywords": {
                "jaccard":      round(j_score, 4),
                "ref_coverage": round(r_cov, 4),
            },
            "matched_keywords": sorted(matched_kws)[:40],
            "semantic":         sem_result.summary(),
            "verdict":          verdict.to_dict(),
        }

        flagged.append(result)
        logger.info(
            f"  {df.group_title}: {verdict.label} "
            f"(conf={verdict.confidence:.0%})"
        )

    # Sort by confidence descending
    flagged.sort(key=lambda r: -r["verdict"]["confidence"])
    logger.info(f"Stage 6 complete — {len(flagged)} results judged")
    return flagged


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_pipeline(
    reference_path: str,
    groups: list[str],
    threshold: float,
    ref_metadata: dict,
):
    logger.info("=" * 60)
    logger.info("  Paper Leak Detection Pipeline starting")
    logger.info(f"  Reference: {reference_path}")
    logger.info(f"  Groups:    {len(groups)}")
    logger.info(f"  Threshold: {threshold:.0%}")
    logger.info("=" * 60)

    # Pre-load reference text & keywords
    logger.info("Extracting reference paper text …")
    ref_text = extract_text(reference_path)
    if not ref_text.strip():
        logger.critical("Reference paper produced no text — aborting.")
        sys.exit(1)
    ref_keywords = extract_keywords(ref_text)
    logger.info(f"Reference: {len(ref_keywords)} keywords extracted")

    # Stage 1 — Scrape
    downloaded = await stage1_scrape(groups)
    if not downloaded:
        logger.warning("No files downloaded — check your group list and Telegram session.")
        _save_results([])
        return []

    # Stage 2 — ViT filter
    filtered = stage2_vit_filter(downloaded)
    if not filtered:
        logger.warning("No files passed the ViT document filter.")
        _save_results([])
        return []

    # Stage 3 — Text extraction
    with_text = stage3_extract_text(filtered)

    # Stage 4 — Keyword matching
    candidates = stage4_keyword_filter(with_text, ref_keywords, threshold)
    if not candidates:
        logger.info("No candidates passed the keyword threshold.")
        _save_results([])
        return []

    # Stage 5 — Semantic analysis
    analysed = stage5_semantic(candidates, ref_text)

    # Stage 6 — LLM judge
    results = stage6_llm_judge(analysed, ref_metadata)

    # Save
    _save_results(results)

    # Summary
    leaks   = [r for r in results if r["verdict"]["verdict"] == "LIKELY_LEAK"]
    similar = [r for r in results if r["verdict"]["verdict"] == "SIMILAR_TOPIC"]
    logger.info("=" * 60)
    logger.info(f"  🚨 Likely Leaks:  {len(leaks)}")
    logger.info(f"  🟡 Similar Topic: {len(similar)}")
    logger.info(f"  Total flagged:    {len(results)}")
    logger.info("=" * 60)

    return results


def parse_args():
    p = argparse.ArgumentParser(
        description="Telegram Paper Leak Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--reference", required=True,
        help="Path to the reference exam paper (PDF or image)",
    )
    p.add_argument(
        "--groups", required=True,
        help="Path to a text file with Telegram group usernames (one per line), "
             "OR a comma-separated list of group usernames",
    )
    p.add_argument(
        "--threshold", type=float, default=KEYWORD_THRESHOLD,
        help=f"Keyword Jaccard threshold (default: {KEYWORD_THRESHOLD})",
    )
    p.add_argument(
        "--subject", default="Unknown",
        help="Exam subject for LLM context (e.g. 'Physics JEE Mains 2024')",
    )
    p.add_argument(
        "--year", default="Unknown",
        help="Exam year/session for LLM context",
    )
    p.add_argument(
        "--board", default="Unknown",
        help="Exam board/authority for LLM context",
    )
    p.add_argument(
        "--no-dashboard", action="store_true",
        help="Skip launching the review dashboard after pipeline completes",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve groups
    if os.path.isfile(args.groups):
        groups = load_groups_file(args.groups)
    else:
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    if not groups:
        logger.critical("No groups provided. Aborting.")
        sys.exit(1)

    ref_metadata = {
        "subject": args.subject,
        "year":    args.year,
        "board":   args.board,
    }

    # Run pipeline
    asyncio.run(run_pipeline(
        reference_path=args.reference,
        groups=groups,
        threshold=args.threshold,
        ref_metadata=ref_metadata,
    ))

    # Launch dashboard
    if not args.no_dashboard:
        logger.info(f"Launching review dashboard at http://{REPORT_HOST}:{REPORT_PORT}")
        from report import app
        uvicorn.run(app, host=REPORT_HOST, port=REPORT_PORT, log_level="warning")
