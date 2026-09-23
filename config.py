"""
config.py — Shared configuration for the Paper Leak Detection pipeline.

Secrets (Telegram API credentials) are loaded from environment variables.
Copy `.env.example` to `.env` and fill in your values before running.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ── Telegram API (from environment — never commit real values) ───────────────
TELEGRAM_API_ID   = int(os.environ.get("TELEGRAM_API_ID", "0"))
TELEGRAM_API_HASH = os.environ.get("TELEGRAM_API_HASH", "")
TELEGRAM_SESSION  = os.environ.get("TELEGRAM_SESSION", "leak_monitor_session")

# ── ViT Document Classifier ───────────────────────────────────────────────────
VIT_MODEL_NAME = "HAMMALE/vit-tiny-classifier-rvlcdip"

# RVL-CDIP class ids we treat as "potential exam documents"
# 1=form, 2=email, 5=scientific_report, 6=scientific_publication,
# 7=specification, 13=questionnaire
DOCUMENT_CLASS_IDS = {1, 2, 5, 6, 7, 13}

# How many PDF pages to run through ViT (first N pages only)
VIT_PDF_PAGES = 2

# ── Text Extraction ───────────────────────────────────────────────────────────
# Minimum characters to trust PyMuPDF embedded text; below this → OCR fallback
OCR_FALLBACK_THRESHOLD = 100
TESSERACT_LANG = "eng"          # change to "eng+hin" for Hindi papers

# ── Keyword Matching ──────────────────────────────────────────────────────────
KEYWORD_THRESHOLD = 0.30        # Jaccard overlap to pass Stage 4
MIN_KEYWORD_LENGTH = 3          # ignore tokens shorter than this

# ── Semantic Analysis ─────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE        = 400         # words per chunk
CHUNK_OVERLAP     = 50          # word overlap between chunks
TOP_CHUNKS        = 5           # how many top chunk-pairs to report

# Cosine similarity threshold to mark a candidate as "semantically close"
SEMANTIC_THRESHOLD = 0.70

# ── LLM Judge ────────────────────────────────────────────────────────────────
OLLAMA_MODEL  = "gemma3:4b"     # or "mistral:7b", "llama3.2:3b"
OLLAMA_HOST   = "http://localhost:11434"

# ── Pipeline ──────────────────────────────────────────────────────────────────
DOWNLOAD_DIR        = "downloaded_files"
RESULTS_FILE        = "results.json"
VERIFIED_FILE       = "verified_results.json"
PARALLEL_GROUPS     = 1         # scrape groups sequentially (Telethon is single-client)
MAX_MESSAGES_PER_GROUP = 50     # reduced for faster demo scans (was 500)
MAX_MEDIA_DOWNLOADS_PER_GROUP = 10  # cap PDFs/photos per group for demo speed
DOWNLOAD_TIMEOUT_SEC = 90       # skip files that take too long to download
SKIP_EXISTING_DOWNLOADS = True  # reuse files already on disk from prior scans

# ── Report Dashboard ──────────────────────────────────────────────────────────
REPORT_HOST = "127.0.0.1"
REPORT_PORT = 8000
TOP_FLAGGED = 20                # max results to show in the dashboard


def validate_telegram_config() -> None:
    """Raise a clear error if Telegram credentials are missing."""
    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        raise RuntimeError(
            "Telegram API credentials are not configured. "
            "Copy .env.example to .env and set TELEGRAM_API_ID and TELEGRAM_API_HASH. "
            "Get them from https://my.telegram.org/apps"
        )
