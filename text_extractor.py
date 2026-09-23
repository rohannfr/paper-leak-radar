"""
text_extractor.py — Stage 3 of the pipeline.

Extracts plain text from:
  • PDF files  → PyMuPDF embedded text, with pytesseract fallback if sparse
  • Image files → pytesseract OCR directly

Returns a single cleaned string per file.
"""

import logging
import os
import re
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from config import OCR_FALLBACK_THRESHOLD, TESSERACT_LANG

logger = logging.getLogger(__name__)

# Try importing pytesseract; warn gracefully if not installed
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed — OCR fallback disabled.")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """Normalise whitespace and strip junk characters."""
    # collapse multiple spaces / newlines
    text = re.sub(r"[ \t]+", " ", raw)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # remove non-printable chars (keep standard punctuation & digits)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    return text.strip()


def _ocr_image(image: Image.Image) -> str:
    """Run tesseract OCR on a PIL Image and return the result string."""
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        return pytesseract.image_to_string(image, lang=TESSERACT_LANG)
    except Exception as e:
        logger.error(f"pytesseract failed: {e}")
        return ""


def _extract_from_pdf_embedded(pdf_path: str) -> tuple[str, bool]:
    """
    Extract embedded text from a PDF using PyMuPDF.
    Returns (text, is_sufficient) where is_sufficient=False means
    the text is too short to trust and OCR fallback should run.
    """
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text("text"))
    doc.close()

    full_text = "\n".join(pages_text)
    is_sufficient = len(full_text.strip()) >= OCR_FALLBACK_THRESHOLD
    return full_text, is_sufficient


def _ocr_pdf(pdf_path: str) -> str:
    """
    Render each PDF page as an image and OCR it.
    Used as fallback when embedded text is too sparse (scanned PDFs).
    """
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        # render at 2x zoom for better OCR accuracy
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages_text.append(_ocr_image(img))
    doc.close()
    return "\n".join(pages_text)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    Tries embedded text first; falls back to OCR if the result is too short.
    """
    try:
        text, sufficient = _extract_from_pdf_embedded(pdf_path)
        if sufficient:
            logger.info(f"[PDF-embed] {Path(pdf_path).name}: {len(text)} chars")
            return _clean_text(text)

        logger.info(f"[PDF-OCR ] {Path(pdf_path).name}: embedded text sparse, running OCR")
        ocr_text = _ocr_pdf(pdf_path)
        return _clean_text(ocr_text)

    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        return ""


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image file via pytesseract OCR.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        text = _ocr_image(img)
        logger.info(f"[IMG-OCR ] {Path(image_path).name}: {len(text)} chars")
        return _clean_text(text)
    except Exception as e:
        logger.error(f"Failed to OCR image {image_path}: {e}")
        return ""


def extract_text(file_path: str) -> str:
    """
    Auto-detect file type and extract text accordingly.
    Supports: .pdf, .png, .jpg, .jpeg, .bmp, .tiff, .webp
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}:
        return extract_text_from_image(file_path)
    else:
        logger.warning(f"Unsupported file extension: {ext} — skipping {file_path}")
        return ""


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python text_extractor.py <path_to_pdf_or_image>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    result = extract_text(path)
    print(f"\n── Extracted text ({len(result)} chars) ──\n")
    print(result[:2000])
