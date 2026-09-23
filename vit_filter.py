"""
vit_filter.py — Stage 2 of the pipeline.

Uses HAMMALE/vit-tiny-classifier-rvlcdip to decide whether a file
(image or PDF) looks like a document worth analysing.

Only files whose predicted class is in DOCUMENT_CLASS_IDS pass through.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import fitz                         # PyMuPDF — PDF → image rendering
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from config import VIT_MODEL_NAME, DOCUMENT_CLASS_IDS, VIT_PDF_PAGES

logger = logging.getLogger(__name__)

# RVL-CDIP 16-class label map
CLASS_NAMES = [
    "letter",               # 0
    "form",                 # 1  ✓
    "email",                # 2  ✓
    "handwritten",          # 3
    "advertisement",        # 4
    "scientific_report",    # 5  ✓
    "scientific_publication",# 6 ✓
    "specification",        # 7  ✓
    "file_folder",          # 8
    "news_article",         # 9
    "budget",               # 10
    "invoice",              # 11
    "presentation",         # 12
    "questionnaire",        # 13 ✓
    "resume",               # 14
    "memo",                 # 15
]


class ViTDocumentFilter:
    """
    Loads the ViT-tiny classifier once and exposes an `is_document()` method.
    Thread-safe for read-only inference.
    """

    def __init__(self):
        logger.info(f"Loading ViT model: {VIT_MODEL_NAME}")
        self.processor = AutoImageProcessor.from_pretrained(VIT_MODEL_NAME)
        self.model = AutoModelForImageClassification.from_pretrained(VIT_MODEL_NAME)
        self.model.eval()
        logger.info("ViT model loaded.")

    @torch.no_grad()
    def classify_image(self, image: Image.Image) -> tuple[int, str, float]:
        """
        Run inference on a single PIL image.
        Returns (class_id, class_name, confidence).
        """
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1)[0]
        class_id = int(probs.argmax().item())
        confidence = float(probs[class_id].item())
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"
        return class_id, class_name, confidence

    def is_document(self, file_path: str) -> tuple[bool, Optional[str], float]:
        """
        Decide whether *file_path* is a relevant exam document.

        For PDFs: classifies the first VIT_PDF_PAGES page renders.
        For images: classifies directly.

        Returns:
            (passes, class_name, confidence)
            passes=True  → file should continue to text extraction
            passes=False → file is not a document, skip it
        """
        ext = Path(file_path).suffix.lower()

        try:
            if ext == ".pdf":
                return self._classify_pdf(file_path)
            elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}:
                img = Image.open(file_path)
                class_id, class_name, conf = self.classify_image(img)
                passes = class_id in DOCUMENT_CLASS_IDS
                logger.info(
                    f"[ViT-img] {Path(file_path).name}: "
                    f"{class_name} ({conf:.2%}) — {'PASS' if passes else 'SKIP'}"
                )
                return passes, class_name, conf
            else:
                logger.warning(f"Unsupported file type for ViT: {ext}")
                return False, None, 0.0

        except Exception as e:
            logger.error(f"ViT classification failed for {file_path}: {e}")
            return False, None, 0.0

    def _classify_pdf(self, pdf_path: str) -> tuple[bool, Optional[str], float]:
        """
        Render first VIT_PDF_PAGES pages of a PDF and run ViT on each.
        If ANY page passes, the document passes (a multi-page exam paper
        might have a cover page that looks like a presentation).
        Returns the result for the page with the highest confidence among
        document classes.
        """
        doc = fitz.open(pdf_path)
        n_pages = min(len(doc), VIT_PDF_PAGES)

        best_class_id, best_class_name, best_conf = -1, None, 0.0
        any_pass = False

        for page_num in range(n_pages):
            page = doc[page_num]
            mat = fitz.Matrix(1.5, 1.5)   # 1.5× zoom is enough for ViT
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            class_id, class_name, conf = self.classify_image(img)
            passes = class_id in DOCUMENT_CLASS_IDS

            logger.info(
                f"[ViT-pdf] {Path(pdf_path).name} p{page_num+1}: "
                f"{class_name} ({conf:.2%}) — {'PASS' if passes else 'SKIP'}"
            )

            if passes and conf > best_conf:
                best_class_id, best_class_name, best_conf = class_id, class_name, conf
                any_pass = True

        doc.close()
        return any_pass, best_class_name, best_conf


# ── Singleton (lazy-loaded on first use) ──────────────────────────────────────
_filter_instance: Optional[ViTDocumentFilter] = None


def get_filter() -> ViTDocumentFilter:
    """Return the shared ViTDocumentFilter instance (loaded once)."""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = ViTDocumentFilter()
    return _filter_instance


def is_document(file_path: str) -> tuple[bool, Optional[str], float]:
    """Convenience function — uses the shared filter singleton."""
    return get_filter().is_document(file_path)


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python vit_filter.py <path_to_file>")
        sys.exit(1)

    path = sys.argv[1]
    passes, label, conf = is_document(path)
    print(f"\nResult: {'✅ DOCUMENT' if passes else '❌ NOT A DOCUMENT'}")
    print(f"Class:  {label}")
    print(f"Conf:   {conf:.2%}")
