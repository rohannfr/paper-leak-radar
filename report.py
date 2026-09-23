"""
report.py — Web Control Center & FastAPI Server.

Serves the interactive frontend web application for:
  • Uploading reference papers
  • Managing Telegram group target lists
  • Configuring & launching asynchronous paper leak scans
  • Real-time scan progress tracking
  • Human-in-the-loop review dashboard with side-by-side snippet diffs
  • Persisting human verification decisions & exporting reports
"""

import asyncio
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    RESULTS_FILE,
    VERIFIED_FILE,
    TOP_FLAGGED,
    REPORT_HOST,
    REPORT_PORT,
    KEYWORD_THRESHOLD,
    SEMANTIC_THRESHOLD,
    OLLAMA_MODEL,
)
from telegram_scraper import load_groups_file

logger = logging.getLogger(__name__)

app = FastAPI(title="Paper Leak Control Center & Dashboard")

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = BASE_DIR / "uploaded_references"
UPLOAD_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── Global Scan State ─────────────────────────────────────────────────────────

class ScanState:
    def __init__(self):
        self.is_running: bool = False
        self.current_stage: str = "Idle"
        self.progress_pct: int = 0
        self.logs: list[str] = []
        self.error: Optional[str] = None
        self.last_run_time: Optional[str] = None
        self.scan_id: Optional[str] = None

    def log(self, message: str):
        logger.info(message)
        self.logs.append(message)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]

    def reset(self):
        self.is_running = True
        self.current_stage = "Initializing"
        self.progress_pct = 5
        self.logs = []
        self.error = None
        self.scan_id = str(uuid.uuid4())[:8]


scan_state = ScanState()


# ── Data Helpers ──────────────────────────────────────────────────────────────

def load_results() -> list[dict]:
    """Load pipeline results JSON, sorted by confidence descending."""
    if not os.path.exists(RESULTS_FILE):
        return []
    try:
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        verdict_order = {"LIKELY_LEAK": 0, "SIMILAR_TOPIC": 1, "UNRELATED": 2}
        data.sort(
            key=lambda x: (
                verdict_order.get(x.get("verdict", {}).get("verdict", "UNRELATED"), 2),
                -x.get("verdict", {}).get("confidence", 0),
            )
        )
        return data[:TOP_FLAGGED]
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return []


def load_verified() -> dict:
    """Load human verification decisions."""
    if not os.path.exists(VERIFIED_FILE):
        return {}
    try:
        with open(VERIFIED_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_verified(data: dict):
    with open(VERIFIED_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_stored_groups() -> list[str]:
    groups_path = BASE_DIR / "groups.txt"
    if not groups_path.exists():
        return ["IIT_JEE_Mains_Advance_Notes_pdf"]
    return load_groups_file(str(groups_path))


# ── Pipeline Runner (CPU work off the event loop) ─────────────────────────────

def _run_pipeline_sync(
    reference_path: str,
    groups: list[str],
    threshold: float,
    subject: str,
    year: str,
    board: str,
):
    """Run the full scan pipeline in a worker thread so the web UI stays responsive."""
    from main import (
        stage1_scrape,
        stage2_vit_filter,
        stage3_extract_text,
        stage4_keyword_filter,
        stage5_semantic,
        stage6_llm_judge,
        _save_results,
    )
    from text_extractor import extract_text
    from keyword_matcher import extract_keywords

    try:
        scan_state.log(f"Starting scan {scan_state.scan_id} with reference: {Path(reference_path).name}")
        scan_state.log(f"Scanning {len(groups)} groups with threshold {threshold:.0%}")

        scan_state.current_stage = "Processing Reference Paper"
        scan_state.progress_pct = 10
        ref_text = extract_text(reference_path)
        if not ref_text.strip():
            raise ValueError("Reference paper contains no readable text.")
        ref_keywords = extract_keywords(ref_text)
        scan_state.log(f"Reference keywords extracted: {len(ref_keywords)} tokens")

        scan_state.current_stage = "Stage 1/6: Telegram Group Scraping"
        scan_state.progress_pct = 25

        def _scrape_progress(message: str, pct: Optional[int] = None):
            scan_state.log(message)
            if pct is not None:
                scan_state.progress_pct = pct

        scan_state.log("Scraping Telegram groups for photos & PDFs...")
        downloaded = asyncio.run(
            stage1_scrape(groups, progress_callback=_scrape_progress)
        )
        scan_state.progress_pct = 40
        scan_state.log(f"Downloaded {len(downloaded)} media files")

        if not downloaded:
            scan_state.log("No media downloaded. Scan completed with 0 results.")
            _save_results([])
            scan_state.current_stage = "Completed"
            scan_state.progress_pct = 100
            return

        scan_state.current_stage = "Stage 2/6: ViT Document Classification"
        scan_state.progress_pct = 45
        filtered = stage2_vit_filter(downloaded)
        scan_state.log(f"ViT filter kept {len(filtered)} document-like files")

        if not filtered:
            scan_state.log("No documents passed ViT filtering.")
            _save_results([])
            scan_state.current_stage = "Completed"
            scan_state.progress_pct = 100
            return

        scan_state.current_stage = "Stage 3/6: Text Extraction & OCR"
        scan_state.progress_pct = 55
        with_text = stage3_extract_text(filtered)
        scan_state.log(f"Extracted text from {len(with_text)} files")

        scan_state.current_stage = "Stage 4/6: Keyword Matching & Jaccard Scoring"
        scan_state.progress_pct = 70
        candidates = stage4_keyword_filter(with_text, ref_keywords, threshold)
        scan_state.log(f"{len(candidates)} candidates passed keyword threshold")

        if not candidates:
            scan_state.log("No candidates passed keyword threshold.")
            _save_results([])
            scan_state.current_stage = "Completed"
            scan_state.progress_pct = 100
            return

        scan_state.current_stage = "Stage 5/6: BGE Semantic Embeddings & RAG"
        scan_state.progress_pct = 85
        analysed = stage5_semantic(candidates, ref_text)

        scan_state.current_stage = "Stage 6/6: Local LLM Reasoning Judge"
        scan_state.progress_pct = 95
        ref_meta = {"subject": subject, "year": year, "board": board}
        results = stage6_llm_judge(analysed, ref_meta)

        _save_results(results)

        scan_state.current_stage = "Completed"
        scan_state.progress_pct = 100
        scan_state.log(f"Pipeline finished! Flagged {len(results)} potential leaks/similar papers.")

    except Exception as e:
        logger.error(f"Scan error: {e}", exc_info=True)
        scan_state.error = str(e)
        scan_state.current_stage = f"Error: {e}"
        scan_state.log(f"CRITICAL ERROR: {e}")
    finally:
        scan_state.is_running = False


async def _async_run_pipeline_task(
    reference_path: str,
    groups: list[str],
    threshold: float,
    subject: str,
    year: str,
    board: str,
):
    await asyncio.to_thread(
        _run_pipeline_sync,
        reference_path,
        groups,
        threshold,
        subject,
        year,
        board,
    )


# ── REST API & Page Routes ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the unified control panel & review dashboard."""
    results = load_results()
    verified = load_verified()

    for r in results:
        rid = r.get("result_id", "")
        r["human_decision"] = verified.get(rid, None)

    groups = get_stored_groups()

    return templates.TemplateResponse(
        request=request,
        name="report.html",
        context={
            "results": results,
            "total": len(results),
            "likely_leaks": sum(1 for r in results if r.get("verdict", {}).get("verdict") == "LIKELY_LEAK"),
            "similar_topic": sum(1 for r in results if r.get("verdict", {}).get("verdict") == "SIMILAR_TOPIC"),
            "groups": groups,
            "scan_state": scan_state,
        },
    )


@app.get("/api/results")
async def api_results():
    """Return raw results JSON."""
    results = load_results()
    verified = load_verified()
    for r in results:
        rid = r.get("result_id", "")
        r["human_decision"] = verified.get(rid, None)
    return JSONResponse(content=results)


@app.get("/api/groups")
async def api_get_groups():
    return {"groups": get_stored_groups()}


@app.post("/api/groups")
async def api_save_groups(request: Request):
    body = await request.json()
    groups_list = body.get("groups", [])
    if isinstance(groups_list, str):
        groups_list = [g.strip() for g in groups_list.split("\n") if g.strip()]

    groups_path = BASE_DIR / "groups.txt"
    with open(groups_path, "w") as f:
        f.write("# Target Telegram Groups\n")
        for g in groups_list:
            if g.strip() and not g.startswith("#"):
                f.write(f"{g.strip()}\n")

    return {"status": "ok", "groups": get_stored_groups()}


@app.post("/api/upload-reference")
async def api_upload_reference(file: UploadFile = File(...)):
    """Upload ground reference paper (PDF or image)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    target_path = UPLOAD_DIR / file.filename
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"Uploaded reference paper: {target_path}")
    return {
        "status": "ok",
        "filename": file.filename,
        "path": str(target_path),
        "size_bytes": target_path.stat().st_size,
    }


@app.post("/api/start-scan")
async def api_start_scan(
    reference_filename: Optional[str] = Form(None),
    reference_file: Optional[UploadFile] = File(None),
    groups: str = Form(""),
    threshold: float = Form(KEYWORD_THRESHOLD),
    subject: str = Form("General"),
    year: str = Form("2024"),
    board: str = Form("Standard"),
):
    """Trigger an asynchronous paper leak detection scan."""
    if scan_state.is_running:
        raise HTTPException(status_code=400, detail="A scan is already running.")

    scan_state.reset()

    # Determine reference file path
    ref_path = None
    if reference_file and reference_file.filename:
        target_path = UPLOAD_DIR / reference_file.filename
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(reference_file.file, buffer)
        ref_path = str(target_path)
    elif reference_filename:
        target_path = UPLOAD_DIR / reference_filename
        if target_path.exists():
            ref_path = str(target_path)

    if not ref_path:
        # Fallback to any file in UPLOAD_DIR or ask user
        existing_files = list(UPLOAD_DIR.glob("*.*"))
        if existing_files:
            ref_path = str(existing_files[-1])
        else:
            raise HTTPException(status_code=400, detail="Please upload or select a reference paper file.")

    # Parse groups
    groups_list = [g.strip() for g in groups.split("\n") if g.strip() and not g.startswith("#")]
    if not groups_list:
        groups_list = get_stored_groups()

    # Launch background task
    asyncio.create_task(
        _async_run_pipeline_task(
            reference_path=ref_path,
            groups=groups_list,
            threshold=threshold,
            subject=subject,
            year=year,
            board=board,
        )
    )

    return {"status": "started", "scan_id": scan_state.scan_id, "reference": Path(ref_path).name}


@app.get("/api/scan-status")
async def api_scan_status():
    """Poll the current scan progress & logs."""
    return {
        "is_running": scan_state.is_running,
        "stage": scan_state.current_stage,
        "progress_pct": scan_state.progress_pct,
        "logs": scan_state.logs[-30:],
        "error": scan_state.error,
        "scan_id": scan_state.scan_id,
    }


@app.post("/api/verify/{result_id}")
async def api_verify(result_id: str, request: Request):
    body = await request.json()
    decision = body.get("decision", "")
    valid = {"CONFIRMED_LEAK", "FALSE_POSITIVE", "NEEDS_REVIEW"}
    if decision not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid decision. Must be one of {valid}")

    verified = load_verified()
    verified[result_id] = decision
    save_verified(verified)

    logger.info(f"Result {result_id} marked as {decision}")
    return {"status": "ok", "result_id": result_id, "decision": decision}


@app.get("/api/summary")
async def api_summary():
    results = load_results()
    verified = load_verified()
    return {
        "total_flagged": len(results),
        "likely_leaks": sum(1 for r in results if r.get("verdict", {}).get("verdict") == "LIKELY_LEAK"),
        "similar_topic": sum(1 for r in results if r.get("verdict", {}).get("verdict") == "SIMILAR_TOPIC"),
        "unrelated": sum(1 for r in results if r.get("verdict", {}).get("verdict") == "UNRELATED"),
        "human_confirmed": sum(1 for v in verified.values() if v == "CONFIRMED_LEAK"),
        "human_rejected": sum(1 for v in verified.values() if v == "FALSE_POSITIVE"),
    }


@app.get("/api/export")
async def api_export():
    """Download results JSON file."""
    if os.path.exists(RESULTS_FILE):
        return FileResponse(RESULTS_FILE, media_type="application/json", filename="paper_leak_results.json")
    raise HTTPException(status_code=404, detail="No results file found.")


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=REPORT_HOST, port=REPORT_PORT)
