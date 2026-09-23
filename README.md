# Paper Leak Radar

Telegram exam paper leak detection system with a web dashboard for scanning, reviewing, and verifying flagged documents.

## Features

- Scrape photos and PDFs from Telegram groups/channels
- ViT document classification filter
- Keyword matching + semantic similarity (BGE embeddings)
- Local LLM judge via Ollama (with rule-based fallback)
- Web UI for scan control and human-in-the-loop review

## Setup

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Also install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) and optionally [Ollama](https://ollama.com) for LLM judging.

### 2. Configure Telegram API

1. Copy `.env.example` to `.env`
2. Get credentials from [my.telegram.org/apps](https://my.telegram.org/apps)
3. Fill in `TELEGRAM_API_ID` and `TELEGRAM_API_HASH`

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

### 3. First-time Telegram login

```bash
python telegram_scraper.py IIT_JEE_Mains_Advance_Notes_pdf
```

Follow the phone login prompt once. A local `.session` file is created (gitignored).

### 4. Run the dashboard

```bash
python report.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000)

## CLI usage

```bash
python main.py --reference path/to/paper.pdf --groups groups.txt
```

## Project structure

```
├── config.py              # Settings (secrets via .env)
├── main.py                # CLI pipeline orchestrator
├── report.py              # FastAPI web dashboard
├── telegram_scraper.py    # Stage 1: Telegram download
├── vit_filter.py          # Stage 2: Document classification
├── text_extractor.py      # Stage 3: PDF/OCR extraction
├── keyword_matcher.py     # Stage 4: Keyword overlap
├── semantic_analyzer.py   # Stage 5: Embedding similarity
├── llm_judge.py           # Stage 6: Leak verdict
├── templates/report.html  # Dashboard UI
└── groups.txt             # Default Telegram targets
```

## Push to GitHub

```bash
git init
git add .
git status                 # confirm .env and *.session are excluded
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```
