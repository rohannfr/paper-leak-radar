"""
telegram_scraper.py — Stage 1 of the pipeline.

Downloads photos AND PDF documents from one or more Telegram groups/channels.
Returns structured metadata for every file so later stages know where it came from.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from telethon import TelegramClient
from telethon.tl.types import MessageMediaDocument, MessageMediaPhoto

from config import (
    TELEGRAM_API_ID,
    TELEGRAM_API_HASH,
    TELEGRAM_SESSION,
    DOWNLOAD_DIR,
    MAX_MESSAGES_PER_GROUP,
    MAX_MEDIA_DOWNLOADS_PER_GROUP,
    DOWNLOAD_TIMEOUT_SEC,
    SKIP_EXISTING_DOWNLOADS,
    PARALLEL_GROUPS,
    validate_telegram_config,
)

ProgressCallback = Optional[Callable[[str, Optional[int]], None]]

logger = logging.getLogger(__name__)


@dataclass
class DownloadedFile:
    """Metadata for a single file downloaded from Telegram."""
    group_username: str
    group_title: str
    message_id: int
    sender_id: Optional[int]
    date: Optional[str]
    file_path: str                      # absolute local path
    file_type: str                      # "photo" | "pdf" | "document"
    original_filename: Optional[str] = None
    caption: Optional[str] = None
    extra: dict = field(default_factory=dict)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _is_pdf(message) -> bool:
    """True if the message carries a PDF document."""
    if not isinstance(message.media, MessageMediaDocument):
        return False
    doc = message.media.document
    if doc is None:
        return False
    # check MIME type
    if doc.mime_type == "application/pdf":
        return True
    # fallback: check file extension in attributes
    for attr in doc.attributes:
        fname = getattr(attr, "file_name", "") or ""
        if fname.lower().endswith(".pdf"):
            return True
    return False


def _is_image_document(message) -> bool:
    """True if the message is a document but with an image MIME type."""
    if not isinstance(message.media, MessageMediaDocument):
        return False
    doc = message.media.document
    if doc is None:
        return False
    img_mimes = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
    return doc.mime_type in img_mimes


def _original_filename(message) -> Optional[str]:
    if not isinstance(message.media, MessageMediaDocument):
        return None
    doc = message.media.document
    if doc is None:
        return None
    for attr in doc.attributes:
        fname = getattr(attr, "file_name", None)
        if fname:
            return fname
    return None


# ── Per-group scraper ─────────────────────────────────────────────────────────

def _notify(progress_callback: ProgressCallback, message: str, pct: Optional[int] = None):
    if progress_callback:
        progress_callback(message, pct)


async def scrape_group(
    client: TelegramClient,
    group: str,
    output_dir: str,
    progress_callback: ProgressCallback = None,
) -> list[DownloadedFile]:
    """
    Download all photos + PDFs from *group*.
    Returns a list of DownloadedFile objects.
    """
    results: list[DownloadedFile] = []
    group_dir = Path(output_dir) / _safe_name(group)
    group_dir.mkdir(parents=True, exist_ok=True)

    try:
        entity = await client.get_entity(group)
        group_title = getattr(entity, "title", group)
        logger.info(f"Scraping: {group_title} ({group})")
        _notify(
            progress_callback,
            f"Connected to {group_title} — scanning up to {MAX_MESSAGES_PER_GROUP} messages",
        )

        msg_count = 0
        async for message in client.iter_messages(entity, limit=MAX_MESSAGES_PER_GROUP):
            msg_count += 1
            if len(results) >= MAX_MEDIA_DOWNLOADS_PER_GROUP:
                _notify(
                    progress_callback,
                    f"{group_title}: reached download cap ({MAX_MEDIA_DOWNLOADS_PER_GROUP} files)",
                )
                break

            file_info = await _handle_message(
                client, message, group, group_title, group_dir
            )
            if file_info:
                results.append(file_info)
                _notify(
                    progress_callback,
                    f"{group_title}: saved {len(results)}/{MAX_MEDIA_DOWNLOADS_PER_GROUP} "
                    f"({file_info.file_type}) msg={file_info.message_id}",
                )
            elif msg_count % 10 == 0:
                _notify(
                    progress_callback,
                    f"{group_title}: scanned {msg_count} messages, {len(results)} files so far",
                )

        logger.info(
            f"Done: {group_title} — {msg_count} msgs scanned, {len(results)} files saved"
        )
        _notify(
            progress_callback,
            f"Finished {group_title}: {msg_count} messages scanned, {len(results)} files",
        )

    except Exception as e:
        logger.error(f"Failed to scrape group '{group}': {e}")

    return results


async def _handle_message(
    client: TelegramClient,
    message,
    group_username: str,
    group_title: str,
    group_dir: Path,
) -> Optional[DownloadedFile]:
    """Download media from a single message. Returns None if no relevant media."""

    date_str = message.date.isoformat() if message.date else None
    sender_id = message.sender_id

    # ── Case 1: Compressed photo ──────────────────────────────────────────────
    if isinstance(message.media, MessageMediaPhoto) and message.photo:
        filename = f"photo_{message.id}.jpg"
        file_path = group_dir / filename
        if not await _ensure_downloaded(client, message, file_path):
            return None
        logger.debug(f"  Photo  msg={message.id} → {filename}")
        return DownloadedFile(
            group_username=group_username,
            group_title=group_title,
            message_id=message.id,
            sender_id=sender_id,
            date=date_str,
            file_path=str(file_path),
            file_type="photo",
            caption=message.text,
        )

    # ── Case 2: PDF document ──────────────────────────────────────────────────
    if _is_pdf(message):
        orig = _original_filename(message) or f"doc_{message.id}.pdf"
        # ensure .pdf extension
        if not orig.lower().endswith(".pdf"):
            orig += ".pdf"
        file_path = group_dir / f"{message.id}_{orig}"
        if not await _ensure_downloaded(client, message, file_path):
            return None
        logger.debug(f"  PDF    msg={message.id} → {file_path.name}")
        return DownloadedFile(
            group_username=group_username,
            group_title=group_title,
            message_id=message.id,
            sender_id=sender_id,
            date=date_str,
            file_path=str(file_path),
            file_type="pdf",
            original_filename=orig,
            caption=message.text,
        )

    # ── Case 3: Image sent as document (uncompressed) ─────────────────────────
    if _is_image_document(message):
        orig = _original_filename(message) or f"img_{message.id}.png"
        file_path = group_dir / f"{message.id}_{orig}"
        if not await _ensure_downloaded(client, message, file_path):
            return None
        logger.debug(f"  ImgDoc msg={message.id} → {file_path.name}")
        return DownloadedFile(
            group_username=group_username,
            group_title=group_title,
            message_id=message.id,
            sender_id=sender_id,
            date=date_str,
            file_path=str(file_path),
            file_type="document",
            original_filename=orig,
            caption=message.text,
        )

    return None   # not a relevant media type


async def _ensure_downloaded(client: TelegramClient, message, file_path: Path) -> bool:
    """Download media with timeout; reuse existing local file when possible."""
    if SKIP_EXISTING_DOWNLOADS and file_path.exists() and file_path.stat().st_size > 0:
        logger.debug(f"  Reusing cached file → {file_path.name}")
        return True

    try:
        await asyncio.wait_for(
            message.download_media(file=str(file_path)),
            timeout=DOWNLOAD_TIMEOUT_SEC,
        )
        return file_path.exists() and file_path.stat().st_size > 0
    except asyncio.TimeoutError:
        logger.warning(f"  Download timed out after {DOWNLOAD_TIMEOUT_SEC}s → {file_path.name}")
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        return False
    except Exception as e:
        logger.warning(f"  Download failed for msg={message.id}: {e}")
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        return False


# ── Multi-group orchestrator ──────────────────────────────────────────────────

async def scrape_groups(
    groups: list[str],
    output_dir: str = DOWNLOAD_DIR,
    progress_callback: ProgressCallback = None,
) -> list[DownloadedFile]:
    """
    Scrape multiple groups in parallel batches of PARALLEL_GROUPS.
    Uses a single Telethon session to avoid flooding.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_files: list[DownloadedFile] = []
    validate_telegram_config()

    async with TelegramClient(TELEGRAM_SESSION, TELEGRAM_API_ID, TELEGRAM_API_HASH) as client:
        if not await client.is_user_authorized():
            raise RuntimeError(
                "Telegram session is not authorized. Run `python telegram_scraper.py <group>` "
                "once in a terminal to log in, then retry the scan."
            )

        _notify(progress_callback, f"Telegram connected — scraping {len(groups)} group(s)")

        for idx, group in enumerate(groups):
            pct = 25 + int((idx / max(len(groups), 1)) * 14)
            _notify(progress_callback, f"Starting group {idx + 1}/{len(groups)}: {group}", pct)
            try:
                group_files = await scrape_group(
                    client, group, output_dir, progress_callback=progress_callback
                )
                all_files.extend(group_files)
            except Exception as e:
                logger.error(f"Failed to scrape group '{group}': {e}")
                _notify(progress_callback, f"Failed to scrape {group}: {e}")

    logger.info(f"Total files downloaded: {len(all_files)}")
    return all_files


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_name(s: str) -> str:
    """Convert a group username/URL into a safe directory name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def load_groups_file(path: str) -> list[str]:
    """Read group usernames from a text file (one per line, # = comment)."""
    groups = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                groups.append(line)
    return groups


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    test_groups = sys.argv[1:] or ["IIT_JEE_Mains_Advance_Notes_pdf"]
    print(f"Scraping groups: {test_groups}")

    files = asyncio.run(scrape_groups(test_groups))
    print(f"\nDownloaded {len(files)} files:")
    for f in files:
        print(f"  [{f.file_type}] {f.group_title} → {Path(f.file_path).name}")
