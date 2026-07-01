"""Download YouTube audio for the isiXhosa speech-to-text pipeline.

This module is the first stage of the ASR data pipeline. It fetches
audio-only streams (and optional subtitles) from YouTube using yt-dlp
and stores each video under ``data/raw/<video_id>/`` with a sidecar
``metadata.json`` for downstream dataset construction.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yt_dlp
from yt_dlp.utils import DownloadError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DEFAULT_LOG_LEVEL,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOGS_DIR,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)

# File extensions treated as downloaded audio assets.
SUPPORTED_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".m4a", ".mp3", ".opus", ".ogg", ".wav", ".webm", ".aac"}
)

# Subtitle file extensions written by yt-dlp.
SUBTITLE_EXTENSIONS: frozenset[str] = frozenset({".vtt", ".srt", ".ass", ".lrc"})

# Sidecar metadata filename stored beside each audio file.
METADATA_FILENAME: str = "metadata.json"

# Subtitle languages to request (isiXhosa first, then common fallbacks).
SUBTITLE_LANGUAGES: list[str] = ["xh", "en", "en-US", "en-GB", "zu", "af"]

# Scalar fields copied from the yt-dlp info dict into metadata.json.
# Large nested fields (subtitles, automatic_captions) are stored separately.
METADATA_FIELDS: tuple[str, ...] = (
    "id",
    "title",
    "webpage_url",
    "original_url",
    "uploader",
    "uploader_id",
    "channel",
    "channel_id",
    "upload_date",
    "duration",
    "description",
    "language",
    "playlist",
    "playlist_id",
    "playlist_title",
    "playlist_index",
    "n_entries",
    "ext",
    "format_id",
    "vcodec",
    "acodec",
    "abr",
    "asr",
)


@dataclass(frozen=True)
class DownloadSummary:
    """Aggregate outcome of a download run.

    Attributes:
        downloaded: Videos newly saved to disk in this run.
        skipped: Videos already present and left unchanged.
        repaired: Videos whose audio existed but metadata was missing.
        failed: Videos that could not be finalized.
        video_dirs: Paths to successfully handled video directories.
    """

    downloaded: int = 0
    skipped: int = 0
    repaired: int = 0
    failed: int = 0
    video_dirs: tuple[Path, ...] = ()

    @property
    def total_handled(self) -> int:
        """Return count of videos that were downloaded, skipped, or repaired."""
        return self.downloaded + self.skipped + self.repaired

    @property
    def succeeded(self) -> bool:
        """Return True when at least one video was handled and none failed."""
        return self.total_handled > 0 and self.failed == 0


def setup_logging(log_level: str) -> None:
    """Configure console and file logging for the download script.

    Idempotent: repeated calls (e.g. during tests) only adjust the level
    instead of attaching duplicate handlers.

    Args:
        log_level: Logging level name (e.g. ``INFO``, ``DEBUG``).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()

    if root_logger.handlers:
        # CHANGE: avoid duplicate log lines when setup_logging runs twice.
        root_logger.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / "download_audio.log", encoding="utf-8"),
        ],
    )


def get_video_directory(output_dir: Path, video_id: str) -> Path:
    """Return the on-disk folder for a single YouTube video.

    Each video is stored in its own subdirectory named after the video ID
    so titles can change without breaking skip logic or metadata paths.

    Args:
        output_dir: Root raw-data directory (typically ``data/raw``).
        video_id: YouTube video identifier.

    Returns:
        Path to the video-specific directory.
    """
    return output_dir / video_id


def get_metadata_path(output_dir: Path, video_id: str) -> Path:
    """Return the expected path to a video's metadata sidecar file.

    Args:
        output_dir: Root raw-data directory.
        video_id: YouTube video identifier.

    Returns:
        Path to ``metadata.json`` for the given video.
    """
    return get_video_directory(output_dir, video_id) / METADATA_FILENAME


def find_audio_file(video_dir: Path) -> Path | None:
    """Locate the downloaded audio file inside a video directory.

    Ignores subtitle files and ``metadata.json`` when searching.

    Args:
        video_dir: Directory that should contain one audio file.

    Returns:
        Path to the audio file, or ``None`` if none is found.
    """
    if not video_dir.is_dir():
        return None

    audio_files = [
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
    if not audio_files:
        return None

    # Prefer the file whose stem matches the folder (video ID), else newest.
    preferred_suffixes = {video_dir / f"{video_dir.name}{path.suffix}" for path in audio_files}
    for path in audio_files:
        if path in preferred_suffixes:
            return path
    return max(audio_files, key=lambda path: path.stat().st_mtime)


def has_downloaded_audio(output_dir: Path, video_id: str) -> bool:
    """Return True when an audio file already exists for a video.

    Used by the yt-dlp skip filter so re-runs do not re-download audio.

    Args:
        output_dir: Root raw-data directory.
        video_id: YouTube video identifier.

    Returns:
        ``True`` if audio is already on disk.
    """
    return find_audio_file(get_video_directory(output_dir, video_id)) is not None


def has_complete_record(output_dir: Path, video_id: str) -> bool:
    """Return True when both audio and metadata exist for a video.

    Args:
        output_dir: Root raw-data directory.
        video_id: YouTube video identifier.

    Returns:
        ``True`` if the video is fully recorded and can be skipped entirely.
    """
    metadata_path = get_metadata_path(output_dir, video_id)
    return metadata_path.is_file() and has_downloaded_audio(output_dir, video_id)


def list_subtitle_files(video_dir: Path) -> list[str]:
    """List subtitle filenames stored alongside a downloaded video.

    Args:
        video_dir: Directory containing audio and optional subtitle files.

    Returns:
        Sorted list of subtitle filenames.
    """
    if not video_dir.is_dir():
        return []

    return sorted(
        path.name
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUBTITLE_EXTENSIONS
    )


def build_metadata_payload(
    info: dict[str, Any],
    output_dir: Path,
    audio_path: Path | None,
) -> dict[str, Any]:
    """Build a JSON-serializable metadata document from yt-dlp info.

    Copies a curated set of scalar fields and stores subtitle *language
    keys* only (not full format trees, which bloat metadata files).

    Args:
        info: Info dictionary returned by yt-dlp after extraction/download.
        output_dir: Root raw-data directory.
        audio_path: Resolved path to the saved audio file, if known.

    Returns:
        Dictionary ready to be written as ``metadata.json``.
    """
    payload: dict[str, Any] = {
        field: info[field]
        for field in METADATA_FIELDS
        if field in info and info[field] is not None
    }

    payload["downloaded_at"] = datetime.now(timezone.utc).isoformat()
    payload["video_id"] = info.get("id")
    payload["audio_file"] = audio_path.name if audio_path else None
    payload["audio_path"] = str(audio_path) if audio_path else None

    # CHANGE: store language keys only; full subtitle format maps are huge.
    manual_subs = info.get("subtitles") or {}
    auto_subs = info.get("automatic_captions") or {}
    if manual_subs:
        payload["available_subtitles"] = sorted(manual_subs.keys())
    if auto_subs:
        payload["available_auto_captions"] = sorted(auto_subs.keys())

    video_id = info.get("id")
    if video_id:
        subtitle_files = list_subtitle_files(get_video_directory(output_dir, video_id))
        if subtitle_files:
            payload["subtitle_files"] = subtitle_files

    return payload


def save_metadata(info: dict[str, Any], output_dir: Path) -> Path:
    """Write ``metadata.json`` for a downloaded video atomically.

    Writes to a temporary file first, then replaces the target so a
    partial write never leaves corrupt metadata on disk.

    Args:
        info: Info dictionary from yt-dlp for a single video entry.
        output_dir: Root raw-data directory.

    Returns:
        Path to the written metadata file.

    Raises:
        ValueError: If the info dict does not contain a video ``id``.
        TypeError: If metadata contains non-JSON-serializable values.
        OSError: If the file cannot be written.
    """
    video_id = info.get("id")
    if not video_id:
        raise ValueError("Cannot save metadata without a video id.")

    video_dir = get_video_directory(output_dir, video_id)
    video_dir.mkdir(parents=True, exist_ok=True)

    audio_path = find_audio_file(video_dir)
    metadata = build_metadata_payload(info, output_dir, audio_path)
    metadata_path = get_metadata_path(output_dir, video_id)
    temp_path = metadata_path.with_suffix(".json.tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
        temp_path.replace(metadata_path)
    except Exception:
        # CHANGE: remove partial temp file so the next run can retry cleanly.
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise

    logger.info("Saved metadata: %s", metadata_path)
    return metadata_path


def create_skip_filter(
    output_dir: Path,
) -> Callable[..., str | None]:
    """Create a yt-dlp match filter that skips videos with existing audio.

    yt-dlp calls the returned function for each playlist entry (or single
    video). Returning a non-empty string tells yt-dlp to skip that entry.

    Args:
        output_dir: Root raw-data directory.

    Returns:
        Callable compatible with yt-dlp's ``match_filter`` option.
    """
    def skip_if_downloaded(
        info_dict: dict[str, Any],
        *,
        incomplete: bool,
    ) -> str | None:
        # Incomplete entries lack an id until fully parsed; never skip those.
        if incomplete:
            return None

        video_id = info_dict.get("id")
        if video_id and has_downloaded_audio(output_dir, video_id):
            title = info_dict.get("title", video_id)
            logger.debug("yt-dlp skip filter: audio exists for %s (%s)", title, video_id)
            return "Already downloaded"
        return None

    return skip_if_downloaded


def build_ydl_options(output_dir: Path) -> dict[str, Any]:
    """Build yt-dlp options for audio-only downloads with subtitles.

    Configures:
    - best available audio stream (no video)
    - per-video output folders under ``data/raw/<video_id>/``
    - manual and automatic subtitles when available
    - playlist support (single URLs and playlist URLs)
    - skip logic for items whose audio is already on disk

    Args:
        output_dir: Root raw-data directory.

    Returns:
        Dictionary of yt-dlp options.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Audio and subtitles share the same folder template; extensions differ.
    output_template = str(output_dir / "%(id)s" / "%(id)s.%(ext)s")

    return {
        # Audio-only: never download a video stream.
        "format": "bestaudio/best",
        "extract_audio": False,
        # Allow both single videos and full playlists.
        "noplaylist": False,
        "ignoreerrors": True,
        "restrictfilenames": True,
        "outtmpl": {
            "default": output_template,
            "subtitle": output_template,
        },
        # Subtitles: download manual and auto-generated tracks when present.
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": SUBTITLE_LANGUAGES,
        "subtitlesformat": "best",
        # Reduce console noise; we log through the logging module instead.
        "quiet": True,
        "no_warnings": True,
        # Skip videos whose audio file is already present on disk.
        "match_filter": create_skip_filter(output_dir),
    }


def iter_video_entries(info: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten yt-dlp info into a list of individual video entries.

    For a single video URL the list has one item. For playlists each
    child entry is returned separately. Entries that failed extraction
    (``None`` values when ``ignoreerrors`` is enabled) are omitted.

    Args:
        info: Top-level info dict from ``extract_info``.

    Returns:
        List of per-video info dictionaries.
    """
    if info.get("_type") == "playlist":
        entries = info.get("entries") or []
        return [entry for entry in entries if isinstance(entry, dict)]
    return [info]


def snapshot_existing_audio_ids(output_dir: Path) -> frozenset[str]:
    """Collect video IDs that already have audio on disk before a download run.

    Args:
        output_dir: Root raw-data directory.

    Returns:
        Frozen set of video IDs with at least one audio file present.
    """
    if not output_dir.is_dir():
        return frozenset()

    return frozenset(
        path.name
        for path in output_dir.iterdir()
        if path.is_dir() and find_audio_file(path) is not None
    )


def finalize_video_entry(
    entry: dict[str, Any],
    output_dir: Path,
    *,
    preexisting_audio_ids: frozenset[str],
) -> tuple[Path | None, str]:
    """Validate a video folder and persist metadata when needed.

    Handles three post-download states:
    - complete record (audio + metadata): skip
    - audio without metadata: repair metadata from yt-dlp info
    - missing audio after download attempt: mark as failed

    Args:
        entry: yt-dlp info dict for one video.
        output_dir: Root raw-data directory.
        preexisting_audio_ids: Video IDs that already had audio before this run.

    Returns:
        Tuple of ``(video_dir, status)`` where status is one of
        ``"downloaded"``, ``"skipped"``, ``"repaired"``, or ``"failed"``.
        ``video_dir`` is ``None`` when finalization fails.
    """
    video_id = entry.get("id")
    title = entry.get("title", video_id or "unknown")

    if not video_id:
        logger.warning("Skipping entry without video id: %s", title)
        return None, "failed"

    video_dir = get_video_directory(output_dir, video_id)
    audio_path = find_audio_file(video_dir)

    if audio_path is None:
        logger.warning(
            "No audio file found for %s (%s) after download attempt.",
            title,
            video_id,
        )
        return None, "failed"

    if has_complete_record(output_dir, video_id):
        logger.info("Already on disk: %s (%s)", title, video_id)
        return video_dir, "skipped"

    save_metadata(entry, output_dir)

    # CHANGE: compare against pre-run snapshot to distinguish new vs repaired.
    status = "repaired" if video_id in preexisting_audio_ids else "downloaded"
    logger.info(
        "%s: %s -> %s",
        "Metadata repaired" if status == "repaired" else "Download complete",
        title,
        audio_path,
    )
    return video_dir, status


def download_youtube_audio(url: str, output_dir: Path) -> DownloadSummary:
    """Download audio (and subtitles) from a YouTube URL or playlist.

    Supports:
    - single video URLs
    - playlist URLs (every item is processed)
    - skipping items whose audio already exists
    - repairing missing ``metadata.json`` for existing audio
    - writing ``metadata.json`` beside each audio file

    Args:
        url: YouTube video or playlist URL.
        output_dir: Directory where files will be saved (default ``data/raw``).

    Returns:
        Summary describing downloaded, skipped, repaired, and failed items.

    Raises:
        DownloadError: If yt-dlp cannot extract any information from the URL.
        ValueError: If extraction succeeds but yields no video entries.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = build_ydl_options(output_dir)

    logger.info("Starting download: %s", url)
    logger.debug("Output directory: %s", output_dir)

    # Snapshot before yt-dlp runs so we can tell new downloads from repairs.
    preexisting_audio_ids = snapshot_existing_audio_ids(output_dir)

    downloaded = skipped = repaired = failed = 0
    video_dirs: list[Path] = []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError:
        logger.exception("yt-dlp failed to download from URL: %s", url)
        raise
    except Exception:
        logger.exception("Unexpected error while downloading: %s", url)
        raise

    if info is None:
        raise DownloadError(f"No information extracted for URL: {url}")

    entries = iter_video_entries(info)
    if not entries:
        raise ValueError(f"No video entries found for URL: {url}")

    logger.info("Processing %d video entry/entries.", len(entries))

    for entry in entries:
        title = entry.get("title", entry.get("id", "unknown"))
        try:
            video_dir, status = finalize_video_entry(
                entry,
                output_dir,
                preexisting_audio_ids=preexisting_audio_ids,
            )
        except Exception:
            # CHANGE: one bad entry must not abort an entire playlist run.
            logger.exception("Failed to finalize download for %s", title)
            failed += 1
            continue

        if video_dir is None:
            failed += 1
            continue

        video_dirs.append(video_dir)
        if status == "downloaded":
            downloaded += 1
        elif status == "skipped":
            skipped += 1
        elif status == "repaired":
            repaired += 1
        else:
            failed += 1

    summary = DownloadSummary(
        downloaded=downloaded,
        skipped=skipped,
        repaired=repaired,
        failed=failed,
        video_dirs=tuple(video_dirs),
    )

    logger.info(
        "Finished processing URL. downloaded=%d skipped=%d repaired=%d failed=%d",
        summary.downloaded,
        summary.skipped,
        summary.repaired,
        summary.failed,
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the download script.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download YouTube audio (and subtitles when available) into "
            "data/raw using yt-dlp. Supports single videos and playlists."
        ),
    )
    parser.add_argument(
        "url",
        help="YouTube video or playlist URL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help=f"Output directory (default: {RAW_DATA_DIR}).",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help=f"Logging level (default: {DEFAULT_LOG_LEVEL}).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point when the script is run from the command line."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        summary = download_youtube_audio(args.url, args.output_dir)
    except DownloadError:
        logger.error("Download failed for URL: %s", args.url)
        raise SystemExit(1) from None
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from None
    except Exception:
        logger.exception("Unhandled error while downloading %s", args.url)
        raise SystemExit(1) from None

    # CHANGE: non-zero exit when nothing succeeded or any item failed.
    if summary.failed > 0:
        logger.error(
            "%d video(s) failed. See logs/download_audio.log for details.",
            summary.failed,
        )
        raise SystemExit(1)
    if summary.total_handled == 0:
        logger.error("No videos were downloaded or found on disk.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
