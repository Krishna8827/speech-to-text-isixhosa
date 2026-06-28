"""Download YouTube audio for the isiXhosa speech-to-text pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yt_dlp

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

SUPPORTED_AUDIO_EXTENSIONS = {".m4a", ".mp3", ".opus", ".ogg", ".wav", ".webm"}


def setup_logging(log_level: str) -> None:
    """Configure logging for the download script.

    Args:
        log_level: Logging level name (e.g. ``INFO``, ``DEBUG``).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / "download_audio.log", encoding="utf-8"),
        ],
    )


def download_youtube_audio(url: str, output_dir: Path) -> Path:
    """Download audio from a YouTube URL using yt-dlp.

    Args:
        url: YouTube video URL.
        output_dir: Directory where the audio file will be saved.

    Returns:
        Path to the downloaded audio file.

    Raises:
        FileNotFoundError: If yt-dlp completes but no audio file is found.
        yt_dlp.utils.DownloadError: If the download fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(title)s.%(ext)s")

    ydl_opts: dict[str, object] = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "restrictfilenames": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [],
    }

    logger.info("Downloading audio from: %s", url)
    logger.debug("Output directory: %s", output_dir)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise FileNotFoundError(f"Could not extract info for URL: {url}")

        title = info.get("title", "unknown")
        ext = info.get("ext", "m4a")
        downloaded_path = output_dir / f"{title}.{ext}"

        if downloaded_path.exists():
            logger.info("Download complete: %s", downloaded_path)
            return downloaded_path

        candidates = sorted(
            path
            for path in output_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        )
        if not candidates:
            raise FileNotFoundError(
                f"Download finished but no audio file found in {output_dir}"
            )

        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        logger.info("Download complete: %s", latest)
        return latest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Download YouTube audio into data/raw using yt-dlp.",
    )
    parser.add_argument(
        "url",
        help="YouTube video URL to download.",
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
    """Entry point for the download script."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        download_youtube_audio(args.url, args.output_dir)
    except Exception:
        logger.exception("Failed to download audio from %s", args.url)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
