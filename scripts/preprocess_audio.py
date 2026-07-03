"""Preprocess raw audio into 16 kHz mono WAV files."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import librosa
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DEFAULT_LOG_LEVEL,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOGS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TARGET_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

SUPPORTED_INPUT_EXTENSIONS: frozenset[str] = frozenset(
    {".m4a", ".mp3", ".opus", ".ogg", ".wav", ".webm", ".flac", ".aac"}
)


def setup_logging(log_level: str) -> None:
    """Configure logging for the preprocess script.

    Args:
        log_level: Logging level name (e.g. ``INFO``, ``DEBUG``).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()

    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / "preprocess_audio.log", encoding="utf-8"),
        ],
    )


def preprocess_audio_file(input_path: Path, output_path: Path) -> Path:
    """Convert a single audio file to 16 kHz mono WAV.

    Uses librosa for resampling and mono conversion. FFmpeg is used
    internally by librosa/audioread when decoding compressed formats.

    Args:
        input_path: Path to the source audio file.
        output_path: Path for the output WAV file.

    Returns:
        Path to the written WAV file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the loaded audio is empty.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Processing: %s", input_path)
    audio, _sample_rate = librosa.load(
        input_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError(f"No audio samples loaded from {input_path}")

    sf.write(output_path, audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
    logger.info("Saved: %s", output_path)
    return output_path


def discover_audio_files(input_dir: Path) -> list[Path]:
    """Find supported audio files under a directory tree.

    Recursively scans ``input_dir`` so nested layouts produced by
    ``download_audio.py`` (``data/raw/<video_id>/<video_id>.webm``) are
    discovered alongside any flat files placed directly in ``data/raw/``.

    Args:
        input_dir: Root directory to scan for audio files.

    Returns:
        Sorted list of audio file paths.
    """
    if not input_dir.is_dir():
        return []

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
    )


def resolve_output_path(
    input_path: Path,
    input_root: Path,
    output_root: Path,
) -> Path:
    """Map a raw audio file to its processed WAV output path.

    Preserves YouTube video IDs from the download layout:
    ``data/raw/<video_id>/<file>`` -> ``data/processed/<video_id>/<video_id>.wav``

    Flat files in ``data/raw/`` continue to map to
    ``data/processed/<stem>.wav``.

    Args:
        input_path: Source audio file path.
        input_root: Root raw-data directory (typically ``data/raw``).
        output_root: Root processed-data directory (typically ``data/processed``).

    Returns:
        Destination path for the converted WAV file.
    """
    input_path = input_path.resolve()
    input_root = input_root.resolve()
    output_root = output_root.resolve()

    try:
        relative = input_path.relative_to(input_root)
    except ValueError:
        return output_root / f"{input_path.stem}.wav"

    # Nested layout from download_audio.py: <video_id>/<filename>.<ext>
    if len(relative.parts) >= 2:
        video_id = relative.parts[0]
        return output_root / video_id / f"{video_id}.wav"

    return output_root / f"{input_path.stem}.wav"


def should_skip_processing(input_path: Path, output_path: Path) -> bool:
    """Return True when an up-to-date processed file already exists.

    Skips re-processing when the output WAV is at least as new as the
    source file, so re-running the script is safe after partial runs.

    Args:
        input_path: Source audio file path.
        output_path: Expected processed WAV path.

    Returns:
        ``True`` if processing can be skipped.
    """
    if not output_path.is_file():
        return False

    return output_path.stat().st_mtime >= input_path.stat().st_mtime


def preprocess_directory(input_dir: Path, output_dir: Path) -> list[Path]:
    """Preprocess all supported audio files under a directory tree.

    Args:
        input_dir: Directory containing raw audio files (flat or nested).
        output_dir: Directory where processed WAV files are written.

    Returns:
        List of paths to successfully processed or already-valid WAV files.
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = discover_audio_files(input_dir)
    if not audio_files:
        logger.warning("No supported audio files found under %s", input_dir)
        return []

    processed_paths: list[Path] = []
    skipped_count = 0

    for input_path in audio_files:
        output_path = resolve_output_path(input_path, input_dir, output_dir)

        if should_skip_processing(input_path, output_path):
            logger.info("Skipping up-to-date file: %s", output_path)
            processed_paths.append(output_path)
            skipped_count += 1
            continue

        try:
            processed_paths.append(
                preprocess_audio_file(input_path, output_path),
            )
        except Exception:
            logger.exception("Failed to preprocess %s", input_path)

    newly_processed = len(processed_paths) - skipped_count
    logger.info(
        "Preprocessed %d file(s), skipped %d up-to-date file(s), "
        "out of %d source file(s).",
        newly_processed,
        skipped_count,
        len(audio_files),
    )
    return processed_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Convert raw audio to 16 kHz mono WAV in data/processed.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Single input audio file. If omitted, all files in --input-dir are processed.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help=f"Directory of raw audio files (default: {RAW_DATA_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help=f"Output directory for WAV files (default: {PROCESSED_DATA_DIR}).",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help=f"Logging level (default: {DEFAULT_LOG_LEVEL}).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the preprocess script."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        if args.input is not None:
            output_path = resolve_output_path(
                args.input,
                args.input_dir,
                args.output_dir,
            )
            if should_skip_processing(args.input, output_path):
                logger.info("Skipping up-to-date file: %s", output_path)
            else:
                preprocess_audio_file(args.input, output_path)
        else:
            preprocess_directory(args.input_dir, args.output_dir)
    except Exception:
        logger.exception("Audio preprocessing failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
