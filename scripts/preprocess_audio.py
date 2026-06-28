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

SUPPORTED_INPUT_EXTENSIONS = {".m4a", ".mp3", ".opus", ".ogg", ".wav", ".webm", ".flac"}


def setup_logging(log_level: str) -> None:
    """Configure logging for the preprocess script.

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
    """Find supported audio files in a directory.

    Args:
        input_dir: Directory to scan for audio files.

    Returns:
        Sorted list of audio file paths.
    """
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
    )


def preprocess_directory(input_dir: Path, output_dir: Path) -> list[Path]:
    """Preprocess all supported audio files in a directory.

    Args:
        input_dir: Directory containing raw audio files.
        output_dir: Directory where processed WAV files are written.

    Returns:
        List of paths to successfully processed files.
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = discover_audio_files(input_dir)
    if not audio_files:
        logger.warning("No supported audio files found in %s", input_dir)
        return []

    processed_paths: list[Path] = []
    for input_path in audio_files:
        output_path = output_dir / f"{input_path.stem}.wav"
        try:
            processed_paths.append(
                preprocess_audio_file(input_path, output_path),
            )
        except Exception:
            logger.exception("Failed to preprocess %s", input_path)

    logger.info(
        "Preprocessed %d of %d file(s).",
        len(processed_paths),
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
            output_path = args.output_dir / f"{args.input.stem}.wav"
            preprocess_audio_file(args.input, output_path)
        else:
            preprocess_directory(args.input_dir, args.output_dir)
    except Exception:
        logger.exception("Audio preprocessing failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
