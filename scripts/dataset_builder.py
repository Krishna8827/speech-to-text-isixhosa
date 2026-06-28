"""Build train, validation, and test datasets from processed audio."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DEFAULT_LOG_LEVEL,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOGS_DIR,
    PROCESSED_DATA_DIR,
    TEST_DATA_DIR,
    TRAIN_DATA_DIR,
    VALIDATION_DATA_DIR,
)

logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """Configure logging for the dataset builder script.

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
            logging.FileHandler(LOGS_DIR / "dataset_builder.log", encoding="utf-8"),
        ],
    )


def build_dataset(
    processed_dir: Path,
    train_dir: Path,
    validation_dir: Path,
    test_dir: Path,
) -> None:
    """Create train, validation, and test splits from processed audio.

    Args:
        processed_dir: Directory containing preprocessed WAV files.
        train_dir: Output directory for training samples.
        validation_dir: Output directory for validation samples.
        test_dir: Output directory for test samples.
    """
    # TODO: Load or create transcript metadata (CSV/JSON) aligned with audio files.
    # TODO: Validate that every audio file has a corresponding transcript.
    # TODO: Split samples into train/validation/test (e.g. 80/10/10) with a fixed seed.
    # TODO: Copy or symlink audio files into data/train, data/validation, data/test.
    # TODO: Export manifest files (audio_path, transcript, duration, split) for training.
    # TODO: Log split statistics (sample count, total duration, vocabulary size).
    raise NotImplementedError("Dataset builder is not yet implemented.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build train/validation/test datasets from processed audio.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help=f"Directory of processed WAV files (default: {PROCESSED_DATA_DIR}).",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=TRAIN_DATA_DIR,
        help=f"Training output directory (default: {TRAIN_DATA_DIR}).",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=VALIDATION_DATA_DIR,
        help=f"Validation output directory (default: {VALIDATION_DATA_DIR}).",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=TEST_DATA_DIR,
        help=f"Test output directory (default: {TEST_DATA_DIR}).",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help=f"Logging level (default: {DEFAULT_LOG_LEVEL}).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the dataset builder script."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        build_dataset(
            args.processed_dir,
            args.train_dir,
            args.validation_dir,
            args.test_dir,
        )
    except NotImplementedError:
        logger.error("Dataset builder is not yet implemented.")
        raise SystemExit(1) from None
    except Exception:
        logger.exception("Dataset build failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
