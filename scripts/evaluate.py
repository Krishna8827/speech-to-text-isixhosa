"""Evaluate a trained isiXhosa speech-to-text model."""

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
    MODELS_DIR,
    RESULTS_DIR,
    TEST_DATA_DIR,
)

logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """Configure logging for the evaluation script.

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
            logging.FileHandler(LOGS_DIR / "evaluate.log", encoding="utf-8"),
        ],
    )


def evaluate(
    model_path: Path,
    test_dir: Path,
    results_dir: Path,
) -> None:
    """Evaluate model performance on the test set.

    Args:
        model_path: Path to a trained model checkpoint or directory.
        test_dir: Directory containing test samples and manifests.
        results_dir: Directory where evaluation reports will be written.
    """
    # TODO: Load the trained model and processor from model_path.
    # TODO: Load the test manifest produced by dataset_builder.
    # TODO: Run batch inference on all test audio files.
    # TODO: Compute Word Error Rate (WER) and Character Error Rate (CER).
    # TODO: Optionally compute per-sample metrics and confidence scores.
    # TODO: Save predictions, references, and aggregate metrics to results_dir.
    # TODO: Generate a human-readable evaluation report (JSON and/or Markdown).
    raise NotImplementedError("Evaluation script is not yet implemented.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate an isiXhosa speech-to-text model on the test set.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR,
        help=f"Path to model checkpoint (default: {MODELS_DIR}).",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=TEST_DATA_DIR,
        help=f"Test data directory (default: {TEST_DATA_DIR}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Results output directory (default: {RESULTS_DIR}).",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help=f"Logging level (default: {DEFAULT_LOG_LEVEL}).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the evaluation script."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        evaluate(args.model_path, args.test_dir, args.results_dir)
    except NotImplementedError:
        logger.error("Evaluation script is not yet implemented.")
        raise SystemExit(1) from None
    except Exception:
        logger.exception("Evaluation failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
