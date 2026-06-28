"""Run inference on new audio with a trained isiXhosa ASR model."""

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
    TARGET_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """Configure logging for the inference script.

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
            logging.FileHandler(LOGS_DIR / "inference.log", encoding="utf-8"),
        ],
    )


def transcribe(
    audio_path: Path,
    model_path: Path,
) -> str:
    """Transcribe a single audio file.

    Args:
        audio_path: Path to the input audio file.
        model_path: Path to a trained model checkpoint or directory.

    Returns:
        Predicted transcript text.
    """
    # TODO: Load the trained model and processor from model_path.
    # TODO: Preprocess input audio to TARGET_SAMPLE_RATE mono if needed.
    # TODO: Run model.generate() or equivalent inference forward pass.
    # TODO: Decode token IDs to isiXhosa text with the model tokenizer.
    # TODO: Apply optional post-processing (punctuation, casing normalization).
    # TODO: Return the final transcript string.
    raise NotImplementedError("Inference script is not yet implemented.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio using a trained isiXhosa ASR model.",
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR,
        help=f"Path to model checkpoint (default: {MODELS_DIR}).",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help=f"Logging level (default: {DEFAULT_LOG_LEVEL}).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the inference script."""
    args = parse_args()
    setup_logging(args.log_level)

    logger.debug(
        "Target sample rate for inference preprocessing: %d Hz",
        TARGET_SAMPLE_RATE,
    )

    try:
        transcript = transcribe(args.audio_path, args.model_path)
        logger.info("Transcript: %s", transcript)
    except NotImplementedError:
        logger.error("Inference script is not yet implemented.")
        raise SystemExit(1) from None
    except Exception:
        logger.exception("Inference failed for %s", args.audio_path)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
