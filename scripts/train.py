"""Train an isiXhosa speech-to-text model."""

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
    TRAIN_DATA_DIR,
    VALIDATION_DATA_DIR,
)

logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """Configure logging for the training script.

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
            logging.FileHandler(LOGS_DIR / "train.log", encoding="utf-8"),
        ],
    )


def train(
    train_dir: Path,
    validation_dir: Path,
    models_dir: Path,
) -> None:
    """Fine-tune an ASR model on isiXhosa data.

    Args:
        train_dir: Directory containing training samples and manifests.
        validation_dir: Directory containing validation samples and manifests.
        models_dir: Directory where model checkpoints will be saved.
    """
    # TODO: Choose a base ASR architecture (e.g. Whisper, Wav2Vec2, MMS).
    # TODO: Load training and validation manifests from dataset_builder output.
    # TODO: Define a PyTorch/Hugging Face Dataset or custom DataLoader.
    # TODO: Configure tokenizer/processor for isiXhosa text normalization.
    # TODO: Set hyperparameters (batch size, learning rate, epochs, warmup steps).
    # TODO: Implement training loop with gradient accumulation and mixed precision.
    # TODO: Run validation each epoch and track WER/CER on the validation set.
    # TODO: Save best checkpoint and final model weights to models_dir.
    # TODO: Write a training summary (config, metrics, checkpoint paths) to results/.
    raise NotImplementedError("Training script is not yet implemented.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train an isiXhosa speech-to-text model.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=TRAIN_DATA_DIR,
        help=f"Training data directory (default: {TRAIN_DATA_DIR}).",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=VALIDATION_DATA_DIR,
        help=f"Validation data directory (default: {VALIDATION_DATA_DIR}).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"Model checkpoint directory (default: {MODELS_DIR}).",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help=f"Logging level (default: {DEFAULT_LOG_LEVEL}).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the training script."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        train(args.train_dir, args.validation_dir, args.models_dir)
    except NotImplementedError:
        logger.error("Training script is not yet implemented.")
        raise SystemExit(1) from None
    except Exception:
        logger.exception("Training failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
