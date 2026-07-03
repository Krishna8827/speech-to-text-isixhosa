"""Evaluate a trained isiXhosa speech-to-text model."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    DATASET_AUDIO_COLUMN,
    DATASET_TRANSCRIPTION_COLUMN,
    DEFAULT_LOG_LEVEL,
    FINAL_MODEL_DIR,
    HF_DATASET_NAME,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOGS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TARGET_SAMPLE_RATE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_NAME,
)
from training_utils import (  # noqa: E402
    compute_wer,
    decode_audio_value,
    load_isixhosa_dataset,
    resample_audio_array,
    resolve_device,
    resolve_model_path,
    save_json_report,
)

logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """Configure logging for the evaluation script.

    Args:
        log_level: Logging level name (e.g. ``INFO``, ``DEBUG``).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    level = getattr(logging, log_level.upper(), logging.INFO)

    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / "evaluate.log", encoding="utf-8"),
        ],
    )


def run_batch_inference(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_arrays: list[np.ndarray],
    sampling_rates: list[int],
    device: torch.device,
    language: str = WHISPER_LANGUAGE,
) -> list[str]:
    """Generate transcripts for a batch of audio arrays.

    Args:
        model: Loaded Whisper model.
        processor: Matching Whisper processor.
        audio_arrays: List of one-dimensional audio arrays.
        sampling_rates: Sample rate for each audio array.
        device: Target inference device.
        language: Whisper language code for transcription.

    Returns:
        List of predicted transcript strings.
    """
    predictions: list[str] = []

    for audio_array, sampling_rate in zip(audio_arrays, sampling_rates, strict=True):
        resampled = resample_audio_array(audio_array, sampling_rate, TARGET_SAMPLE_RATE)
        inputs = processor(
            resampled,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device)

        try:
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=language,
                task="transcribe",
            )
        except ValueError:
            logger.warning(
                "Language '%s' is not supported during generation; using task-only prompts.",
                language,
            )
            forced_decoder_ids = processor.get_decoder_prompt_ids(task="transcribe")

        with torch.inference_mode():
            generated_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,
            )

        prediction = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()
        predictions.append(prediction)

    return predictions


def evaluate(
    model_path: Path | None,
    results_dir: Path,
    dataset_name: str,
    split: str,
    batch_size: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Evaluate model performance on a dataset split.

    Args:
        model_path: Optional path to a trained model directory or Hub id.
        results_dir: Directory where evaluation reports will be written.
        dataset_name: Hugging Face dataset identifier.
        split: Dataset split to evaluate (e.g. ``test`` or ``validation``).
        batch_size: Number of samples processed per inference batch.
        seed: Random seed used when creating dataset splits.

    Returns:
        Dictionary of aggregate evaluation metrics.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    resolved_model = resolve_model_path(
        model_path=model_path,
        final_model_dir=FINAL_MODEL_DIR,
        default_pretrained=WHISPER_MODEL_NAME,
    )

    device = resolve_device()
    logger.info("Loading model from '%s'", resolved_model)
    processor = WhisperProcessor.from_pretrained(resolved_model)
    model = WhisperForConditionalGeneration.from_pretrained(resolved_model)
    model.to(device)
    model.eval()

    dataset = load_isixhosa_dataset(dataset_name=dataset_name, seed=seed)
    if split not in dataset:
        raise ValueError(
            f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}"
        )

    eval_split = dataset[split]
    logger.info("Evaluating on '%s' split (%d samples)", split, len(eval_split))

    references: list[str] = []
    predictions: list[str] = []
    per_sample: list[dict[str, str | int]] = []

    audio_arrays: list[np.ndarray] = []
    sampling_rates: list[int] = []
    batch_references: list[str] = []
    batch_indices: list[int] = []

    for index, sample in enumerate(eval_split):
        audio = sample[DATASET_AUDIO_COLUMN]
        array, sampling_rate = decode_audio_value(audio)
        reference = sample[DATASET_TRANSCRIPTION_COLUMN].strip()

        audio_arrays.append(array)
        sampling_rates.append(sampling_rate)
        batch_references.append(reference)
        batch_indices.append(index)

        if len(audio_arrays) < batch_size and index < len(eval_split) - 1:
            continue

        batch_predictions = run_batch_inference(
            model=model,
            processor=processor,
            audio_arrays=audio_arrays,
            sampling_rates=sampling_rates,
            device=device,
            language=WHISPER_LANGUAGE,
        )

        for sample_index, prediction, reference_text in zip(
            batch_indices,
            batch_predictions,
            batch_references,
            strict=True,
        ):
            predictions.append(prediction)
            references.append(reference_text)
            per_sample.append(
                {
                    "index": sample_index,
                    "reference": reference_text,
                    "prediction": prediction,
                }
            )

        audio_arrays = []
        sampling_rates = []
        batch_references = []
        batch_indices = []

    wer = compute_wer(predictions, references)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": resolved_model,
        "dataset": dataset_name,
        "split": split,
        "num_samples": len(references),
        "wer": wer,
        "language": WHISPER_LANGUAGE,
        "device": str(device),
        "samples": per_sample,
    }

    json_path = results_dir / f"evaluation_{split}.json"
    save_json_report(json_path, report)

    markdown_path = results_dir / f"evaluation_{split}.md"
    markdown_lines = [
        "# isiXhosa ASR Evaluation Report",
        "",
        f"- **Timestamp (UTC):** {report['timestamp_utc']}",
        f"- **Model:** `{resolved_model}`",
        f"- **Dataset:** `{dataset_name}`",
        f"- **Split:** `{split}`",
        f"- **Samples:** {len(references)}",
        f"- **WER:** {wer:.4f}",
        "",
        "## Sample Predictions",
        "",
    ]
    for sample in per_sample[:10]:
        markdown_lines.extend(
            [
                f"### Sample {sample['index']}",
                "",
                f"- **Reference:** {sample['reference']}",
                f"- **Prediction:** {sample['prediction']}",
                "",
            ]
        )
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    logger.info("Saved Markdown report to %s", markdown_path)

    logger.info("Evaluation complete. WER on '%s': %.4f", split, wer)
    return {"wer": wer, "num_samples": len(references), "split": split}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate an isiXhosa speech-to-text model.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Path to a trained model directory or Hugging Face model id. "
            f"Defaults to {FINAL_MODEL_DIR} when present, otherwise "
            f"{WHISPER_MODEL_NAME}."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=HF_DATASET_NAME,
        help=f"Hugging Face dataset id (default: {HF_DATASET_NAME}).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("validation", "test"),
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Results output directory (default: {RESULTS_DIR}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for dataset splits (default: {RANDOM_SEED}).",
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
        evaluate(
            model_path=args.model_path,
            results_dir=args.results_dir,
            dataset_name=args.dataset_name,
            split=args.split,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    except Exception:
        logger.exception("Evaluation failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
