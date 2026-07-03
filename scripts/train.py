"""Train an isiXhosa speech-to-text model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    CHECKPOINTS_DIR,
    DEFAULT_LOG_LEVEL,
    FINAL_MODEL_DIR,
    HF_DATASET_NAME,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOGS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TRAINING_BATCH_SIZE,
    TRAINING_EVAL_STEPS,
    TRAINING_GENERATION_MAX_LENGTH,
    TRAINING_GRADIENT_ACCUMULATION_STEPS,
    TRAINING_LEARNING_RATE,
    TRAINING_LOGGING_STEPS,
    TRAINING_NUM_EPOCHS,
    TRAINING_OUTPUT_DIR,
    TRAINING_PREDICT_WITH_GENERATE,
    TRAINING_SAVE_STEPS,
    TRAINING_SAVE_TOTAL_LIMIT,
    TRAINING_WARMUP_STEPS,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_NAME,
)
from training_utils import (  # noqa: E402
    DataCollatorSpeechSeq2SeqWithPadding,
    build_compute_metrics,
    load_isixhosa_dataset,
    load_whisper_model_and_processor,
    preprocess_dataset_splits,
    resolve_device,
    save_json_report,
)

logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """Configure logging for the training script.

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
            logging.FileHandler(LOGS_DIR / "train.log", encoding="utf-8"),
        ],
    )


def train(
    models_dir: Path,
    dataset_name: str,
    model_name: str,
    output_dir: Path,
    final_model_dir: Path,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_steps: int,
    seed: int,
) -> None:
    """Fine-tune Whisper on the isiXhosa ASR dataset.

    Args:
        models_dir: Root directory for saved model artifacts.
        dataset_name: Hugging Face dataset identifier.
        model_name: Base Whisper model to fine-tune.
        output_dir: Directory for training checkpoints.
        final_model_dir: Directory for the exported final model.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Optimizer learning rate.
        warmup_steps: Learning-rate warmup steps.
        seed: Random seed for reproducibility.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    use_fp16 = device.type == "cuda"

    logger.info("Loading base model '%s'", model_name)
    processor, model = load_whisper_model_and_processor(model_name, language=WHISPER_LANGUAGE)

    raw_dataset = load_isixhosa_dataset(dataset_name=dataset_name, seed=seed)
    dataset = preprocess_dataset_splits(raw_dataset, processor)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = build_compute_metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=TRAINING_LOGGING_STEPS,
        eval_steps=TRAINING_EVAL_STEPS,
        save_steps=TRAINING_SAVE_STEPS,
        save_total_limit=TRAINING_SAVE_TOTAL_LIMIT,
        predict_with_generate=TRAINING_PREDICT_WITH_GENERATE,
        generation_max_length=TRAINING_GENERATION_MAX_LENGTH,
        fp16=use_fp16,
        logging_dir=str(LOGS_DIR / "tensorboard"),
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=seed,
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    logger.info("Starting Whisper fine-tuning on %s", dataset_name)
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    logger.info("Saving final model to %s", final_model_dir)
    trainer.save_model(str(final_model_dir))
    processor.save_pretrained(str(final_model_dir))

    metrics_path = output_dir / "train_results.json"
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", eval_metrics)
    trainer.save_state()

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "base_model": model_name,
        "language": WHISPER_LANGUAGE,
        "checkpoints_dir": str(output_dir.resolve()),
        "final_model_dir": str(final_model_dir.resolve()),
        "device": str(device),
        "train_samples": len(dataset["train"]),
        "validation_samples": len(dataset["validation"]),
        "test_samples": len(dataset["test"]),
        "hyperparameters": {
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "seed": seed,
        },
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }

    summary_path = RESULTS_DIR / "training_summary.json"
    save_json_report(summary_path, summary)

    with (LOGS_DIR / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    logger.info("Training complete. Validation WER: %.4f", eval_metrics.get("eval_wer", float("nan")))
    logger.info("Final model saved to %s", final_model_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper on the isiXhosa ASR dataset.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"Root model directory (default: {MODELS_DIR}).",
    )
    parser.add_argument(
        "--dataset-name",
        default=HF_DATASET_NAME,
        help=f"Hugging Face dataset id (default: {HF_DATASET_NAME}).",
    )
    parser.add_argument(
        "--model-name",
        default=WHISPER_MODEL_NAME,
        help=f"Base Whisper model (default: {WHISPER_MODEL_NAME}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRAINING_OUTPUT_DIR,
        help=f"Checkpoint output directory (default: {TRAINING_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--final-model-dir",
        type=Path,
        default=FINAL_MODEL_DIR,
        help=f"Final model output directory (default: {FINAL_MODEL_DIR}).",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=TRAINING_NUM_EPOCHS,
        help=f"Training epochs (default: {TRAINING_NUM_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAINING_BATCH_SIZE,
        help=f"Per-device train batch size (default: {TRAINING_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=TRAINING_GRADIENT_ACCUMULATION_STEPS,
        help=(
            "Gradient accumulation steps "
            f"(default: {TRAINING_GRADIENT_ACCUMULATION_STEPS})."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=TRAINING_LEARNING_RATE,
        help=f"Learning rate (default: {TRAINING_LEARNING_RATE}).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=TRAINING_WARMUP_STEPS,
        help=f"Warmup steps (default: {TRAINING_WARMUP_STEPS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED}).",
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
        train(
            models_dir=args.models_dir,
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            output_dir=args.output_dir,
            final_model_dir=args.final_model_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
        )
    except Exception:
        logger.exception("Training failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
