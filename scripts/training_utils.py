"""Shared utilities for Whisper training, evaluation, and dataset handling."""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import (
    DATASET_AUDIO_COLUMN,
    DATASET_TRANSCRIPTION_COLUMN,
    HF_DATASET_NAME,
    RANDOM_SEED,
    TARGET_SAMPLE_RATE,
    TEST_SPLIT_RATIO,
    VALIDATION_SPLIT_RATIO,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_NAME,
)

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Collate speech features and tokenized labels for Whisper seq2seq training."""

    processor: WhisperProcessor

    def __call__(
        self,
        features: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """Pad a batch of preprocessed samples for the trainer.

        Args:
            features: List of dicts with ``input_features`` and ``labels``.

        Returns:
            Batched tensors ready for ``Seq2SeqTrainer``.
        """
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )

        if (
            labels.size(1) > 0
            and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def decode_audio_value(audio_value: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Decode a Hugging Face audio feature to a float array and sample rate.

    Uses ``soundfile`` so audio loading works on Windows without ``torchcodec``.

    Args:
        audio_value: Raw or decoded audio dictionary from ``datasets``.

    Returns:
        Tuple of ``(audio_array, sampling_rate)``.

    Raises:
        ValueError: If the audio dictionary does not contain decodable data.
    """
    if "array" in audio_value and "sampling_rate" in audio_value:
        array = np.asarray(audio_value["array"], dtype=np.float32)
        return array, int(audio_value["sampling_rate"])

    if audio_value.get("bytes"):
        array, sampling_rate = sf.read(io.BytesIO(audio_value["bytes"]))
        return np.asarray(array, dtype=np.float32), int(sampling_rate)

    if audio_value.get("path"):
        path = Path(str(audio_value["path"]))
        if not path.is_file():
            raise ValueError(f"Audio path does not exist: {path}")
        array, sampling_rate = sf.read(path)
        return np.asarray(array, dtype=np.float32), int(sampling_rate)

    raise ValueError("Audio value is missing 'array', 'bytes', and 'path' fields.")


def cast_dataset_audio_for_local_decoding(dataset: DatasetDict) -> DatasetDict:
    """Disable automatic torchcodec decoding for all audio columns.

    Args:
        dataset: Dataset splits returned by ``load_dataset``.

    Returns:
        Dataset with ``Audio(decode=False)`` casting applied.
    """
    casted = DatasetDict(
        {
            split_name: split_data.cast_column(
                DATASET_AUDIO_COLUMN,
                Audio(sampling_rate=TARGET_SAMPLE_RATE, decode=False),
            )
            for split_name, split_data in dataset.items()
        }
    )
    return casted


def resolve_device() -> torch.device:
    """Select CUDA when available, otherwise CPU.

    Returns:
        PyTorch device for model execution.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        return device

    logger.info("CUDA not available; using CPU.")
    return torch.device("cpu")


def load_whisper_model_and_processor(
    model_name_or_path: str,
    language: str = WHISPER_LANGUAGE,
) -> tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    """Load a Whisper processor and model from Hugging Face Hub or a local path.

    Args:
        model_name_or_path: Hugging Face model id or local directory.
        language: Language code for forced decoder ids during generation.

    Returns:
        Tuple of ``(processor, model)``.
    """
    logger.info("Loading Whisper model from '%s'", model_name_or_path)
    processor = WhisperProcessor.from_pretrained(model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    return processor, model


def resample_audio_array(
    audio_array: np.ndarray,
    sampling_rate: int,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Resample an audio array to the target sample rate.

    Args:
        audio_array: One-dimensional audio samples.
        sampling_rate: Current sample rate in Hz.
        target_rate: Desired sample rate in Hz.

    Returns:
        Resampled mono float32 audio array.
    """
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=0)

    if sampling_rate == target_rate:
        return audio_array.astype(np.float32, copy=False)

    return librosa.resample(
        audio_array.astype(np.float32),
        orig_sr=sampling_rate,
        target_sr=target_rate,
    )


def prepare_dataset_batch(
    batch: dict[str, Any],
    processor: WhisperProcessor,
    audio_column: str = DATASET_AUDIO_COLUMN,
    transcription_column: str = DATASET_TRANSCRIPTION_COLUMN,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> dict[str, Any]:
    """Preprocess a batch: resample audio to 16 kHz and tokenize transcriptions.

    Args:
        batch: Raw dataset batch from Hugging Face ``datasets``.
        processor: Whisper processor for feature extraction and tokenization.
        audio_column: Name of the audio field in the batch.
        transcription_column: Name of the transcription field in the batch.
        target_rate: Target sample rate for Whisper input features.

    Returns:
        Batch with ``input_features`` and ``labels`` ready for training.
    """
    audio_inputs = batch[audio_column]
    is_batched = isinstance(audio_inputs, list)
    if not is_batched:
        audio_inputs = [audio_inputs]

    transcription_inputs = batch[transcription_column]
    if not is_batched:
        transcription_inputs = [transcription_inputs]
    elif not isinstance(transcription_inputs, list):
        transcription_inputs = [transcription_inputs]

    input_features: list[np.ndarray] = []
    for audio in audio_inputs:
        array, sampling_rate = decode_audio_value(audio)
        resampled = resample_audio_array(array, sampling_rate, target_rate)
        features = processor.feature_extractor(
            resampled,
            sampling_rate=target_rate,
        ).input_features[0]
        input_features.append(features)

    labels = processor.tokenizer(
        transcription_inputs,
        truncation=True,
        max_length=processor.tokenizer.model_max_length,
    ).input_ids

    if not is_batched:
        return {
            "input_features": input_features[0],
            "labels": labels[0] if labels and isinstance(labels[0], list) else labels,
        }

    return {
        "input_features": input_features,
        "labels": labels,
    }


def load_isixhosa_dataset(
    dataset_name: str = HF_DATASET_NAME,
    validation_ratio: float = VALIDATION_SPLIT_RATIO,
    test_ratio: float = TEST_SPLIT_RATIO,
    seed: int = RANDOM_SEED,
) -> DatasetDict:
    """Load the isiXhosa ASR dataset and ensure train/validation/test splits exist.

    Args:
        dataset_name: Hugging Face dataset identifier.
        validation_ratio: Fraction held out for validation when splitting.
        test_ratio: Fraction held out for test when splitting.
        seed: Random seed for reproducible splits.

    Returns:
        Dataset dictionary with ``train``, ``validation``, and ``test`` splits.
    """
    logger.info("Loading Hugging Face dataset '%s'", dataset_name)
    dataset = load_dataset(dataset_name)

    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})

    dataset = cast_dataset_audio_for_local_decoding(dataset)

    if "validation" not in dataset and "test" not in dataset:
        logger.info(
            "No validation/test split found; creating %.0f%% validation and "
            "%.0f%% test splits (seed=%d).",
            validation_ratio * 100,
            test_ratio * 100,
            seed,
        )
        first_split = dataset["train"].train_test_split(
            test_size=validation_ratio + test_ratio,
            seed=seed,
        )
        relative_test_size = test_ratio / (validation_ratio + test_ratio)
        second_split = first_split["test"].train_test_split(
            test_size=relative_test_size,
            seed=seed,
        )
        dataset = DatasetDict(
            {
                "train": first_split["train"],
                "validation": second_split["train"],
                "test": second_split["test"],
            }
        )
    elif "validation" not in dataset:
        logger.info("Creating validation split from train (ratio=%.2f).", validation_ratio)
        split = dataset["train"].train_test_split(test_size=validation_ratio, seed=seed)
        dataset = DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
                "test": dataset["test"],
            }
        )
    elif "test" not in dataset:
        logger.info("Creating test split from validation (ratio=%.2f).", test_ratio)
        split = dataset["validation"].train_test_split(test_size=test_ratio, seed=seed)
        dataset = DatasetDict(
            {
                "train": dataset["train"],
                "validation": split["train"],
                "test": split["test"],
            }
        )

    for split_name, split_data in dataset.items():
        logger.info("Split '%s': %d samples", split_name, len(split_data))

    return dataset


def preprocess_dataset_splits(
    dataset: DatasetDict,
    processor: WhisperProcessor,
    num_proc: int | None = None,
) -> DatasetDict:
    """Apply Whisper preprocessing to all dataset splits.

    Args:
        dataset: Raw dataset splits with audio and transcription columns.
        processor: Whisper processor used for feature extraction and tokenization.
        num_proc: Optional number of processes for ``Dataset.map``.

    Returns:
        Dataset with ``input_features`` and ``labels`` columns.
    """
    import os

    if num_proc is None:
        num_proc = 1 if os.name == "nt" else None

    audio_column = DATASET_AUDIO_COLUMN
    transcription_column = DATASET_TRANSCRIPTION_COLUMN

    remove_columns = [
        column
        for column in dataset["train"].column_names
        if column not in {audio_column, transcription_column}
    ]

    def _prepare_batch(batch: dict[str, Any]) -> dict[str, Any]:
        return prepare_dataset_batch(
            batch,
            processor=processor,
            audio_column=audio_column,
            transcription_column=transcription_column,
        )

    processed = dataset.map(
        _prepare_batch,
        remove_columns=remove_columns,
        num_proc=num_proc,
        desc="Preprocessing dataset",
        load_from_cache_file=True,
    )
    return processed


def build_compute_metrics(processor: WhisperProcessor):
    """Create a WER metric function for ``Seq2SeqTrainer``.

    Args:
        processor: Whisper processor used to decode generated token ids.

    Returns:
        Callable compatible with ``Seq2SeqTrainer.compute_metrics``.
    """
    import jiwer

    def compute_metrics(pred) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        decoded_preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        wer = jiwer.wer(decoded_labels, decoded_preds)
        return {"wer": wer}

    return compute_metrics


def compute_wer(
    predictions: list[str],
    references: list[str],
) -> float:
    """Compute word error rate between predictions and references.

    Args:
        predictions: Model-generated transcripts.
        references: Ground-truth transcripts.

    Returns:
        Word error rate as a float between 0 and 1+.
    """
    import jiwer

    return jiwer.wer(
        [ref.strip() for ref in references],
        [pred.strip() for pred in predictions],
    )


def save_json_report(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON report to disk.

    Args:
        path: Destination file path.
        payload: Serializable dictionary to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("Saved report to %s", path)


def resolve_model_path(
    model_path: Path | None,
    final_model_dir: Path,
    default_pretrained: str = WHISPER_MODEL_NAME,
) -> str:
    """Resolve a model path, preferring a locally trained final model.

    Args:
        model_path: Explicit model path from CLI arguments.
        final_model_dir: Directory containing the fine-tuned final model.
        default_pretrained: Fallback Hugging Face model id.

    Returns:
        Model directory path or Hugging Face model identifier.
    """
    if model_path is not None:
        resolved = model_path.expanduser().resolve()
        if resolved.is_dir() and (resolved / "config.json").exists():
            logger.info("Using model from explicit path: %s", resolved)
            return str(resolved)
        if resolved.is_dir():
            logger.warning(
                "Explicit model path '%s' does not contain config.json; "
                "falling back to pretrained '%s'.",
                resolved,
                default_pretrained,
            )
            return default_pretrained
        logger.info("Using Hugging Face model id: %s", resolved)
        return str(resolved)

    final_dir = final_model_dir.expanduser().resolve()
    if final_dir.is_dir() and (final_dir / "config.json").exists():
        logger.info("Using fine-tuned model from %s", final_dir)
        return str(final_dir)

    logger.info(
        "No fine-tuned model found at %s; using pretrained '%s'.",
        final_dir,
        default_pretrained,
    )
    return default_pretrained
