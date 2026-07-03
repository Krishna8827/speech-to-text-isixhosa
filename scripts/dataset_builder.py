"""Build train, validation, and test datasets from processed audio."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    DATASET_AUDIO_COLUMN,
    DATASET_TRANSCRIPTION_COLUMN,
    DEFAULT_LOG_LEVEL,
    HF_DATASET_NAME,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOGS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TARGET_SAMPLE_RATE,
    TEST_DATA_DIR,
    TRAIN_DATA_DIR,
    VALIDATION_DATA_DIR,
)
from training_utils import (  # noqa: E402
    decode_audio_value,
    load_isixhosa_dataset,
    save_json_report,
)

logger = logging.getLogger(__name__)

MANIFEST_FILENAME: str = "manifest.json"


def setup_logging(log_level: str) -> None:
    """Configure logging for the dataset builder script.

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
            logging.FileHandler(LOGS_DIR / "dataset_builder.log", encoding="utf-8"),
        ],
    )


def export_split_manifest(
    split_name: str,
    split_data,
    output_dir: Path,
    copy_audio: bool,
) -> dict[str, int | float]:
    """Export one dataset split to a local directory with a manifest file.

    Args:
        split_name: Name of the split (train, validation, or test).
        split_data: Hugging Face ``Dataset`` split.
        output_dir: Destination directory for exported artifacts.
        copy_audio: When True, write WAV files locally; otherwise store metadata only.

    Returns:
        Summary statistics for the exported split.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    if copy_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, str | float | int]] = []
    total_duration_seconds = 0.0

    for index, sample in enumerate(split_data):
        audio = sample[DATASET_AUDIO_COLUMN]
        transcription = sample[DATASET_TRANSCRIPTION_COLUMN].strip()
        file_id = str(sample.get("file_id", index))
        array, sampling_rate = decode_audio_value(audio)
        duration_seconds = len(array) / float(sampling_rate)
        total_duration_seconds += duration_seconds

        if copy_audio:
            audio_path = audio_dir / f"{file_id}.wav"
            _write_wav(audio_path, array, sampling_rate)
            relative_audio_path = str(audio_path.relative_to(output_dir))
        else:
            relative_audio_path = f"hf://{split_name}/{index}"

        entries.append(
            {
                "id": file_id,
                "audio_path": relative_audio_path,
                "transcription": transcription,
                "duration_seconds": round(duration_seconds, 3),
                "split": split_name,
            }
        )

    manifest = {
        "split": split_name,
        "dataset": HF_DATASET_NAME,
        "sample_count": len(entries),
        "total_duration_seconds": round(total_duration_seconds, 3),
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "entries": entries,
    }

    manifest_path = output_dir / MANIFEST_FILENAME
    save_json_report(manifest_path, manifest)

    return {
        "sample_count": len(entries),
        "total_duration_seconds": total_duration_seconds,
    }


def _write_wav(path: Path, audio_array, sampling_rate: int) -> None:
    """Write an audio array to a WAV file, resampling to 16 kHz mono when needed.

    Args:
        path: Output WAV path.
        audio_array: Audio samples as a NumPy array.
        sampling_rate: Current sample rate of the audio array.
    """
    import librosa
    import soundfile as sf

    audio = audio_array
    if sampling_rate != TARGET_SAMPLE_RATE:
        audio = librosa.resample(
            audio.astype("float32"),
            orig_sr=sampling_rate,
            target_sr=TARGET_SAMPLE_RATE,
        )
        sampling_rate = TARGET_SAMPLE_RATE

    sf.write(path, audio, sampling_rate, subtype="PCM_16")


def build_dataset(
    processed_dir: Path,
    train_dir: Path,
    validation_dir: Path,
    test_dir: Path,
    dataset_name: str,
    seed: int,
    copy_audio: bool,
) -> None:
    """Create train, validation, and test splits from the Hugging Face dataset.

    When ``processed_dir`` contains WAV files and sidecar ``*.json`` transcripts,
    those local samples are appended to the exported manifests after the Hugging
    Face samples are written.

    Args:
        processed_dir: Directory containing preprocessed WAV files (optional local data).
        train_dir: Output directory for training samples.
        validation_dir: Output directory for validation samples.
        test_dir: Output directory for test samples.
        dataset_name: Hugging Face dataset identifier.
        seed: Random seed for reproducible splits.
        copy_audio: Whether to copy audio files into each split directory.
    """
    dataset = load_isixhosa_dataset(dataset_name=dataset_name, seed=seed)

    split_dirs = {
        "train": train_dir,
        "validation": validation_dir,
        "test": test_dir,
    }

    summaries: dict[str, dict[str, int | float]] = {}
    for split_name, output_dir in split_dirs.items():
        logger.info("Exporting '%s' split to %s", split_name, output_dir)
        summaries[split_name] = export_split_manifest(
            split_name=split_name,
            split_data=dataset[split_name],
            output_dir=output_dir,
            copy_audio=copy_audio,
        )

    local_samples = _discover_local_samples(processed_dir)
    if local_samples:
        logger.info(
            "Found %d local processed samples; appending to train manifest.",
            len(local_samples),
        )
        _append_local_samples_to_manifest(train_dir, local_samples)

    overall = {
        "dataset": dataset_name,
        "seed": seed,
        "copy_audio": copy_audio,
        "splits": summaries,
        "local_processed_samples": len(local_samples),
    }
    save_json_report(LOGS_DIR / "dataset_builder_summary.json", overall)

    for split_name, stats in summaries.items():
        logger.info(
            "Split '%s': %d samples, %.1f seconds total",
            split_name,
            stats["sample_count"],
            stats["total_duration_seconds"],
        )


def _discover_local_samples(processed_dir: Path) -> list[dict[str, str]]:
    """Find WAV files with matching transcript JSON files under processed_dir.

    Args:
        processed_dir: Root directory to search recursively.

    Returns:
        List of local sample metadata dictionaries.
    """
    if not processed_dir.exists():
        return []

    samples: list[dict[str, str]] = []
    for wav_path in sorted(processed_dir.rglob("*.wav")):
        transcript_path = wav_path.with_suffix(".json")
        if not transcript_path.exists():
            continue

        with transcript_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        transcription = (
            payload.get("transcription")
            or payload.get("text")
            or payload.get("transcript")
        )
        if not transcription:
            continue

        samples.append(
            {
                "audio_path": str(wav_path.resolve()),
                "transcription": str(transcription).strip(),
            }
        )

    return samples


def _append_local_samples_to_manifest(
    train_dir: Path,
    local_samples: list[dict[str, str]],
) -> None:
    """Append locally processed samples to an existing train manifest.

    Args:
        train_dir: Training split output directory.
        local_samples: Local audio/transcription metadata.
    """
    manifest_path = train_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        save_json_report(
            manifest_path,
            {
                "split": "train",
                "dataset": "local/processed",
                "sample_count": 0,
                "total_duration_seconds": 0.0,
                "target_sample_rate": TARGET_SAMPLE_RATE,
                "entries": [],
            },
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    entries = manifest.setdefault("entries", [])
    start_index = len(entries)

    import librosa

    total_duration = float(manifest.get("total_duration_seconds", 0.0))
    for offset, sample in enumerate(local_samples):
        source_path = Path(sample["audio_path"])
        destination_dir = train_dir / "audio" / "local"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / source_path.name

        if not destination_path.exists():
            shutil.copy2(source_path, destination_path)

        duration_seconds = librosa.get_duration(path=source_path)
        total_duration += duration_seconds
        entries.append(
            {
                "id": f"local_{start_index + offset}",
                "audio_path": str(destination_path.relative_to(train_dir)),
                "transcription": sample["transcription"],
                "duration_seconds": round(duration_seconds, 3),
                "split": "train",
            }
        )

    manifest["sample_count"] = len(entries)
    manifest["total_duration_seconds"] = round(total_duration, 3)
    save_json_report(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build train/validation/test datasets from the Hugging Face corpus.",
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
        "--dataset-name",
        default=HF_DATASET_NAME,
        help=f"Hugging Face dataset id (default: {HF_DATASET_NAME}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for dataset splits (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--copy-audio",
        action="store_true",
        help="Copy audio files into each split directory (default: manifest only).",
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
            dataset_name=args.dataset_name,
            seed=args.seed,
            copy_audio=args.copy_audio,
        )
    except Exception:
        logger.exception("Dataset build failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
