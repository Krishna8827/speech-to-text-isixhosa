"""Run inference on new audio with OpenAI Whisper via Hugging Face."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DEFAULT_LOG_LEVEL,
    FINAL_MODEL_DIR,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOGS_DIR,
    TARGET_SAMPLE_RATE,
    WHISPER_MODEL_NAME,
)

logger = logging.getLogger(__name__)

DEFAULT_WHISPER_MODEL: str = WHISPER_MODEL_NAME
DEFAULT_LANGUAGE: str = "xh"
SUPPORTED_AUDIO_EXTENSION: str = ".wav"

# Preferred Whisper language identifiers for isiXhosa (checked in order).
PREFERRED_ISIXHOSA_CODES: tuple[str, ...] = ("xh", "xho")
ISIXHOSA_LANGUAGE_NAMES: tuple[str, ...] = ("xhosa", "isixhosa")
AUTO_LANGUAGE_ALIASES: frozenset[str] = frozenset({"auto", "automatic", "detect"})


def setup_logging(log_level: str) -> None:
    """Configure logging for the inference script.

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
            logging.FileHandler(LOGS_DIR / "inference.log", encoding="utf-8"),
        ],
    )


def resolve_device() -> torch.device:
    """Select the best available PyTorch device for inference.

    Returns:
        ``cuda`` when a GPU is available, otherwise ``cpu``.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        return device

    logger.info("CUDA not available; using CPU.")
    return torch.device("cpu")


def validate_audio_path(audio_path: Path) -> Path:
    """Validate that the input path refers to an existing WAV file.

    Args:
        audio_path: User-supplied path to an audio file.

    Returns:
        Resolved absolute path to the WAV file.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a file or not a WAV file.
    """
    resolved = audio_path.expanduser().resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved}")

    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {resolved}")

    if resolved.suffix.lower() != SUPPORTED_AUDIO_EXTENSION:
        raise ValueError(
            f"Expected a WAV file (.wav), got '{resolved.suffix}' for: {resolved}"
        )

    return resolved


def get_whisper_language_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Load Whisper language tables from the installed Transformers library.

    Returns:
        Tuple of ``(LANGUAGES, TO_LANGUAGE_CODE)`` from the Whisper tokenizer.
    """
    from transformers.models.whisper.tokenization_whisper import (  # noqa: PLC0415
        LANGUAGES,
        TO_LANGUAGE_CODE,
    )

    return LANGUAGES, TO_LANGUAGE_CODE


def get_supported_whisper_languages() -> frozenset[str]:
    """Return all language names and codes accepted by Whisper.

    Returns:
        Frozen set of lowercase language codes and full language names.
    """
    languages, to_language_code = get_whisper_language_maps()
    supported = (
        set(languages.keys())
        | set(languages.values())
        | set(to_language_code.keys())
        | set(to_language_code.values())
    )
    return frozenset(lang.lower() for lang in supported)


def find_isixhosa_language_code(processor: WhisperProcessor) -> str | None:
    """Detect whether the loaded Whisper model supports isiXhosa.

    Checks preferred ISO codes and known language names, then verifies the
    candidate works with ``get_decoder_prompt_ids``.

    Args:
        processor: Loaded Whisper processor.

    Returns:
        A Whisper-compatible isiXhosa language identifier, or ``None``.
    """
    languages, to_language_code = get_whisper_language_maps()
    candidates: list[str] = []

    for code in PREFERRED_ISIXHOSA_CODES:
        if code in languages or code in to_language_code or code in to_language_code.values():
            candidates.append(code)

    for code, name in languages.items():
        name_lower = name.lower()
        if any(term in name_lower for term in ISIXHOSA_LANGUAGE_NAMES):
            candidates.extend([code, name])

    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)

        try:
            processor.get_decoder_prompt_ids(language=candidate, task="transcribe")
        except ValueError:
            continue

        return candidate

    return None


def model_supports_isixhosa(processor: WhisperProcessor) -> bool:
    """Return True when the loaded Whisper model supports isiXhosa.

    Args:
        processor: Loaded Whisper processor.

    Returns:
        ``True`` if a valid isiXhosa language code is available.
    """
    return find_isixhosa_language_code(processor) is not None


def resolve_transcription_language(
    processor: WhisperProcessor,
    requested_language: str,
) -> str | None:
    """Resolve the Whisper language setting for transcription.

    If isiXhosa is requested but unsupported, returns ``None`` so Whisper
    performs automatic language detection instead of raising an error.

    Args:
        processor: Loaded Whisper processor.
        requested_language: User-requested language code or ``auto``.

    Returns:
        Whisper language identifier, or ``None`` for automatic detection.
    """
    normalized = requested_language.strip().lower()

    if normalized in AUTO_LANGUAGE_ALIASES:
        logger.info("Using automatic language detection.")
        return None

    is_isixhosa_request = (
        normalized in PREFERRED_ISIXHOSA_CODES
        or normalized in ISIXHOSA_LANGUAGE_NAMES
        or normalized == DEFAULT_LANGUAGE
    )

    if is_isixhosa_request:
        isixhosa_code = find_isixhosa_language_code(processor)
        if isixhosa_code is not None:
            logger.info(
                "isiXhosa is supported by this model; using language code '%s'.",
                isixhosa_code,
            )
            return isixhosa_code

        logger.warning(
            "isiXhosa (requested language '%s') is not supported by the loaded "
            "Whisper model. Falling back to automatic language detection.",
            requested_language,
        )
        return None

    if normalized not in get_supported_whisper_languages():
        logger.warning(
            "Language '%s' is not supported by the loaded Whisper model. "
            "Falling back to automatic language detection.",
            requested_language,
        )
        return None

    try:
        processor.get_decoder_prompt_ids(language=requested_language, task="transcribe")
    except ValueError as exc:
        logger.warning(
            "Language '%s' could not be applied (%s). "
            "Falling back to automatic language detection.",
            requested_language,
            exc,
        )
        return None

    logger.info("Using requested language '%s'.", requested_language)
    return requested_language


def resolve_inference_model_name(model_name: str | None) -> str:
    """Resolve the model to load, preferring a locally trained final model.

    Args:
        model_name: Optional explicit model path or Hugging Face model id.

    Returns:
        Resolved model directory path or Hugging Face model identifier.
    """
    if model_name and model_name != DEFAULT_WHISPER_MODEL:
        return model_name

    final_dir = FINAL_MODEL_DIR.expanduser().resolve()
    if final_dir.is_dir() and (final_dir / "config.json").exists():
        logger.info("Using fine-tuned model from %s", final_dir)
        return str(final_dir)

    logger.info(
        "No fine-tuned model found at %s; using pretrained '%s'.",
        final_dir,
        DEFAULT_WHISPER_MODEL,
    )
    return DEFAULT_WHISPER_MODEL


def load_whisper_model(
    model_name: str,
    device: torch.device,
) -> tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    """Load a Whisper processor and model from Hugging Face Hub or a local path.

    Args:
        model_name: Hugging Face model identifier or local model directory.
        device: Target device for model weights.

    Returns:
        Tuple of ``(processor, model)`` ready for inference.
    """
    print("Model loading...")
    logger.info("Loading Whisper model '%s' on %s", model_name, device)

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()

    if model_supports_isixhosa(processor):
        isixhosa_code = find_isixhosa_language_code(processor)
        logger.info("Whisper model supports isiXhosa via '%s'.", isixhosa_code)
    else:
        logger.warning(
            "Whisper model does not list isiXhosa as a supported language; "
            "automatic language detection will be used unless another "
            "supported language is requested.",
        )

    logger.info("Model loaded successfully.")
    return processor, model


def load_audio(audio_path: Path) -> np.ndarray:
    """Load and resample audio to mono 16 kHz for Whisper.

    Args:
        audio_path: Path to the input WAV file.

    Returns:
        One-dimensional float32 audio array.

    Raises:
        ValueError: If no audio samples are loaded.
    """
    print("Audio loading...")
    logger.info("Loading audio from: %s", audio_path)

    audio, _sample_rate = librosa.load(
        audio_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError(f"No audio samples loaded from {audio_path}")

    logger.info("Loaded %d samples at %d Hz.", audio.size, TARGET_SAMPLE_RATE)
    return audio


def transcribe(
    audio_path: Path,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """Transcribe a single WAV file with Whisper.

    Args:
        audio_path: Path to the input WAV file.
        model_name: Hugging Face Whisper model identifier.
        language: ISO language code passed to Whisper (default: isiXhosa).

    Returns:
        Predicted transcript text.
    """
    validated_path = validate_audio_path(audio_path)
    device = resolve_device()
    resolved_model_name = resolve_inference_model_name(model_name)

    processor, model = load_whisper_model(resolved_model_name, device)
    audio = load_audio(validated_path)

    resolved_language = resolve_transcription_language(processor, language)

    print("Transcribing...")
    if resolved_language is None:
        logger.info("Running Whisper inference with automatic language detection.")
    else:
        logger.info("Running Whisper inference (language=%s).", resolved_language)

    inputs = processor(
        audio,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        task="transcribe",
        language=resolved_language,
    )

    with torch.inference_mode():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
        )

    transcript = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0].strip()

    print(f"Final transcript: {transcript}")
    logger.info("Transcription complete.")
    return transcript


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe a WAV file using openai/whisper-small.",
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to the WAV file to transcribe.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_WHISPER_MODEL,
        help=f"Hugging Face model id (default: {DEFAULT_WHISPER_MODEL}).",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=(
            "Whisper language code (default: xh). Use 'auto' for automatic "
            "language detection when isiXhosa is unsupported."
        ),
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

    try:
        transcribe(
            audio_path=args.audio_path,
            model_name=args.model_name,
            language=args.language,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from None
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from None
    except Exception:
        logger.exception("Inference failed for %s", args.audio_path)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
