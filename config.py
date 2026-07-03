"""Project configuration for speech-to-text-isixhosa."""

from pathlib import Path

# Project root directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Data directories
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
TRAIN_DATA_DIR: Path = DATA_DIR / "train"
VALIDATION_DATA_DIR: Path = DATA_DIR / "validation"
TEST_DATA_DIR: Path = DATA_DIR / "test"

# Output directories
MODELS_DIR: Path = PROJECT_ROOT / "models"
CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"
FINAL_MODEL_DIR: Path = MODELS_DIR / "final"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Audio processing settings
TARGET_SAMPLE_RATE: int = 16000
TARGET_CHANNELS: int = 1  # mono

# Model and dataset settings
WHISPER_MODEL_NAME: str = "openai/whisper-small"
HF_DATASET_NAME: str = "zionia/isixhosa-asr"
WHISPER_LANGUAGE: str = "xh"
DATASET_AUDIO_COLUMN: str = "audio"
DATASET_TRANSCRIPTION_COLUMN: str = "transcription"

# Dataset split settings (used when the HF dataset has no validation/test split)
VALIDATION_SPLIT_RATIO: float = 0.1
TEST_SPLIT_RATIO: float = 0.1
RANDOM_SEED: int = 42

# Training hyperparameters
TRAINING_OUTPUT_DIR: Path = CHECKPOINTS_DIR
TRAINING_NUM_EPOCHS: int = 3
TRAINING_BATCH_SIZE: int = 4
TRAINING_GRADIENT_ACCUMULATION_STEPS: int = 2
TRAINING_LEARNING_RATE: float = 1e-5
TRAINING_WARMUP_STEPS: int = 50
TRAINING_LOGGING_STEPS: int = 10
TRAINING_EVAL_STEPS: int = 100
TRAINING_SAVE_STEPS: int = 100
TRAINING_SAVE_TOTAL_LIMIT: int = 3
TRAINING_GENERATION_MAX_LENGTH: int = 225
TRAINING_PREDICT_WITH_GENERATE: bool = True

# Logging settings
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL: str = "INFO"
