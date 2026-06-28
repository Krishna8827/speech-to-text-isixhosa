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
RESULTS_DIR: Path = PROJECT_ROOT / "results"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Audio processing settings
TARGET_SAMPLE_RATE: int = 16000
TARGET_CHANNELS: int = 1  # mono

# Logging settings
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL: str = "INFO"
