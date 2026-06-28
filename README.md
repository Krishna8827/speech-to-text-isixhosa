# speech-to-text-isixhosa

Automatic speech recognition (ASR) pipeline for isiXhosa speech-to-text.

## Requirements

- Python 3.11
- [FFmpeg](https://ffmpeg.org/download.html) (required by yt-dlp and librosa for audio decoding)

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Project Structure

```
speech-to-text-isixhosa/
├── config.py              # Shared paths and settings
├── data/
│   ├── raw/               # Downloaded source audio
│   ├── processed/         # 16 kHz mono WAV files
│   ├── train/
│   ├── validation/
│   └── test/
├── scripts/
│   ├── download_audio.py  # Download YouTube audio
│   ├── preprocess_audio.py
│   ├── dataset_builder.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── models/                # Saved model checkpoints
├── results/               # Evaluation outputs
└── logs/                  # Log files
```

## Usage

### Download audio from YouTube

```bash
python scripts/download_audio.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Optional flags:

- `--output-dir` — override output directory (default: `data/raw`)
- `--log-level` — logging level (default: `INFO`)

### Preprocess audio

Convert all files in `data/raw` to 16 kHz mono WAV:

```bash
python scripts/preprocess_audio.py
```

Process a single file:

```bash
python scripts/preprocess_audio.py --input path/to/audio.mp3
```

Optional flags:

- `--input-dir` — source directory (default: `data/raw`)
- `--output-dir` — destination directory (default: `data/processed`)
- `--log-level` — logging level (default: `INFO`)

## Pipeline (planned)

1. **Download** — fetch isiXhosa audio from YouTube (`download_audio.py`)
2. **Preprocess** — normalize to 16 kHz mono WAV (`preprocess_audio.py`)
3. **Build dataset** — create train/validation/test splits (`dataset_builder.py`)
4. **Train** — fine-tune an ASR model (`train.py`)
5. **Evaluate** — measure WER/CER on held-out data (`evaluate.py`)
6. **Inference** — transcribe new audio (`inference.py`)

## License

TBD
