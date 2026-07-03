# speech-to-text-isixhosa

Automatic speech recognition (ASR) pipeline for isiXhosa speech-to-text, built around OpenAI Whisper fine-tuning on the [zionia/isixhosa-asr](https://huggingface.co/datasets/zionia/isixhosa-asr) dataset.

## Requirements

- Python 3.11
- [FFmpeg](https://ffmpeg.org/download.html) (required by yt-dlp and librosa for audio decoding)
- Optional: NVIDIA GPU with CUDA for faster training and inference

On Windows, audio from Hugging Face datasets is decoded with `soundfile` (via `scripts/training_utils.py`) so training works without the optional `torchcodec` backend.

## Installation

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
├── config.py                 # Shared paths, model, and training settings
├── data/
│   ├── raw/                  # Downloaded source audio
│   ├── processed/            # 16 kHz mono WAV files
│   ├── train/                # Exported train manifests (optional)
│   ├── validation/           # Exported validation manifests (optional)
│   └── test/                 # Exported test manifests (optional)
├── scripts/
│   ├── download_audio.py     # Download YouTube audio
│   ├── preprocess_audio.py   # Normalize audio to 16 kHz mono WAV
│   ├── dataset_builder.py    # Export HF dataset splits and manifests
│   ├── train.py              # Fine-tune Whisper on isiXhosa ASR data
│   ├── evaluate.py           # Compute WER on validation/test splits
│   ├── inference.py          # Transcribe new WAV files
│   └── training_utils.py     # Shared training/evaluation utilities
├── models/
│   ├── checkpoints/          # Training checkpoints (created during training)
│   └── final/                # Exported fine-tuned model
├── results/                  # Evaluation and training summaries
└── logs/                     # Log files and TensorBoard events
```

## Training

Fine-tune `openai/whisper-small` on the Hugging Face dataset `zionia/isixhosa-asr`:

```bash
python scripts/train.py
```

The script will:

- Load the dataset (`audio`, `transcription` fields)
- Create train/validation/test splits when they are not provided
- Resample audio to 16 kHz and tokenize transcriptions
- Train with `Seq2SeqTrainer` and compute WER during evaluation
- Save checkpoints to `models/checkpoints/`
- Save the final model to `models/final/`
- Write logs to `logs/train.log` and metrics to `results/training_summary.json`

CUDA is used automatically when available.

Common options:

```bash
python scripts/train.py --num-train-epochs 5 --batch-size 4 --learning-rate 1e-5
python scripts/train.py --dataset-name zionia/isixhosa-asr --model-name openai/whisper-small
```

## Evaluation

Evaluate the fine-tuned model (or fall back to the base Whisper model):

```bash
python scripts/evaluate.py
python scripts/evaluate.py --split validation
python scripts/evaluate.py --model-path models/final
```

Reports are written to `results/evaluation_test.json` and `results/evaluation_test.md`.

## Inference

Transcribe a WAV file. When `models/final/` exists, the fine-tuned model is used automatically; otherwise the script falls back to `openai/whisper-small`.

```bash
python scripts/inference.py path/to/audio.wav
python scripts/inference.py path/to/audio.wav --model-name models/final
python scripts/inference.py path/to/audio.wav --language auto
```

## Data Pipeline

### 1. Download audio from YouTube

```bash
python scripts/download_audio.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Optional flags:

- `--output-dir` — override output directory (default: `data/raw`)
- `--log-level` — logging level (default: `INFO`)

### 2. Preprocess audio

Convert all files in `data/raw` to 16 kHz mono WAV:

```bash
python scripts/preprocess_audio.py
```

Process a single file:

```bash
python scripts/preprocess_audio.py --input path/to/audio.mp3
```

### 3. Build local dataset manifests (optional)

Export Hugging Face splits to local directories for inspection or offline use:

```bash
python scripts/dataset_builder.py
python scripts/dataset_builder.py --copy-audio
```

## Full Pipeline

1. **Download** — fetch isiXhosa audio from YouTube (`download_audio.py`)
2. **Preprocess** — normalize to 16 kHz mono WAV (`preprocess_audio.py`)
3. **Build dataset** — export HF splits and manifests (`dataset_builder.py`)
4. **Train** — fine-tune Whisper (`train.py`)
5. **Evaluate** — measure WER on held-out data (`evaluate.py`)
6. **Inference** — transcribe new audio (`inference.py`)

## Configuration

Shared settings live in `config.py`, including:

- Model: `openai/whisper-small`
- Dataset: `zionia/isixhosa-asr`
- Audio sample rate: 16 kHz
- Checkpoint directory: `models/checkpoints/`
- Final model directory: `models/final/`
- Training hyperparameters (epochs, batch size, learning rate)

## License

TBD
