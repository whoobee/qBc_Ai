# qBc_Ai

Unified AI service stack for the qB-Companion robot. This module manages the shared Vision-Language Model (VLM) server and hosts modular AI feature handlers (Voice Assistant, Visual Exploration).

![Python](https://img.shields.io/badge/Python-3.13-blue) ![AI](https://img.shields.io/badge/AI-Qwen3.5--4B-purple) ![Whisper](https://img.shields.io/badge/STT-Faster%20Whisper-orange) ![Piper](https://img.shields.io/badge/TTS-Piper-yellow)

## Overview

The `qBc_Ai` service integrates multiple AI capabilities into a single MQTT-controllable node:
- **Speech-to-Text (STT)**: Uses `faster-whisper` for offline voice transcription.
- **Large Language Model (LLM/VLM)**: Manages an `axllm` server running a local model (default: Qwen3.5-4B), potentially accelerated via an M5Stack LLM8850 module.
- **Text-to-Speech (TTS)**: Uses `piper` to generate high-quality voices locally.

## Architecture

The main service (`main.py`) acts as the infrastructure layer:
1. Validates and loads STT (`Whisper`) and TTS (`Piper`) models.
2. Spawns and manages the `axllm serve` subprocess for the LLM.
3. Dynamically loads and runs AI handlers.

### Feature Handlers

- **`voice_handler.py`**: Implements a Voice Assistant workflow: Wake Word/Audio Capture (from `qBc_Audio`) → Whisper STT → LLM generation → Piper TTS → Audio playback.
- **`exploration_handler.py`**: Visual Exploration workflow: Captures camera frames (via `qBc_Vision`) → analyzes them using the VLM → generates spoken descriptions.

## Prerequisites

### Dependencies
```bash
cd qBc_Ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Requires: faster-whisper, openai, paho-mqtt, Pillow
```

### External Tools
- **axllm**: Must be installed and available in the system `PATH`.
- **Piper TTS**: Must be installed (e.g., `pip install piper-tts`).
- **Piper Voice Model**: You need an ONNX voice model.
  ```bash
  mkdir -p piper_models && cd piper_models
  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
  ```
- **Qwen3.5-4B Model**: The default target model for `axllm`.

## Usage

```bash
python3 main.py --piper-model piper_models/en_US-lessac-medium.onnx
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--api-url` | `http://127.0.0.1:8000/v1` | LLM API Base URL (`axllm` server) |
| `--model` | `AXERA-TECH/Qwen3.5-4B...` | Model identifier |
| `--model-dir` | `Qwen3.5-4B` | Path to the local `axllm` model weights |
| `--whisper-model`| `tiny` | Whisper model size (`tiny`, `base`, `small`) |
| `--whisper-device`| `cpu` | Execution device for Whisper |

## MQTT Interface

The service exposes its state globally.

### Published Topics
- `robot/ai/state`: Retained service state (`{"status": "online"}`).
- `robot/ai/current_state`: Human-readable state indicating the active handler (e.g., `loading_llm`, `ready`, `voice_handler:listening`, `exploration_handler:analyzing`).
- `robot/ai/error_info`: Any encountered errors.
- `robot/system/heartbeat/ai`: 1 Hz keepalive.
