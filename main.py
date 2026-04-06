#!/usr/bin/env python3
"""
qBc_Ai Voice Service

Voice assistant pipeline using MQTT:
    1. Receives voice recordings from qBc_Audio (wake word triggered)
    2. Transcribes speech using faster-whisper
    3. Generates response via LLM (OpenAI-compatible API / axllm)
    4. Synthesizes speech using Piper TTS
    5. Sends audio to qBc_Audio for playback

MQTT topics:
    Subscribe:
        robot/audio/recording_ready   Voice recording path from qBc_Audio

    Publish:
        robot/audio/play              Play TTS audio via qBc_Audio
        robot/ai/voice/state          (RETAIN) Service state
        robot/system/heartbeat/ai_voice   Keepalive (1 Hz)

Prerequisites:
    1. Install faster-whisper:
       pip install faster-whisper

    2. Install Piper TTS (aarch64 binary or pip):
       pip install piper-tts
       # -or- download from https://github.com/rhasspy/piper/releases

    3. Download a Piper voice model into piper_models/:
       mkdir -p piper_models && cd piper_models
       wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
       wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

    4. axllm serve is auto-managed (started if not already running).
       Override model dir with --model-dir if needed.

Usage:
    python3 voice_service.py [--piper-model piper_models/en_US-lessac-medium.onnx]
"""

import argparse
import atexit
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import paho.mqtt.client as mqtt
from faster_whisper import WhisperModel
from openai import OpenAI

logger = logging.getLogger("qBc_Ai_Voice")

SCRIPT_DIR = Path(__file__).parent
PLAYBACK_DIR = SCRIPT_DIR.parent / "qBc_Audio" / "resources" / "playback"
DEFAULT_PIPER_MODEL = str(SCRIPT_DIR / "piper_models" / "en_US-lessac-medium.onnx")
DEFAULT_MODEL_DIR = str(SCRIPT_DIR / "Qwen3.5-4B")

# MQTT topics
TOPIC_RECORDING_READY = "robot/audio/recording_ready"
TOPIC_PLAY = "robot/audio/play"
TOPIC_STATE = "robot/ai/voice/state"
TOPIC_HEARTBEAT = "robot/system/heartbeat/ai_voice"

# LLM defaults
DEFAULT_API_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "AXERA-TECH/Qwen3.5-4B-AX650-GPTQ-Int4-C128-P1152-CTX2047"

SYSTEM_PROMPT = (
    "You are qB, a friendly and helpful robot companion. "
    "Give short, natural, conversational responses. "
    "Keep answers to 1-3 sentences unless more detail is needed. "
    "Do NOT use <think> tags or output internal reasoning."
)

# Playback volume (0-100%)
PLAYBACK_VOLUME = 20



class VoiceService:
    _PROGRESS_RE = re.compile(r"(\d+)%\s*\|")

    def __init__(
        self,
        broker="localhost",
        port=1883,
        api_url=DEFAULT_API_URL,
        model_name=DEFAULT_MODEL,
        model_dir=DEFAULT_MODEL_DIR,
        whisper_model_size="tiny",
        whisper_device="cpu",
        whisper_compute_type="int8",
        piper_model_path=DEFAULT_PIPER_MODEL,
        piper_binary="piper",
    ):
        self.broker = broker
        self.port = port
        self.api_url = api_url
        self.model_name = model_name
        self.model_dir = model_dir
        # Resolve piper model path (relative paths resolved against script dir)
        piper_path = Path(piper_model_path)
        if not piper_path.is_absolute():
            piper_path = SCRIPT_DIR / piper_path
        self.piper_model_path = str(piper_path.resolve())
        self.piper_binary = piper_binary

        self._processing = False
        self._lock = threading.Lock()
        self._running = False
        self._axllm_proc = None

        # ── Load all models at startup ──

        # 1. LLM server (axllm)
        if not self._ensure_server():
            raise RuntimeError("LLM server unavailable — cannot start voice service")

        self._llm = OpenAI(api_key="not-needed", base_url=api_url)

        # 2. Whisper STT model
        logger.info("Loading Whisper model: %s (device=%s, compute=%s)",
                     whisper_model_size, whisper_device, whisper_compute_type)
        self._whisper = WhisperModel(
            whisper_model_size,
            device=whisper_device,
            compute_type=whisper_compute_type,
        )
        logger.info("Whisper model loaded")

        # 3. Piper TTS model — validate files exist
        if not os.path.isfile(self.piper_model_path):
            raise FileNotFoundError(
                f"Piper voice model not found: {self.piper_model_path}\n"
                "Download with:\n"
                "  cd qBc_Ai/piper_models\n"
                "  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                "en/en_US/lessac/medium/en_US-lessac-medium.onnx\n"
                "  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
            )
        piper_config = self.piper_model_path + ".json"
        if not os.path.isfile(piper_config):
            raise FileNotFoundError(f"Piper config not found: {piper_config}")
        logger.info("Piper TTS model verified: %s", self.piper_model_path)

        logger.info("All models ready")

        # MQTT client
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="qbc_ai_voice",
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.will_set(
            TOPIC_STATE,
            json.dumps({"status": "offline"}),
            qos=1, retain=True,
        )

    # ------------------------------------------------------------------
    # axllm server management
    # ------------------------------------------------------------------

    def _ensure_server(self):
        """Start axllm serve if not already reachable."""
        try:
            url = self.api_url.rstrip("/") + "/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2):
                logger.info("LLM server already running at %s", self.api_url)
                return True
        except Exception:
            pass

        axllm = shutil.which("axllm")
        if not axllm:
            logger.error("axllm not found in PATH — install it first")
            return False
        if not os.path.isdir(self.model_dir):
            logger.error("Model directory not found: %s", self.model_dir)
            return False

        logger.info("Starting axllm serve %s ...", self.model_dir)
        self._axllm_proc = subprocess.Popen(
            [axllm, "serve", self.model_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        atexit.register(self._stop_server)

        server_ready = threading.Event()

        def _reader():
            for raw in self._axllm_proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                if server_ready.is_set():
                    # Server running — just drain stdout silently
                    continue
                m = self._PROGRESS_RE.search(line)
                if m:
                    logger.info("Loading model: %s%%", m.group(1))
                if "starting" in line.lower() and "server" in line.lower():
                    logger.info("LLM server ready")
                    server_ready.set()

        threading.Thread(target=_reader, daemon=True).start()

        # Poll HTTP endpoint as fallback (up to 180s)
        for _ in range(360):
            if server_ready.is_set():
                return True
            time.sleep(0.5)
            try:
                url = self.api_url.rstrip("/") + "/models"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=2):
                    logger.info("LLM server ready (HTTP check)")
                    return True
            except Exception:
                if self._axllm_proc.poll() is not None:
                    logger.error("axllm exited with code %d",
                                 self._axllm_proc.returncode)
                    self._axllm_proc = None
                    return False

        logger.error("LLM server did not start in time")
        return False

    def _stop_server(self):
        """Terminate the axllm serve subprocess if we started it."""
        if self._axllm_proc and self._axllm_proc.poll() is None:
            logger.info("Stopping axllm serve...")
            self._axllm_proc.terminate()
            try:
                self._axllm_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._axllm_proc.kill()
            self._axllm_proc = None

    # ------------------------------------------------------------------
    # MQTT callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, connect_flags, reason_code, properties):
        if reason_code.is_failure:
            logger.error("MQTT connection failed: %s", reason_code)
            return
        logger.info("Connected to MQTT broker %s:%d", self.broker, self.port)
        client.subscribe(TOPIC_RECORDING_READY, qos=1)
        self._publish_state()

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        if reason_code.is_failure:
            logger.warning("Disconnected from MQTT broker: %s", reason_code)

    def _on_message(self, client, userdata, msg):
        if msg.topic != TOPIC_RECORDING_READY:
            return
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Invalid JSON on %s", msg.topic)
            return

        file_path = data.get("file")
        if not file_path:
            logger.warning("No file in recording_ready message")
            return

        # Process in a separate thread to not block MQTT
        threading.Thread(
            target=self._process_voice,
            args=(file_path,),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _publish_state(self):
        with self._lock:
            processing = self._processing
        state = {"status": "online", "processing": processing}
        self._client.publish(TOPIC_STATE, json.dumps(state), qos=1, retain=True)

    # ------------------------------------------------------------------
    # Voice pipeline
    # ------------------------------------------------------------------

    def _process_voice(self, recording_path):
        with self._lock:
            if self._processing:
                logger.warning("Already processing, skipping: %s", recording_path)
                return
            self._processing = True
        self._publish_state()

        try:
            # 1. Transcribe with Whisper
            logger.info("Transcribing: %s", recording_path)
            text = self._transcribe(recording_path)
            if not text or not text.strip():
                logger.info("Empty transcription, skipping")
                return

            logger.info("Transcription: %s", text)

            # 2. Query LLM
            logger.info("Querying LLM...")
            response = self._query_llm(text)
            if not response or not response.strip():
                logger.warning("Empty LLM response")
                return

            logger.info("LLM response: %s", response)

            # 3. Synthesize speech with Piper
            logger.info("Synthesizing speech...")
            audio_filename = self._synthesize(response)
            if not audio_filename:
                logger.error("TTS synthesis failed")
                return

            logger.info("TTS output: %s", audio_filename)

            # 4. Send playback command to qBc_Audio
            self._client.publish(
                TOPIC_PLAY,
                json.dumps({
                    "file": audio_filename,
                    "volume": PLAYBACK_VOLUME,
                    "voice": True,
                }),
                qos=1,
            )
            logger.info("Playback command sent")

        except Exception as e:
            logger.error("Voice pipeline error: %s", e, exc_info=True)
        finally:
            with self._lock:
                self._processing = False
            self._publish_state()

    def _transcribe(self, audio_path):
        """Transcribe audio file using faster-whisper."""
        if not os.path.isfile(audio_path):
            logger.error("Recording not found: %s", audio_path)
            return ""

        segments, info = self._whisper.transcribe(
            audio_path,
            language="en",
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments)
        return text.strip()

    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

    def _query_llm(self, user_text):
        """Send transcribed text to the LLM and return the response."""
        response = self._llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            max_tokens=256,
        )
        text = response.choices[0].message.content
        # Strip <think>...</think> tags that some models emit
        text = self._THINK_RE.sub("", text).strip()
        return text

    def _synthesize(self, text):
        """Synthesize speech using Piper TTS, save to playback directory.

        Returns the filename (not full path) for the audio service to resolve.
        """
        PLAYBACK_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"response_{timestamp}.wav"
        output_path = PLAYBACK_DIR / filename

        cmd = [
            self.piper_binary,
            "--model", self.piper_model_path,
            "--output_file", str(output_path),
        ]

        try:
            proc = subprocess.run(
                cmd,
                input=text,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode != 0:
                logger.error("Piper error: %s", proc.stderr)
                return None
            if not output_path.exists():
                logger.error("Piper did not create output file")
                return None
            return filename
        except subprocess.TimeoutExpired:
            logger.error("Piper TTS timed out")
            return None
        except FileNotFoundError:
            logger.error("Piper binary not found: %s (install with: pip install piper-tts)",
                         self.piper_binary)
            return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        self._running = True

        self._client.connect(self.broker, self.port)
        self._client.loop_start()

        logger.info("qBc_Ai Voice service on MQTT %s:%d", self.broker, self.port)

        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())

        while not stop.is_set():
            self._client.publish(TOPIC_HEARTBEAT, b"1", qos=0)
            stop.wait(1.0)

        logger.info("Shutting down...")
        self._running = False
        self._client.publish(
            TOPIC_STATE,
            json.dumps({"status": "offline"}),
            qos=1, retain=True,
        )
        self._client.loop_stop()
        self._client.disconnect()
        self._stop_server()


def main():
    parser = argparse.ArgumentParser(description="qBc_Ai Voice Service")
    parser.add_argument("--mqtt-broker", default="localhost",
                        help="MQTT broker address")
    parser.add_argument("--mqtt-port", type=int, default=1883,
                        help="MQTT broker port")
    parser.add_argument("--api-url", default=DEFAULT_API_URL,
                        help="LLM API base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="LLM model name")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="Path to axllm model directory")
    parser.add_argument("--whisper-model", default="tiny",
                        help="Whisper model size (tiny, base, small, medium)")
    parser.add_argument("--whisper-device", default="cpu",
                        help="Whisper device (cpu, cuda)")
    parser.add_argument("--whisper-compute", default="int8",
                        help="Whisper compute type (int8, float16, float32)")
    parser.add_argument("--piper-model", default=DEFAULT_PIPER_MODEL,
                        help="Path to Piper voice model (.onnx)")
    parser.add_argument("--piper-binary", default="piper",
                        help="Path to piper binary")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    service = VoiceService(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        api_url=args.api_url,
        model_name=args.model,
        model_dir=args.model_dir,
        whisper_model_size=args.whisper_model,
        whisper_device=args.whisper_device,
        whisper_compute_type=args.whisper_compute,
        piper_model_path=args.piper_model,
        piper_binary=args.piper_binary,
    )
    service.run()


if __name__ == "__main__":
    main()
