#!/usr/bin/env python3
"""
qBc_Ai — Unified AI service for qB Companion.

Manages the shared axllm VLM server and hosts modular AI feature handlers.
Each handler lives in its own file and registers MQTT topics via the service.

Current features:
    - Voice assistant (voice_handler.py): STT → LLM → TTS
    - Visual exploration (exploration_handler.py): Camera → VLM → TTS

MQTT topics:
    Publish:
        robot/ai/state               (RETAIN) Overall service state
        robot/system/heartbeat/ai    Keepalive (1 Hz)

Prerequisites:
    pip install faster-whisper openai paho-mqtt Pillow
    axllm binary in PATH
    Piper TTS binary + voice model in piper_models/

Usage:
    python3 main.py [--piper-model piper_models/en_US-lessac-medium.onnx]
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

logger = logging.getLogger("qBc_Ai")

SCRIPT_DIR = Path(__file__).parent
PLAYBACK_DIR = SCRIPT_DIR.parent / "qBc_Audio" / "resources" / "playback"
DEFAULT_PIPER_MODEL = str(SCRIPT_DIR / "piper_models" / "en_US-lessac-medium.onnx")
DEFAULT_MODEL_DIR = str(SCRIPT_DIR / "Qwen3.5-4B")

DEFAULT_API_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "AXERA-TECH/Qwen3.5-4B-AX650-GPTQ-Int4-C128-P1152-CTX2047"

TOPIC_STATE = "robot/ai/state"
TOPIC_HEARTBEAT = "robot/system/heartbeat/ai"
TOPIC_CURRENT_STATE = "robot/ai/current_state"
TOPIC_ERROR_INFO = "robot/ai/error_info"

PLAYBACK_VOLUME = 50

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_PROGRESS_RE = re.compile(r"(\d+)%\s*\|")


class AiService:
    """Unified AI service — shared infrastructure for all AI handlers."""

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

        piper_path = Path(piper_model_path)
        if not piper_path.is_absolute():
            piper_path = SCRIPT_DIR / piper_path
        self.piper_model_path = str(piper_path.resolve())
        self.piper_binary = piper_binary

        self._running = False
        self._axllm_proc = None
        self._current_state = "starting"
        self._error_info = "E_OK"
        self._handlers = []

        # ── MQTT client (connect early for state reporting) ──
        self.mqtt = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="qbc_ai",
        )
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_disconnect = self._on_disconnect
        self.mqtt.will_set(
            TOPIC_STATE,
            json.dumps({"status": "offline"}),
            qos=1, retain=True,
        )
        self.mqtt.connect(broker, port)
        self.mqtt.loop_start()

        try:
            # ── 1. LLM server (axllm) ──
            self._publish_current_state("loading_llm")
            if not self._ensure_server():
                self._publish_error("LLM server unavailable")
                raise RuntimeError("LLM server unavailable — cannot start AI service")

            self.llm = OpenAI(api_key="not-needed", base_url=api_url)

            # ── 2. Whisper STT ──
            self._publish_current_state("loading_stt")
            logger.info(
                "Loading Whisper: %s (device=%s, compute=%s)",
                whisper_model_size, whisper_device, whisper_compute_type,
            )
            self.whisper = WhisperModel(
                whisper_model_size,
                device=whisper_device,
                compute_type=whisper_compute_type,
            )
            logger.info("Whisper model loaded")

            # ── 3. Piper TTS — validate ──
            self._publish_current_state("validating_tts")
            if not os.path.isfile(self.piper_model_path):
                self._publish_error("Piper model not found")
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
                self._publish_error("Piper config not found")
                raise FileNotFoundError(f"Piper config not found: {piper_config}")
            logger.info("Piper TTS verified: %s", self.piper_model_path)

            # ── Feature handlers ──
            from voice_handler import VoiceHandler
            from exploration_handler import ExplorationHandler

            self._handlers = [
                VoiceHandler(self),
                ExplorationHandler(self),
            ]
            for h in self._handlers:
                h.register_callbacks(self.mqtt)
            for h in self._handlers:
                h.subscribe(self.mqtt)
            for h in self._handlers:
                h.publish_state()

            self._publish_current_state("ready")
            logger.info("AI service ready — %d handler(s) loaded", len(self._handlers))

        except Exception:
            self.mqtt.loop_stop()
            self.mqtt.disconnect()
            raise

    # ------------------------------------------------------------------
    # State / error publishing
    # ------------------------------------------------------------------

    def _publish_current_state(self, state):
        """Publish current_state topic (retained)."""
        self._current_state = state
        self.mqtt.publish(TOPIC_CURRENT_STATE, state, qos=1, retain=True)

    def _publish_error(self, error):
        """Publish error_info topic (retained)."""
        self._error_info = error
        self.mqtt.publish(TOPIC_ERROR_INFO, error, qos=1, retain=True)

    def set_handler_state(self, handler_name, state):
        """Called by handlers to update the service's current_state."""
        if state == "idle":
            self._publish_current_state("ready")
        else:
            self._publish_current_state(f"{handler_name}:{state}")

    def set_handler_error(self, handler_name, error):
        """Called by handlers to report an error."""
        if error is None:
            self._publish_error("E_OK")
        else:
            self._publish_error(f"{handler_name}: {error}"[:100])

    # ------------------------------------------------------------------
    # Shared utilities for handlers
    # ------------------------------------------------------------------

    def strip_think_tags(self, text):
        """Remove <think>...</think> blocks from LLM output."""
        return _THINK_RE.sub("", text).strip()

    def synthesize(self, text, prefix="response"):
        """Synthesize speech with Piper TTS.

        Returns the filename (relative to playback dir) or None on failure.
        """
        PLAYBACK_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.wav"
        output_path = PLAYBACK_DIR / filename

        cmd = [
            self.piper_binary,
            "--model", self.piper_model_path,
            "--output_file", str(output_path),
        ]
        try:
            proc = subprocess.run(
                cmd, input=text,
                capture_output=True, text=True, timeout=30,
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
            logger.error(
                "Piper binary not found: %s (install with: pip install piper-tts)",
                self.piper_binary,
            )
            return None

    def publish_audio(self, filename, volume=PLAYBACK_VOLUME):
        """Send playback command to qBc_Audio."""
        self.mqtt.publish(
            "robot/audio/play",
            json.dumps({"file": filename, "volume": volume, "voice": True}),
            qos=1,
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
                    continue
                m = _PROGRESS_RE.search(line)
                if m:
                    logger.info("Loading model: %s%%", m.group(1))
                if "starting" in line.lower() and "server" in line.lower():
                    logger.info("LLM server ready")
                    server_ready.set()

        threading.Thread(target=_reader, daemon=True).start()

        for _ in range(360):  # up to 180 s
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
                    logger.error(
                        "axllm exited with code %d",
                        self._axllm_proc.returncode,
                    )
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

        # Subscribe handler topics
        for h in self._handlers:
            h.subscribe(client)

        # Publish states
        client.publish(
            TOPIC_STATE,
            json.dumps({"status": "online"}),
            qos=1, retain=True,
        )
        client.publish(TOPIC_CURRENT_STATE, self._current_state, qos=1, retain=True)
        client.publish(TOPIC_ERROR_INFO, self._error_info, qos=1, retain=True)
        for h in self._handlers:
            h.publish_state()

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        if reason_code.is_failure:
            logger.warning("Disconnected from MQTT broker: %s", reason_code)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        self._running = True
        # MQTT already connected in __init__

        logger.info("qBc_Ai service running on MQTT %s:%d", self.broker, self.port)

        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())

        while not stop.is_set():
            self.mqtt.publish(TOPIC_HEARTBEAT, b"1", qos=0)
            stop.wait(1.0)

        logger.info("Shutting down...")
        self._publish_current_state("shutting_down")
        self._running = False
        self.mqtt.publish(
            TOPIC_STATE,
            json.dumps({"status": "offline"}),
            qos=1, retain=True,
        )
        self._publish_current_state("offline")
        for h in self._handlers:
            h.shutdown()
        self.mqtt.loop_stop()
        self.mqtt.disconnect()
        self._stop_server()


def main():
    parser = argparse.ArgumentParser(description="qBc_Ai — Unified AI Service")
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

    service = AiService(
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
