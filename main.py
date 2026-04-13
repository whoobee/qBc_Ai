#!/usr/bin/env python3
"""
qBc_Ai — Unified AI service for qB Companion.

Manages NPU-accelerated AI servers and hosts modular AI feature handlers.
Each handler lives in its own file and registers MQTT topics via the service.

Current features:
    - Voice assistant (voice_handler.py): STT → LLM → TTS
    - Visual exploration (exploration_handler.py): Camera → VLM → TTS

All three inference stages (STT, LLM, TTS) are accessed via configurable HTTP
endpoints, allowing them to run locally on the NPU or on a remote server.

MQTT topics:
    Publish:
        robot/ai/state               (RETAIN) Overall service state
        robot/system/heartbeat/ai    Keepalive (1 Hz)

Prerequisites:
    pip install openai paho-mqtt Pillow
    axllm binary in PATH (for local LLM mode)
    whisper.axcl server (STT, default port 8801)
    melotts.axcl server (TTS, default port 8802)

Usage:
    python3 main.py [--stt-url http://127.0.0.1:8801] [--tts-url http://127.0.0.1:8802]
"""

import argparse
import atexit
import json
import logging
import os
import base64
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import paho.mqtt.client as mqtt
from openai import OpenAI

logger = logging.getLogger("qBc_Ai")

SCRIPT_DIR = Path(__file__).parent
PLAYBACK_DIR = SCRIPT_DIR.parent / "qBc_Audio" / "resources" / "playback"
DEFAULT_MODEL_DIR = str(SCRIPT_DIR / "models" / "Qwen3.5-4B")

DEFAULT_STT_URL = "http://127.0.0.1:8801"
DEFAULT_LLM_URL = "http://127.0.0.1:8000/v1"
DEFAULT_TTS_URL = "http://127.0.0.1:8802"
DEFAULT_MODEL = "AXERA-TECH/Qwen3.5-4B-AX650-GPTQ-Int4-C128-P1152-CTX2047"

TOPIC_STATE = "robot/ai/state"
TOPIC_HEARTBEAT = "robot/system/heartbeat/ai"
TOPIC_CURRENT_STATE = "robot/ai/current_state"
TOPIC_ERROR_INFO = "robot/ai/error_info"
TOPIC_TRANSCRIPT = "robot/ai/transcript"
TOPIC_LOADING_PROGRESS = "robot/system/loading_progress"

PLAYBACK_VOLUME = 50  # default, overridden by MQTT settings

TOPIC_SETTINGS_AUDIO = "robot/settings/audio"
TOPIC_SETTINGS_AI = "robot/settings/ai"

# ── Language configuration ──
# Maps language code → STT language, TTS language code, LLM instruction
LANGUAGE_CONFIG = {
    "en": {
        "stt_lang": "en",
        "tts_lang": "en",
        "name": "English",
        "instruction": "",
    },
    "ro": {
        "stt_lang": "ro",
        "tts_lang": "ro",
        "name": "Romanian",
        "instruction": "You MUST respond in Romanian (limba romana).",
    },
    "de": {
        "stt_lang": "de",
        "tts_lang": "de",
        "name": "German",
        "instruction": "You MUST respond in German (Deutsch).",
    },
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_PROGRESS_RE = re.compile(r"(\d+)%\s*\|")


class AiService:
    """Unified AI service — shared infrastructure for all AI handlers."""

    def __init__(
        self,
        broker="localhost",
        port=1883,
        stt_url=DEFAULT_STT_URL,
        llm_url=DEFAULT_LLM_URL,
        tts_url=DEFAULT_TTS_URL,
        model_name=DEFAULT_MODEL,
        model_dir=DEFAULT_MODEL_DIR,
    ):
        self.broker = broker
        self.port = port
        self.stt_url = stt_url.rstrip("/")
        self.llm_url = llm_url.rstrip("/")
        self.tts_url = tts_url.rstrip("/")
        self.model_name = model_name
        self.model_dir = model_dir

        self._running = False
        self._axllm_proc = None
        self._current_state = "starting"
        self._error_info = "E_OK"
        self._handlers = []

        # Volume settings (updated dynamically via MQTT)
        self._global_volume = 100
        self._ai_reply_volume = PLAYBACK_VOLUME

        # Language settings (updated dynamically via MQTT)
        self._language = "en"

        # ── MQTT client (connect early for state reporting) ──
        self.mqtt = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="qbc_ai",
        )
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_disconnect = self._on_disconnect
        self.mqtt.on_message = self._on_message
        self.mqtt.will_set(
            TOPIC_STATE,
            json.dumps({"status": "offline"}),
            qos=1, retain=True,
        )
        self.mqtt.connect(broker, port)
        self.mqtt.loop_start()

        try:
            # ── 1. LLM server ──
            self._publish_current_state("loading_llm")
            self._publish_progress("loading_llm", 0, "Starting LLM server...")
            if not self._wait_for_server(self.llm_url + "/models", "LLM",
                                         timeout=180):
                self._publish_error("LLM server unavailable")
                raise RuntimeError("LLM server unavailable — cannot start AI service")

            self.llm = OpenAI(api_key="not-needed", base_url=llm_url)
            self._publish_progress("loading_llm", 70, "LLM server ready")

            # ── 2. STT server (whisper.axcl) ──
            self._publish_current_state("loading_stt")
            self._publish_progress("loading_stt", 72, "Connecting to STT server...")
            if not self._wait_for_server(self.stt_url + "/health", "STT",
                                         timeout=120):
                self._publish_error("STT server unavailable")
                raise RuntimeError(
                    f"STT server not reachable at {self.stt_url}\n"
                    "Start whisper.axcl:  cd ~/whisper.axcl && bash serve.sh"
                )
            logger.info("STT server ready at %s", self.stt_url)
            self._publish_progress("loading_stt", 85, "Speech recognition ready")

            # ── 3. TTS server (melotts.axcl) ──
            self._publish_current_state("loading_tts")
            self._publish_progress("loading_tts", 88, "Connecting to TTS server...")
            if not self._wait_for_server(self.tts_url + "/health", "TTS",
                                         timeout=120):
                self._publish_error("TTS server unavailable")
                raise RuntimeError(
                    f"TTS server not reachable at {self.tts_url}\n"
                    "Start melotts.axcl:  cd ~/melotts.axcl && bash serve.sh"
                )
            logger.info("TTS server ready at %s", self.tts_url)
            self._publish_progress("loading_tts", 92, "TTS server ready")

            # ── 4. MCP Tool Server ──
            self._publish_progress("loading_tools", 94, "Loading tool server...")
            from mcp_server import McpServer
            self.mcp = McpServer(mqtt_client=self.mqtt)
            logger.info("MCP Tool Server loaded with %d tools", len(self.mcp.get_tools_schema()))

            # ── Feature handlers ──
            self._publish_progress("loading_handlers", 96, "Loading handlers...")
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
            self._publish_progress("ready", 100, "AI service ready")
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

    def _publish_progress(self, stage, percent, message=""):
        """Publish loading progress for boot screen / dashboard."""
        self.mqtt.publish(
            TOPIC_LOADING_PROGRESS,
            json.dumps({
                "service": "ai",
                "stage": stage,
                "percent": min(100, max(0, int(percent))),
                "message": message or stage,
            }),
            qos=0,
        )

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

    def transcribe(self, audio_path):
        """Transcribe audio via the whisper.axcl STT server.

        Sends the audio as base64-encoded data to the /recognize endpoint
        and returns the transcribed text.
        """
        if not os.path.isfile(audio_path):
            logger.error("Recording not found: %s", audio_path)
            return ""

        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            payload = json.dumps({
                "base64": base64.b64encode(audio_data).decode("ascii"),
            }).encode("utf-8")

            url = f"{self.stt_url}/recognize"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = resp.read().decode("utf-8").strip()

            data = json.loads(result)
            return data.get("recognition", data.get("text", "")).strip()
        except Exception as e:
            logger.error("STT request failed: %s", e)
            return ""

    def synthesize(self, text, prefix="response"):
        """Synthesize speech via the melotts.axcl TTS server.

        Posts text to the /synthesize endpoint and decodes the base64 audio
        response into a WAV file.  The result is resampled to 16 kHz mono
        S16_LE to match the ReSpeaker's native playback format.

        Returns the filename (relative to playback dir) or None on failure.
        """
        PLAYBACK_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.wav"
        output_path = PLAYBACK_DIR / filename

        try:
            payload = json.dumps({
                "sentence": text,
                "base64": True,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.tts_url}/synthesize",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            if not result.get("success"):
                logger.error("TTS server returned failure: %s", result)
                return None

            audio_data = base64.b64decode(result["base64"])
            if not audio_data or len(audio_data) < 100:
                logger.error("TTS server returned empty or invalid audio")
                return None

            # Write raw TTS output to a temp file, then resample with sox
            raw_path = str(output_path) + ".raw.wav"
            with open(raw_path, "wb") as f:
                f.write(audio_data)

            # Resample to 16 kHz mono (ReSpeaker native format)
            try:
                import subprocess as _sp
                result_sox = _sp.run(
                    ["sox", raw_path, "-r", "16000", "-c", "1",
                     "-b", "16", str(output_path)],
                    capture_output=True, text=True, timeout=15,
                )
                os.unlink(raw_path)
                if result_sox.returncode != 0:
                    logger.warning("Sox resample failed: %s", result_sox.stderr)
                    # Fall back to the original file
                    os.rename(raw_path, str(output_path))
            except FileNotFoundError:
                logger.warning("Sox not available — using TTS output as-is")
                os.rename(raw_path, str(output_path))

            return filename
        except Exception as e:
            logger.error("TTS request failed: %s", e)
            return None

    def get_effective_ai_volume(self):
        """Compute effective AI reply volume: ai_reply_volume * global_volume / 100."""
        return round(self._ai_reply_volume * self._global_volume / 100)

    def publish_transcript(self, data):
        """Publish an AI transcript event for the debug viewer."""
        data.setdefault("timestamp", time.time())
        self.mqtt.publish(
            TOPIC_TRANSCRIPT,
            json.dumps(data),
            qos=1,
        )

    @property
    def stt_language(self):
        """Current STT language code (e.g. 'en', 'ro', 'de')."""
        return LANGUAGE_CONFIG.get(self._language, LANGUAGE_CONFIG["en"])["stt_lang"]

    @property
    def tts_language(self):
        """Current TTS language code."""
        return LANGUAGE_CONFIG.get(self._language, LANGUAGE_CONFIG["en"])["tts_lang"]

    @property
    def language_instruction(self):
        """LLM instruction for the current language (empty for English)."""
        return LANGUAGE_CONFIG.get(self._language, LANGUAGE_CONFIG["en"])["instruction"]

    def _apply_language(self, lang):
        """Switch language at runtime (called from MQTT settings handler)."""
        if lang not in LANGUAGE_CONFIG:
            logger.warning("Unknown language code: %s — ignoring", lang)
            return
        old = self._language
        self._language = lang
        logger.info("Language switched: %s → %s", old, lang)

    def publish_audio(self, filename, volume=None):
        """Send playback command to qBc_Audio."""
        if volume is None:
            volume = self.get_effective_ai_volume()
        self.mqtt.publish(
            "robot/audio/play",
            json.dumps({"file": filename, "volume": volume, "voice": True}),
            qos=1,
        )

    # ------------------------------------------------------------------
    # Server management
    # ------------------------------------------------------------------

    def _wait_for_server(self, url, name, method="GET", timeout=180,
                         local_launcher=None):
        """Wait for an HTTP server to become reachable.

        If the server is already up, returns immediately.  If *local_launcher*
        is provided and the server is not reachable, calls it to start a local
        process before polling.  Returns True when ready, False on timeout.
        """
        # Quick check — already running?
        try:
            req = urllib.request.Request(url, method=method)
            with urllib.request.urlopen(req, timeout=2):
                logger.info("%s server already running at %s", name, url)
                return True
        except Exception:
            pass

        # Attempt local launch if a launcher is provided
        if local_launcher:
            if not local_launcher():
                return False

        # Poll until ready
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.5)
            try:
                req = urllib.request.Request(url, method=method)
                with urllib.request.urlopen(req, timeout=2):
                    logger.info("%s server ready at %s", name, url)
                    return True
            except Exception:
                pass
            # Check if local process died
            if self._axllm_proc and self._axllm_proc.poll() is not None:
                logger.error("axllm exited with code %d",
                             self._axllm_proc.returncode)
                self._axllm_proc = None
                return False

        logger.error("%s server did not become ready in %ds", name, timeout)
        return False

    def _launch_axllm(self):
        """Start axllm serve locally.  Returns True if launched."""
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

        # Read stdout in background for progress reporting
        def _reader():
            for raw in self._axllm_proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                m = _PROGRESS_RE.search(line)
                if m:
                    pct = int(m.group(1))
                    logger.info("Loading model: %s%%", pct)
                    overall = int(pct * 0.68)
                    self._publish_progress(
                        "loading_llm", overall,
                        f"Loading LLM model: {pct}%",
                    )

        threading.Thread(target=_reader, daemon=True).start()
        return True

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

        # Subscribe to settings
        client.subscribe(TOPIC_SETTINGS_AUDIO, qos=1)
        client.subscribe(TOPIC_SETTINGS_AI, qos=1)

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

    def _on_message(self, client, userdata, msg):
        """Handle messages not matched by per-topic callbacks."""
        if msg.topic == TOPIC_SETTINGS_AUDIO:
            try:
                data = json.loads(msg.payload)
                self._global_volume = int(data.get("global_volume", self._global_volume))
                self._ai_reply_volume = int(data.get("ai_reply_volume", self._ai_reply_volume))
                logger.info("Volume settings updated: global=%d, ai_reply=%d",
                            self._global_volume, self._ai_reply_volume)
            except Exception as e:
                logger.warning("Failed to parse audio settings: %s", e)
        elif msg.topic == TOPIC_SETTINGS_AI:
            try:
                data = json.loads(msg.payload)
                lang = data.get("language")
                if lang and lang != self._language:
                    self._apply_language(lang)
            except Exception as e:
                logger.warning("Failed to parse AI settings: %s", e)

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
    parser.add_argument("--stt-url", default=DEFAULT_STT_URL,
                        help="STT server base URL (whisper.axcl)")
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL,
                        help="LLM API base URL (axllm serve)")
    parser.add_argument("--tts-url", default=DEFAULT_TTS_URL,
                        help="TTS server base URL (melotts.axcl)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="LLM model name")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="Path to axllm model directory (local mode only)")
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
        stt_url=args.stt_url,
        llm_url=args.llm_url,
        tts_url=args.tts_url,
        model_name=args.model,
        model_dir=args.model_dir,
    )
    service.run()


if __name__ == "__main__":
    main()
