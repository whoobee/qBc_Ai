"""
Visual exploration handler for qBc_Ai.

Pipeline: explore command → request camera frame → VLM image analysis → Piper TTS → playback.

MQTT topics:
    Subscribe:  robot/ai/explore/cmd, robot/vision/frame_ready
    Publish:    robot/vision/cmd, robot/ai/explore/state, robot/ai/explore/result
"""

import base64
import io
import json
import logging
import os
import threading
import time

from PIL import Image

logger = logging.getLogger("qBc_Ai.explore")

TOPIC_EXPLORE_CMD = "robot/ai/explore/cmd"
TOPIC_EXPLORE_STATE = "robot/ai/explore/state"
TOPIC_EXPLORE_RESULT = "robot/ai/explore/result"
TOPIC_VISION_CMD = "robot/vision/cmd"
TOPIC_FRAME_READY = "robot/vision/frame_ready"

SYSTEM_PROMPT = (
    "You are qB, a curious and adventurous robot companion exploring your surroundings. "
    "You love discovering new things and are always excited about what you see. "
    "Do NOT use <think> tags or output internal reasoning."
)

EXPLORE_PROMPT = (
    "Look at this image from my camera. "
    "Describe what you see, point out anything interesting, "
    "and suggest what I should explore or look at next. "
    "Be curious and enthusiastic! Keep it to 2-3 sentences."
)

ENCODE_SIZE = (384, 384)
ENCODE_QUALITY = 85
FRAME_TIMEOUT = 10.0


class ExplorationHandler:
    """Handles AI-powered visual exploration."""

    subscriptions = [
        (TOPIC_EXPLORE_CMD, 1),
        (TOPIC_FRAME_READY, 1),
    ]

    def __init__(self, service):
        self._svc = service
        self._processing = False
        self._lock = threading.Lock()

        # Frame synchronization
        self._waiting_for_frame = False
        self._frame_path = None
        self._frame_event = threading.Event()

    def register_callbacks(self, client):
        """Register per-topic MQTT callbacks (called once at init)."""
        client.message_callback_add(TOPIC_EXPLORE_CMD, self._on_explore_cmd)
        client.message_callback_add(TOPIC_FRAME_READY, self._on_frame_ready)

    def subscribe(self, client):
        """Subscribe to topics (called on every MQTT connect)."""
        for topic, qos in self.subscriptions:
            client.subscribe(topic, qos=qos)

    def publish_state(self):
        with self._lock:
            processing = self._processing
        self._svc.mqtt.publish(
            TOPIC_EXPLORE_STATE,
            json.dumps({"status": "online", "processing": processing}),
            qos=1, retain=True,
        )

    def shutdown(self):
        self._svc.mqtt.publish(
            TOPIC_EXPLORE_STATE,
            json.dumps({"status": "offline"}),
            qos=1, retain=True,
        )

    # ------------------------------------------------------------------
    # MQTT callbacks
    # ------------------------------------------------------------------

    def _on_explore_cmd(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Invalid JSON on %s", msg.topic)
            return

        if data.get("command") != "explore":
            logger.debug("Ignoring unknown command: %s", data.get("command"))
            return

        threading.Thread(target=self._explore, daemon=True).start()

    def _on_frame_ready(self, client, userdata, msg):
        if not self._waiting_for_frame:
            return
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        file_path = data.get("file")
        if file_path:
            self._frame_path = file_path
            self._frame_event.set()

    # ------------------------------------------------------------------
    # Exploration pipeline
    # ------------------------------------------------------------------

    def _explore(self):
        with self._lock:
            if self._processing:
                logger.warning("Already processing, skipping explore request")
                return
            self._processing = True
        self.publish_state()
        self._svc.set_handler_error("explore", None)

        try:
            # 1. Request frame capture
            self._svc.set_handler_state("explore", "capturing")
            logger.info("Requesting frame capture...")
            self._frame_event.clear()
            self._frame_path = None
            self._waiting_for_frame = True

            self._svc.mqtt.publish(
                TOPIC_VISION_CMD,
                json.dumps({"command": "capture_frame"}),
                qos=1,
            )

            # 2. Wait for frame
            if not self._frame_event.wait(timeout=FRAME_TIMEOUT):
                logger.error("Frame capture timed out")
                self._svc.set_handler_error("explore", "frame capture timed out")
                return
            self._waiting_for_frame = False

            frame_path = self._frame_path
            if not frame_path or not os.path.isfile(frame_path):
                logger.error("Frame file not found: %s", frame_path)
                self._svc.set_handler_error("explore", "frame file not found")
                return
            logger.info("Frame received: %s", frame_path)

            # 3. Encode image for VLM
            b64 = self._encode_image(frame_path)
            if not b64:
                logger.error("Failed to encode image")
                self._svc.set_handler_error("explore", "image encoding failed")
                return

            # 4. Analyze with VLM
            self._svc.set_handler_state("explore", "analyzing")
            logger.info("Analyzing image with VLM...")
            analysis = self._analyze_image(b64)
            if not analysis or not analysis.strip():
                logger.warning("Empty analysis result")
                self._svc.set_handler_error("explore", "empty VLM analysis")
                return
            logger.info("Exploration analysis: %s", analysis)

            # 5. Publish result
            self._svc.mqtt.publish(
                TOPIC_EXPLORE_RESULT,
                json.dumps({
                    "analysis": analysis,
                    "frame": frame_path,
                    "timestamp": time.time(),
                }),
                qos=1,
            )

            # 6. Synthesize narration
            self._svc.set_handler_state("explore", "synthesizing")
            logger.info("Synthesizing exploration narration...")
            audio_filename = self._svc.synthesize(analysis, prefix="explore")
            if not audio_filename:
                logger.error("TTS synthesis failed")
                self._svc.set_handler_error("explore", "TTS synthesis failed")
                return

            # 7. Play
            self._svc.publish_audio(audio_filename)
            logger.info("Exploration narration sent for playback")

        except Exception as e:
            logger.error("Exploration pipeline error: %s", e, exc_info=True)
            self._svc.set_handler_error("explore", str(e)[:80])
        finally:
            self._waiting_for_frame = False
            with self._lock:
                self._processing = False
            self.publish_state()
            self._svc.set_handler_state("explore", "idle")

    def _encode_image(self, image_path):
        """Load image, resize for VLM input, return base64 JPEG."""
        try:
            img = Image.open(image_path)
            resized = img.resize(ENCODE_SIZE, Image.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=ENCODE_QUALITY)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            logger.error("Image encoding error: %s", e)
            return None

    def _analyze_image(self, b64_image):
        """Send image to VLM for exploration analysis."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXPLORE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                        },
                    },
                ],
            },
        ]

        response = self._svc.llm.chat.completions.create(
            model=self._svc.model_name,
            messages=messages,
            max_tokens=256,
        )
        text = response.choices[0].message.content
        return self._svc.strip_think_tags(text)
