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
import re
import threading
import time

from PIL import Image

logger = logging.getLogger("qBc_Ai.explore")

TOPIC_EXPLORE_CMD = "robot/ai/explore/cmd"
TOPIC_EXPLORE_STATE = "robot/ai/explore/state"
TOPIC_EXPLORE_RESULT = "robot/ai/explore/result"
TOPIC_NAV_WAYPOINTS = "robot/navigation/waypoints"
TOPIC_VISION_CMD = "robot/vision/cmd"
TOPIC_FRAME_READY = "robot/vision/frame_ready"

SYSTEM_PROMPT = (
    "You are qB, a curious robot exploring your surroundings. "
    "Always respond in EXACTLY this format:\n"
    "Line 1: A JSON object with waypoints\n"
    "Line 2: ---\n"
    "Line 3+: Your narration\n"
    "Do NOT use <think> tags."
)

EXPLORE_PROMPT = (
    "Look at this image from my camera. "
    "Pick a point you want to explore and create 2-5 waypoints to get there, "
    "avoiding obstacles. Waypoints are in image coordinates where (0,0) is top-left "
    "and (1,1) is bottom-right. x is horizontal, y is vertical.\n\n"
    "Respond in EXACTLY this format (JSON on first line, then ---, then narration):\n"
    '{"waypoints": [{"x": 0.5, "y": 0.8}, {"x": 0.5, "y": 0.4}]}\n'
    "---\n"
    "I see a hallway ahead and I want to explore it!"
)

_WAYPOINT_JSON_RE = re.compile(
    r'\{\s*"waypoints"\s*:\s*\[.*?\]\s*\}', re.DOTALL
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
        t0 = time.monotonic()

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
            raw_analysis = self._analyze_image(b64)
            if not raw_analysis or not raw_analysis.strip():
                logger.warning("Empty analysis result")
                self._svc.set_handler_error("explore", "empty VLM analysis")
                return
            logger.info("Raw VLM response: %s", raw_analysis)

            # 4b. Parse waypoints and narration
            waypoints, narration = self._parse_vlm_response(raw_analysis)
            logger.info("Parsed %d waypoints, narration: %s", len(waypoints), narration[:80])

            # 4c. Publish waypoints for navigation service
            if waypoints:
                self._svc.mqtt.publish(
                    TOPIC_NAV_WAYPOINTS,
                    json.dumps({
                        "waypoints": waypoints,
                        "frame": frame_path,
                        "timestamp": time.time(),
                    }),
                    qos=1,
                )
                logger.info("Published %d waypoints to %s", len(waypoints), TOPIC_NAV_WAYPOINTS)
            else:
                logger.warning("No valid waypoints parsed from VLM response")

            # 5. Publish transcript for debug viewer
            self._svc.publish_transcript({
                "type": "exploration",
                "prompt": "Visual exploration",
                "response": narration,
                "waypoints": waypoints,
                "frame": frame_path,
                "duration_ms": int((time.monotonic() - t0) * 1000),
            })

            # 6. Publish result (includes both waypoints and narration)
            self._svc.mqtt.publish(
                TOPIC_EXPLORE_RESULT,
                json.dumps({
                    "analysis": narration,
                    "waypoints": waypoints,
                    "frame": frame_path,
                    "timestamp": time.time(),
                }),
                qos=1,
            )

            # 7. Synthesize narration
            self._svc.set_handler_state("explore", "synthesizing")
            logger.info("Synthesizing exploration narration...")
            audio_filename = self._svc.synthesize(narration, prefix="explore")
            if not audio_filename:
                logger.error("TTS synthesis failed")
                self._svc.set_handler_error("explore", "TTS synthesis failed")
                return

            # 8. Play
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

    def _parse_vlm_response(self, text):
        """Parse VLM response into (waypoints_list, narration_text).

        Expected format: JSON on first line, '---' separator, then narration.
        Falls back to regex extraction if format doesn't match.
        Returns ([], full_text) on parse failure so TTS still works.
        """
        waypoints = []
        narration = text

        # Try splitting on --- separator first
        parts = text.split("---", 1)
        json_part = parts[0].strip()
        if len(parts) > 1:
            narration = parts[1].strip()

        # Try parsing JSON from the first part
        parsed = None
        try:
            parsed = json.loads(json_part)
        except (json.JSONDecodeError, ValueError):
            # Fallback: regex search in the full text
            m = _WAYPOINT_JSON_RE.search(text)
            if m:
                try:
                    parsed = json.loads(m.group())
                    # Remove JSON from narration
                    narration = text[:m.start()].strip(" \n") + " " + text[m.end():].strip(" \n")
                    narration = narration.replace("---", "").strip()
                except (json.JSONDecodeError, ValueError):
                    pass

        if parsed and isinstance(parsed.get("waypoints"), list):
            for wp in parsed["waypoints"]:
                if isinstance(wp, dict):
                    x = wp.get("x")
                    y = wp.get("y")
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                            waypoints.append({"x": float(x), "y": float(y)})

        if not narration or not narration.strip():
            narration = text

        return waypoints, narration.strip()

    def _analyze_image(self, b64_image):
        """Send image to VLM for exploration analysis."""
        # Build system prompt with optional language instruction for narration
        prompt = SYSTEM_PROMPT
        lang_instr = self._svc.language_instruction
        if lang_instr:
            prompt = f"{prompt}\n{lang_instr} The JSON waypoints must stay in English format, but write the narration in the requested language."

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt}],
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
            max_tokens=384,
        )
        text = response.choices[0].message.content
        return self._svc.strip_think_tags(text)
