"""
Voice assistant handler for qBc_Ai.

Pipeline: wake-word recording → Whisper STT → LLM chat → Piper TTS → playback.

MQTT topics:
    Subscribe:  robot/audio/recording_ready
    Publish:    robot/ai/voice/state  (RETAIN)
"""

import json
import logging
import os
import threading
import time

logger = logging.getLogger("qBc_Ai.voice")

TOPIC_RECORDING_READY = "robot/audio/recording_ready"
TOPIC_STATE = "robot/ai/voice/state"

SYSTEM_PROMPT = (
    "You are qB, a friendly and helpful robot companion. "
    "You MUST use the provided tools to fetch the time or the weather if asked. "
    "Do not answer without using the tools if you need that information. "
    "Give short, natural, conversational responses. "
    "Keep answers to 1-3 sentences unless more detail is needed. "
    "Do NOT use <think> tags or output internal reasoning."
)


class VoiceHandler:
    """Handles voice interactions: transcribe → LLM → speak."""

    subscriptions = [(TOPIC_RECORDING_READY, 1)]

    def __init__(self, service):
        self._svc = service
        self._processing = False
        self._lock = threading.Lock()

    def register_callbacks(self, client):
        """Register per-topic MQTT callbacks (called once at init)."""
        client.message_callback_add(TOPIC_RECORDING_READY, self._on_recording)

    def subscribe(self, client):
        """Subscribe to topics (called on every MQTT connect)."""
        for topic, qos in self.subscriptions:
            client.subscribe(topic, qos=qos)

    def publish_state(self):
        with self._lock:
            processing = self._processing
        self._svc.mqtt.publish(
            TOPIC_STATE,
            json.dumps({"status": "online", "processing": processing}),
            qos=1, retain=True,
        )

    def shutdown(self):
        self._svc.mqtt.publish(
            TOPIC_STATE,
            json.dumps({"status": "offline"}),
            qos=1, retain=True,
        )

    # ------------------------------------------------------------------
    # MQTT callback
    # ------------------------------------------------------------------

    def _on_recording(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Invalid JSON on %s", msg.topic)
            return

        file_path = data.get("file")
        if not file_path:
            logger.warning("No file in recording_ready message")
            return

        threading.Thread(
            target=self._process_voice,
            args=(file_path,),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Voice pipeline
    # ------------------------------------------------------------------

    def _process_voice(self, recording_path):
        with self._lock:
            if self._processing:
                logger.warning("Already processing, skipping: %s", recording_path)
                return
            self._processing = True
        self.publish_state()
        self._svc.set_handler_error("voice", None)

        try:
            # 1. Transcribe
            self._svc.set_handler_state("voice", "transcribing")
            logger.info("Transcribing: %s", recording_path)
            text = self._transcribe(recording_path)
            if not text or not text.strip():
                logger.info("Empty transcription, skipping")
                return
            logger.info("Transcription: %s", text)

            # 2. Query LLM
            self._svc.set_handler_state("voice", "querying_llm")
            logger.info("Querying LLM...")
            response = self._query_llm(text)
            if not response or not response.strip():
                logger.warning("Empty LLM response")
                self._svc.set_handler_error("voice", "empty LLM response")
                return
            logger.info("LLM response: %s", response)

            # 3. Synthesize speech
            self._svc.set_handler_state("voice", "synthesizing")
            logger.info("Synthesizing speech...")
            audio_filename = self._svc.synthesize(response, prefix="voice")
            if not audio_filename:
                logger.error("TTS synthesis failed")
                self._svc.set_handler_error("voice", "TTS synthesis failed")
                return

            # 4. Play
            self._svc.publish_audio(audio_filename)
            logger.info("Playback command sent")

            # 5. Check if it's a question, re-trigger mic if so
            if response.strip().endswith("?"):
                try:
                    import wave
                    from pathlib import Path
                    playback_dir = Path(__file__).parent.parent / "qBc_Audio" / "resources" / "playback"
                    wav_path = playback_dir / audio_filename
                    with wave.open(str(wav_path), 'rb') as wf:
                        duration = wf.getnframes() / float(wf.getframerate())
                    
                    logger.info("Question detected. Waiting %.1fs to re-trigger microphone...", duration)
                    time.sleep(duration + 0.5)
                    self._svc.mqtt.publish("robot/audio/cmd", json.dumps({"command": "record"}), qos=1)
                    logger.info("Microphone re-triggered for conversational continuation.")
                except Exception as e:
                    logger.error("Failed to re-trigger microphone: %s", e)

        except Exception as e:
            logger.error("Voice pipeline error: %s", e, exc_info=True)
            self._svc.set_handler_error("voice", str(e)[:80])
        finally:
            with self._lock:
                self._processing = False
            self.publish_state()
            self._svc.set_handler_state("voice", "idle")

    def _transcribe(self, audio_path):
        """Transcribe audio file using faster-whisper."""
        if not os.path.isfile(audio_path):
            logger.error("Recording not found: %s", audio_path)
            return ""

        segments, _info = self._svc.whisper.transcribe(
            audio_path,
            language="en",
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()

    def _query_llm(self, user_text):
        """Send transcribed text to LLM and return the response."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        
        response = self._svc.llm.chat.completions.create(
            model=self._svc.model_name,
            messages=messages,
            tools=self._svc.mcp.get_tools_schema(),
            max_tokens=256,
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            # We must serialize the message correctly for the next request.
            messages.append(message.model_dump(exclude_unset=True))
            for tool_call in message.tool_calls:
                result = self._svc.mcp.execute_tool(
                    tool_call.function.name, 
                    tool_call.function.arguments
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
                
            response = self._svc.llm.chat.completions.create(
                model=self._svc.model_name,
                messages=messages,
                max_tokens=256,
            )
            message = response.choices[0].message

        text = message.content or ""
        return self._svc.strip_think_tags(text)
