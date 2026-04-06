#!/usr/bin/env python3
"""
qB Companion Vision - Continuous AI-powered surroundings analysis.

Uses the Raspberry Pi camera and Qwen3.5-4B VLM running on an LLM8850
NPU accelerator to continuously analyze the environment.

Prerequisites:
    1. Install axllm:
       https://github.com/AXERA-TECH/ax-llm

    2. Start the model server:
       axllm serve /path/to/qBc_Ai/Qwen3.5-4B/

    3. Install dependencies:
       pip install -r requirements.txt

    4. Run:
       python main.py

Keys:
    q  - Quit
    p  - Pause / Resume
    r  - Force immediate re-capture
"""

import argparse
import atexit
import base64
import io
import os
import re
import shutil
import signal
import subprocess
import time

import numpy as np
from PIL import Image
from openai import OpenAI

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Label, ProgressBar
from textual import work

from rich.text import Text
from rich.style import Style
from rich.color import Color

try:
    from picamera2 import Picamera2

    HAS_CAMERA = True
except ImportError:
    HAS_CAMERA = False

# ── defaults ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "Qwen3.5-4B")
DEFAULT_API_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "AXERA-TECH/Qwen3.5-4B-AX650-GPTQ-Int4-C128-P1152-CTX2047"
CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 480

SYSTEM_PROMPT = (
    "You are a vision assistant for a robot companion. "
    "Describe what you see concisely and clearly. "
    "Focus on identifying objects, people, activities, and the environment. "
    "Do NOT use <think> tags or output internal reasoning."
)
USER_PROMPT = (
    "Analyze this image from my camera. "
    "What objects, people, or notable things do you see? "
    "Describe the surroundings briefly."
)


# ── image → half-block terminal rendering ───────────────────────────────────
def render_image(img: Image.Image, width: int, max_rows: int = 0) -> Text:
    """Return a Rich Text with coloured ▀ half-block characters."""
    if width <= 0:
        return Text("(no space)")

    img = img.convert("RGB")
    aspect = img.height / img.width
    rows = max(1, int(width * aspect / 2))
    if max_rows > 0:
        rows = min(rows, max_rows)

    img = img.resize((width, rows * 2), Image.LANCZOS)
    px = np.asarray(img)

    result = Text()
    for y in range(0, rows * 2, 2):
        line = Text()
        top_row = px[y]
        bot_row = px[y + 1] if y + 1 < rows * 2 else np.zeros_like(top_row)

        for x in range(width):
            tr, tg, tb = int(top_row[x][0]), int(top_row[x][1]), int(top_row[x][2])
            br, bg, bb = int(bot_row[x][0]), int(bot_row[x][1]), int(bot_row[x][2])
            line.append(
                "▀",
                style=Style(
                    color=Color.from_rgb(tr, tg, tb),
                    bgcolor=Color.from_rgb(br, bg, bb),
                ),
            )
        result.append_text(line)
        if y + 2 < rows * 2:
            result.append("\n")
    return result


# ── image widget ────────────────────────────────────────────────────────────
class ImageWidget(Static):
    """Renders a PIL Image inside a Textual Static widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._image: Image.Image | None = None

    def set_image(self, img: Image.Image) -> None:
        self._image = img
        self._repaint()

    def _repaint(self) -> None:
        if self._image is None:
            self.update("[dim]Waiting for camera…[/dim]")
            return
        w = self.content_size.width
        h = self.content_size.height
        if w > 0:
            self.update(render_image(self._image, w, h))

    def on_resize(self) -> None:
        self._repaint()


# ── main application ────────────────────────────────────────────────────────
class VisionApp(App):

    CSS = """
    Screen {
        layout: vertical;
    }
    #main-area {
        layout: horizontal;
        height: 1fr;
    }

    /* ── camera panel ── */
    #camera-box {
        width: 1fr;
        height: 1fr;
        border: round $primary;
        overflow: hidden;
    }
    #camera-box > Label {
        dock: top;
        width: 100%;
        text-style: bold;
        color: $text;
        background: $primary-background;
        padding: 0 1;
    }
    #camera-view {
        width: 100%;
        height: 1fr;
    }

    /* ── analysis panel ── */
    #analysis-box {
        width: 1fr;
        height: 1fr;
        border: round $secondary;
        overflow-y: auto;
    }
    #analysis-box > Label {
        dock: top;
        width: 100%;
        text-style: bold;
        color: $text;
        background: $secondary-background;
        padding: 0 1;
    }
    #analysis-text {
        width: 100%;
        height: auto;
        padding: 0 1;
    }

    /* ── status bar ── */
    #status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }

    /* ── loading bar inside analysis ── */
    #loading-bar {
        width: 100%;
        margin: 1 0;
    }
    """

    TITLE = "qB Companion Vision"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("p", "toggle_pause", "Pause / Resume"),
        Binding("r", "force_capture", "Re-capture"),
    ]

    def __init__(self, api_url: str, model: str, model_dir: str):
        super().__init__()
        self.api_url = api_url
        self.model_name = model
        self.model_dir = model_dir
        self.camera: "Picamera2 | None" = None
        self.client: OpenAI | None = None
        self._axllm_proc: subprocess.Popen | None = None
        self.paused = False
        self._running = True
        self._capture_num = 0

    # ── compose ─────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-area"):
            with Vertical(id="camera-box"):
                yield Label(" Camera")
                yield ImageWidget(id="camera-view")
            with Vertical(id="analysis-box"):
                yield Label(" Analysis")
                yield ProgressBar(id="loading-bar", total=100, show_eta=False)
                yield Static(
                    "[dim]Initialising…[/dim]", id="analysis-text"
                )
        yield Label(" Initialising…", id="status-bar")
        yield Footer()

    # ── lifecycle ───────────────────────────────────────────────────────
    def on_mount(self) -> None:
        self._startup_worker()

    def on_unmount(self) -> None:
        self._running = False
        if self.camera:
            try:
                self.camera.stop()
            except Exception:
                pass
        self._stop_server()

    # ── startup worker (threaded) ───────────────────────────────────────
    @work(exclusive=True, thread=True)
    def _startup_worker(self) -> None:
        """Start the server in the background, then kick off analysis."""
        self._init_camera()
        self._ensure_server()
        self.client = OpenAI(api_key="not-needed", base_url=self.api_url)
        # Hide progress bar, start analysis
        self.call_from_thread(self._hide_loading_bar)
        self._run_analysis()

    def _hide_loading_bar(self) -> None:
        try:
            self.query_one("#loading-bar", ProgressBar).display = False
        except Exception:
            pass

    def _set_loading(self, progress: float, status: str) -> None:
        """Update loading bar and status in the analysis panel."""
        try:
            bar = self.query_one("#loading-bar", ProgressBar)
            bar.update(progress=progress)
            self.query_one("#analysis-text", Static).update(
                f"[bold]Loading Model[/bold]\n\n{status}"
            )
            self.query_one("#status-bar", Label).update(f" {status}")
        except Exception:
            pass

    # ── axllm server management ─────────────────────────────────────────
    _PROGRESS_RE = re.compile(r"(\d+)%\s*\|")

    def _ensure_server(self) -> None:
        """Start axllm serve if not already reachable."""
        import urllib.request
        try:
            url = self.api_url.rstrip("/") + "/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2):
                self.call_from_thread(self._set_loading, 100, "Model server already running")
                return
        except Exception:
            pass

        axllm = shutil.which("axllm")
        if not axllm:
            self.call_from_thread(self._set_loading, 0, "axllm not found in PATH – install it first")
            return
        if not os.path.isdir(self.model_dir):
            self.call_from_thread(self._set_loading, 0, f"Model dir not found: {self.model_dir}")
            return

        self.call_from_thread(self._set_loading, 0, "Starting axllm serve…")
        self._axllm_proc = subprocess.Popen(
            [axllm, "serve", self.model_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        atexit.register(self._stop_server)

        # Read output lines in a separate thread to parse progress
        import threading
        last_pct = [0]
        last_line = ["Loading…"]
        server_ready = threading.Event()

        def _reader():
            for raw in self._axllm_proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                m = self._PROGRESS_RE.search(line)
                if m:
                    last_pct[0] = int(m.group(1))
                # Trim verbose prefixes for display
                short = line.strip()
                if len(short) > 80:
                    short = short[:77] + "…"
                last_line[0] = short
                self.call_from_thread(self._set_loading, last_pct[0], short)
                if "starting" in line.lower() and "server" in line.lower():
                    last_pct[0] = 100
                    self.call_from_thread(self._set_loading, 100, "Server ready!")
                    server_ready.set()

        reader_t = threading.Thread(target=_reader, daemon=True)
        reader_t.start()

        # Also poll the HTTP endpoint as a fallback
        for i in range(360):  # up to 180 s
            if server_ready.is_set():
                return
            time.sleep(0.5)
            try:
                url = self.api_url.rstrip("/") + "/models"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=2):
                    self.call_from_thread(self._set_loading, 100, "Model server ready")
                    return
            except Exception:
                if self._axllm_proc.poll() is not None:
                    self.call_from_thread(
                        self._set_loading, 0,
                        f"axllm exited with code {self._axllm_proc.returncode}",
                    )
                    self._axllm_proc = None
                    return

        self.call_from_thread(self._set_loading, last_pct[0], "Model server did not start in time")

    def _stop_server(self) -> None:
        """Terminate the axllm serve subprocess if we started it."""
        if self._axllm_proc and self._axllm_proc.poll() is None:
            self._axllm_proc.terminate()
            try:
                self._axllm_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._axllm_proc.kill()
            self._axllm_proc = None

    # ── helpers (main-thread only) ──────────────────────────────────────
    def _set_status(self, msg: str) -> None:
        try:
            self.query_one("#status-bar", Label).update(f" {msg}")
        except Exception:
            pass

    def _set_analysis(self, markup: str) -> None:
        try:
            self.query_one("#analysis-text", Static).update(markup)
            self.query_one("#analysis-box").scroll_end(animate=False)
        except Exception:
            pass

    def _set_camera(self, img: Image.Image) -> None:
        try:
            self.query_one("#camera-view", ImageWidget).set_image(img)
        except Exception:
            pass

    # ── camera ──────────────────────────────────────────────────────────
    def _init_camera(self) -> None:
        if not HAS_CAMERA:
            self.call_from_thread(
                self._set_loading, 0,
                "picamera2 not available — running in test-pattern mode",
            )
            return
        try:
            self.call_from_thread(self._set_loading, 0, "Initialising camera…")
            self.camera = Picamera2()
            cfg = self.camera.create_video_configuration(
                main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
            )
            self.camera.configure(cfg)
            self.camera.start()
            # Allow auto-exposure / white-balance to settle
            time.sleep(2)
            self.call_from_thread(self._set_loading, 0, "Camera ready ✓")
        except Exception as exc:
            self.camera = None
            self.call_from_thread(self._set_loading, 0, f"Camera error: {exc}")

    def _capture_frame(self) -> Image.Image:
        """Grab one frame (camera or synthetic test pattern)."""
        if self.camera:
            return Image.fromarray(self.camera.capture_array())
        # Synthetic gradient for testing without a camera
        r = np.linspace(0, 255, CAPTURE_WIDTH, dtype=np.uint8)[np.newaxis, :]
        r = np.broadcast_to(r, (CAPTURE_HEIGHT, CAPTURE_WIDTH)).copy()
        g = np.linspace(0, 255, CAPTURE_HEIGHT, dtype=np.uint8)[:, np.newaxis]
        g = np.broadcast_to(g, (CAPTURE_HEIGHT, CAPTURE_WIDTH)).copy()
        b = np.full((CAPTURE_HEIGHT, CAPTURE_WIDTH), 128, dtype=np.uint8)
        return Image.fromarray(np.stack([r, g, b], axis=-1))

    @staticmethod
    def _encode_jpeg_b64(img: Image.Image) -> str:
        """Resize to model input size and return base64 JPEG."""
        resized = img.resize((384, 384), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # ── main loop (runs inside the startup worker thread) ─────────────
    def _run_analysis(self) -> None:
        while self._running:
            if self.paused:
                time.sleep(0.5)
                continue
            self._cycle()
            # Short pause between cycles
            time.sleep(0.5)

    def _cycle(self) -> None:
        """One capture → analyse → display cycle (runs in worker thread)."""
        try:
            # ── capture ─────────────────────────────────────────────────
            self.call_from_thread(self._set_status, "Capturing…")
            img = self._capture_frame()
            self._capture_num += 1
            cap = self._capture_num

            self.call_from_thread(self._set_camera, img)
            self.call_from_thread(
                self._set_analysis,
                f"[bold]Capture #{cap}[/bold]\n[dim]Sending to model…[/dim]",
            )

            # ── encode & send ───────────────────────────────────────────
            self.call_from_thread(self._set_status, "Analysing…")
            b64 = self._encode_jpeg_b64(img)

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            },
                        },
                    ],
                },
            ]

            t0 = time.time()

            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                max_tokens=512,
            )

            # ── stream tokens ───────────────────────────────────────────
            tokens = 0
            t_first: float | None = None
            full_text = ""
            last_ui = 0.0

            for event in stream:
                if not self._running:
                    return
                delta = getattr(event.choices[0], "delta", None)
                if delta and getattr(delta, "content", None):
                    chunk = delta.content
                    if t_first is None:
                        t_first = time.time()
                    tokens += 1
                    full_text += chunk

                    # Throttle UI updates to ~7 fps
                    now = time.time()
                    if now - last_ui > 0.15:
                        escaped = full_text.replace("[", "\\[")
                        self.call_from_thread(
                            self._set_analysis,
                            f"[bold]Capture #{cap}[/bold]\n\n{escaped}",
                        )
                        last_ui = now

            # ── final update ────────────────────────────────────────────
            escaped = full_text.replace("[", "\\[")
            self.call_from_thread(
                self._set_analysis,
                f"[bold]Capture #{cap}[/bold]\n\n{escaped}",
            )

            elapsed = time.time() - t0
            if t_first and tokens:
                ttft = t_first - t0
                gen_time = time.time() - t_first
                tps = tokens / gen_time if gen_time > 0 else 0
                self.call_from_thread(
                    self._set_status,
                    f"#{cap}  |  {elapsed:.1f}s total  |  TTFT {ttft:.1f}s  "
                    f"|  {tps:.1f} tok/s  |  {tokens} tokens",
                )
            else:
                self.call_from_thread(
                    self._set_status,
                    f"#{cap}  |  {elapsed:.1f}s  |  (no tokens received)",
                )

        except Exception as exc:
            msg = str(exc)
            if "onnect" in msg or "refused" in msg:
                self.call_from_thread(
                    self._set_status,
                    f"Cannot connect to {self.api_url} – is axllm serve running?",
                )
                self.call_from_thread(
                    self._set_analysis,
                    "[bold red]Connection Error[/bold red]\n\n"
                    f"Cannot reach the model server at {self.api_url}\n\n"
                    "Start the server first:\n"
                    "[bold]axllm serve /path/to/qBc_Ai/Qwen3.5-4B/[/bold]",
                )
            else:
                self.call_from_thread(self._set_status, f"Error: {msg}")
                escaped = msg.replace("[", "\\[")
                self.call_from_thread(
                    self._set_analysis,
                    f"[bold red]Error[/bold red]\n\n{escaped}",
                )
            time.sleep(5)

    # ── key bindings ────────────────────────────────────────────────────
    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.paused:
            self._set_status("Paused  —  press  p  to resume")
        else:
            self._set_status("Resumed")

    def action_force_capture(self) -> None:
        if self.paused:
            self.paused = False
            self._set_status("Resumed (forced)")


# ── entry point ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="qB Companion Vision – continuous AI surroundings analysis"
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"axllm serve base URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name for the API",
    )
    parser.add_argument(
        "--model-dir",
        default=MODEL_DIR,
        help=f"Path to model directory (default: {MODEL_DIR})",
    )
    args = parser.parse_args()

    app = VisionApp(api_url=args.api_url, model=args.model, model_dir=args.model_dir)
    app.run()


if __name__ == "__main__":
    main()
