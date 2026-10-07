"""Private persistent worker for the branded Gradio interface."""

from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from model_config import missing_files


HERE = Path(__file__).resolve().parent
CORE_ROOT = HERE / "vendor" / "comfy_core"
MODEL_ROOT = HERE / "models"
OUTPUT_ROOT = HERE / "outputs"


def model_status() -> str:
    missing = missing_files(MODEL_ROOT)
    if missing:
        return "Missing model files (run the installer):\n" + "\n".join(
            str(path.relative_to(HERE)) for path in missing
        )
    return "All five required model files are present."


class Worker:
    def __init__(self):
        self.process = None
        self.log = None
        self.lines: queue.Queue[str] = queue.Queue()
        self.lock = threading.RLock()

    def _reader(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put("")

    def _next(self, timeout: int | None) -> dict:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError("The LTX worker stopped responding; see outputs/core-worker.log.") from error
        if not line:
            raise RuntimeError("The LTX worker exited; see outputs/core-worker.log.")
        return json.loads(line)

    def start(self):
        if self.process and self.process.poll() is None:
            return
        if not (CORE_ROOT / "nodes.py").is_file():
            raise RuntimeError("Bundled inference core is missing. Reinstall the app files.")
        if missing_files(MODEL_ROOT):
            raise RuntimeError(model_status())
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        self.log = (OUTPUT_ROOT / "core-worker.log").open("a", encoding="utf-8")
        self.lines = queue.Queue()
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(HERE / "core_worker.py"),
             "--core-root", str(CORE_ROOT), "--model-root", str(MODEL_ROOT)],
            cwd=HERE, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            encoding="utf-8", text=True, bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        threading.Thread(target=self._reader, daemon=True).start()
        try:
            ready = self._next(120)
            if not ready.get("ready"):
                raise RuntimeError(ready.get("error", "LTX worker did not start."))
        except Exception:
            self.stop()
            raise

    def request(self, payload: dict, on_event=None) -> dict:
        with self.lock:
            self.start()
            try:
                self.process.stdin.write(json.dumps(payload) + "\n")
                self.process.stdin.flush()
                while True:
                    # Long custom clips can take over 30 minutes in a single stage.
                    # EOF still reports a crashed worker immediately.
                    answer = self._next(None)
                    if "event" in answer:
                        if on_event:
                            on_event(str(answer["event"]))
                        continue
                    if not answer.get("ok"):
                        if answer.get("kind") == "memory":
                            self.stop()
                            raise RuntimeError(
                                "GPU or system memory ran out. The inference worker was "
                                "released; your installed models and saved videos are safe. "
                                "Try a shorter clip, smaller output size, or disable the "
                                "high-detail pass. Original error: "
                                + str(answer.get("error", "memory allocation failed"))
                            )
                        raise RuntimeError(answer.get("error", "LTX generation failed."))
                    return answer
            except (OSError, TimeoutError, json.JSONDecodeError):
                self.stop()
                raise
            except RuntimeError as error:
                if "worker exited" in str(error).lower():
                    self.stop()
                    raise RuntimeError(
                        "The inference worker exited unexpectedly. It may have run "
                        "out of GPU or system memory. Try a shorter clip, smaller "
                        "output size, or disable the high-detail pass; the next "
                        "generation will start a fresh worker."
                    ) from error
                raise

    def stop(self):
        with self.lock:
            process = self.process
            self.process = None
            if process and process.poll() is None:
                try:
                    process.stdin.write('{"command":"stop"}\n')
                    process.stdin.flush()
                    process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
            if self.log:
                self.log.close()
                self.log = None


WORKER = Worker()
atexit.register(WORKER.stop)
