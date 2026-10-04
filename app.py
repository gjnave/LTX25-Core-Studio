"""Get Going Fast LTX 2.5 Core Studio — local, serverless inference UI."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import gradio as gr

from audio_tools import audio_info, read_selection
from network_settings import read_settings, save_settings, verify_login
from runtime import OUTPUT_ROOT, WORKER, model_status


TITLE = "LTX 2.5 Core Studio"
LOCAL_VERSION = (Path(__file__).with_name("VERSION")).read_text(encoding="utf-8").strip()
VERSION_URLS = (
    "https://codeberg.org/Cognibuild/LTX25-Core-Studio/raw/branch/main/VERSION",
    "https://raw.githubusercontent.com/gjnave/LTX25-Core-Studio/main/VERSION",
)
ACTIVE_MODE = "local"
ACTIVE_URL = ""
SIZES = {
    "Fast · 768 × 512": (768, 512),
    "Fast portrait · 512 × 768": (512, 768),
    "Landscape · 1024 × 576": (1024, 576),
    "Landscape · 1152 × 640": (1152, 640),
    "Landscape · 1280 × 704": (1280, 704),
    "Portrait · 768 × 1152": (768, 1152),
}
DURATIONS = {
    "2 seconds · 49 frames": 49,
    "4 seconds · 97 frames": 97,
    "5 seconds · 121 frames": 121,
    "6 seconds · 145 frames": 145,
    "8 seconds · 193 frames": 193,
    "10 seconds · 241 frames": 241,
    "15 seconds · 361 frames (experimental)": 361,
    "20 seconds · 481 frames (experimental)": 481,
    "30 seconds · 721 frames (experimental)": 721,
    "45 seconds · 1081 frames (experimental)": 1081,
    "60 seconds · 1441 frames (experimental)": 1441,
}


def audio_description(path):
    if not path:
        return "Upload a song, then type a start and end time below."
    try:
        _, seconds = audio_info(path)
        return f"Uploaded audio: **{seconds:.2f} seconds**. Times accept `12.5` or `1:12.5`."
    except Exception as error:
        return f"Could not read the uploaded audio: {error}"


def preview_audio(path, start, end):
    if not path:
        raise gr.Error("Upload an audio file first.")
    try:
        samples, sample_rate, (first, last, _) = read_selection(
            path, start, end, max_seconds=60,
        )
    except Exception as error:
        raise gr.Error(str(error)) from error
    return gr.update(value=(sample_rate, samples), visible=True), f"Previewing {first:.2f}–{last:.2f} seconds."


def duration_advice(duration):
    if DURATIONS[duration] > 241:
        return (
            "⚠️ **Experimental long clip (one continuous pass).** 15–60 seconds can use much more GPU "
            "and system memory than a 10-second clip, take several minutes, "
            "or fail. Try Fast size and turn off the detail pass first. "
            "A memory error releases the worker so you can retry."
        )
    return "Longer clips (15–60 seconds) are available but experimental."


def generate(image, audio, audio_start, audio_end, prompt, size, duration, detail_pass, seed, progress=gr.Progress()):
    if not image:
        raise gr.Error("Upload a first-frame image.")
    width, height = SIZES[size]
    frames = DURATIONS[duration]
    progress(0.02, desc="Starting app-local LTX core")
    stages = {
        "Loading": 0.08, "Encoding": 0.2, "Generating": 0.3,
        "Upscaling": 0.65, "Decoding": 0.9,
    }

    def on_event(message):
        value = next((amount for key, amount in stages.items() if message.startswith(key)), 0.15)
        progress(value, desc=message)

    try:
        result = WORKER.request({
            "image_path": image,
            "audio_path": audio or "",
            "audio_start": audio_start,
            "audio_end": audio_end,
            "prompt": (prompt or "").strip(),
            "width": width,
            "height": height,
            "frames": frames,
            "fps": 24,
            "seed": int(seed),
            "detail_pass": bool(detail_pass),
            "output_dir": str(OUTPUT_ROOT),
        }, on_event)
    except Exception as error:
        raise gr.Error(str(error)) from error
    progress(1, desc="Saved")
    source = "uploaded audio" if result["used_uploaded_audio"] else "model-generated audio"
    details = (
        f"Saved: {result['output']}\n"
        f"{result['width']} × {result['height']} · {result['frames']} frames at {result['fps']} fps · {source}\n"
        f"Seed: {result['seed']} · Total: {result['total_seconds']:.1f} s "
        f"(initial model load: {result['load_seconds']:.1f} s)"
    )
    return result["output"], details


def release_models():
    WORKER.stop()
    return "Models released from GPU memory. Saved videos remain on disk."


def check_for_updates():
    for url in VERSION_URLS:
        try:
            request = Request(url, headers={"User-Agent": "LTX25-Core-Studio-update-check"})
            with urlopen(request, timeout=8) as response:
                remote = response.read(80).decode("utf-8").strip()
            if not remote or len(remote) > 40 or any(c not in "0123456789." for c in remote):
                continue
            if tuple(map(int, remote.split("."))) > tuple(map(int, LOCAL_VERSION.split("."))):
                return (f"**Update available: {remote}** (installed: {LOCAL_VERSION}). "
                        "Close the app, then run `3-UPDATE-LTX25-Core-Studio.bat` beside the app folder. "
                        "Your models and outputs will be kept.")
            return f"Up to date: **{LOCAL_VERSION}**."
        except (OSError, UnicodeError, ValueError):
            continue
    return "Could not reach the update mirrors. Your installed app is unchanged; try again later."


def save_network_mode(mode, username, password):
    try:
        saved = save_settings(mode, username, password)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    label = {"local": "Local only", "lan": "Local network", "public": "Temporary public link"}[saved["mode"]]
    return f"Saved **{label}**. Close this server and start `run.bat` again to apply it. The current link stays unchanged until restart.", ""


def active_network_status():
    label = {"local": "Local only", "lan": "Local network", "public": "Temporary public link"}[ACTIVE_MODE]
    address = ACTIVE_URL or "Starting..."
    return f"**Running mode:** {label}  \n**Current address:** {address}"


CSS = """
body, .gradio-container { background: #071020 !important; color: #e9f1ff !important; }
.gradio-container { max-width: 1440px !important; }
.ggf-hero { padding: 26px 32px; border: 1px solid #37516e; border-radius: 18px;
  background: linear-gradient(110deg,#0d1c32,#192451); margin: 8px 0 22px; }
.ggf-kicker { color: #ffbf42; letter-spacing: .15em; font-size: 12px; font-weight: 800; }
.ggf-title { color: #ffb536; font-size: 42px; font-weight: 800; margin: 5px 0; }
.ggf-sub { color: #dce9ff; font-size: 16px; margin: 6px 0 16px; }
.ggf-links a { color: #89caff !important; margin-right: 23px; font-weight: 650; }
#generate { background: #ffbb40 !important; color: #081321 !important; font-weight: 800 !important; }
"""


def build_demo():
    with gr.Blocks(title=f"Get Going Fast · {TITLE}") as demo:
        gr.HTML("""
        <div class="ggf-hero">
          <div class="ggf-kicker">GET GOING FAST · LOCAL AI</div>
          <div class="ggf-title">LTX 2.5 Core Studio</div>
          <div class="ggf-sub">Image-to-video with synchronized audio · app-local inference · no ComfyUI installation or server</div>
          <div class="ggf-links"><a href="https://getgoingfast.pro" target="_blank">GetGoingFast.pro ↗</a>
          <a href="https://youtube.com/@theaihobbyguy" target="_blank">TheAIHobbyGuy on YouTube ↗</a></div>
        </div>""")
        with gr.Tabs():
          with gr.Tab("Create"):
            gr.Markdown("Use only images and audio you have the right and consent to edit.")
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(label="First-frame image · required", type="filepath", sources=["upload", "clipboard"], height=320)
                with gr.Column(scale=1):
                    audio = gr.Audio(label="Audio track · optional (leave blank for model-generated audio)", type="filepath", sources=["upload"], format="wav", editable=False)
                    audio_details = gr.Markdown(audio_description(None))
                    with gr.Row():
                        audio_start = gr.Textbox(label="Start time", value="0", placeholder="0 or 1:12.5")
                        audio_end = gr.Textbox(label="End time · blank = file end", value="", placeholder="e.g. 1:17.5")
                    with gr.Row():
                        preview_button = gr.Button("Preview selected audio")
                        audio_preview_note = gr.Markdown("")
                    audio_preview = gr.Audio(label="Selected audio preview", interactive=False, visible=False)
            generate_button = gr.Button("Generate video", variant="primary", elem_id="generate")
            output = gr.Video(label="Output", interactive=False)
            audio.change(audio_description, inputs=audio, outputs=audio_details)
            preview_button.click(preview_audio, [audio, audio_start, audio_end], [audio_preview, audio_preview_note])
          with gr.Tab("Settings"):
            with gr.Accordion("Generation options", open=True):
                prompt = gr.Textbox(label="Motion and sound prompt · optional", lines=3,
                                placeholder="Blank lets the model choose motion from the first frame.")
                with gr.Row():
                    size = gr.Dropdown(label="Output size", choices=list(SIZES), value="Fast · 768 × 512")
                    duration = gr.Dropdown(label="Clip length", choices=list(DURATIONS), value="4 seconds · 97 frames")
                    seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                detail_pass = gr.Checkbox(label="High-detail second pass (slower, full selected size)", value=True)
                duration_note = gr.Markdown(duration_advice("4 seconds · 97 frames"))
                duration.change(duration_advice, inputs=duration, outputs=duration_note)
            with gr.Accordion("Generation details and GPU memory", open=False):
                details = gr.Textbox(label="Last saved file and timing", lines=3, interactive=False)
                status = gr.Textbox(label="Model status", value=model_status, lines=7, interactive=False)
                with gr.Row():
                    gr.Button("Refresh model status").click(model_status, outputs=status)
                    gr.Button("Release models / free GPU memory").click(release_models, outputs=status)
            with gr.Accordion("App updates", open=False):
                update_result = gr.Markdown(f"Installed version: **{LOCAL_VERSION}**")
                gr.Button("Check for updates").click(check_for_updates, outputs=update_result)
            with gr.Accordion("Network access", open=False):
                gr.Markdown("Saving a change requires a server restart. Anyone with the login can change these settings, so use a private password.")
                current_network = gr.Markdown(active_network_status())
                gr.Button("Refresh current address").click(active_network_status, outputs=current_network)
                settings = read_settings()
                network_mode = gr.Dropdown(label="Access mode", choices=[("This computer only", "local"), ("Local network (LAN)", "lan"), ("Temporary public link", "public")], value=settings["mode"])
                network_user = gr.Textbox(label="Username for LAN or public access", value=settings["username"], max_lines=1)
                network_password = gr.Textbox(label="Password (leave blank to keep existing password)", type="password", value="", max_lines=1)
                gr.Markdown("LAN access listens on all network interfaces and may prompt for a Windows Firewall rule. Public access uses a temporary Gradio share link. Both require a username and a password of at least 12 characters. Passwords are stored locally as salted hashes, not included in the ZIP.")
                network_saved = gr.Markdown("")
                gr.Button("Save network settings").click(save_network_mode, [network_mode, network_user, network_password], [network_saved, network_password])
            gr.Markdown("LTX 2.5 model by Lightricks. Bundled inference components retain their upstream licenses; model weights download separately. Outputs are saved locally in `outputs`.")
        generate_button.click(generate, [image, audio, audio_start, audio_end, prompt, size, duration, detail_pass, seed], [output, details], concurrency_limit=1)
        demo.load(active_network_status, outputs=current_network)
    return demo


if __name__ == "__main__":
    settings = read_settings()
    ACTIVE_MODE = settings["mode"]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    port = int(os.environ["GRADIO_SERVER_PORT"]) if os.environ.get("GRADIO_SERVER_PORT") else None
    access = {"local": "127.0.0.1", "lan": "0.0.0.0", "public": "127.0.0.1"}[ACTIVE_MODE]
    auth = None if ACTIVE_MODE == "local" else lambda username, password: verify_login(username, password, settings)
    demo = build_demo().queue()
    _, local_url, public_url = demo.launch(
        server_name=access, server_port=port, inbrowser=ACTIVE_MODE == "local",
        share=ACTIVE_MODE == "public", auth=auth, show_error=True,
        prevent_thread_lock=True, css=CSS, theme=gr.themes.Base(),
        allowed_paths=[str(OUTPUT_ROOT.resolve())],
    )
    ACTIVE_URL = public_url or local_url
    print(f"LTX 2.5 Core Studio running in {ACTIVE_MODE} mode at {ACTIVE_URL}", flush=True)
    if ACTIVE_MODE == "public" and not public_url:
        demo.close()
        raise RuntimeError("Public share link could not be created. Nothing was exposed without a login.")
    timeout = int(os.environ.get("LTX_PUBLIC_MAX_SECONDS", "0")) if ACTIVE_MODE == "public" else 0
    if timeout > 0:
        threading.Timer(timeout, demo.close).start()
    threading.Event().wait()
