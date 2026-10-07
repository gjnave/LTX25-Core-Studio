"""GGF Spokesman — talking avatars powered by local LTX 2.5 inference."""

from __future__ import annotations

import os
import shutil
import threading
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

import gradio as gr

from audio_tools import audio_info, parse_time, read_selection
from network_settings import read_settings, save_settings, verify_login, launch_access_servers, install_upload_disconnect_handling
from runtime import MODEL_ROOT, OUTPUT_ROOT, WORKER, model_status


TITLE = "GGF Spokesman"
LOCAL_VERSION = (Path(__file__).with_name("VERSION")).read_text(encoding="utf-8").strip()
VERSION_URLS = (
    "https://codeberg.org/Cognibuild/LTX25-Core-Studio/raw/branch/main/VERSION",
    "https://raw.githubusercontent.com/gjnave/LTX25-Core-Studio/main/VERSION",
)
ACTIVE_MODE = "local"
ACTIVE_URL = ""
ACTIVE_LOCAL_URL = ""
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
DURATION_HELP = "Choose a preset or type seconds (90) or minutes:seconds (2:00). Longer clips need more memory and time."


def duration_frames(duration):
    """Accept existing presets or a custom duration; align to LTX's 8*k+1 frames."""
    if duration in DURATIONS:
        return DURATIONS[duration]
    try:
        seconds = parse_time(duration, "Clip length")
        if seconds is None or seconds < 1:
            raise ValueError("Clip length must be at least one second.")
        return round(seconds * 3) * 8 + 1
    except (ValueError, OverflowError) as error:
        raise gr.Error(str(error)) from error


def lora_choices():
    root = MODEL_ROOT / "loras"
    return [("Off · base model", "")] + [
        (path.name, path.name) for path in sorted(root.glob("*.safetensors")) if path.is_file()
    ]


def refresh_loras(selected):
    choices = lora_choices()
    return gr.update(choices=choices, value=selected if selected in dict(choices).values() else "")


def import_lora(upload):
    if not upload:
        raise gr.Error("Choose an LTX 2.5 LoRA .safetensors file first.")
    source = Path(upload)
    if source.suffix.lower() != ".safetensors":
        raise gr.Error("Only .safetensors LoRAs are supported.")
    try:
        from safetensors import safe_open
        with safe_open(str(source), framework="pt", device="cpu") as weights:
            if not any("lora" in key.lower() for key in weights.keys()):
                raise ValueError("This file does not contain recognizable LoRA weights.")
        root = MODEL_ROOT / "loras"
        root.mkdir(parents=True, exist_ok=True)
        destination = root / f"{source.stem}-{uuid.uuid4().hex[:8]}.safetensors"
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    except Exception as error:
        raise gr.Error(f"Could not import LoRA: {error}") from error
    return gr.update(choices=lora_choices(), value=destination.name), "LoRA imported and selected. Add any required trigger words to your generation prompt."


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
        samples, sample_rate, (first, last, _) = read_selection(path, start, end)
    except Exception as error:
        raise gr.Error(str(error)) from error
    return gr.update(value=(sample_rate, samples), visible=True), f"Previewing {first:.2f}–{last:.2f} seconds."


def duration_advice(duration):
    try:
        frames = duration_frames(duration)
    except gr.Error:
        return "Enter clip length in seconds (90) or minutes:seconds (2:00)."
    if frames > 241:
        return (
            "**Long clip (one continuous pass).** Longer videos use more GPU "
            "and system memory than a 10-second clip, take several minutes, "
            "or fail. Try Fast size and turn off the detail pass first. "
            "A memory error releases the worker so you can retry."
        )
    return "Choose a preset through one minute, or type your own longer duration."


def generate(image, audio, audio_start, audio_end, prompt, size, duration, detail_pass, seed, lora_name="", lora_strength=1.0, progress=gr.Progress()):
    if not image:
        raise gr.Error("Upload a first-frame image.")
    width, height = SIZES[size]
    frames = duration_frames(duration)
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
            "lora_name": lora_name or "",
            "lora_strength": lora_strength,
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


def _generate_new_mode(mode, image, audio, audio_start, audio_end, prompt,
                       size, duration, detail_pass, seed, progress, lora_name="", lora_strength=1.0):
    if mode == "text_video" and not (prompt or "").strip():
        raise gr.Error("Describe the video you want to create.")
    if mode == "audio_video" and not audio:
        raise gr.Error("Upload an audio track.")
    width, height = SIZES[size]
    frames = duration_frames(duration)
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
            "mode": mode,
            "lora_name": lora_name or "",
            "lora_strength": lora_strength,
            "image_path": image or "",
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


def generate_text(prompt, size, duration, detail_pass, seed, lora_name="", lora_strength=1.0, progress=gr.Progress()):
    return _generate_new_mode("text_video", "", "", "0", "", prompt,
                              size, duration, detail_pass, seed, progress, lora_name, lora_strength)


def generate_audio_only(audio, audio_start, audio_end, prompt,
                        size, duration, detail_pass, seed, lora_name="", lora_strength=1.0, progress=gr.Progress()):
    return _generate_new_mode("audio_video", "", audio, audio_start, audio_end,
                              prompt, size, duration, detail_pass, seed, progress, lora_name, lora_strength)


def generate_image_only(image, prompt, size, duration, detail_pass, seed,
                        lora_name="", lora_strength=1.0, progress=gr.Progress()):
    if not image:
        raise gr.Error("Upload a first-frame image.")
    return _generate_new_mode("image_audio", image, "", "0", "", prompt,
                              size, duration, detail_pass, seed, progress, lora_name, lora_strength)


def release_models():
    WORKER.stop()
    return "Models released from GPU memory. Saved videos remain on disk."


def check_for_updates():
    from update_app import check_update
    return check_update()


def update_and_restart():
    from update_app import start_restart
    if not WORKER.lock.acquire(blocking=False):
        return 'Finish the current generation before updating.'
    try:
        WORKER.stop()
        start_restart()
    except Exception as error:
        WORKER.lock.release()
        return f'Update could not start: {error}'
    timer=threading.Timer(3,lambda: os._exit(0))
    timer.daemon=True
    timer.start()
    return 'Updating and restarting. The local app will reopen; temporary public phone links may change.'


def save_network_mode(mode, username, password):
    try:
        saved = save_settings(mode, username, password)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    label = {"local": "Local only", "lan": "Local network", "public": "Temporary public link"}[saved["mode"]]
    login_note = f"Remote login: **{saved['username']}**." if saved["digest"] else "**No login required.**"
    return f"Saved **{label}**. {login_note} Close this server and start `run.bat` again to apply it. The current link stays unchanged until restart.", ""


def active_network_status():
    label = {"local": "Local only", "lan": "Local network", "public": "Temporary public link"}[ACTIVE_MODE]
    return (f"**Running mode:** {label}  \n**Local app (no login):** {ACTIVE_LOCAL_URL or 'Starting...'}  \n"
            f"**Shared address:** {ACTIVE_URL if ACTIVE_MODE != 'local' and ACTIVE_URL != ACTIVE_LOCAL_URL else 'Not active'}")


CSS = """
body, .gradio-container { background: #071020 !important; color: #e9f1ff !important; }
.gradio-container { max-width: 1440px !important; }
.ggf-hero { padding: 26px 32px; border: 1px solid #37516e; border-radius: 18px;
  background: linear-gradient(110deg,#0d1c32,#192451); margin: 8px 0 22px; }
.ggf-kicker { color: #ffbf42; letter-spacing: .15em; font-size: 12px; font-weight: 800; }
.ggf-title { color: #ffb536; font-size: 42px; font-weight: 800; margin: 5px 0; }
.ggf-sub { color: #dce9ff; font-size: 16px; margin: 6px 0 16px; }
.ggf-links a { color: #89caff !important; margin-right: 23px; font-weight: 650; }
.ggf-powered, .ggf-powered p { color: #9fb1cb !important; font-size: 12px; }
#generate { background: #ffbb40 !important; color: #081321 !important; font-weight: 800 !important; }
#studio-tabs [role="tablist"] { gap: 6px; padding: 8px 0 12px; border-color: #37516e; }
#studio-tabs button[role="tab"] {
  color: #e9f1ff !important; background: #172842 !important;
  border: 1px solid #45607f !important; border-radius: 8px;
  padding: 10px 16px; min-height: 44px; font-weight: 650; opacity: 1;
}
#studio-tabs button[role="tab"][aria-selected="true"] {
  color: #081321 !important; background: #ffbb40 !important; border-color: #ffbb40 !important;
}
#studio-tabs button[role="tab"]:hover { border-color: #ffbb40 !important; }
#studio-tabs button[role="tab"]:focus-visible { outline: 2px solid #89caff; outline-offset: 2px; }
#studio-tabs .overflow-menu > button,
#studio-tabs .overflow-dropdown button {
  color: #e9f1ff !important; background: #172842 !important;
  min-height: 44px; border-radius: 6px; padding: 10px 12px;
}
#studio-tabs .overflow-dropdown { background: #172842 !important; border: 1px solid #45607f !important; }
#studio-tabs .overflow-dropdown button:hover { background: #284365 !important; }
.ggf-intro, .ggf-intro h3, .ggf-intro p { color: #e9f1ff !important; }
#studio-tabs > .tabitem { border-color: #37516e; }
@media (max-width: 600px) {
  #studio-tabs button[role="tab"] { padding: 10px 12px; font-size: 14px; }
  .ggf-hero { padding: 18px; margin-bottom: 12px; }
  .ggf-title { font-size: 30px; }
}
"""


def build_demo(public_preview=False):
    install_upload_disconnect_handling()
    with gr.Blocks(title=f"Get Going Fast · {TITLE}") as demo:
        gr.HTML(f"""
        <div class="ggf-hero">
          <div class="ggf-kicker">GET GOING FAST · LOCAL AI</div>
          <div class="ggf-title">{TITLE}</div>
          <div class="ggf-sub">Bring a person or character to life with your photo and voice.</div>
          <div class="ggf-links"><a href="https://getgoingfast.pro" target="_blank">GetGoingFast.pro ↗</a>
          <a href="https://youtube.com/@theaihobbyguy" target="_blank">TheAIHobbyGuy on YouTube ↗</a></div>
        </div>""")
        if public_preview:
            gr.Markdown("**Temporary public preview:** no login is required. Do not upload private media to this shared test link.")
        with gr.Tabs(selected="lip-sync", elem_id="studio-tabs"):
          with gr.Tab("Lip Sync", id="lip-sync", elem_id="tab-lip-sync"):
            gr.Markdown("### Lip Sync\nAnimate a person or character using a reference image and speech or music. Use only media you have the right and consent to edit.", elem_classes="ggf-intro")
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(label="Person or character · reference image", type="filepath", sources=["upload", "clipboard"], height=320)
                with gr.Column(scale=1):
                    audio = gr.Audio(label="Speech or music · leave blank to generate sound", type="filepath", sources=["upload"], format="wav", editable=False)
                    audio_details = gr.Markdown(audio_description(None))
                    with gr.Row():
                        audio_start = gr.Textbox(label="Start time", value="0", placeholder="0 or 1:12.5")
                        audio_end = gr.Textbox(label="End time · blank = file end", value="", placeholder="e.g. 1:17.5")
                    with gr.Row():
                        preview_button = gr.Button("Preview selected audio")
                        audio_preview_note = gr.Markdown("")
                    audio_preview = gr.Audio(label="Selected audio preview", interactive=False, visible=False)
            with gr.Row():
                size = gr.Dropdown(label="Output size", choices=list(SIZES), value="Fast · 768 × 512")
                duration = gr.Dropdown(label="Clip length", choices=list(DURATIONS), value="4 seconds · 97 frames", allow_custom_value=True, info=DURATION_HELP)
            with gr.Accordion("Motion and quality (optional)", open=False):
                prompt = gr.Textbox(label="Motion and sound prompt · optional", lines=3,
                                placeholder="Blank lets the model choose motion from your reference image.")
                detail_pass = gr.Checkbox(label="High-detail second pass (slower, full selected size)", value=True)
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                duration_note = gr.Markdown(duration_advice("4 seconds · 97 frames"))
            duration.change(duration_advice, inputs=duration, outputs=duration_note)
            generate_button = gr.Button("Generate lip-sync video", variant="primary", elem_id="generate")
            output = gr.Video(label="Output", interactive=False)
            audio.change(audio_description, inputs=audio, outputs=audio_details)
            preview_button.click(preview_audio, [audio, audio_start, audio_end], [audio_preview, audio_preview_note])
          with gr.Tab("Audio to Video", id="audio-video", elem_id="tab-audio-video"):
            gr.Markdown("### Audio to Video\nUpload audio and describe the visuals. The original audio is kept in the result.", elem_classes="ggf-intro")
            audio_only = gr.Audio(label="Audio track", type="filepath", sources=["upload"], format="wav", editable=False)
            audio_only_prompt = gr.Textbox(label="What should the video show?", lines=3,
                placeholder="A singer performs on a warmly lit stage, moving in time with the music.")
            with gr.Accordion("Trim audio (optional)", open=False):
                with gr.Row():
                    audio_only_start = gr.Textbox(label="Start time", value="0", placeholder="0 or 1:12.5")
                    audio_only_end = gr.Textbox(label="End time · blank = file end", value="")
                audio_only_preview_button = gr.Button("Preview selected audio")
                audio_only_preview = gr.Audio(label="Selected audio preview", interactive=False, visible=False)
                audio_only_preview_note = gr.Markdown("")
            with gr.Row():
                audio_only_size = gr.Dropdown(label="Shape and size", choices=list(SIZES), value="Fast · 768 × 512")
                audio_only_duration = gr.Dropdown(label="Clip length", choices=list(DURATIONS), value="4 seconds · 97 frames", allow_custom_value=True, info=DURATION_HELP)
            with gr.Accordion("Optional quality and seed", open=False):
                audio_only_detail = gr.Checkbox(label="Full-size detail pass (turn off for a half-size quick preview)", value=True)
                audio_only_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
            audio_only_generate = gr.Button("Create video from audio", variant="primary", elem_id="generate-audio")
            audio_only_output = gr.Video(label="Your video", interactive=False)
            audio_only_preview_button.click(preview_audio, [audio_only, audio_only_start, audio_only_end],
                                            [audio_only_preview, audio_only_preview_note])
          with gr.Tab("Text to Video", id="text-video", elem_id="tab-text-video"):
            gr.Markdown("### Text to Video\nDescribe a short scene; the app creates video and sound. No reference image is needed.", elem_classes="ggf-intro")
            text_prompt = gr.Textbox(label="What should happen?", lines=4,
                placeholder="A golden retriever runs through a sunny meadow. The camera follows at ground level. Birds chirp softly.")
            with gr.Row():
                text_size = gr.Dropdown(label="Shape and size", choices=list(SIZES), value="Fast · 768 × 512")
                text_duration = gr.Dropdown(label="Clip length", choices=list(DURATIONS), value="4 seconds · 97 frames", allow_custom_value=True, info=DURATION_HELP)
            with gr.Accordion("Optional quality and seed", open=False):
                text_detail = gr.Checkbox(label="Full-size detail pass (turn off for a half-size quick preview)", value=True)
                text_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
            text_generate = gr.Button("Create video", variant="primary", elem_id="generate-text")
            text_output = gr.Video(label="Your video", interactive=False)
          with gr.Tab("Image to Video", id="image-video", elem_id="tab-image-video"):
            gr.Markdown("### Image to Video\nUpload a still image and describe how it should move. Sound is generated automatically.", elem_classes="ggf-intro")
            image_only = gr.Image(label="Starting image", type="filepath", sources=["upload", "clipboard"], height=320)
            image_prompt = gr.Textbox(label="Motion (optional)", lines=3,
                placeholder="The subject turns toward the camera while a light breeze moves their hair.")
            with gr.Row():
                image_size = gr.Dropdown(label="Shape and size", choices=list(SIZES), value="Fast · 768 × 512")
                image_duration = gr.Dropdown(label="Clip length", choices=list(DURATIONS), value="4 seconds · 97 frames", allow_custom_value=True, info=DURATION_HELP)
            with gr.Accordion("Optional quality and seed", open=False):
                image_detail = gr.Checkbox(label="Full-size detail pass (turn off for a half-size quick preview)", value=True)
                image_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
            image_generate = gr.Button("Animate image", variant="primary", elem_id="generate-image")
            image_output = gr.Video(label="Your video", interactive=False)
          with gr.Tab("Settings", id="settings", elem_id="tab-settings"):
            gr.Markdown("### Settings\nOptional LoRA, app updates, network access, and GPU status.", elem_classes="ggf-intro")
            with gr.Accordion("LoRA (optional · all video tabs)", open=True):
                gr.Markdown("Use an **LTX 2.5-compatible model LoRA**. Off keeps the original model unchanged. Your selection applies to every video tab in this browser session. Add the LoRA's trigger words to your prompt if required; other model families are not supported.")
                lora_name = gr.Dropdown(label="LoRA", choices=lora_choices(), value="")
                lora_strength = gr.Slider(label="LoRA strength · 0 disables it", minimum=-2, maximum=2, value=1.0, step=0.05)
                gr.Button("Refresh LoRA list").click(refresh_loras, lora_name, lora_name)
                gr.Markdown("Upload a .safetensors file below, or copy it into the app's `models/loras` folder and refresh the list. LoRAs can increase memory use and change quality.")
                lora_upload = gr.File(label="Upload LoRA", file_types=[".safetensors"], type="filepath")
                lora_note = gr.Markdown("")
                gr.Button("Import and select LoRA").click(import_lora, lora_upload, [lora_name, lora_note], concurrency_limit=1)
            with gr.Accordion("Generation details and GPU memory", open=False):
                details = gr.Textbox(label="Last saved file and timing", lines=3, interactive=False)
                status = gr.Textbox(label="Model status", value=model_status, lines=7, interactive=False)
                with gr.Row():
                    gr.Button("Refresh model status").click(model_status, outputs=status)
                    if not public_preview:
                        gr.Button("Release models / free GPU memory").click(release_models, outputs=status)
            with gr.Accordion("App updates", open=False):
                update_result = gr.Markdown(f"Installed version: **{LOCAL_VERSION}**")
                gr.Button("Check for updates").click(check_for_updates, outputs=update_result)
                gr.Button("Update and restart").click(update_and_restart, outputs=update_result, queue=False)
            if not public_preview:
                with gr.Accordion("Network access", open=False):
                    gr.Markdown("Saving a change requires a server restart. Leave the password blank to turn login off.")
                    current_network = gr.Markdown(active_network_status())
                    gr.Button("Refresh current address").click(active_network_status, outputs=current_network)
                    settings = read_settings()
                    network_mode = gr.Dropdown(label="Access mode", choices=[("This computer only", "local"), ("Local network (LAN)", "lan"), ("Temporary public link", "public")], value=settings["mode"])
                    network_user = gr.Textbox(label="Username for LAN or public access", value=settings["username"], max_lines=1)
                    network_password = gr.Textbox(label="Password · blank means no login", type="password", value="", max_lines=1)
                    gr.Markdown("The local app always opens without a login. LAN/public access uses a separate address with your saved login. LAN access may prompt for a Windows Firewall rule. Public access uses a temporary Gradio share link. Remote passwords are stored locally as salted hashes, not included in the ZIP.")
                    network_saved = gr.Markdown("")
                    gr.Button("Save network settings").click(save_network_mode, [network_mode, network_user, network_password], [network_saved, network_password])
        gr.Markdown("Powered by **LTX 2.5** from Lightricks. Bundled inference components retain their upstream licenses; model weights download separately. Outputs are saved locally in `outputs`.", elem_classes="ggf-powered")
        generate_button.click(generate, [image, audio, audio_start, audio_end, prompt, size, duration, detail_pass, seed, lora_name, lora_strength], [output, details], concurrency_limit=1)
        text_generate.click(generate_text, [text_prompt, text_size, text_duration, text_detail, text_seed, lora_name, lora_strength], [text_output, details], concurrency_limit=1)
        image_generate.click(generate_image_only, [image_only, image_prompt, image_size, image_duration, image_detail, image_seed, lora_name, lora_strength], [image_output, details], concurrency_limit=1)
        audio_only_generate.click(generate_audio_only,
            [audio_only, audio_only_start, audio_only_end, audio_only_prompt, audio_only_size,
             audio_only_duration, audio_only_detail, audio_only_seed, lora_name, lora_strength], [audio_only_output, details], concurrency_limit=1)
        if not public_preview:
            demo.load(active_network_status, outputs=current_network)
    return demo


if __name__ == "__main__":
    Path(__file__).with_name('app.lock').write_text(str(os.getpid()))
    settings = read_settings()
    ACTIVE_MODE = settings["mode"]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("GRADIO_SERVER_PORT") or "7861")
    auth = None if ACTIVE_MODE == "local" or not settings["digest"] else lambda username, password: verify_login(username, password, settings)
    local_demo, remote_demo, local_url, remote_url = launch_access_servers(
        lambda: build_demo().queue(), mode=ACTIVE_MODE, preferred_port=port,
        inbrowser=True, auth=auth, show_error=True,
        prevent_thread_lock=True, css=CSS, theme=gr.themes.Base(),
        allowed_paths=[str(OUTPUT_ROOT.resolve())],
    )
    ACTIVE_LOCAL_URL = local_url
    ACTIVE_URL = remote_url or local_url
    print(f"{TITLE} running in {ACTIVE_MODE} mode at {ACTIVE_URL}", flush=True)
    timeout = int(os.environ.get("LTX_PUBLIC_MAX_SECONDS", "0")) if ACTIVE_MODE == "public" else 0
    if timeout > 0 and remote_demo is not None:
        threading.Timer(timeout, remote_demo.close).start()
    threading.Event().wait()
