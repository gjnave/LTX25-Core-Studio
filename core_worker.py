"""Serverless, app-local LTX 2.5 inference using selected Comfy core modules.

JSON lines on stdout are the private protocol. All upstream diagnostics go to
stderr. No ComfyUI server, frontend, custom-node manager, or external checkout
is imported or launched.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

from audio_tools import read_selection
from model_config import (
    AUDIO_VAE, DIFFUSION, SPATIAL_UPSCALER, TEXT_ENCODER, VIDEO_VAE,
    missing_files,
)


FIRST_PASS_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
DETAIL_PASS_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
NEGATIVE_PROMPT = "video game, cartoon, childish, ugly, distorted, low quality"


def validate_request(request: dict) -> dict:
    prompt = str(request.get("prompt") or "").strip()
    image_path = Path(str(request.get("image_path") or ""))
    if not image_path.is_file():
        raise ValueError("Upload a first-frame image.")
    width = int(request.get("width", 1024))
    height = int(request.get("height", 576))
    frames = int(request.get("frames", 97))
    fps = int(request.get("fps", 24))
    if not (256 <= width <= 1536 and 256 <= height <= 1536):
        raise ValueError("Width and height must be between 256 and 1536 pixels.")
    if width % 64 or height % 64:
        raise ValueError("Width and height must be divisible by 64 for the two-stage workflow.")
    if frames < 9 or frames > 1441 or (frames - 1) % 8:
        raise ValueError("Frames must be 8*k+1, between 9 and 1441 (one minute).")
    if fps != 24:
        raise ValueError("This workflow currently uses 24 fps.")
    seed = int(request.get("seed", -1))
    if seed < 0:
        seed = random.randrange(2**63)
    if seed >= 2**63:
        raise ValueError("Seed is too large.")
    audio_path = str(request.get("audio_path") or "").strip()
    if audio_path and not Path(audio_path).is_file():
        raise ValueError("Uploaded audio file was not found.")
    audio_start = request.get("audio_start", "0")
    audio_end = request.get("audio_end", "")
    output_dir = Path(str(request.get("output_dir") or "")).resolve()
    return {
        "prompt": prompt, "image_path": str(image_path), "audio_path": audio_path,
        "audio_start": audio_start, "audio_end": audio_end,
        "width": width, "height": height, "frames": frames, "fps": fps,
        "seed": seed, "detail_pass": bool(request.get("detail_pass", True)),
        "output_dir": output_dir,
    }


def load_image(path: str):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        if source.width * source.height > 32_000_000 or max(source.size) > 8192:
            raise ValueError("Image exceeds 32 megapixels or 8192 pixels on one side.")
        source.load()
        image = ImageOps.exif_transpose(source).convert("RGB")
        if max(image.size) > 1536:
            image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        return torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255).unsqueeze(0)


def load_audio(path: str, frames: int, fps: int, start, end):
    import torch

    audio, sample_rate, _ = read_selection(
        path, start, end, max_seconds=frames / fps,
    )
    waveform = torch.from_numpy(audio.T.copy())
    if waveform.ndim != 2 or waveform.numel() == 0:
        raise ValueError("Audio has no decodable samples.")
    if waveform.shape[0] > 2:
        waveform = waveform[:2]
    duration_samples = round(sample_rate * frames / fps)
    if waveform.shape[1] > duration_samples:
        waveform = waveform[:, :duration_samples]
    elif waveform.shape[1] < duration_samples:
        waveform = torch.nn.functional.pad(waveform, (0, duration_samples - waveform.shape[1]))
    return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}


class Engine:
    def __init__(self, core_root: Path, model_root: Path):
        if not (core_root / "nodes.py").is_file() or not (core_root / "comfy_extras" / "nodes_lt.py").is_file():
            raise RuntimeError(f"Bundled LTX core is missing: {core_root}")
        os.chdir(core_root)
        sys.path.insert(0, str(core_root))
        import torch
        from comfy.cli_args import args as core_args
        import folder_paths
        import nodes
        from comfy_extras.nodes_custom_sampler import KSamplerSelect, ManualSigmas, RandomNoise, SamplerCustomAdvanced
        from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
        from comfy_extras.nodes_lt import (
            EmptyLTXVLatentVideo, LTXVConcatAVLatent, LTXVConditioning,
            LTXVDualCFGGuider, LTXVImgToVideoInplace, LTXVPreprocess,
            LTXVSeparateAVLatent,
        )
        from comfy_extras.nodes_lt_audio import LTXVAudioVAEDecode, LTXVEmptyLatentAudio, LTXVAudioVAEEncode
        from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
        from comfy_extras.nodes_video import CreateVideo, SaveVideo

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this app's Python environment.")
        # SaveVideo normally receives prompt metadata from Comfy's graph runner.
        # This app calls it directly, so there is no graph metadata to attach.
        core_args.disable_metadata = True
        for kind in ("diffusion_models", "text_encoders", "vae", "latent_upscale_models"):
            folder_paths.add_model_folder_path(kind, str(model_root / kind), is_default=True)
        self.torch = torch
        self.folder_paths = folder_paths
        self.nodes = nodes
        self.KSamplerSelect = KSamplerSelect
        self.ManualSigmas = ManualSigmas
        self.RandomNoise = RandomNoise
        self.SamplerCustomAdvanced = SamplerCustomAdvanced
        self.LatentUpscaleModelLoader = LatentUpscaleModelLoader
        self.EmptyLTXVLatentVideo = EmptyLTXVLatentVideo
        self.LTXVConcatAVLatent = LTXVConcatAVLatent
        self.LTXVConditioning = LTXVConditioning
        self.LTXVDualCFGGuider = LTXVDualCFGGuider
        self.LTXVImgToVideoInplace = LTXVImgToVideoInplace
        self.LTXVPreprocess = LTXVPreprocess
        self.LTXVSeparateAVLatent = LTXVSeparateAVLatent
        self.LTXVAudioVAEDecode = LTXVAudioVAEDecode
        self.LTXVEmptyLatentAudio = LTXVEmptyLatentAudio
        self.LTXVAudioVAEEncode = LTXVAudioVAEEncode
        self.LTXVLatentUpsampler = LTXVLatentUpsampler
        self.CreateVideo = CreateVideo
        self.SaveVideo = SaveVideo
        self.model_root = model_root
        self.model = self.clip = self.video_vae = self.audio_vae = self.upscaler = None
        self.load_seconds = 0.0
        self.text_cache = {}

    def load_models(self, event) -> None:
        if self.model is not None:
            return
        missing = missing_files(self.model_root)
        if missing:
            raise RuntimeError("Missing LTX 2.5 models. Run the installer:\n" + "\n".join(map(str, missing)))
        started = time.perf_counter()
        event("Loading the app-local INT8 model and VAEs")
        self.model = self.nodes.UNETLoader().load_unet(DIFFUSION, "default")[0]
        self.clip = self.nodes.CLIPLoader().load_clip(TEXT_ENCODER, "ltxv", "default")[0]
        self.video_vae = self.nodes.VAELoader().load_vae(VIDEO_VAE)[0]
        self.audio_vae = self.nodes.VAELoader().load_vae(AUDIO_VAE)[0]
        self.upscaler = self.LatentUpscaleModelLoader.execute(SPATIAL_UPSCALER)[0]
        self.load_seconds = time.perf_counter() - started

    def text_conditioning(self, prompt: str, fps: int):
        key = (prompt, fps)
        if key not in self.text_cache:
            positive = self.nodes.CLIPTextEncode().encode(self.clip, prompt)[0]
            negative = self.nodes.CLIPTextEncode().encode(self.clip, NEGATIVE_PROMPT)[0]
            self.text_cache[key] = self.LTXVConditioning.execute(positive, negative, fps)
            if len(self.text_cache) > 2:
                self.text_cache.pop(next(iter(self.text_cache)))
        return self.text_cache[key]

    def sample(self, latent: dict, positive, negative, sigmas: str, seed: int):
        guider = self.LTXVDualCFGGuider.execute(self.model, positive, negative, 1.0, 1.0)[0]
        noise = self.RandomNoise.execute(seed)[0]
        sampler = self.KSamplerSelect.execute("euler_ancestral")[0]
        schedule = self.ManualSigmas.execute(sigmas)[0]
        return self.SamplerCustomAdvanced.execute(noise, guider, sampler, schedule, latent)[0]

    def generate(self, request: dict, event) -> dict:
        cfg = validate_request(request)
        started = time.perf_counter()
        self.load_models(event)
        image = load_image(cfg["image_path"])
        source_audio = load_audio(
            cfg["audio_path"], cfg["frames"], cfg["fps"],
            cfg["audio_start"], cfg["audio_end"],
        ) if cfg["audio_path"] else None
        cfg["output_dir"].mkdir(parents=True, exist_ok=True)
        self.folder_paths.set_output_directory(str(cfg["output_dir"]))
        with self.torch.inference_mode():
            event("Encoding image and text")
            positive, negative = self.text_conditioning(cfg["prompt"], cfg["fps"])
            preprocessed = self.LTXVPreprocess.execute(image, 18)[0]
            latent_video = self.EmptyLTXVLatentVideo.execute(
                cfg["width"] // 2, cfg["height"] // 2, cfg["frames"], 1,
            )[0]
            latent_video = self.LTXVImgToVideoInplace.execute(
                self.video_vae, preprocessed, latent_video, 0.7, False,
            )[0]
            empty_audio = self.LTXVEmptyLatentAudio.execute(
                cfg["frames"], cfg["fps"], 1, self.audio_vae,
            )[0]
            if source_audio:
                encoded_audio = self.LTXVAudioVAEEncode.execute(source_audio, self.audio_vae)[0]
                expected = empty_audio["samples"]
                actual = encoded_audio["samples"]
                if actual.shape[2] > expected.shape[2]:
                    actual = actual[:, :, :expected.shape[2], :]
                elif actual.shape[2] < expected.shape[2]:
                    actual = self.torch.nn.functional.pad(actual, (0, 0, 0, expected.shape[2] - actual.shape[2]))
                encoded_audio["samples"] = actual
                encoded_audio["noise_mask"] = self.torch.zeros_like(actual)
                latent_audio = encoded_audio
            else:
                latent_audio = empty_audio
            av_latent = self.LTXVConcatAVLatent.execute(latent_video, latent_audio)[0]
            event("Generating video and audio — first pass")
            sampled = self.sample(av_latent, positive, negative, FIRST_PASS_SIGMAS, cfg["seed"])
            video_latent, audio_latent = self.LTXVSeparateAVLatent.execute(sampled)
            if cfg["detail_pass"]:
                event("Upscaling and refining — detail pass")
                video_latent = self.LTXVLatentUpsampler.execute(video_latent, self.upscaler, self.video_vae)[0]
                video_latent = self.LTXVImgToVideoInplace.execute(
                    self.video_vae, preprocessed, video_latent, 1.0, False,
                )[0]
                av_latent = self.LTXVConcatAVLatent.execute(video_latent, audio_latent)[0]
                sampled = self.sample(av_latent, positive, negative, DETAIL_PASS_SIGMAS, 42)
                video_latent, audio_latent = self.LTXVSeparateAVLatent.execute(sampled)
            event("Decoding and saving MP4")
            pixels = self.nodes.VAEDecodeTiled().decode(
                self.video_vae, video_latent, tile_size=768, overlap=64,
                temporal_size=4096, temporal_overlap=32,
            )[0]
            output_audio = source_audio or self.LTXVAudioVAEDecode.execute(audio_latent, self.audio_vae)[0]
            video = self.CreateVideo.execute(pixels, cfg["fps"], output_audio, 8, "sRGB", "none")[0]
            prefix = f"ltx25-core-{int(time.time())}-seed-{cfg['seed']}"
            self.SaveVideo.execute(video, prefix, {"format": "mp4", "codec": {"codec": "h264"}})
        matches = sorted(cfg["output_dir"].glob(prefix + "*.mp4"), key=lambda p: p.stat().st_mtime_ns)
        if not matches:
            raise RuntimeError("Video encoder returned without creating an MP4.")
        return {
            "output": str(matches[-1]), "seed": cfg["seed"],
            "frames": cfg["frames"], "fps": cfg["fps"],
            "width": int(pixels.shape[2]), "height": int(pixels.shape[1]),
            "used_uploaded_audio": bool(source_audio),
            "load_seconds": round(self.load_seconds, 2),
            "total_seconds": round(time.perf_counter() - started, 2),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = sys.stdout
    sys.stdout = sys.stderr
    sys.argv = [sys.argv[0], "--models-directory", str(args.model_root.resolve())]
    try:
        engine = Engine(args.core_root.resolve(), args.model_root.resolve())
    except Exception as error:
        protocol.write(json.dumps({"ready": False, "error": str(error)}) + "\n")
        protocol.flush()
        traceback.print_exc()
        return 1
    protocol.write('{"ready": true}\n')
    protocol.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "stop":
                break
            def event(message):
                protocol.write(json.dumps({"event": message}) + "\n")
                protocol.flush()
            result = engine.generate(request, event)
            protocol.write(json.dumps({"ok": True, **result}) + "\n")
        except Exception as error:
            traceback.print_exc()
            message = str(error)
            memory_error = isinstance(error, MemoryError) or any(
                phrase in message.lower() for phrase in (
                    "out of memory", "not enough memory", "allocation failed",
                    "cannot allocate memory", "cuda error: out of memory",
                )
            )
            protocol.write(json.dumps({
                "ok": False, "error": message,
                "kind": "memory" if memory_error else "error",
            }) + "\n")
        protocol.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
