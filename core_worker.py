"""Serverless, app-local LTX 2.5 inference using selected Comfy core modules.

JSON lines on stdout are the private protocol. All upstream diagnostics go to
stderr. No ComfyUI server, frontend, custom-node manager, or external checkout
is imported or launched.
"""

from __future__ import annotations

import argparse
import json
import math
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


def safe_vae_output(torch, process_output):
    """Core tiled decode returns inference tensors to an in-place normalization."""
    def normalize(image):
        if torch.is_inference(image):
            with torch.inference_mode(False), torch.no_grad():
                return process_output(image.clone())
        return process_output(image)
    return normalize


def validate_request(request: dict) -> dict:
    mode = str(request.get("mode") or "image_audio")
    if mode not in {"image_audio", "text_video", "audio_video"}:
        raise ValueError("Unknown generation mode.")
    prompt = str(request.get("prompt") or "").strip()
    image_value = str(request.get("image_path") or "").strip()
    image_path = Path(image_value) if image_value else None
    if mode == "image_audio" and (image_path is None or not image_path.is_file()):
        raise ValueError("Upload a first-frame image.")
    if image_path is not None and not image_path.is_file():
        raise ValueError("The first-frame image was not found.")
    if mode in {"text_video", "audio_video"} and not prompt:
        raise ValueError("Describe the video you want to create.")
    width = int(request.get("width", 1024))
    height = int(request.get("height", 576))
    frames = int(request.get("frames", 97))
    fps = int(request.get("fps", 24))
    if not (256 <= width <= 1536 and 256 <= height <= 1536):
        raise ValueError("Width and height must be between 256 and 1536 pixels.")
    if width % 64 or height % 64:
        raise ValueError("Width and height must be divisible by 64 for the two-stage workflow.")
    if frames < 9 or (frames - 1) % 8:
        raise ValueError("Frames must be 8*k+1, with at least 9 frames.")
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
    if mode == "audio_video" and not audio_path:
        raise ValueError("Upload an audio track.")
    audio_start = request.get("audio_start", "0")
    audio_end = request.get("audio_end", "")
    output_dir = Path(str(request.get("output_dir") or "")).resolve()
    lora_name = str(request.get("lora_name") or "")
    lora_strength = float(request.get("lora_strength", 1.0))
    if not math.isfinite(lora_strength) or not -2 <= lora_strength <= 2:
        raise ValueError("LoRA strength must be between -2 and 2.")
    if lora_name and (Path(lora_name).name != lora_name or ":" in lora_name
                      or Path(lora_name).suffix.lower() != ".safetensors"):
        raise ValueError("Select a .safetensors LoRA from models/loras.")
    return {
        "mode": mode, "prompt": prompt,
        "image_path": str(image_path) if image_path else "", "audio_path": audio_path,
        "audio_start": audio_start, "audio_end": audio_end,
        "width": width, "height": height, "frames": frames, "fps": fps,
        "seed": seed, "detail_pass": bool(request.get("detail_pass", True)),
        "output_dir": output_dir,
        "lora_name": lora_name, "lora_strength": lora_strength,
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
            LTXVAddGuide, LTXVCropGuides,
        )
        from comfy_extras.nodes_lt_audio import LTXVAudioVAEDecode, LTXVEmptyLatentAudio, LTXVAudioVAEEncode
        from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
        from comfy_extras.nodes_video import CreateVideo, SaveVideo

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this app's Python environment.")
        # SaveVideo normally receives prompt metadata from Comfy's graph runner.
        # This app calls it directly, so there is no graph metadata to attach.
        core_args.disable_metadata = True
        for kind in ("diffusion_models", "text_encoders", "vae", "latent_upscale_models", "loras"):
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
        self.LTXVAddGuide = LTXVAddGuide
        self.LTXVCropGuides = LTXVCropGuides
        self.LTXVAudioVAEDecode = LTXVAudioVAEDecode
        self.LTXVEmptyLatentAudio = LTXVEmptyLatentAudio
        self.LTXVAudioVAEEncode = LTXVAudioVAEEncode
        self.LTXVLatentUpsampler = LTXVLatentUpsampler
        self.CreateVideo = CreateVideo
        self.SaveVideo = SaveVideo
        self.model_root = model_root
        self.model = self.clip = self.video_vae = self.audio_vae = self.upscaler = None
        self.base_model = None
        self.active_lora_key = None
        self.load_seconds = 0.0
        self.text_cache = {}
        self.alpha_model = None
        self.alpha_model_key = None

    def load_models(self, event) -> None:
        if self.model is not None:
            return
        missing = missing_files(self.model_root)
        if missing:
            raise RuntimeError("Missing LTX 2.5 models. Run the installer:\n" + "\n".join(map(str, missing)))
        started = time.perf_counter()
        event("Loading the app-local INT8 model and VAEs")
        self.model = self.nodes.UNETLoader().load_unet(DIFFUSION, "default")[0]
        self.base_model = self.model
        self.clip = self.nodes.CLIPLoader().load_clip(TEXT_ENCODER, "ltxv", "default")[0]
        self.video_vae = self.nodes.VAELoader().load_vae(VIDEO_VAE)[0]
        self.audio_vae = self.nodes.VAELoader().load_vae(AUDIO_VAE)[0]
        self.video_vae.process_output = safe_vae_output(self.torch,self.video_vae.process_output)
        self.audio_vae.process_output = safe_vae_output(self.torch,self.audio_vae.process_output)
        self.upscaler = self.LatentUpscaleModelLoader.execute(SPATIAL_UPSCALER)[0]
        self.load_seconds = time.perf_counter() - started

    def select_lora(self, cfg, event):
        name, strength = cfg["lora_name"], cfg["lora_strength"]
        if not name or strength == 0:
            self.model = self.base_model
            self.active_lora_key = None
            return
        root = (self.model_root / "loras").resolve()
        path = (root / name).resolve()
        if path.parent != root or not path.is_file():
            raise ValueError("Selected LoRA was not found in models/loras. Refresh the LoRA list in Settings.")
        key = (str(path), path.stat().st_mtime_ns, path.stat().st_size, strength)
        if self.active_lora_key == key:
            return
        event(f"Loading LoRA: {name} (strength {strength:g})")
        # Always patch the clean base; switching adapters never stacks old LoRAs.
        self.model = self.base_model
        self.active_lora_key = None
        patched = self.nodes.LoraLoaderModelOnly().load_lora_model_only(self.base_model, name, strength)[0]
        if sum(map(len, patched.patches.values())) <= sum(map(len, self.base_model.patches.values())):
            raise ValueError("No compatible model weights matched this LoRA. Use an LTX 2.5-compatible model LoRA.")
        self.model = patched
        self.active_lora_key = key

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
        if request.get("mode") == "alpha_matte":
            return self.generate_matte(request, event)
        cfg = validate_request(request)
        started = time.perf_counter()
        self.load_models(event)
        self.select_lora(cfg, event)
        image = load_image(cfg["image_path"]) if cfg["image_path"] else None
        source_audio = load_audio(
            cfg["audio_path"], cfg["frames"], cfg["fps"],
            cfg["audio_start"], cfg["audio_end"],
        ) if cfg["audio_path"] else None
        cfg["output_dir"].mkdir(parents=True, exist_ok=True)
        self.folder_paths.set_output_directory(str(cfg["output_dir"]))
        # INT8 weight offloading re-wraps quantized tensors as Parameters.
        # inference_mode makes those tensors incompatible with that operation.
        # no_grad retains inference without creating inference-only tensors.
        with self.torch.no_grad():
            event("Encoding references and text" if image is not None else "Encoding text")
            positive, negative = self.text_conditioning(cfg["prompt"], cfg["fps"])
            preprocessed = self.LTXVPreprocess.execute(image, 18)[0] if image is not None else None
            latent_video = self.EmptyLTXVLatentVideo.execute(
                cfg["width"] // 2, cfg["height"] // 2, cfg["frames"], 1,
            )[0]
            if preprocessed is not None:
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
                if preprocessed is not None:
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

    def generate_matte(self, request: dict, event) -> dict:
        from matte_tools import (
            ALPHA_FILE, adapter_ready, prepare_video, video_chunks,
            padded_frames, MatteExports,
        )
        import uuid
        started = time.perf_counter()
        source = str(request.get('video_path') or '')
        if not source or not Path(source).is_file():
            raise ValueError('Upload a video and wait for its preview first.')
        if not adapter_ready(self.model_root):
            raise ValueError('Download the optional Background Removal adapter in its tab first.')
        start, duration = float(request.get('start',0)), float(request.get('duration',0))
        short_edge = int(request.get('short_edge',384))
        if short_edge not in {0,384,512,768}:
            raise ValueError('Select a Background Removal size from its menu.')
        seed = int(request.get('seed',-1))
        if seed < 0:
            seed = random.randrange(2**63)
        if seed >= 2**63:
            raise ValueError('Seed is too large.')
        output_root = Path(request['output_dir']).resolve()
        output_root.mkdir(parents=True,exist_ok=True)
        prefix = f'GGF-Spokesman-matte-{int(time.time())}-{uuid.uuid4().hex[:10]}'
        work = output_root/'matte-work'/prefix
        work.mkdir(parents=True)
        event('Preparing your video for background removal')
        meta = prepare_video(source,work/'source.mkv',short_edge,start,duration)
        self.load_models(event)
        previous_model, previous_key = self.model, self.active_lora_key
        exporter = None
        try:
            path = self.model_root/'loras'/ALPHA_FILE
            alpha_key = (str(path),path.stat().st_mtime_ns,path.stat().st_size,1.0)
            if self.alpha_model_key != alpha_key:
                self.select_lora({'lora_name':ALPHA_FILE,'lora_strength':1.0},event)
                self.alpha_model, self.alpha_model_key = self.model, alpha_key
            else:
                self.model, self.active_lora_key = self.alpha_model, alpha_key
            with self.torch.no_grad():
                # Alpha Gen requires empty text, full-resolution guidance, and one stage.
                empty = self.nodes.CLIPTextEncode().encode(self.clip,'')[0]
                base_positive, base_negative = self.LTXVConditioning.execute(empty,empty,float(meta['fps']))
                exporter = MatteExports(work,output_root,prefix,meta,
                                        request.get('background_path'),bool(request.get('transparent',True)))
                for index, rgb in enumerate(video_chunks(work/'source.mkv')):
                    count = len(rgb)
                    frames = padded_frames(rgb)
                    length = len(frames)
                    event(f'Generating background mask · section {index+1} · {count} frames')
                    reference = self.torch.from_numpy(frames.copy()).float().div_(255)
                    latent = self.EmptyLTXVLatentVideo.execute(meta['width'],meta['height'],length,1)[0]
                    positive, negative, latent = self.LTXVAddGuide.execute(
                        base_positive,base_negative,self.video_vae,latent,reference,0,1.0,
                        iclora_parameters={'reference_downscale_factor':1})
                    audio = self.LTXVEmptyLatentAudio.execute(length,float(meta['fps']),1,self.audio_vae)[0]
                    av_latent = self.LTXVConcatAVLatent.execute(latent,audio)[0]
                    sampled = self.sample(av_latent,positive,negative,FIRST_PASS_SIGMAS,seed)
                    video_latent, _ = self.LTXVSeparateAVLatent.execute(sampled)
                    _, _, video_latent = self.LTXVCropGuides.execute(positive,negative,video_latent)
                    event(f'Decoding background mask · section {index+1}')
                    pixels = self.nodes.VAEDecodeTiled().decode(
                        self.video_vae,video_latent,tile_size=512,overlap=64,
                        temporal_size=64,temporal_overlap=8)[0]
                    if pixels.ndim == 5:
                        pixels = pixels[0]
                    if len(pixels) < count:
                        raise RuntimeError('The matte has fewer frames than the source section.')
                    exporter.write(rgb,pixels[:count].float().cpu().numpy())
                    del reference, latent, audio, av_latent, sampled, video_latent, pixels, positive, negative
                event('Saving your preview and background-removal downloads')
                result = exporter.finish(source,start)
                exporter = None
            return {**result,'total_seconds':round(time.perf_counter()-started,2),'seed':seed}
        finally:
            # Restore the exact previous adapter/base; keep Alpha Gen cached for reuse.
            self.model, self.active_lora_key = previous_model, previous_key
            if exporter is not None:
                exporter.close(abort=True)


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
