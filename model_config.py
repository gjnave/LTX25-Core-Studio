"""Model filenames for the LTX 2.5 app-local Comfy core."""

from pathlib import Path


DIFFUSION = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
TEXT_ENCODER = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
SPATIAL_UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

MODEL_FILES = {
    "diffusion_models": DIFFUSION,
    "text_encoders": TEXT_ENCODER,
    "vae": (VIDEO_VAE, AUDIO_VAE),
    "latent_upscale_models": SPATIAL_UPSCALER,
}

EXPECTED_SIZES = {
    DIFFUSION: 21504034224,
    TEXT_ENCODER: 15372969374,
    VIDEO_VAE: 1472223346,
    AUDIO_VAE: 364866540,
    SPATIAL_UPSCALER: 995778752,
}


def required_files(root: Path) -> list[Path]:
    files = []
    for directory, names in MODEL_FILES.items():
        for name in (names,) if isinstance(names, str) else names:
            files.append(root / directory / name)
    return files


def missing_files(root: Path) -> list[Path]:
    return [
        path for path in required_files(root)
        if not path.is_file() or path.stat().st_size != EXPECTED_SIZES[path.name]
    ]
