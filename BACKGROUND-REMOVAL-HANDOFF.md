# Background Removal tab — October 9, 2026

The new tab is separate from the four creation tabs. app.py adds optional adapter
download, source-video upload, optional replacement background, processing size,
clip range, transparent export, and one-click handoff from each creation output.
Existing creation request validation and sampling paths are unchanged.

Adapter: Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen,
ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors, 1308787472 bytes.
Pinned upstream revision: 6184df14b1b560bf9b6447d3ee126bd7e4a88513.
SHA256: d9e143f979e0756f0c83d772e6a8890f4c6b933db1f7d93fc06e25c179f3fd36.
Download uses the customer's accepted Hugging Face access and optional read
token. The app does not save this token. Adapter download is optional and never
part of the standard model downloads. models/loras is private and preserved.

The tab now provides an explicit three-step gated-access flow: open the model
page and accept/request access, supply a same-account read token (or reuse the
PC's existing HF login), then Check access. A HEAD request for the exact pinned
weight file confirms access and expected metadata without downloading weights.
Only successful checks enable Download. The download repeats that check before
starting. No local "I agreed" checkbox substitutes for actual server permission.
Invalid credentials, missing/pending permission, and network errors have
distinct sanitized messages. Editing the token disables Download until checked
again. Browser login alone cannot authenticate the PC app. Tokens are never
saved by this app and are cleared from the field when downloading begins.

The existing persistent Engine reuses its clean distilled INT8 base, text
encoder and VAEs. Alpha Gen is a separate cached patch at strength 1.0; the
exact previously selected model/LoRA is restored in finally, including failures.
Releasing models, restarting, or OOM ends the worker and clears those caches.
Ordinary Comfy RAM/VRAM offloading may still occur between operations.

Matting uses empty text, full native processing resolution in stage 1, reference
strength 1.0, LTXVAddGuide with reference_downscale_factor 1, the existing
distilled eight-step schedule, and LTXVCropGuides before decoding. There is no
latent upscale/detail pass. The selected LoRA in Settings is not applied to this
path. Source video is normalized to its frame rate, resized proportionally, and
padded to 32-pixel multiples. Exports remove that padding. Sections have at most
145 frames, tail frames are padded to 8n+1, and only original frames are written.
Joins between independently generated sections may need review.

matte_tools.py streams bounded-size sections and exports a mask MP4, preview MP4
(checkerboard or provided background), and optional alpha-channel VP9 WebM.
Preview/WebM keep the selected source audio. Saved exports have unique names
under outputs; processing material is preserved in outputs/matte-work.
imageio-ffmpeg was added as a bundled-executable fallback for PCs without ffmpeg.
The existing updater installs changed requirements, so no installer BAT change
is necessary to obtain this feature.

Verified without GPU inference:
- UI builds with the six correct tab IDs and four handoff buttons.
- Four existing creation-mode validation tests pass.
- Four new CPU checks pass: frame padding/145+5 chunking, portrait shape,
  one-click tab selection, real MP4/WebM exports with audio and decoded alpha,
  and a simulated sampling path proving cache reuse/LoRA restoration on failure.
- CPU checks cover 17-frame and 150-frame inputs and unique result names.

## Real GPU smoke test — October 9

The user granted Hugging Face access for gjnave and authorized closing FireRed.
FireRed's identified server/worker tree was stopped; VRAM usage fell from about
20 GB to 0.9 GB. The adapter was downloaded and its exact size/SHA256 verified.
This installation uses Python 3.10.11; checksum verification now reads chunks
instead of the Python-3.11-only hashlib.file_digest.

The first actual run completed sampling but failed during tiled VAE decoding:
the core's inference-mode tiler returned an inference tensor to an in-place
normalization. safe_vae_output in core_worker.py clones inference tensors to
normal tensors before applying the original normalization. Video/audio VAE
wrappers preserve upstream normalization and avoid changing the bundled core.
A regression check verifies numerical output and that the original tensor is
not mutated.

RTX 4090, existing distilled convrot INT8 model, empty prompt, Alpha Gen 1.0,
512x320, 17 source frames at 24 fps (~0.71 seconds), eight-step matte:
- Cold matte: 54.22 s wall time, including worker/base/LoRA loading.
- Warm matte: 30.47 s. No base/LoRA loading events on the second request.
- Return to ordinary image-video path, LoRA off: 26.49 s, no base reload event.
- Source/mask/preview/WebM frame counts matched (17), original audio remained.
- VP9 alpha was decoded and verified: 0..255 range; actual transparent regions.
- Visual review showed isolated subject, microphone, and hair edges, with teal
  background replacement. This is a short smoke test, not broad quality proof.

Evidence: outputs/alpha-smoke-20261009/gpu-smoke-report.json, results.json,
alpha-comparison.png, source.mp4. Export paths are recorded in results.json.
The first failed report is preserved in gpu-smoke-first-error.json.
Test worker was stopped afterward; VRAM returned to about 0.9 GB.
Still unverified: long-video GPU section joins and other GPUs/resolutions.
Production publication/installer rebuild has not been performed for this feature.
