# Get Going Fast · LTX 2.5 Core Studio

Local Gradio video generation with four simple creation modes: Lip Sync
(image + audio), Text to Video, Image to Video, and Audio to Video. The audio
mode keeps the uploaded track in the MP4; the text and image modes generate
synchronized audio with LTX 2.5. All four modes use the same five model files.
No ComfyUI installation or server is required: this app bundles only the
Python inference components it calls.

The Windows installer downloads this source into a `LTX25-Core-Studio` folder
beside `installer.bat` and `run.bat`. Its two-stage image-to-video
path follows the active part of the supplied `LTX2.5.json`: distilled INT8
model, 8-step first pass, latent 2× upscale, and 3-step detail pass. The
optional uploaded-audio path is an added feature, not present in that active
workflow branch.

## Install and run

Double-click the parent-folder `installer.bat`. The complete ZIP uses its
bundled app files; an installer-only copy downloads source from Codeberg,
then GitHub if needed, then a Google Drive fallback when publicly available.
You do not need to copy this repository beside the installer.
It creates a private `.venv`, installs
CUDA PyTorch and the other dependencies, and downloads the five required model
files. You must first accept the [LTX 2.5 model access terms](https://huggingface.co/Lightricks/LTX-2.5).
If asked to authenticate, run the indicated `hf auth login` command, then
rerun the installer. Afterward, double-click the parent-folder `run.bat`
or `LTX25-Core-Studio\run.bat`. To update, close the app and run the
parent-folder `3-UPDATE-LTX25-Core-Studio.bat`. The updater keeps downloaded
models, generated videos, and local network settings.

The model weights are **not** bundled with the app source. The installer
downloads them into `LTX25-Core-Studio\models`. Saved MP4 files and the
worker log go into `LTX25-Core-Studio\outputs`.

The Lip Sync tab preserves the original first-frame image, optional audio,
Start/End trim times, Generate button, and video output in one short flow. Its
optional motion prompt, output size, clip length, seed, and high-detail pass
remain under Settings → Lip Sync options. The audio preview appears only
when requested.

Text to Video needs only a scene description. Image to Video needs a starting
image and can use an optional motion description. Audio to Video needs an audio
file plus a visual description; its original audio is retained. These three
tabs show only short 2–10 second lengths to make first runs predictable.
Their optional quality/seed controls are collapsed by default. The full-size
detail pass is on by default; turn it off for a quicker, half-size preview.

Turning off the high-detail pass skips the latent-upscale/detail pass and
outputs at half the selected width and height. The two-stage mode outputs the
full selected size. Frames use the LTX `8*k+1` rule at 24 fps. Uploaded audio
is trimmed or padded to the selected duration. On a phone, use the Start/End
time fields below the audio upload instead of waveform handles. Times accept
seconds or `mm:ss`; the Preview button plays the selected range. The prompt
is optional: blank text lets the model infer motion from the first frame.

Fast portrait is 512 × 768, matching Fast landscape's pixel count. Each output
video appears directly below its Generate button; saved-file and timing details
are in Settings.

First/Last Frame and Alpha Gen matte are not included yet. The former needs a
separately validated two-frame conditioning path; the latter needs an optional
IC-LoRA plus video-guidance code. Neither is a reason to download more models
or add nonworking controls to the starter app.

The 15-, 20-, 30-, 45-, and 60-second choices are experimental single-pass runs. Long clips
may run out of GPU or system memory, especially at larger sizes or with the
detail pass. A memory error releases the worker and leaves saved videos and
downloaded models untouched. Start with Fast size and no detail pass.

## Network settings

The app starts on this computer only. To make it reachable on the local network
or through a temporary public Gradio link, use the **Settings** tab. Choose an
access mode and provide a username and password of at least 12 characters.
Save, close the server, then relaunch `run.bat`; network changes do not apply
to the running process. The password is kept in `network_settings.json` as a
salted hash and is never included in the installer ZIP. Anyone with this login
can generate videos and change Settings, so do not share it broadly. Switch
back to local-only mode when network access is no longer needed.

## Performance and limits

This is a local inference experiment. On an RTX 4090, a 768×512, 49-frame,
two-stage generation completed in 37.8 seconds including initial model loading;
the repeated warm run in the same worker completed in 14.3 seconds.
The exact memory and speed depend on GPU, resolution, duration, driver, and
PyTorch build. A smaller 512×320, 17-frame two-stage run with uploaded audio
completed in 34.5 seconds. These are smoke tests, not quality or broad
compatibility guarantees; measure your own GPU and review outputs before use.
In this Studio update, 49-frame Text to Video and Audio to Video single-stage
smoke tests completed in about 29 seconds each including model loading;
full-size two-stage Text to Video completed in about 34 seconds. Those runs
verify execution, not output quality on every prompt.

## Bundled code and model licenses

`vendor/comfy_core` is a selected source snapshot from ComfyUI commit
`b0f4b7b294ce482a2e071d9d762c133d38c7aa07`, copied without frontend,
server, custom-node manager, or model weights. Its `LICENSE` is GPL-3.0.
LTX 2.5 weights remain subject to [Lightricks' separate terms](https://huggingface.co/Lightricks/LTX-2.5).
Review both sets of terms before distributing this experiment.
# Repository updates

This app follows original Headliner: Codeberg and GitHub generate the downloadable source archive directly from repository commits. Maintainers push individual files; no source ZIP or checksum manifest needs rebuilding. Settings checks repository revisions and offers Update and restart. With the server closed, run `.venv\Scripts\python.exe update_app.py`, or the outer `3-UPDATE-LTX25-Core-Studio.bat`. Models, outputs and private settings are kept; replaced source is backed up. Git is not required for customer updates.

Source: https://codeberg.org/Cognibuild/LTX25-Core-Studio and https://github.com/gjnave/LTX25-Core-Studio.
