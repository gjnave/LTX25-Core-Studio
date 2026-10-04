# Get Going Fast · LTX 2.5 Core Studio

Local Gradio video generation from a first-frame image and prompt. Add an audio
file to condition the video and carry that track into the MP4, or omit it to
generate synchronized audio with LTX 2.5. No ComfyUI installation or server is
required: this app bundles only the Python inference components it calls.

The Windows installer downloads this source into a `LTX25-Core-Studio` folder
beside `installer.bat` and `run.bat`. Its two-stage image-to-video
path follows the active part of the supplied `LTX2.5.json`: distilled INT8
model, 8-step first pass, latent 2× upscale, and 3-step detail pass. The
optional uploaded-audio path is an added feature, not present in that active
workflow branch.

## Install and run

Double-click the parent-folder `installer.bat`. It downloads the app from
Codeberg, then GitHub if needed, then a Google Drive fallback when publicly
available. You do not need to copy this repository beside the installer.
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

The Create tab keeps the first-frame image, optional audio, Start/End trim times,
Generate button, and video output in one short flow. The optional motion prompt,
output size, clip length, seed, and high-detail pass are under Settings →
Generation options. The audio preview appears only when requested.

Turning off the high-detail pass skips the latent-upscale/detail pass and
outputs at half the selected width and height. The two-stage mode outputs the
full selected size. Frames use the LTX `8*k+1` rule at 24 fps. Uploaded audio
is trimmed or padded to the selected duration. On a phone, use the Start/End
time fields below the audio upload instead of waveform handles. Times accept
seconds or `mm:ss`; the Preview button plays the selected range. The prompt
is optional: blank text lets the model infer motion from the first frame.

Fast portrait is 512 × 768, matching Fast landscape's pixel count. The output
video appears below Generate; saved-file and timing details are in Settings.

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

## Bundled code and model licenses

`vendor/comfy_core` is a selected source snapshot from ComfyUI commit
`b0f4b7b294ce482a2e071d9d762c133d38c7aa07`, copied without frontend,
server, custom-node manager, or model weights. Its `LICENSE` is GPL-3.0.
LTX 2.5 weights remain subject to [Lightricks' separate terms](https://huggingface.co/Lightricks/LTX-2.5).
Review both sets of terms before distributing this experiment.
