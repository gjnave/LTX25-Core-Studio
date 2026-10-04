# Release mirrors and update contract

This source tree is published to:

1. Codeberg: `https://codeberg.org/Cognibuild/LTX25-Core-Studio` (source ZIP plus README and VERSION while Git push authentication is unavailable)
2. GitHub: `https://github.com/gjnave/LTX25-Core-Studio`
3. Google Drive (daysinging account): `Software/LTX25-Core-Studio/` with a source ZIP intended as the installer's last-resort fallback. Anonymous access must be enabled and tested before this fallback can be claimed operational.

Every code release must update `VERSION`, run a source-only package audit, publish the same tracked source to GitHub and the matching source archive to Codeberg, replace the Drive source ZIP with a ZIP of the same revision, and verify all three readbacks. If Codeberg Git authentication is restored, publish the full Git tree there too. Do not include `.venv`, `models`, `outputs`, `.gradio`, `network_settings.json`, or local Git metadata in the Drive archive. Keep the root installer ZIP separate from these public source mirrors.

The root `installer.bat` tries Codeberg, then GitHub, then Google Drive. It stages and validates a ZIP before backing up or copying installed code. Existing models, outputs, and private environment remain in place. The root updater calls the installer with `/update`; the app's check button only reports that an update exists and tells the user to close the server and run the updater.

LoRA support is not enabled merely by placing a file in a model folder. The current app-local core has a Comfy model-only LoRA loader, but a particular LoRA must be compatible with the LTX 2.5 architecture and the installed INT8 checkpoint and be tested before a UI control is shipped.
