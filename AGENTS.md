# LTX 2.5 Core Studio release rules

- Preserve user models, outputs, the private `.venv`, and `network_settings.json` during updates.
- Never delete a file or folder without the user's explicit permission.
- The public source has three synchronized download mirrors: Codeberg, GitHub, and the daysinging Google Drive Software/LTX25-Core-Studio source archive. A release is not complete until all three contain the same VERSION and source, or the missing mirror is reported explicitly.
- Do not upload model weights, outputs, credentials, `.venv`, or `network_settings.json`.
- The root `installer.bat` is the single Windows CMD installer, including model downloads. `3-UPDATE-LTX25-Core-Studio.bat` calls that installer in update mode.
- See `RELEASE-MIRRORS.md` for endpoints and verification steps. Do not assume a local commit has reached any mirror.
