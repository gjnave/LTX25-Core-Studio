# LTX 2.5 Core Studio release rules

- Preserve user models, outputs, the private `.venv`, and `network_settings.json` during updates.
- Never delete a file or folder without the user's explicit permission.
- Publish individual source files to Codeberg Cognibuild/LTX25-Core-Studio and GitHub gjnave/LTX25-Core-Studio. Updates download host-generated repository archives like original Headliner. Never require a manually rebuilt source ZIP or manifest. Drive is optional backup storage. Verify both repository revisions after pushing.
- Do not upload model weights, outputs, credentials, `.venv`, or `network_settings.json`.
- The root `installer.bat` is the single Windows CMD installer, including model downloads. `3-UPDATE-LTX25-Core-Studio.bat` uses update_app.py when available, with installer update mode for older installations.
- See `RELEASE-MIRRORS.md` for endpoints and verification steps. Do not assume a local commit has reached any mirror.
