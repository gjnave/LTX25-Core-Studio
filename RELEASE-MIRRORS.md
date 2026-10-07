# Source publication and updates

Publish the same individual source files and commit to Codeberg (primary) and GitHub. The updater checks the branch commit, downloads the host-generated archive for that revision, validates the app files and paths, backs up replaced source, and applies it. No manually uploaded source ZIP or generated hash manifest is needed. A normal source commit is enough for update detection.

Settings offers Check for updates and Update and restart. The outer update BAT works when the app is closed. Models, outputs, workspaces, environments, and private settings are preserved. Customer installer ZIPs remain separate member downloads. Google Drive may keep backups, but it is not a manually maintained prerequisite for current code updates.

- https://codeberg.org/Cognibuild/LTX25-Core-Studio
- https://github.com/gjnave/LTX25-Core-Studio
