# Sculpt Pose Chain Tool

Blender extension (Blender 4.3+). All code is in `__init__.py`; metadata is in `blender_manifest.toml`.

## Git identity
Commits and pushes in this repo must use the GitHub account **thegrasstrainer** only. This is set in the repo's local git config:
- `user.name = thegrasstrainer`
- `user.email = 32713112+thegrasstrainer@users.noreply.github.com`
- `credential.https://github.com.username = thegrasstrainer`

Check `gh auth status` shows thegrasstrainer before using `gh`. Never commit as or push with another account.

## Releasing a usable version
Every usable version goes to this repo's GitHub **Releases** as an installable zip. The owner later uploads that same zip to the Blender Extensions platform (extensions.blender.org).

1. Bump `version` in `blender_manifest.toml`, and update README.md if features or limits changed.
2. Commit and push.
3. Build and validate the zip with Blender (it puts the manifest at the top level of the zip, which Blender and the Extensions platform require):
   ```
   & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --factory-startup --command extension validate
   & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --factory-startup --command extension build --output-dir dist
   ```
   Output: `dist/sculpt_pose_chain_tool-<version>.zip` (`dist/` is gitignored).
4. Tag the release commit `v<version>` and publish:
   ```
   gh release create v<version> dist/sculpt_pose_chain_tool-<version>.zip --title "v<version> - <summary>" --notes "<what changed>"
   ```
   Attach only the zip, not loose `__init__.py` / manifest files.
5. Do not upload to the Blender Extensions platform; the owner does that when it is ready to go public.
