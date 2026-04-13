---
name: soar-package-submission
description: Use when packaging or refreshing a SOAR submission draft tarball, especially for MiniCPM-SALA / SGLang competition packages under /Users/ql/cursor/openbmb/submission-drafts. Rebuild hashes, strip pycache, avoid macOS xattr tar noise, reuse existing local wheels such as flash_attn, and return the final tar path, size, and sha256.
---

# SOAR Submission Packaging

Use this skill for SOAR competition packaging work when the user asks to:

- "打个包"
- refresh a submission tarball
- rebuild `SHA256SUMS.txt`
- verify a draft contains required files
- avoid re-downloading local wheels like `flash_attn`

## Workflow

1. Confirm the draft directory and expected tarball name.
2. Check whether required files exist:
   - `prepare_env.sh`
   - `prepare_model.sh`
   - any route-specific scripts or calibration files the draft depends on
3. Reuse local wheels already in the draft; do not re-download them unless the user explicitly asks.
4. Package manually:
   - remove `__pycache__` and `*.pyc`
   - refresh `SHA256SUMS.txt`
   - build tar with `COPYFILE_DISABLE=1`
   - compute final sha256
5. Report:
   - final tar path
   - final size
   - sha256
   - any important caveats

## Notes

- Always build the tar with `COPYFILE_DISABLE=1` to avoid macOS extended-attribute noise.
- Always remove `__pycache__` and `*.pyc` before packaging.
- Always refresh `SHA256SUMS.txt` before building the tar.
- Prefer returning one clean package artifact, not multiple competing tarballs.
- If the user wants this behavior every time, the most reliable way is to explicitly invoke this skill in future prompts or automations:
  - `Use [$soar-package-submission](/Users/ql/.codex/skills/soar-package-submission/SKILL.md).`
