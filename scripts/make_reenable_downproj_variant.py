#!/usr/bin/env python3
"""
Clone the current winning no-attn submission package and re-enable selected
MLP down_proj layers by shrinking the dynamic skip set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


DEFAULT_SRC = (
    "/Users/ql/cursor/openbmb/"
    "submission-w4a16-marlin-publichybrid-noattn-skipdown151617-v1"
)
DEFAULT_DST = (
    "/Users/ql/cursor/openbmb/"
    "submission-w4a16-marlin-publichybrid-noattn-skipdown17-v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--dst", default=DEFAULT_DST)
    parser.add_argument(
        "--reenable-layers",
        default="15,16",
        help="Comma-separated down_proj layers to re-enable. Default: 15,16",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination directory if it already exists.",
    )
    return parser.parse_args()


def should_ignore(_: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if name == "__pycache__" or name.endswith(".pyc") or name == ".DS_Store":
            ignored.add(name)
    return ignored


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_sha256sums(dst: Path) -> None:
    lines: list[str] = []
    for path in sorted(dst.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.name == "SHA256SUMS.txt":
            continue
        rel = path.relative_to(dst).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}")
    (dst / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")


def build_skip_config(reenable_layers: set[int]) -> dict[str, dict]:
    keep_skipped = [layer for layer in (15, 16, 17) if layer not in reenable_layers]
    config = {
        "-:^model\\.layers\\.\\d+\\.self_attn\\.(q_proj|k_proj|v_proj|o_proj)$": {},
        "-:^model\\.layers\\.\\d+\\.self_attn\\.qkv_proj$": {},
    }
    if keep_skipped:
        keep_expr = "|".join(str(layer) for layer in keep_skipped)
        config[f"-:^model\\.layers\\.({keep_expr})\\.mlp\\.down_proj$"] = {}
    return config


def patch_prepare_model(path: Path, config_name: str, attempt_name: str) -> None:
    text = path.read_text()
    text = re.sub(
        r'"publichybrid_noattn_skipdown151617_v1"',
        f'"{attempt_name}"',
        text,
        count=1,
    )
    text = re.sub(
        r'configs/selective_marlin/noattn-skipdown151617-gs128\.json',
        f"configs/selective_marlin/{config_name}",
        text,
        count=1,
    )
    path.write_text(text)


def patch_readme_like(path: Path, old: str, new: str) -> None:
    if not path.exists():
        return
    text = path.read_text()
    text = text.replace(old, new)
    path.write_text(text)


def main() -> int:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"Source package directory not found: {src}")
    if dst.exists():
        if not args.force:
            raise SystemExit(f"Destination already exists: {dst} (use --force)")
        shutil.rmtree(dst)

    reenable_layers = {
        int(item.strip())
        for item in args.reenable_layers.split(",")
        if item.strip()
    }
    invalid = sorted(layer for layer in reenable_layers if layer not in {15, 16, 17})
    if invalid:
        raise SystemExit(f"Only layers 15/16/17 are supported here, got: {invalid}")

    shutil.copytree(
        src,
        dst,
        ignore=should_ignore,
        symlinks=True,
        ignore_dangling_symlinks=True,
    )

    keep_skipped = [layer for layer in (15, 16, 17) if layer not in reenable_layers]
    skip_suffix = "".join(str(layer) for layer in keep_skipped) or "none"
    config_name = f"noattn-skipdown{skip_suffix}-gs128.json"
    attempt_name = f"publichybrid_noattn_skipdown{skip_suffix}_v1"

    config_dir = dst / "configs" / "selective_marlin"
    new_config_path = config_dir / config_name
    new_config_path.write_text(
        json.dumps(build_skip_config(reenable_layers), indent=2, ensure_ascii=False) + "\n"
    )

    patch_prepare_model(dst / "prepare_model.sh", config_name=config_name, attempt_name=attempt_name)
    patch_readme_like(
        dst / "README.md",
        "submission-w4a16-marlin-publichybrid-noattn-skipdown151617-v1",
        dst.name,
    )
    patch_readme_like(
        dst / "RESULTS.md",
        "submission-w4a16-marlin-publichybrid-noattn-skipdown151617-v1",
        dst.name,
    )

    variant_note = dst / "VARIANT_NOTES.md"
    variant_note.write_text(
        "\n".join(
            [
                f"# {dst.name}",
                "",
                "Derived from the current 99.97 no-attn submission package.",
                "",
                f"Re-enabled down_proj layers: {sorted(reenable_layers)}",
                f"Still skipped down_proj layers: {keep_skipped}",
                "",
                "Everything else is intentionally unchanged:",
                "- calibration file",
                "- quant env / eval env split",
                "- all-attention skip",
                "- serve route",
                "- cpu-cache-layer-outputs",
                "",
            ]
        )
        + "\n"
    )

    rewrite_sha256sums(dst)

    print(json.dumps(
        {
            "src": str(src),
            "dst": str(dst),
            "reenabled_down_proj_layers": sorted(reenable_layers),
            "remaining_skipped_down_proj_layers": keep_skipped,
            "new_config": str(new_config_path),
            "prepare_model": str(dst / "prepare_model.sh"),
            "sha256sums": str(dst / "SHA256SUMS.txt"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
