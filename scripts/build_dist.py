#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build (and optionally validate) the ydd_symmetric_edit extension ZIP.

Supports native Windows and WSL execution. On WSL, Windows Blender
executables are launched through ``cmd.exe`` and WSL paths are converted with
``wslpath``. On native Windows, Blender is launched directly and paths are
already in the format it expects. Supports a plain build and a trial build
(name suffix + trial.py constants patched).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR_NAME = "ydd_symmetric_edit"
PACKAGE_ROOT = REPOSITORY_ROOT / PACKAGE_DIR_NAME
DIST_ROOT = REPOSITORY_ROOT / "dist"

DEFAULT_TRIAL_DAYS = 14

BASE_NAME = "ydd Symmetric Edit"
TRIAL_NAME = "ydd Symmetric Edit (Trial)"
MANIFEST_NAME_OLD = f'name = "{BASE_NAME}"'
MANIFEST_NAME_NEW = f'name = "{TRIAL_NAME}"'
TRIAL_BUILD_OLD = "TRIAL_BUILD = False"
TRIAL_BUILD_NEW = "TRIAL_BUILD = True"
TRIAL_DAYS_OLD = "TRIAL_DAYS = 14"

# Complete allowlist of ZIP contents (docs/testing.md keeps the same list).
EXPECTED_PACKAGE_FILES = frozenset(
    {
        "blender_manifest.toml",
        "LICENSE",
        "__init__.py",
        "_types.py",
        "backup.py",
        "delete_dissolve.py",
        "element_pairs.py",
        "extrude.py",
        "extrude_menu.py",
        "face_mapping.py",
        "gc_gate.py",
        "gizmo_adopt.py",
        "history.py",
        "inset_bevel.py",
        "keymaps.py",
        "layer_names.py",
        "matching.py",
        "operators.py",
        "replay.py",
        "rip.py",
        "selection.py",
        "session.py",
        "session_state.py",
        "snapshot.py",
        "stitch_common.py",
        "stitch_crossings.py",
        "stitch_offset.py",
        "stitch_pathedges.py",
        "stitch_pstitch.py",
        "stitch_reflect.py",
        "trial.py",
        "ui.py",
        "watcher.py",
    }
)


class BuildError(RuntimeError):
    pass


def running_on_windows() -> bool:
    """Whether this Python process can invoke Windows executables directly."""
    return os.name == "nt"


def replace_unique(text: str, old: str, new: str) -> str:
    """Replace old with new, requiring exactly one occurrence of old in text."""
    count = text.count(old)
    if count != 1:
        raise BuildError(f"expected exactly one occurrence of {old!r}, found {count}")
    return text.replace(old, new, 1)


def patch_manifest_name(text: str) -> str:
    """Rewrite the manifest display name to the trial variant."""
    return replace_unique(text, MANIFEST_NAME_OLD, MANIFEST_NAME_NEW)


def patch_trial_module(text: str, days: int) -> str:
    """Flip trial.py's TRIAL_BUILD flag on and set TRIAL_DAYS to days."""
    text = replace_unique(text, TRIAL_BUILD_OLD, TRIAL_BUILD_NEW)
    text = replace_unique(text, TRIAL_DAYS_OLD, f"TRIAL_DAYS = {days}")
    return text


def read_version(manifest_path: Path) -> str:
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest["version"]


def default_out_name(trial: bool) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "trial" if trial else "candidate"
    return f"{prefix}_{stamp}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", action="store_true", help="Build a trial (nag-only) package")
    parser.add_argument("--days", type=int, default=None, help="Trial length in days (requires --trial)")
    parser.add_argument("--out", default=None, help="Output directory name under dist/")
    parser.add_argument("--blender52", default=None, help="Windows path to the Blender 5.2 executable")
    parser.add_argument("--blender42", default=None, help="Windows path to the Blender 4.2 executable")
    parser.add_argument("--skip-validate", action="store_true", help="Skip extension validate steps")

    args = parser.parse_args(argv)

    if args.days is not None and not args.trial:
        parser.error("--days requires --trial")
    if args.days is None:
        args.days = DEFAULT_TRIAL_DAYS

    return args


def normalize_executable_path(value: str | None, setting: str) -> str | None:
    """Normalize a Blender executable setting without altering its inner path."""
    if value is None:
        return None

    normalized = value.strip()
    if not normalized or all(character in {"'", '"'} for character in normalized):
        raise BuildError(f"{setting} executable path is empty")

    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
        normalized = normalized[1:-1].strip()
        if not normalized:
            raise BuildError(f"{setting} executable path is empty")

    return normalized


def resolve_blender_paths(args: argparse.Namespace) -> tuple[str, str | None]:
    """Executable paths are machine-local: flags or env vars only, no shipped defaults."""
    blender52 = normalize_executable_path(
        args.blender52 if args.blender52 is not None else os.environ.get("YSE_BLENDER52"),
        "Blender 5.2",
    )
    blender42 = None
    if not args.skip_validate:
        blender42 = normalize_executable_path(
            args.blender42 if args.blender42 is not None else os.environ.get("YSE_BLENDER42"),
            "Blender 4.2",
        )
    if not blender52:
        raise BuildError("no Blender 5.2 executable configured; pass --blender52 or set YSE_BLENDER52")
    if not blender42 and not args.skip_validate:
        raise BuildError(
            "no Blender 4.2 executable configured; pass --blender42, set YSE_BLENDER42, or use --skip-validate"
        )
    return blender52, blender42


def to_windows_path(path: Path) -> str:
    """Return a path suitable for Blender's Windows command-line interface."""
    if running_on_windows():
        return str(path)

    try:
        result = subprocess.run(
            ["wslpath", "-w", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise BuildError("wslpath is required when build_dist.py runs outside native Windows") from error
    except subprocess.CalledProcessError as error:
        raise BuildError(f"wslpath failed to convert {path}") from error
    return result.stdout.strip()


def blender_command(blender_exe: str, arguments: list[str]) -> list[str]:
    """Build the subprocess command for the current host environment."""
    if running_on_windows():
        return [blender_exe, *arguments]
    return ["cmd.exe", "/c", blender_exe, *arguments]


def run_blender_command(blender_exe: str, arguments: list[str], cwd: Path) -> None:
    command = blender_command(blender_exe, arguments)
    try:
        result = subprocess.run(command, cwd=str(cwd))
    except OSError as error:
        raise BuildError(f"failed to execute Blender command {command}: {error}") from error
    if result.returncode != 0:
        raise BuildError(f"Blender command failed (exit {result.returncode}): {command}")


def build_extension(blender_exe: str, source_dir: Path, output_zip: Path, cwd: Path) -> None:
    run_blender_command(
        blender_exe,
        [
            "--factory-startup",
            "--command",
            "extension",
            "build",
            "--source-dir",
            to_windows_path(source_dir),
            "--output-filepath",
            to_windows_path(output_zip),
        ],
        cwd,
    )


def validate_extension(blender_exe: str, package_zip: Path, cwd: Path) -> None:
    run_blender_command(
        blender_exe,
        [
            "--factory-startup",
            "--command",
            "extension",
            "validate",
            to_windows_path(package_zip),
        ],
        cwd,
    )


def prepare_trial_source(staging_dir: Path, days: int) -> None:
    shutil.copytree(
        PACKAGE_ROOT,
        staging_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.zip"),
    )

    manifest_path = staging_dir / "blender_manifest.toml"
    manifest_path.write_text(
        patch_manifest_name(manifest_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    trial_path = staging_dir / "trial.py"
    trial_path.write_text(
        patch_trial_module(trial_path.read_text(encoding="utf-8"), days),
        encoding="utf-8",
    )


def verify_package(package_zip: Path, *, trial: bool) -> int:
    with zipfile.ZipFile(package_zip) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]

        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise BuildError(f"{package_zip} contains duplicate entries: {duplicates}")
        for info in infos:
            name = info.filename
            if name.startswith("/") or "\\" in name or ".." in name.split("/"):
                raise BuildError(f"{package_zip} contains an unsafe entry name: {name!r}")
            # Directory entries can be marked by name, Unix mode bits, or the
            # MS-DOS directory attribute; a symlink only by Unix mode bits.
            if name.endswith("/") or (info.external_attr >> 16) & 0o170000 == 0o040000 or info.external_attr & 0x10:
                raise BuildError(f"{package_zip} contains a directory entry: {name!r}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise BuildError(f"{package_zip} contains a symlink entry: {name!r}")

        extra = sorted(set(names) - EXPECTED_PACKAGE_FILES)
        missing = sorted(EXPECTED_PACKAGE_FILES - set(names))
        if extra:
            raise BuildError(f"{package_zip} contains unexpected entries: {extra}")
        if missing:
            raise BuildError(f"{package_zip} is missing expected entries: {missing}")

        manifest = tomllib.loads(archive.read("blender_manifest.toml").decode("utf-8"))
        expected_name = TRIAL_NAME if trial else BASE_NAME
        if manifest.get("name") != expected_name:
            raise BuildError(f"{package_zip} manifest name is {manifest.get('name')!r}, expected {expected_name!r}")

        trial_source = archive.read("trial.py").decode("utf-8")
        expected_flag = TRIAL_BUILD_NEW if trial else TRIAL_BUILD_OLD
        if expected_flag not in trial_source:
            raise BuildError(f"{package_zip} trial.py does not contain {expected_flag!r}")

        return len(names)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        blender52, blender42 = resolve_blender_paths(args)
    except BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    manifest_path = PACKAGE_ROOT / "blender_manifest.toml"
    version = read_version(manifest_path)

    out_name = args.out or default_out_name(args.trial)
    out_dir = DIST_ROOT / out_name
    if out_dir.exists():
        print(f"error: output directory already exists: {out_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True)

    if args.trial:
        source_dir = out_dir / "src"
        prepare_trial_source(source_dir, args.days)
        zip_name = f"{PACKAGE_DIR_NAME}-{version}-trial.zip"
    else:
        source_dir = PACKAGE_ROOT
        zip_name = f"{PACKAGE_DIR_NAME}-{version}.zip"

    package_zip = out_dir / zip_name

    try:
        build_extension(blender52, source_dir, package_zip, REPOSITORY_ROOT)

        if not args.skip_validate:
            # resolve_blender_paths only returns blender42=None under --skip-validate.
            assert blender42 is not None
            for blender_exe in (blender42, blender52):
                validate_extension(blender_exe, package_zip, REPOSITORY_ROOT)

        entry_count = verify_package(package_zip, trial=args.trial)
    except BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"{package_zip.resolve()} ({entry_count} entries)")
    print("BUILD_VALIDATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
