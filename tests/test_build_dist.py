# SPDX-License-Identifier: GPL-3.0-or-later

"""Host-platform command construction tests for scripts/build_dist.py."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIST_PATH = REPOSITORY_ROOT / "scripts" / "build_dist.py"


def _load_build_dist():
    spec = importlib.util.spec_from_file_location("build_dist_platform_tests", BUILD_DIST_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_dist = _load_build_dist()


def test_native_windows_path_does_not_require_wslpath(monkeypatch, tmp_path):
    monkeypatch.setattr(build_dist, "running_on_windows", lambda: True)

    def unexpected_wslpath(*args, **kwargs):
        raise AssertionError("wslpath must not be called on native Windows")

    monkeypatch.setattr(build_dist.subprocess, "run", unexpected_wslpath)

    path = tmp_path / "package.zip"
    assert build_dist.to_windows_path(path) == str(path)


def test_wsl_path_conversion_uses_wslpath(monkeypatch, tmp_path):
    monkeypatch.setattr(build_dist, "running_on_windows", lambda: False)
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="C:\\repo\\package.zip\n", stderr="")

    monkeypatch.setattr(build_dist.subprocess, "run", fake_run)

    path = tmp_path / "package.zip"
    assert build_dist.to_windows_path(path) == "C:\\repo\\package.zip"
    assert seen["command"] == ["wslpath", "-w", str(path)]
    assert seen["kwargs"] == {"capture_output": True, "text": True, "check": True}


def test_resolve_blender_paths_unquotes_argument_and_environment(monkeypatch):
    monkeypatch.setattr(build_dist, "running_on_windows", lambda: True)
    monkeypatch.setenv("YSE_BLENDER52", '"C:/Program Files/Blender/5.2/blender.exe"')
    args = build_dist.parse_args(["--blender42", "'D:/Program Files/Blender/4.2/blender.exe'"])

    blender52, blender42 = build_dist.resolve_blender_paths(args)

    assert blender52 == "C:/Program Files/Blender/5.2/blender.exe"
    assert blender42 == "D:/Program Files/Blender/4.2/blender.exe"
    assert build_dist.blender_command(blender52, ["--version"])[0] == blender52


def test_resolve_blender_paths_skips_unused_invalid_42_setting(monkeypatch):
    monkeypatch.setenv("YSE_BLENDER52", "C:/Program Files/Blender/5.2/blender.exe")
    monkeypatch.setenv("YSE_BLENDER42", '""')
    args = build_dist.parse_args(["--skip-validate"])

    blender52, blender42 = build_dist.resolve_blender_paths(args)

    assert blender52 == "C:/Program Files/Blender/5.2/blender.exe"
    assert blender42 is None


@pytest.mark.parametrize("value", ["", "   ", '""', "''", "\"'"])
def test_normalize_executable_path_rejects_empty_or_quote_only_values(value):
    with pytest.raises(build_dist.BuildError, match="executable path is empty"):
        build_dist.normalize_executable_path(value, "Blender 5.2")


@pytest.mark.parametrize(
    ("native_windows", "expected_prefix"),
    ((True, []), (False, ["cmd.exe", "/c"])),
)
def test_blender_command_uses_host_launcher(monkeypatch, native_windows, expected_prefix):
    monkeypatch.setattr(build_dist, "running_on_windows", lambda: native_windows)

    arguments = ["--factory-startup", "--command", "extension"]
    command = build_dist.blender_command("C:\\Program Files\\Blender\\blender.exe", arguments)

    assert command == [*expected_prefix, "C:\\Program Files\\Blender\\blender.exe", *arguments]


def test_run_blender_command_invokes_native_executable_directly(monkeypatch, tmp_path):
    monkeypatch.setattr(build_dist, "running_on_windows", lambda: True)
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(build_dist.subprocess, "run", fake_run)

    build_dist.run_blender_command("blender.exe", ["--version"], tmp_path)

    assert seen == {"command": ["blender.exe", "--version"], "kwargs": {"cwd": str(tmp_path)}}


def test_run_blender_command_converts_spawn_error_to_build_error(monkeypatch, tmp_path):
    monkeypatch.setattr(build_dist, "running_on_windows", lambda: True)

    def missing_executable(*args, **kwargs):
        raise FileNotFoundError("Blender executable not found")

    monkeypatch.setattr(build_dist.subprocess, "run", missing_executable)

    with pytest.raises(build_dist.BuildError, match=r"failed to execute Blender command .*blender\.exe"):
        build_dist.run_blender_command("blender.exe", ["--version"], tmp_path)
