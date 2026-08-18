"""Blender-free tests for the manifest runner's control plane."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_tests.py"
spec = importlib.util.spec_from_file_location("yse_test_runner", RUNNER_PATH)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def test_manifest_is_complete() -> None:
    entries = runner.load_manifest()
    runner.check_completeness(entries, Path(__file__).parent)
    assert len(entries) == len(runner.discover_test_files(Path(__file__).parent))


def test_completeness_fails_closed_for_unclassified_file(tmp_path: Path) -> None:
    (tmp_path / "test_extra.py").write_text("", encoding="utf-8")
    entries = [runner.TestSpec("test_known.py", "fast-python", (), (), ("fast",))]
    (tmp_path / "test_known.py").write_text("", encoding="utf-8")
    with pytest.raises(runner.ManifestError, match="unclassified"):
        runner.check_completeness(entries, tmp_path)


def test_completeness_fails_closed_for_missing_manifest_file(tmp_path: Path) -> None:
    (tmp_path / "test_known.py").write_text("", encoding="utf-8")
    entries = [
        runner.TestSpec("test_known.py", "fast-python", (), (), ("fast",)),
        runner.TestSpec("test_ghost.py", "fast-python", (), (), ("fast",)),
    ]
    with pytest.raises(runner.ManifestError, match="not found"):
        runner.check_completeness(entries, tmp_path)


def test_completeness_sees_uppercase_test_files(tmp_path: Path) -> None:
    (tmp_path / "test_known.py").write_text("", encoding="utf-8")
    (tmp_path / "TEST_sneaky.py").write_text("", encoding="utf-8")
    entries = [runner.TestSpec("test_known.py", "fast-python", (), (), ("fast",))]
    with pytest.raises(runner.ManifestError, match="unclassified"):
        runner.check_completeness(entries, tmp_path)


def _write_manifest(tmp_path: Path, body: str) -> Path:
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(body, encoding="utf-8")
    return manifest


def test_manifest_rejects_empty_markers_for_blender_modes(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        '[[tests]]\nfile = "test_x.py"\nmode = "background"\nmarkers = []\nversions = ["5.2"]\n',
    )
    with pytest.raises(runner.ManifestError, match="markers must not be empty"):
        runner.load_manifest(manifest)


def test_manifest_rejects_non_list_markers(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        '[[tests]]\nfile = "test_x.py"\nmode = "background"\nmarkers = "OK"\nversions = ["5.2"]\n',
    )
    with pytest.raises(runner.ManifestError, match="list of non-empty strings"):
        runner.load_manifest(manifest)


def test_manifest_restricts_staged_gui_to_preferences_test(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        '[[tests]]\nfile = "test_core.py"\nmode = "staged-gui"\nmarkers = ["X_OK"]\nversions = ["5.2"]\n',
    )
    with pytest.raises(runner.ManifestError, match="staged-gui is reserved"):
        runner.load_manifest(manifest)


def test_marker_and_exit_code_evaluation() -> None:
    entry = runner.TestSpec("test_x.py", "background", ("A_OK", "B_OK"), ("4.2",), ())
    assert runner.evaluate_execution(entry, runner.ExecutionResult(0, "A_OK B_OK")) == ("pass", [])
    assert runner.evaluate_execution(entry, runner.ExecutionResult(0, "A_OK")) == ("fail", ["B_OK"])
    assert runner.evaluate_execution(entry, runner.ExecutionResult(3, "A_OK B_OK")) == ("fail", [])
    # Markers split across the stdout/stderr boundary must not assemble.
    assert runner.evaluate_execution(entry, runner.ExecutionResult(0, "B_OK A_", "OK")) == ("fail", ["A_OK"])
    # A superstring marker is not the marker.
    assert runner.evaluate_execution(entry, runner.ExecutionResult(0, "A_OK_EXTRA B_OK")) == ("fail", ["A_OK"])
    # But stderr-only markers still count.
    assert runner.evaluate_execution(entry, runner.ExecutionResult(0, "A_OK", "B_OK")) == ("pass", [])


def test_execute_process_timeout_fails_instead_of_hanging() -> None:
    result = runner.execute_process(["sleep", "5"], Path.cwd(), timeout=0.2)
    assert result.returncode != 0
    assert "TIMEOUT" in result.stderr


def test_injected_executor_covers_pass_fail_missing_and_exception(tmp_path: Path) -> None:
    names = ("test_pass.py", "test_fail.py", "test_missing.py", "test_exception.py")
    for name in names:
        (tmp_path / name).write_text("", encoding="utf-8")
    entries = [runner.TestSpec(name, "fast-python", ("OK",), (), ("fast",)) for name in names]

    def fake_executor(command, cwd):
        name = command[-1]
        if name == "tests/test_pass.py":
            return runner.ExecutionResult(0, "OK")
        if name == "tests/test_fail.py":
            return runner.ExecutionResult(2, "OK")
        if name == "tests/test_missing.py":
            return runner.ExecutionResult(0, "no marker")
        raise RuntimeError("injected process failure")

    code, results = runner.run_manifest(
        entries,
        repository_root=tmp_path,
        tests_root=tmp_path,
        executor=fake_executor,
        output_path=tmp_path / "result.json",
    )
    assert code == 1
    assert [result["status"] for result in results] == ["pass", "fail", "fail", "fail"]
    assert results[-1]["exit_code"] is None


@pytest.mark.skipif(shutil.which("wslpath") is None, reason="runner drives Blender via cmd.exe and is WSL-only")
def test_background_mode_builds_blender_command_and_captures_identity(tmp_path: Path) -> None:
    (tmp_path / "test_bg.py").write_text("", encoding="utf-8")
    entry = runner.TestSpec("test_bg.py", "background", ("BG_OK",), ("5.2",), ("core",))
    seen_commands = []

    def fake_executor(command, cwd):
        seen_commands.append(list(command))
        return runner.ExecutionResult(0, "Blender 5.2.0 LTS (hash abc)\nBG_OK")

    code, results = runner.run_manifest(
        [entry],
        repository_root=tmp_path,
        tests_root=tmp_path,
        blender_paths={"4.2": "X:\\b42.exe", "5.2": "X:\\b52.exe"},
        executor=fake_executor,
        output_path=tmp_path / "result.json",
    )
    assert code == 0
    assert results[0]["status"] == "pass"
    assert seen_commands[0][:2] == ["cmd.exe", "/c"]
    assert "--background" in seen_commands[0]
    import json

    provenance = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert "Blender 5.2.0 LTS" in provenance["blender"]["5.2"]


def test_staged_gui_is_recorded_as_skip(tmp_path: Path) -> None:
    (tmp_path / "test_preferences.py").write_text("", encoding="utf-8")
    entry = runner.TestSpec("test_preferences.py", "staged-gui", ("STAGE2_OK",), ("5.2",), ("staged",))

    def unexpected_executor(command, cwd):
        raise AssertionError(f"staged test executed: {command}")

    code, results = runner.run_manifest(
        [entry],
        tests_root=tmp_path,
        repository_root=tmp_path,
        executor=unexpected_executor,
        output_path=tmp_path / "result.json",
    )
    assert code == 0
    assert results[0]["status"] == "skip"
    assert "documented" in results[0]["reason"]


def test_tree_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    (tmp_path / "ydd_symmetric_edit").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    tracked = tmp_path / "ydd_symmetric_edit" / "module.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    first = runner.tree_sha256(tmp_path)
    assert first == runner.tree_sha256(tmp_path)
    tracked.write_text("value = 2\n", encoding="utf-8")
    assert first != runner.tree_sha256(tmp_path)


def test_tag_and_version_filters() -> None:
    specs = [
        runner.TestSpec("test_fast.py", "fast-python", (), (), ("fast",)),
        runner.TestSpec("test_gui.py", "gui", (), ("4.2", "5.2"), ("gui", "perf")),
        runner.TestSpec("test_core.py", "background", (), ("4.2",), ("core",)),
    ]
    assert [item.file for item in runner.select_specs(specs, tags=("fast",))] == ["test_fast.py"]
    assert [item.file for item in runner.select_specs(specs, tags=("perf",), versions=("5.2",))] == ["test_gui.py"]
    assert runner.select_specs(specs, versions=("5.2",))[0].file == "test_fast.py"
