#!/usr/bin/env python3
"""Run the Wave 1 test manifest from WSL.

The manifest is deliberately the only source of test classification.  Blender
is driven through ``cmd.exe``; this module never imports Blender or any
third-party package so that its planning and unit tests work on WSL alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "tests" / "manifest.toml"
TESTS_ROOT = REPOSITORY_ROOT / "tests"
MODES = {"fast-python", "fast-pytest", "background", "gui", "staged-gui"}
VERSION_ORDER = ("4.2", "5.2")
# A hung GUI modal must fail one test, not stall the whole sweep.
DEFAULT_TEST_TIMEOUT = 900.0


class ManifestError(ValueError):
    """The manifest and the source tree do not describe the same tests."""


@dataclass(frozen=True)
class TestSpec:
    file: str
    mode: str
    markers: tuple[str, ...]
    versions: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def output(self) -> str:
        # The separator keeps a marker from assembling across the
        # stdout/stderr boundary.
        return f"{self.stdout}\n{self.stderr}"


Executor = Callable[[Sequence[str], Path], ExecutionResult]


def execute_process(command: Sequence[str], cwd: Path, timeout: float = DEFAULT_TEST_TIMEOUT) -> ExecutionResult:
    """The sole process-execution seam used by test runs.

    Unit tests inject a callable with this same small protocol.  In
    particular, Blender invocation is not spread across the runner.
    """

    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            # cmd.exe children consume an inherited stdin (known WSL interop
            # pitfall); never let a test read the runner's stdin.
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # Killing cmd.exe may orphan the Windows-side blender.exe; the sweep
        # still continues and the record names the timeout.
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return ExecutionResult(-1, stdout, f"{stderr}\nTIMEOUT after {timeout:.0f}s")
    return ExecutionResult(completed.returncode, completed.stdout or "", completed.stderr or "")


def load_manifest(path: Path = MANIFEST_PATH) -> list[TestSpec]:
    import tomllib

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("tests")
    if not isinstance(entries, list):
        raise ManifestError("manifest must contain [[tests]] entries")
    specs: list[TestSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise ManifestError(f"manifest entry {index} is not a table")
        try:
            file = str(entry["file"])
            mode = str(entry["mode"])
        except (KeyError, TypeError) as exc:
            raise ManifestError(f"invalid manifest entry {index}: {exc}") from exc
        for field in ("markers", "versions", "tags"):
            raw = entry.get(field, [])
            if not isinstance(raw, list) or any(not isinstance(value, str) or not value for value in raw):
                raise ManifestError(f"{file}: {field} must be a list of non-empty strings")
        markers = tuple(entry.get("markers", []))
        versions = tuple(entry.get("versions", []))
        tags = tuple(entry.get("tags", []))
        if file in seen:
            raise ManifestError(f"duplicate manifest entry: {file}")
        if mode not in MODES:
            raise ManifestError(f"{file}: unsupported mode {mode!r}")
        if not file.startswith("test_") or not file.endswith(".py") or Path(file).name != file:
            raise ManifestError(f"{file}: file must be a tests/ direct child named test_*.py")
        if any(version not in VERSION_ORDER for version in versions):
            raise ManifestError(f"{file}: unsupported Blender version in manifest")
        if not versions and mode not in {"fast-python", "fast-pytest"}:
            raise ManifestError(f"{file}: Blender tests need at least one version")
        # Pass/fail must never degrade to exit-code-only for Blender tests.
        if not markers and mode != "fast-pytest":
            raise ManifestError(f"{file}: markers must not be empty for mode {mode!r}")
        # staged-gui skips execution, so an arbitrary entry using it would
        # silently drop that test from the gate.
        if mode == "staged-gui" and file != "test_preferences_persistence.py":
            raise ManifestError(f"{file}: staged-gui is reserved for test_preferences_persistence.py")
        seen.add(file)
        specs.append(TestSpec(file, mode, markers, versions, tags))
    return specs


def discover_test_files(tests_root: Path = TESTS_ROOT) -> set[str]:
    # Case-insensitive so a stray TEST_*.py still trips the completeness
    # check instead of silently never running.
    return {path.name for path in tests_root.glob("*.py") if path.is_file() and path.name.lower().startswith("test_")}


def check_completeness(specs: Iterable[TestSpec], tests_root: Path = TESTS_ROOT) -> None:
    """Fail closed unless manifest and ``test_*.py`` sets are identical."""

    manifest_files = [spec.file for spec in specs]
    manifest_set = set(manifest_files)
    actual_set = discover_test_files(tests_root)
    missing_entries = sorted(actual_set - manifest_set)
    missing_files = sorted(manifest_set - actual_set)
    if missing_entries or missing_files:
        details = []
        if missing_entries:
            details.append(f"unclassified test files: {', '.join(missing_entries)}")
        if missing_files:
            details.append(f"manifest files not found: {', '.join(missing_files)}")
        raise ManifestError("; ".join(details))


def parse_values(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(dict.fromkeys(result))


def select_specs(
    specs: Iterable[TestSpec], *, tags: Sequence[str] = (), versions: Sequence[str] = ()
) -> list[TestSpec]:
    wanted_tags = set(tags)
    wanted_versions = set(versions)
    selected: list[TestSpec] = []
    for spec in specs:
        if wanted_tags and not wanted_tags.intersection(spec.tags):
            continue
        if wanted_versions and spec.mode not in {"fast-python", "fast-pytest"}:
            if not wanted_versions.intersection(spec.versions):
                continue
        selected.append(spec)
    return selected


def resolve_blender_paths(blender42: str | None = None, blender52: str | None = None) -> dict[str, str]:
    """Executable paths are machine-local: flags or env vars only, no shipped defaults.

    Versions without a configured path are omitted; ``run_manifest`` fails loudly
    (never silently skips) when a selected test needs an omitted version.
    """
    candidates = {
        "4.2": blender42 or os.environ.get("YSE_BLENDER42"),
        "5.2": blender52 or os.environ.get("YSE_BLENDER52"),
    }
    return {version: path for version, path in candidates.items() if path}


def to_windows_path(path: Path) -> str:
    converted = subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True, check=True)
    return converted.stdout.strip()


def command_for(
    spec: TestSpec,
    *,
    version: str | None,
    repository_root: Path,
    blender_paths: dict[str, str],
    convert_test_path: bool = True,
) -> list[str]:
    test_path = repository_root / "tests" / spec.file
    if spec.mode == "fast-python":
        return ["uv", "run", "python", str(test_path.relative_to(repository_root))]
    if spec.mode == "fast-pytest":
        return ["uv", "run", "pytest", "-q", str(test_path.relative_to(repository_root))]
    if version not in blender_paths:
        raise ValueError(f"no Blender path configured for version {version!r}")
    blender_test_path = to_windows_path(test_path) if convert_test_path else str(test_path)
    flags = ["--factory-startup"]
    if spec.mode == "background":
        flags += ["--background", "--disable-crash-handler"]
    elif spec.mode == "gui":
        flags += [
            "--enable-event-simulate",
            "--disable-crash-handler",
            "--no-window-focus",
            "-p",
            "40",
            "40",
            "960",
            "600",
        ]
    else:
        raise ValueError(f"staged-gui has no direct command: {spec.file}")
    return ["cmd.exe", "/c", blender_paths[version], *flags, "--python", blender_test_path]


def marker_present(marker: str, output: str) -> bool:
    # Word-bounded so YSE_X_OK does not match inside YSE_X_OK_EXTRA.
    return re.search(rf"(?<![A-Z0-9_]){re.escape(marker)}(?![A-Z0-9_])", output) is not None


def evaluate_execution(spec: TestSpec, execution: ExecutionResult) -> tuple[str, list[str]]:
    missing = [marker for marker in spec.markers if not marker_present(marker, execution.output)]
    status = "pass" if execution.returncode == 0 and not missing else "fail"
    return status, missing


TREE_ROOTS = ("ydd_symmetric_edit", "tests", "scripts", "pyproject.toml")


def _git_listed_files(repository_root: Path) -> list[Path] | None:
    """Tracked + untracked-but-not-ignored files, honoring .gitignore.

    Machine-specific files (tests/run_*.bat) and generated outputs
    (tests/benchmarks/) would otherwise make the tree hash irreproducible.
    """

    try:
        listing = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "--", *TREE_ROOTS],
            cwd=str(repository_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if listing.returncode != 0:
        return None
    paths = [repository_root / line for line in listing.stdout.splitlines() if line.strip()]
    return [path for path in paths if path.is_file()]


def tree_sha256(repository_root: Path = REPOSITORY_ROOT) -> str:
    candidates = _git_listed_files(repository_root)
    if candidates is None:
        candidates = []
        for name in TREE_ROOTS:
            root = repository_root / name
            if root.is_file():
                candidates.append(root)
            elif root.is_dir():
                candidates.extend(
                    path
                    for path in root.rglob("*")
                    if path.is_file()
                    and "__pycache__" not in path.parts
                    and path.suffix not in {".pyc", ".pyo"}
                    # Mirror the .gitignore exclusions the git listing honors.
                    and "benchmarks" not in path.parts
                    and not (path.name.startswith("run_") and path.suffix == ".bat")
                )
    digest = hashlib.sha256()
    for path in sorted(set(candidates), key=lambda item: item.relative_to(repository_root).as_posix()):
        relative = path.relative_to(repository_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes() if path.is_file() else b"").hexdigest()


def git_provenance(repository_root: Path) -> tuple[str | None, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repository_root), capture_output=True, text=True, check=False
        )
        source_commit = commit.stdout.strip() if commit.returncode == 0 else None
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=str(repository_root),
            capture_output=True,
            text=True,
            check=False,
        )
        return source_commit, dirty.returncode != 0 or bool(dirty.stdout.strip())
    except OSError:
        return None, True


def _result_record(
    spec: TestSpec,
    version: str | None,
    status: str,
    exit_code: int | None,
    missing: Sequence[str],
    duration: float,
    *,
    reason: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "file": spec.file,
        "version": version,
        "status": status,
        "exit_code": exit_code,
        "missing_markers": list(missing),
        "duration_sec": round(duration, 3),
    }
    if reason:
        record["reason"] = reason
    return record


def run_manifest(
    specs: Sequence[TestSpec],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    tests_root: Path = TESTS_ROOT,
    blender_paths: dict[str, str] | None = None,
    tags: Sequence[str] = (),
    versions: Sequence[str] = (),
    executor: Executor | None = None,
    dry_run: bool = False,
    output_path: Path | None = None,
) -> tuple[int, list[dict[str, object]]]:
    """Run selected entries and return ``(exit_code, result_records)``.

    ``executor`` is injectable specifically so tests can exercise pass/fail,
    missing-marker, and exception paths without Blender.
    """

    check_completeness(specs, tests_root)
    selected = select_specs(specs, tags=tags, versions=versions)
    paths = blender_paths or resolve_blender_paths()
    run_process = executor or execute_process
    records: list[dict[str, object]] = []
    blender_identity = {version: paths[version] for version in VERSION_ORDER if version in paths}
    started_at = datetime.now(UTC).isoformat()
    requested_versions = set(versions)
    missing_versions = sorted(
        {
            version
            for spec in selected
            if spec.mode not in {"fast-python", "fast-pytest", "staged-gui"}
            for version in spec.versions
            if version not in paths and (not requested_versions or version in requested_versions)
        }
    )
    if missing_versions:
        raise ValueError(
            "no Blender executable configured for version(s) "
            + ", ".join(missing_versions)
            + "; pass --blender42/--blender52 or set YSE_BLENDER42/YSE_BLENDER52"
        )

    def versions_for(spec: TestSpec) -> list[str | None]:
        if spec.mode in {"fast-python", "fast-pytest"}:
            return [None]
        return [
            version
            for version in VERSION_ORDER
            if version in spec.versions
            and version in paths
            and (not requested_versions or version in requested_versions)
        ]

    if dry_run:
        for spec in selected:
            if spec.mode == "staged-gui":
                print(f"SKIP {spec.file}: staged GUI procedure is documented, not automated")
                continue
            for version in versions_for(spec):
                command = command_for(
                    spec,
                    version=version,
                    repository_root=repository_root,
                    blender_paths=paths,
                    convert_test_path=False,
                )
                print(shlex.join(command))
        return 0, records

    for spec in selected:
        if spec.mode == "staged-gui":
            records.append(
                _result_record(
                    spec,
                    None,
                    "skip",
                    None,
                    (),
                    0.0,
                    reason="staged GUI requires the documented extension-install persistence procedure",
                )
            )
            continue
        for version in versions_for(spec):
            started = time.monotonic()
            try:
                command = command_for(
                    spec,
                    version=version,
                    repository_root=repository_root,
                    blender_paths=paths,
                )
                execution = run_process(command, repository_root)
                status, missing = evaluate_execution(spec, execution)
                if version is not None and execution.output:
                    first_blender_line = next(
                        (line.strip() for line in execution.output.splitlines() if line.strip().startswith("Blender ")),
                        None,
                    )
                    if first_blender_line:
                        blender_identity[version] = f"{paths[version]} ({first_blender_line})"
            except Exception as exc:  # Keep the sweep going after one failed test.
                execution = None
                status, missing = "fail", list(spec.markers)
                print(f"ERROR {spec.file} {version or ''}: {exc}", file=sys.stderr)
            duration = time.monotonic() - started
            exit_code = execution.returncode if execution is not None else None
            record = _result_record(spec, version, status, exit_code, missing, duration)
            if status == "fail" and execution is not None:
                # A failure without its raw output cannot be diagnosed later.
                failure_log = (
                    repository_root / "tmp" / "test_runs" / f"fail_{spec.file}.{version or 'fast'}.{int(started)}.log"
                )
                try:
                    failure_log.parent.mkdir(parents=True, exist_ok=True)
                    failure_log.write_text(execution.output, encoding="utf-8")
                    record["failure_log"] = str(failure_log)
                except OSError as exc:
                    print(f"ERROR: could not write failure log: {exc}", file=sys.stderr)
            records.append(record)
            print(f"{status.upper()} {spec.file}" + (f" [{version}]" if version else ""))

    finished_at = datetime.now(UTC).isoformat()
    provenance: dict[str, object] = {
        "blender": blender_identity,
        "selection": {"tags": list(tags), "versions": list(versions)},
        "started_at": started_at,
        "finished_at": finished_at,
        "results": records,
    }
    try:
        source_commit, source_dirty = git_provenance(repository_root)
        provenance["source_commit"] = source_commit
        provenance["source_dirty"] = source_dirty
        provenance["source_tree_sha256"] = tree_sha256(repository_root)
        provenance["manifest_sha256"] = file_sha256(repository_root / "tests" / "manifest.toml")
        provenance["runner_sha256"] = file_sha256(repository_root / "scripts" / "run_tests.py")
    except Exception as exc:  # The collected results must survive a provenance failure.
        provenance["provenance_error"] = repr(exc)
    try:
        if output_path is None:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            output_path = repository_root / "tmp" / "test_runs" / f"{stamp}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: could not write provenance JSON: {exc}", file=sys.stderr)
    failed = any(record["status"] == "fail" for record in records)
    print("YSE_SWEEP_FAILED" if failed else "YSE_SWEEP_OK")
    return (1 if failed else 0), records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list selected tests without running them")
    parser.add_argument("--dry-run", action="store_true", help="print selected command lines without running them")
    parser.add_argument("--tags", nargs="+", help="comma- or space-separated tag filter")
    parser.add_argument("--versions", nargs="+", help="Blender versions (4.2 and/or 5.2)")
    parser.add_argument("--blender42", help="Windows path to Blender 4.2")
    parser.add_argument("--blender52", help="Windows path to Blender 5.2")
    parser.add_argument("--output-json", type=Path, help="provenance JSON output path")
    parser.add_argument("--with-staged", action="store_true", help="reserved; staged GUI remains documented-only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        specs = load_manifest()
        check_completeness(specs)
        tags = parse_values(args.tags)
        versions = parse_values(args.versions)
        if any(version not in VERSION_ORDER for version in versions):
            raise ManifestError(f"unsupported Blender version selector: {', '.join(versions)}")
        selected = select_specs(specs, tags=tags, versions=versions)
        paths = resolve_blender_paths(args.blender42, args.blender52)
        if args.list:
            for spec in selected:
                versions_text = ",".join(spec.versions) if spec.mode not in {"fast-python", "fast-pytest"} else "-"
                print(f"{spec.file}\t{spec.mode}\t{versions_text}\t{','.join(spec.tags)}")
            return 0
        code, _ = run_manifest(
            specs,
            blender_paths=paths,
            tags=tags,
            versions=versions,
            dry_run=args.dry_run,
            output_path=args.output_json,
        )
        return code
    except (ManifestError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
