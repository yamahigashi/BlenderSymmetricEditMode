# SPDX-License-Identifier: GPL-3.0-or-later

"""Pure-Python unit tests for trial support: no bpy required.

trial.py is loaded directly from disk (bypassing ydd_symmetric_edit/__init__.py,
which imports bpy) so this file can run under plain pytest.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trial = _load_module("yse_trial", REPOSITORY_ROOT / "ydd_symmetric_edit" / "trial.py")
build_dist = _load_module("build_dist", SCRIPTS_DIR / "build_dist.py")


# --- trial.days_left --------------------------------------------------------


def test_days_left_on_start_day_equals_trial_days():
    start = "2026-08-01"
    assert trial.days_left(start, date(2026, 8, 1)) == trial.TRIAL_DAYS


def test_days_left_on_deadline_day_is_zero():
    start = "2026-08-01"
    deadline = date(2026, 8, 1).fromordinal(date(2026, 8, 1).toordinal() + trial.TRIAL_DAYS)
    assert trial.days_left(start, deadline) == 0


def test_days_left_past_deadline_is_negative():
    start = "2026-08-01"
    past = date(2026, 8, 1).fromordinal(date(2026, 8, 1).toordinal() + trial.TRIAL_DAYS + 5)
    assert trial.days_left(start, past) == -5


def test_days_left_empty_string_is_none():
    assert trial.days_left("", date(2026, 8, 1)) is None


def test_days_left_malformed_string_is_none():
    assert trial.days_left("not-a-date", date(2026, 8, 1)) is None


def test_module_constants_are_default_off():
    assert trial.TRIAL_BUILD is False
    assert trial.TRIAL_DAYS == 14


# --- build_dist patch functions ---------------------------------------------


def test_replace_unique_replaces_single_occurrence():
    result = build_dist.replace_unique("TRIAL_BUILD = False\n", "TRIAL_BUILD = False", "TRIAL_BUILD = True")
    assert result == "TRIAL_BUILD = True\n"


def test_replace_unique_errors_on_zero_occurrences():
    with pytest.raises(build_dist.BuildError):
        build_dist.replace_unique("nothing here\n", "TRIAL_BUILD = False", "TRIAL_BUILD = True")


def test_replace_unique_errors_on_multiple_occurrences():
    text = "TRIAL_BUILD = False\nTRIAL_BUILD = False\n"
    with pytest.raises(build_dist.BuildError):
        build_dist.replace_unique(text, "TRIAL_BUILD = False", "TRIAL_BUILD = True")


def test_patch_manifest_name_appends_trial_suffix():
    text = 'id = "ydd_symmetric_edit"\nname = "ydd Symmetric Edit"\nversion = "0.9.0"\n'
    patched = build_dist.patch_manifest_name(text)
    assert 'name = "ydd Symmetric Edit (Trial)"' in patched
    assert 'name = "ydd Symmetric Edit"\n' not in patched


def test_patch_manifest_name_errors_when_pattern_missing():
    with pytest.raises(build_dist.BuildError):
        build_dist.patch_manifest_name('name = "Something Else"\n')


def test_patch_trial_module_sets_build_flag_and_days():
    text = "TRIAL_BUILD = False\nTRIAL_DAYS = 14\n"
    patched = build_dist.patch_trial_module(text, 30)
    assert "TRIAL_BUILD = True" in patched
    assert "TRIAL_DAYS = 30" in patched
    assert "TRIAL_BUILD = False" not in patched
    assert "TRIAL_DAYS = 14" not in patched


def test_patch_trial_module_errors_when_build_flag_missing():
    with pytest.raises(build_dist.BuildError):
        build_dist.patch_trial_module("TRIAL_DAYS = 14\n", 30)


def test_patch_trial_module_errors_when_days_missing():
    with pytest.raises(build_dist.BuildError):
        build_dist.patch_trial_module("TRIAL_BUILD = False\n", 30)


def test_patch_trial_module_matches_real_source_file():
    real_text = (REPOSITORY_ROOT / "ydd_symmetric_edit" / "trial.py").read_text(encoding="utf-8")
    patched = build_dist.patch_trial_module(real_text, 21)
    assert "TRIAL_BUILD = True" in patched
    assert "TRIAL_DAYS = 21" in patched


def test_patch_manifest_name_matches_real_manifest_file():
    real_text = (REPOSITORY_ROOT / "ydd_symmetric_edit" / "blender_manifest.toml").read_text(encoding="utf-8")
    patched = build_dist.patch_manifest_name(real_text)
    assert 'name = "ydd Symmetric Edit (Trial)"' in patched


# --- build_dist argument parsing --------------------------------------------


def test_parse_args_days_without_trial_errors():
    with pytest.raises(SystemExit):
        build_dist.parse_args(["--days", "30"])


def test_parse_args_days_with_trial_ok():
    args = build_dist.parse_args(["--trial", "--days", "30"])
    assert args.trial is True
    assert args.days == 30


def test_parse_args_default_days_is_trial_days():
    args = build_dist.parse_args(["--trial"])
    assert args.days == build_dist.DEFAULT_TRIAL_DAYS


def test_default_out_name_prefix():
    assert build_dist.default_out_name(trial=False).startswith("candidate_")
    assert build_dist.default_out_name(trial=True).startswith("trial_")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
