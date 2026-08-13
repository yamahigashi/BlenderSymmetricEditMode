# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare the current direct topology result with the revision-3 golden."""

from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(PACKAGE_PARENT))

from fixtures_interior_host import GOLDEN_BUILDERS  # noqa: E402
from generate_interior_host_golden import (  # noqa: E402
    OUTPUT,  # noqa: E402
    run_fixture,  # noqa: E402
)


def _load_golden(path: Path = OUTPUT) -> dict:
    if not path.is_file():
        raise AssertionError(f"golden file is missing: {path}; run generate_interior_host_golden.py in Blender")
    return json.loads(path.read_text(encoding="utf-8"))


def check_golden(path: Path = OUTPUT) -> None:
    golden = _load_golden(path)
    assert golden.get("schema") == 1
    # The canonical corpus remains rev3; rev4/v5 implementations use the
    # same schema and values, so the runner must accept either label while
    # dispatching carrier_frames through generate_interior_host_golden's
    # compatibility wrappers.
    assert golden.get("revision") in {"rev3", "rev4"}
    assert re.fullmatch(r"[0-9a-fA-F]{40,64}", str(golden.get("git_head", "")))
    expected = {entry["name"]: entry for entry in golden.get("fixtures", ())}
    names = {name for name, _builder in GOLDEN_BUILDERS}
    assert set(expected) == names, (set(expected), names)
    for name, builder in GOLDEN_BUILDERS:
        actual = run_fixture(name, builder)
        assert actual == expected[name], (name, actual, expected[name])


def main() -> None:
    check_golden()
    print("YSE_INTERIOR_HOST_GOLDEN_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None
