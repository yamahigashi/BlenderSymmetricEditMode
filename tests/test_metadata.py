# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep extension, legacy add-on, and development metadata synchronized."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "ydd_symmetric_edit"
WORKSPACE_ROOT = REPOSITORY_ROOT


def assigned_literal(module_path: Path, name: str):
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def run() -> None:
    manifest = tomllib.loads((PACKAGE_ROOT / "blender_manifest.toml").read_text(encoding="utf-8"))
    project = tomllib.loads((WORKSPACE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    bl_info = assigned_literal(PACKAGE_ROOT / "__init__.py", "bl_info")

    assert manifest["blender_version_min"] == "4.2.0"
    assert bl_info["blender"] == (4, 2, 0)
    assert manifest["version"] == "0.9.0"
    assert bl_info["version"] == (0, 9, 0)
    assert project["project"]["version"] == manifest["version"]
    assert project["project"]["requires-python"] == ">=3.11"
    print("YSE_METADATA_TEST_OK", flush=True)


if __name__ == "__main__":
    run()
