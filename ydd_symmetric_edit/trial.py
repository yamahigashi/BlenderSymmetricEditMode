# SPDX-License-Identifier: GPL-3.0-or-later
# Pure Python: no bpy import, so this module can be exercised by plain pytest
# and patched line-by-line by scripts/build_dist.py.
from __future__ import annotations

from datetime import date

TRIAL_BUILD = False
TRIAL_DAYS = 14


def days_left(start_iso: str, today: date) -> int | None:
    """Return the number of trial days remaining, or None if start_iso is unusable."""
    if not start_iso:
        return None
    try:
        start = date.fromisoformat(start_iso)
    except ValueError:
        return None
    return TRIAL_DAYS - (today - start).days
