# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import functools
import gc


def gc_disabled_during_execute(execute):
    """Disable cyclic GC for the wrapped ``execute`` body.

    Saves entry state and restores only if it was enabled, so nested/
    re-entrant calls stay idempotent. Wraps every return path, including
    exceptions.
    """

    @functools.wraps(execute)
    def wrapper(self, context, *args, **kwargs):
        prior = gc.isenabled()
        gc.disable()
        try:
            return execute(self, context, *args, **kwargs)
        finally:
            if prior:
                gc.enable()

    return wrapper
