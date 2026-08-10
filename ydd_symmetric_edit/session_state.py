from __future__ import annotations

import time
from collections import OrderedDict

from ._types import (
    HistoryRecord,
    KnifeSession,
)

_SESSIONS: dict[int, KnifeSession] = {}
_HISTORY_RECORDS: OrderedDict[int, HistoryRecord] = OrderedDict()
_NEXT_HISTORY_TOKEN = max(1, int(time.time_ns() & 0x7FFFFFFF))
_HISTORY_REPAIR_QUEUED = False
_HISTORY_REPAIR_BUSY = False
_MAX_HISTORY_RECORDS = 256
_HISTORY_SEQUENCE = 0
_FINISH_REPORTS: list[tuple[str, str]] = []
_PASSTHROUGH_POLL_INTERVAL = 0.01
_PASSTHROUGH_START_GRACE = 0.75


def _new_history_token() -> int:
    global _NEXT_HISTORY_TOKEN

    while True:
        token = _NEXT_HISTORY_TOKEN
        _NEXT_HISTORY_TOKEN = (_NEXT_HISTORY_TOKEN + 1) & 0x7FFFFFFF
        if _NEXT_HISTORY_TOKEN == 0:
            _NEXT_HISTORY_TOKEN = 1
        if token and token not in _HISTORY_RECORDS:
            return token
