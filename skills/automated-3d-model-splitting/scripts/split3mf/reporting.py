"""Human-readable and machine-readable progress for the active pipeline."""

from __future__ import annotations

import json
import time


_STARTED_AT = time.perf_counter()


def runtime_log(stage: str, event: str, message: str, **details) -> None:
    record = {
        "stage": str(stage),
        "event": str(event),
        "message": str(message),
        "elapsed_seconds": round(time.perf_counter() - _STARTED_AT, 3),
        **details,
    }
    suffix = " " + json.dumps(details, ensure_ascii=False, sort_keys=True) if details else ""
    print(f"[{stage}] {message}{suffix}", flush=True)
    print("runtime_step=" + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
