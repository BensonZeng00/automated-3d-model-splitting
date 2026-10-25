"""Human-readable and machine-readable progress for the active pipeline."""

from __future__ import annotations

def runtime_log(stage: str, event: str, message: str, **details) -> None:
    from .execution_log import emit
    emit(stage, event, message, level="WARNING" if stage == "警告" else "INFO", **details)
