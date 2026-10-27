"""JSONL diagnostics with bounded console summaries and legacy-output capture."""
from contextlib import redirect_stdout, redirect_stderr
from contextvars import ContextVar
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import time
import uuid
import re

from .admission import AdmissionError

_ACTIVE = ContextVar("execution_log", default=None)
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}


def emit(stage, event, message, *, level="INFO", **details):
    logger = _ACTIVE.get()
    if logger:
        logger.emit(stage, event, message, level=level, **details)


def summary(record):
    logger = _ACTIVE.get()
    if logger:
        logger.summary.update(record)
        if record.get("object_id"):
            logger.object_id = record["object_id"]
    else:
        print(json.dumps(record, ensure_ascii=False, separators=(",", ":")), flush=True)


class RunLog:
    def __init__(self, path, level, object_id):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a", encoding="utf-8")
        self.console = sys.stdout
        self.level = level
        self.object_id = object_id
        self.started = time.perf_counter()
        self.stage = "preflight"
        self.last_error = ""
        self.summary = {}

    def emit(self, stage, event, message, *, level="INFO", **details):
        self.stage = str(stage)
        raw_message = str(message)
        short_message = raw_message.split("diagnostic=", 1)[0][:400]
        brief_details = {k: v for k, v in details.items()
                         if v is None or type(v) in (int, float, bool) or isinstance(v, str) and len(v) <= 400}
        record = {"time": datetime.now(timezone.utc).isoformat(), "level": level,
                  "stage": self.stage, "event": event, "object_id": self.object_id,
                  "interface_id": details.get("interface_id"),
                  "elapsed_seconds": round(time.perf_counter()-self.started, 3),
                  "message": raw_message if level == "DEBUG" else short_message,
                  "details": details if level == "DEBUG" else brief_details}
        if record["interface_id"] is None:
            match = re.search(r"\bI\d{3,}\b", str(message))
            if match:
                record["interface_id"] = match.group(0)
        self.file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
        if level != "DEBUG" and (raw_message != short_message or len(details) != len(brief_details)):
            diagnostic = {**record, "level": "DEBUG", "event": event + "_details",
                          "message": raw_message, "details": details}
            self.file.write(json.dumps(diagnostic, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
        self.file.flush()
        if level == "ERROR":
            self.last_error = short_message
        if event != "legacy_output" and _LEVELS[level] >= _LEVELS[self.level]:
            print(f"[{level}][{stage}] {short_message[:240]}", file=self.console, flush=True)


class _LineSink(io.TextIOBase):
    def __init__(self, logger, level):
        self.logger, self.level, self.pending = logger, level, ""

    def write(self, text):
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self._record(line)
        return len(text)

    def _record(self, line):
        if not line.strip():
            return
        self.logger.emit(self.logger.stage, "legacy_output", line, level=self.level)
        key, sep, value = line.partition("=")
        if sep and key == "preflight":
            self.logger.summary["preflight"] = json.loads(value)
            return
        if sep and key.endswith(("_required", "_review", "_failure")):
            try:
                data = json.loads(value)
            except ValueError:
                return
            if isinstance(data, dict):
                for field in ("review_json", "table", "manifest_path", "decision_path", "directory", "error"):
                    if field in data:
                        self.logger.summary[field] = str(data[field])[:400]
        elif sep and key in ("stage_artifacts", "stage05_artifacts", "stage04_artifacts", "output_3mf"):
            self.logger.summary[key] = value

    def flush(self):
        if self.pending:
            self._record(self.pending)
            self.pending = ""


def run_logged(callback, path, *, level="WARNING", object_id=None):
    """Apply the same exit/log contract to full runs and artifact replay CLIs."""
    logger = RunLog(path, level, object_id)
    token = _ACTIVE.set(logger)
    out, err = _LineSink(logger, "DEBUG"), _LineSink(logger, "ERROR")
    code = 0
    try:
        with redirect_stdout(out), redirect_stderr(err):
            result = callback()
            code = result if type(result) is int else 0
    except AdmissionError as exc:
        code = 6
        logger.emit(exc.record["stage"], "admission_rejected", str(exc), level="ERROR", diagnostic=exc.record)
        logger.summary.update(exc.record)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 3)
        if isinstance(exc.code, str):
            logger.emit(logger.stage, "execution_failed", exc.code, level="ERROR")
    except Exception as exc:
        code = 3
        logger.emit(logger.stage, "execution_failed", str(exc), level="ERROR",
                    exception_type=type(exc).__name__, diagnostic=getattr(exc, "record", {}))
    finally:
        out.flush()
        err.flush()
        status = {0:"completed", 4:"review_required", 5:"selection_required", 6:"unsupported"}.get(code, "failed")
        logger.summary.update(exit_code=code, status=logger.summary.get("status", status),
                              stage=logger.stage, object_id=logger.object_id, log_path=str(logger.path))
        if code and logger.last_error:
            logger.summary["error"] = logger.last_error.split("diagnostic=", 1)[0][:400]
        result_path = logger.path.with_suffix(".summary.json")
        result_path.write_text(json.dumps(logger.summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({**logger.summary, "summary_path": str(result_path)},
                         ensure_ascii=False, separators=(",", ":")), file=logger.console, flush=True)
        logger.file.close()
        _ACTIVE.reset(token)
    return code


def add_execution_arguments(parser):
    parser.add_argument("--strict-quality", action="store_true", help="Stop on geometry quality failures.")
    parser.add_argument("--complexity-config", type=Path, help="JSON complexity limit overrides.")
    parser.add_argument("--log-file", type=Path, help="Full local JSONL diagnostics.")
    parser.add_argument("--log-level", choices=list(_LEVELS), default="WARNING", help="Console threshold; full log remains local.")


def logged_cli(callback, args, default_root):
    from .admission import ComplexityLimits, limits_scope
    def execute():
        from .quality_report import report_scope
        with limits_scope(ComplexityLimits.load(args.complexity_config)):
            with report_scope(Path(default_root) / (uuid.uuid4().hex + '_split_report.md'),
                              strict=getattr(args, 'strict_quality', False)):
                return callback()
    path = args.log_file or Path(default_root) / (uuid.uuid4().hex + ".jsonl")
    return run_logged(execute, path, level=args.log_level)
