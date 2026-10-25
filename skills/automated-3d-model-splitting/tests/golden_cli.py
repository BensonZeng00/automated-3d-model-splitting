"""Read the shared JSON summary contract without changing golden expectations."""
import json
from pathlib import Path


def artifact_directory(output, key):
    records = []
    for line in output.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "exit_code" in value:
            records.append(value)
    if len(records) != 1:
        raise AssertionError("Expected exactly one CLI JSON summary")
    summary = records[0]
    if summary["exit_code"] != 0 or not isinstance(summary.get(key), str):
        raise AssertionError(f"Missing successful {key}: {summary}")
    return Path(summary[key])


from contextlib import contextmanager
import os
import tempfile


@contextmanager
def golden_directory(prefix):
    """Optionally retain full evidence for a requested regression investigation."""
    root = os.environ.get("GOLDEN_ARTIFACT_ROOT")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        yield tempfile.mkdtemp(prefix=prefix, dir=root)
    else:
        with tempfile.TemporaryDirectory(prefix=prefix) as directory:
            yield directory
