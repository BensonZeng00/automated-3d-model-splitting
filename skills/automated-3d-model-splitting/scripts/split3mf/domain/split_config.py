from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class SplitConfig:
    namespace: argparse.Namespace
    input_path: Path
    preflight_checks: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.namespace, name)
