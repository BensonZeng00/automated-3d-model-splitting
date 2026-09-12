from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class InterfacePreflightFinding:
    parent_index: int
    child_index: int
    loop_index: int
    status: str
    cap_mode: str | None
    parent_thickness_mm: float | None
    safe_maximum_mm: float | None
    effective_minimum_mm: float | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FullTreePreflightReport:
    status: str
    elapsed_seconds: float
    planned_parent_indices: tuple[int, ...]
    findings: tuple[InterfacePreflightFinding, ...]
    error: str | None = None
    blocked_parent_index: int | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
            "planned_parent_indices": list(self.planned_parent_indices),
            "interface_count": len(self.findings),
            "findings": [finding.__dict__ for finding in self.findings],
            "error": self.error,
            "blocked_parent_index": self.blocked_parent_index,
        }


class FullTreePreflightError(ValueError):
    def __init__(self, report: FullTreePreflightReport) -> None:
        super().__init__(report.error or "full-tree interface preflight failed")
        self.report = report


class FullTreePreflightService:
    """Plan every recursive interface before the first expensive Boolean."""

    def run(
        self,
        recursive_steps: list[dict],
        context_loader: Callable[[int], tuple[list[dict], dict, dict]],
    ) -> FullTreePreflightReport:
        started_at = time.perf_counter()
        findings: list[InterfacePreflightFinding] = []
        planned_parents: list[int] = []
        for step in recursive_steps:
            parent_index = int(step["local_body_index"])
            planned_parents.append(parent_index)
            try:
                refs, _union_by_child, _subtree_by_child = context_loader(
                    parent_index
                )
            except ValueError as exc:
                report = FullTreePreflightReport(
                    status="BLOCK",
                    elapsed_seconds=round(time.perf_counter() - started_at, 3),
                    planned_parent_indices=tuple(planned_parents),
                    findings=tuple(findings),
                    error=str(exc),
                    blocked_parent_index=parent_index,
                )
                raise FullTreePreflightError(report) from exc
            for ref in refs:
                decision = ref.get("cap_decision")
                record = dict(getattr(decision, "record", {}) or {})
                safe_maximum = _optional_float(
                    record.get("safe_maximum_inward_depth_mm")
                )
                effective_minimum = _optional_float(
                    record.get("effective_minimum_inward_depth_mm")
                )
                parent_thickness = _optional_float(
                    record.get("parent_thickness_min_mm")
                )
                status = "PASS"
                if safe_maximum is not None and safe_maximum <= 0.0:
                    status = "BLOCK"
                elif (
                    safe_maximum is not None
                    and effective_minimum is not None
                    and safe_maximum + 1e-9 < effective_minimum
                ):
                    status = "BLOCK"
                elif bool(record.get("parent_thickness_is_lower_bound")):
                    status = "WARN"
                findings.append(
                    InterfacePreflightFinding(
                        parent_index=parent_index,
                        child_index=int(ref["component_index"]),
                        loop_index=int(ref.get("loop_index", 0)),
                        status=status,
                        cap_mode=(
                            None if decision is None else str(decision.mode)
                        ),
                        parent_thickness_mm=parent_thickness,
                        safe_maximum_mm=safe_maximum,
                        effective_minimum_mm=effective_minimum,
                        details={
                            "boundary_vertex_count": len(
                                ref.get("global_loop", [])
                            ),
                            "requested_cap_mode": ref.get(
                                "requested_cap_mode"
                            ),
                            "fit_clearance_mm": ref.get("fit_clearance_mm"),
                        },
                    )
                )
        blocked = [finding for finding in findings if finding.status == "BLOCK"]
        report = FullTreePreflightReport(
            status="BLOCK" if blocked else "PASS",
            elapsed_seconds=round(time.perf_counter() - started_at, 3),
            planned_parent_indices=tuple(planned_parents),
            findings=tuple(findings),
            error=(
                "one or more interfaces failed the full-tree safety preflight"
                if blocked
                else None
            ),
            blocked_parent_index=(blocked[0].parent_index if blocked else None),
        )
        if blocked:
            raise FullTreePreflightError(report)
        return report


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
