"""Human approval gate for a frozen Stage 04 interface plan."""

from __future__ import annotations

import json
from pathlib import Path

from .stage_cache import fingerprint_payload
from .stage_table_report import interface_table


ASSEMBLY_REVIEW_SCHEMA_VERSION = 1


class AssemblyReviewRequired(Exception):
    def __init__(self, review_path: Path, table_path: Path) -> None:
        super().__init__(f"Stage 04 interface plan needs user confirmation: {review_path}")
        self.review_path = Path(review_path)
        self.table_path = Path(table_path)


def load_direction_policy_corrections(review_path: str | Path | None) -> dict[str, str]:
    """Read explicit per-interface direction corrections from a prior review."""
    if review_path is None or not Path(review_path).expanduser().is_file():
        return {}
    reviewed = json.loads(Path(review_path).expanduser().read_text(encoding="utf-8"))
    corrections = {}
    for request in reviewed.get("correction_requests", []):
        policy = request.get("direction_policy")
        if policy is None:
            continue
        if policy != "smaller_area_first":
            raise ValueError(f"unsupported Stage 04 direction policy: {policy}")
        interface_id = str(request.get("interface_id") or "").strip()
        if not interface_id:
            raise ValueError("direction correction requires an interface_id")
        corrections[interface_id] = policy
    return corrections


def assembly_plan_fingerprint(plan: dict) -> str:
    """Bind approval to geometry and roles, independent of display tables."""
    return fingerprint_payload({
        "schema": plan.get("schema"),
        "recognized_boundary_fingerprint": plan.get("recognized_boundary_fingerprint"),
        "parts": plan.get("parts"),
        "interfaces": plan.get("interfaces"),
    })


def ensure_assembly_review(
    plan: dict,
    *,
    source_sha256: str,
    template_path: Path,
    table_path: Path,
    review_path: Path | None = None,
) -> dict:
    """Create a review template or validate explicit approval of this exact plan."""
    template_path = Path(template_path)
    selected_path = Path(review_path).expanduser() if review_path is not None else template_path
    fingerprint = assembly_plan_fingerprint(plan)
    payload = {
        "schema_version": ASSEMBLY_REVIEW_SCHEMA_VERSION,
        "source_sha256": str(source_sha256),
        "plan_fingerprint": fingerprint,
        "recognized_boundary_fingerprint": plan.get("recognized_boundary_fingerprint"),
        "interface_count": int(plan.get("interface_count", 0)),
        "user_confirmed": False,
        "correction_requests": [],
        "instruction": (
            "Review 04_interface_table.md and the frozen boundaries. If an interface is "
            "wrong, keep user_confirmed=false and list each correction as "
            '{"interface_id":"I001","issue":"...","requested_change":"..."} '
            "in correction_requests (use interface_id=null for a missing interface). "
            "Revise Stage 04 and review the new plan. "
            "Set user_confirmed=true only when every relation is correct."
        ),
        "interface_table": plan.get("interface_table") or interface_table(plan),
    }
    if selected_path.is_file():
        reviewed = json.loads(selected_path.read_text(encoding="utf-8"))
        if not isinstance(reviewed, dict):
            raise ValueError("Stage 04 review JSON must contain an object")
        requests = reviewed.get("correction_requests", [])
        for field in (
            "schema_version", "source_sha256",
            "recognized_boundary_fingerprint", "interface_count",
        ):
            if reviewed.get(field) != payload[field]:
                raise ValueError(f"Stage 04 review is stale or invalid: {field}")
        if (not requests or reviewed.get("user_confirmed") is True) and reviewed.get(
            "plan_fingerprint"
        ) != payload["plan_fingerprint"]:
            raise ValueError("Stage 04 review is stale or invalid: plan_fingerprint")
        if type(reviewed.get("user_confirmed")) is not bool:
            raise ValueError("Stage 04 review user_confirmed must be a boolean")
        if not isinstance(requests, list) or any(not isinstance(item, dict) for item in requests):
            raise ValueError("Stage 04 correction_requests must be an array of objects")
        known_interfaces = {str(item["interface_id"]) for item in plan.get("interfaces", [])}
        for request in requests:
            interface_id = request.get("interface_id")
            if interface_id not in (None, "") and str(interface_id) not in known_interfaces:
                raise ValueError("Stage 04 correction request references an unknown interface")
            if not str(request.get("issue", "")).strip() or not str(request.get("requested_change", "")).strip():
                raise ValueError("Stage 04 correction request needs issue and requested_change")
        payload["user_confirmed"] = reviewed["user_confirmed"]
        payload["correction_requests"] = requests
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not payload["user_confirmed"]:
        raise AssemblyReviewRequired(template_path, table_path)
    return payload
