"""Hash-indexed interface roles shared by planning and mesh construction."""

from __future__ import annotations

import re


_PART_ID = re.compile(r"^(P\d{2})(?:\s|$)")


def interface_role_index(plan: dict) -> dict[str, dict[str, str]]:
    """Build and verify an O(1) interface-id lookup from the published role table."""
    rows = plan.get("relation_table", {}).get("rows")
    interfaces = plan.get("interfaces")
    if not isinstance(rows, list) or not isinstance(interfaces, list):
        raise ValueError("interface plan must contain role-table rows and interfaces")

    table_index: dict[str, tuple[str, str]] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) < 3:
            raise ValueError("interface role-table row must contain id, tenon and mortise")
        interface_id = str(row[0]).strip()
        if not interface_id or interface_id in table_index:
            raise ValueError(f"interface role table has a duplicate or empty id: {interface_id!r}")
        tenon_match = _PART_ID.match(str(row[1]).strip())
        mortise_match = _PART_ID.match(str(row[2]).strip())
        if tenon_match is None or mortise_match is None:
            raise ValueError(f"{interface_id}: role table contains an invalid part id")
        tenon_part, mortise_part = tenon_match.group(1), mortise_match.group(1)
        if tenon_part == mortise_part:
            raise ValueError(f"{interface_id}: a part cannot be both tenon and mortise")
        table_index[interface_id] = (tenon_part, mortise_part)

    plan_index: dict[str, dict[str, str]] = {}
    for relation in interfaces:
        interface_id = str(relation.get("interface_id", "")).strip()
        if not interface_id or interface_id in plan_index:
            raise ValueError(f"interface plan has a duplicate or empty id: {interface_id!r}")
        tenon_part = str(relation.get("tenon_part", ""))
        mortise_part = str(relation.get("mortise_part", ""))
        if table_index.get(interface_id) != (tenon_part, mortise_part):
            raise ValueError(
                f"{interface_id}: interface table roles do not match the geometry plan "
                f"(table={table_index.get(interface_id)}, "
                f"plan={(tenon_part, mortise_part)})"
            )
        parts = {str(value) for value in relation.get("parts", [])}
        if tenon_part == mortise_part or {tenon_part, mortise_part} != parts:
            raise ValueError(f"{interface_id}: tenon/mortise ids do not match its part pair")
        if not relation.get("contact", {}).get("shared_boundary_loops"):
            raise ValueError(f"{interface_id}: interface has no shared boundary loops")
        plan_index[interface_id] = {
            "tenon_part": tenon_part,
            "mortise_part": mortise_part,
        }

    if set(table_index) != set(plan_index):
        raise ValueError("interface role table and geometry plan contain different interface ids")
    return plan_index
