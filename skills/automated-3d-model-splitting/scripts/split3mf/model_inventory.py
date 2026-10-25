"""Stream mesh inventory without decoding paint or constructing geometry."""
import posixpath
import zipfile
from xml.etree import ElementTree as ET

from .admission import check

CORE = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"


def inspect_models(path, *, entry_filter=None):
    items = []
    with zipfile.ZipFile(path) as archive:
        names = {}
        if "Metadata/model_settings.config" in archive.namelist():
            root = ET.fromstring(archive.read("Metadata/model_settings.config"))
            for part in root.iter("part"):
                name = next((m.get("value") for m in part.findall("metadata") if m.get("key") == "name"), None)
                if name:
                    names[part.get("id")] = name
        for entry in sorted(n for n in archive.namelist() if n.endswith(".model")):
            if entry_filter and entry != entry_filter:
                continue
            check("inventory", model_xml_bytes=archive.getinfo(entry).file_size)
            current = None
            unit = "millimeter"
            with archive.open(entry) as stream:
                for event, node in ET.iterparse(stream, events=("start", "end")):
                    if event == "start" and node.tag == CORE + "model":
                        unit = node.get("unit", "millimeter")
                    if event == "start" and node.tag == CORE + "object":
                        current = {"object_id": entry + "#" + node.get("id", ""), "entry": entry,
                                   "mesh_id": node.get("id"), "unit": unit, "name": node.get("name") or names.get(node.get("id")) or posixpath.basename(entry),
                                   "source_faces": 0, "vertices": 0, "paint_token_chars": 0,
                                   "material_tokens": set(), "bbox_min": None, "bbox_max": None}
                    elif event == "end" and current is not None:
                        if node.tag == CORE + "triangle":
                            current["source_faces"] += 1
                            token = node.get("paint_color", "DEFAULT")
                            current["paint_token_chars"] = max(current["paint_token_chars"], len(token))
                            # Complex selectors are not material IDs; count only literal tokens.
                            if len(token) <= 2 or token == "DEFAULT":
                                current["material_tokens"].add(token)
                        elif node.tag == CORE + "vertex":
                            xyz = [float(node.get(axis)) for axis in ("x", "y", "z")]
                            current["vertices"] += 1
                            if current["bbox_min"] is None:
                                current["bbox_min"], current["bbox_max"] = xyz, xyz.copy()
                            else:
                                current["bbox_min"] = list(map(min, zip(current["bbox_min"], xyz)))
                                current["bbox_max"] = list(map(max, zip(current["bbox_max"], xyz)))
                        elif node.tag == CORE + "object":
                            if current["source_faces"]:
                                current["literal_color_tokens"] = sorted(current.pop("material_tokens"))
                                current["size_source_units"] = [b-a for a,b in zip(current["bbox_min"], current["bbox_max"])]
                                items.append(current)
                            current = None
                        node.clear()
    return items


def select_model(items, object_id=None, model_entry=None):
    if not items:
        raise ValueError("3MF contains no mesh objects")
    candidates = items
    if object_id:
        candidates = [item for item in candidates if item["object_id"] == object_id]
    if model_entry:
        candidates = [item for item in candidates if item["entry"] == model_entry or item["object_id"] == model_entry]
    if not candidates:
        raise ValueError("selected object does not exist or conflicts with --model-entry")
    return candidates[0] if len(candidates) == 1 else None
