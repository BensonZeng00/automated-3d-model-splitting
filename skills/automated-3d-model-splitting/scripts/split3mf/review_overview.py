"""Compact, numbered review sheets; full six-view evidence stays on disk."""
from pathlib import Path
from PIL import Image, ImageDraw


def build_review_overviews(items, directory, *, rows_per_page=5):
    paths = []
    for offset in range(0, len(items), rows_per_page):
        batch = items[offset:offset + rows_per_page]
        sheet = Image.new("RGB", (960, 190 * len(batch)), "white")
        draw = ImageDraw.Draw(sheet)
        for row, item in enumerate(batch):
            with Image.open(item["review_image"]) as source:
                zooms = source.crop((0, source.height // 2, source.width, source.height))
                zooms.thumbnail((960, 160), Image.Resampling.LANCZOS)
                sheet.paste(zooms, (0, row * 190 + 25))
            draw.text((8, row * 190 + 5), f"F{item['fragment_id']:03d}  +X / -X / +Y / -Y / +Z / -Z", fill="black")
        path = Path(directory) / f"review_overview_{len(paths)+1:02d}.png"
        sheet.save(path)
        paths.append(str(path))
    return paths
