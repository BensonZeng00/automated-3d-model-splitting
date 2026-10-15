"""Spool mesh XML and stream it into ZIP without a per-triangle XML tree."""
import re
import tempfile
import zipfile
from xml.sax.saxutils import quoteattr


class MeshPayloadStore:
    def __init__(self):
        self.stream = None
        self.ranges = []

    def __enter__(self):
        self.stream = tempfile.TemporaryFile(mode='w+b')
        return self

    def __exit__(self, *_):
        self.stream.close()

    def add_mesh(self, mesh, colors, group_id, paint_tokens, format_float):
        self.stream.seek(0, 2)
        start = self.stream.tell()
        buffer = ['<mesh><vertices>']

        def append(value):
            buffer.append(value)
            if len(buffer) >= 4096:
                self.stream.write(''.join(buffer).encode('utf-8'))
                buffer.clear()

        for x, y, z in mesh.vertices:
            append(f'<vertex x="{format_float(x)}" y="{format_float(y)}" z="{format_float(z)}" />')
        append('</vertices><triangles>')
        for index, (a, b, c) in enumerate(mesh.faces):
            fields = f'v1="{int(a)}" v2="{int(b)}" v3="{int(c)}"'
            if colors is not None:
                color = int(colors[index])
                fields += f' pid="{group_id}" p1="{color}" p2="{color}" p3="{color}"'
            if paint_tokens is not None:
                fields += ' paint_color=' + quoteattr(paint_tokens[index])
            append('<triangle ' + fields + ' />')
        append('</triangles></mesh>')
        self.stream.write(''.join(buffer).encode('utf-8'))
        self.ranges.append((start, self.stream.tell()-start))
        return len(self.ranges)-1

    def write_model(self, archive, skeleton):
        info = zipfile.ZipInfo('3D/3dmodel.model', date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        cursor = 0
        with archive.open(info, 'w', force_zip64=True) as output:
            for match in re.finditer(rb'<_split3mf_payload index="(\d+)"\s*/>', skeleton):
                output.write(skeleton[cursor:match.start()])
                offset, remaining = self.ranges[int(match[1])]
                self.stream.seek(offset)
                while remaining:
                    block = self.stream.read(min(1024*1024, remaining))
                    if not block:
                        raise ValueError('Incomplete temporary mesh payload')
                    output.write(block)
                    remaining -= len(block)
                cursor = match.end()
            output.write(skeleton[cursor:])
