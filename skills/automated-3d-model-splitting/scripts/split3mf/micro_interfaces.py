"""Keep principal connectors; close secondary print-scale loops without posts."""
import numpy as np
from .print_tolerance import current
from .reporting import runtime_log


def loop_area_span_metrics(vertices, records):
    """Measure variable-length loops in one compiled NumPy pass."""
    lengths = np.asarray([len(record["loop"]) for record in records], dtype=np.int64)
    if not len(lengths):
        return np.empty(0), np.empty(0)
    starts = np.r_[0, np.cumsum(lengths[:-1])]
    flat_ids = np.concatenate([
        np.asarray(record["loop"], dtype=np.int64) for record in records
    ])
    points = np.asarray(vertices)[flat_ids]
    minimum = np.minimum.reduceat(points, starts, axis=0)
    maximum = np.maximum.reduceat(points, starts, axis=0)
    spans = np.linalg.norm(maximum - minimum, axis=1)
    centers = np.add.reduceat(points, starts, axis=0) / lengths[:, None]
    centered = points - np.repeat(centers, lengths, axis=0)
    following = np.arange(len(points), dtype=np.int64) + 1
    following[starts + lengths - 1] = starts
    cross_lengths = np.linalg.norm(
        np.cross(centered, points[following] - np.repeat(centers, lengths, axis=0)),
        axis=1,
    )
    areas = np.add.reduceat(cross_lengths, starts) * 0.5
    return areas, spans


def filter_micro_interface_loops(vertices, records, *, part_index):
    if len(records) < 2 or current().micro_area_mm2 <= 0:
        return records, []
    areas, spans = loop_area_span_metrics(vertices, records)
    principal = int(np.argmax(areas))
    retained,ignored = [],[]
    for index, (record, area, span) in enumerate(zip(records, areas, spans)):
        if index != principal and area <= current().micro_area_mm2 and span <= 5.0:
            ignored.append(dict(loop_index=record['loop_index'],vertices=len(record['loop']),
                                area_mm2=area,span_mm=span,reason='secondary_micro_loop_direct_closure'))
        else:
            retained.append(record)
    if ignored:
        runtime_log('分界','micro_interface_loops_closed_without_connectors',
                    '微小次级回环保留表面并局部封闭，不生成独立榫卯',
                    part_index=int(part_index),ignored_loops=ignored,
                    retained_connector_loops=len(retained))
    return retained,ignored
