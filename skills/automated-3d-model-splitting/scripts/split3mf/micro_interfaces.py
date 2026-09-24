"""Keep principal connectors; close secondary print-scale loops without posts."""
import numpy as np
from .print_tolerance import current
from .reporting import runtime_log


def filter_micro_interface_loops(vertices, records, *, part_index):
    if len(records) < 2 or current().micro_area_mm2 <= 0:
        return records, []
    metrics = []
    for record in records:
        points = np.asarray(vertices)[record['loop']]
        center = points.mean(axis=0)
        area = float(np.linalg.norm(np.cross(points-center, np.roll(points,-1,axis=0)-center),axis=1).sum()/2)
        span = float(np.linalg.norm(np.ptp(points,axis=0)))
        metrics.append((area,span))
    principal = int(np.argmax([area for area,_ in metrics]))
    retained,ignored = [],[]
    for index,(record,(area,span)) in enumerate(zip(records,metrics)):
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
