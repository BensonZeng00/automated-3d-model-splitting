"""Keep source face ownership on the frozen Stage 03 boundary edges."""

from __future__ import annotations

import numpy as np


class SourceContactEdges:
    def __init__(self, vertex_count: int, faces: np.ndarray, components: list):
        self.vertex_count = int(vertex_count)
        owners = np.zeros(len(faces), dtype=np.int32)
        for part, component in enumerate(components, start=1):
            face_ids = np.asarray(component.global_faces, dtype=np.int64)
            if np.any(owners[face_ids]):
                raise ValueError("recognized parts share source faces")
            owners[face_ids] = part
        edges = np.sort(np.concatenate((
            faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
        )), axis=1)
        keys = edges[:, 0] * self.vertex_count + edges[:, 1]
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.owners = np.tile(owners, 3)[order]
        self.face_ids = np.tile(np.arange(len(faces), dtype=np.int64), 3)[order]

    def labels(self, loop: np.ndarray, part: int) -> np.ndarray:
        other = np.roll(loop, -1)
        keys = np.minimum(loop, other) * self.vertex_count + np.maximum(loop, other)
        first = np.searchsorted(self.keys, keys, side="left")
        last = np.searchsorted(self.keys, keys, side="right")
        labels = np.zeros(len(loop), dtype=np.int32)
        for index, (start, stop) in enumerate(zip(first, last)):
            peers = np.unique(self.owners[start:stop])
            peers = peers[(peers != 0) & (peers != part)]
            if len(peers) == 1:
                labels[index] = peers[0]
            elif len(peers) > 1:
                labels[index] = -1
        return labels

    def contact_face_ids(self, loop: np.ndarray, part: int, peer: int) -> np.ndarray:
        """Source faces on one side of edges shared with a given peer."""
        other = np.roll(loop, -1)
        keys = np.minimum(loop, other) * self.vertex_count + np.maximum(loop, other)
        first = np.searchsorted(self.keys, keys, side="left")
        last = np.searchsorted(self.keys, keys, side="right")
        result = []
        for start, stop in zip(first, last):
            owners = self.owners[start:stop]
            if np.any(owners == peer):
                result.extend(self.face_ids[start:stop][owners == part])
        return np.unique(result)


def retained_arc_labels(source_ids: np.ndarray, labels: np.ndarray,
                        frozen_ids: np.ndarray) -> np.ndarray:
    positions = {int(vertex): index for index, vertex in enumerate(source_ids)}
    if len(positions) != len(source_ids):
        raise ValueError("source loop repeats a vertex")
    indices = [positions[int(vertex)] for vertex in frozen_ids]
    result = []
    for first, last in zip(indices, indices[1:] + indices[:1]):
        arc = labels[np.arange(first, last if last > first else last + len(labels)) % len(labels)]
        distinct = np.unique(arc)
        result.append(int(distinct[0]) if len(distinct) == 1 else -1)
    return np.asarray(result, dtype=np.int32)
