"""Incremental incident-face index for local boundary closure."""
from collections import defaultdict


class FaceEdgeIndex:
    def __init__(self, faces):
        self.owners = defaultdict(set)
        self.neighbors = defaultdict(set)
        for index, face in enumerate(faces):
            self.add(index, face)

    @staticmethod
    def edges(face):
        a, b, c = map(int, face)
        return ((min(a,b),max(a,b)), (min(b,c),max(b,c)), (min(c,a),max(c,a)))

    def add(self, index, face):
        for a, b in self.edges(face):
            self.owners[(a,b)].add(index)
            self.neighbors[a].add(b)
            self.neighbors[b].add(a)

    def remove(self, index, face):
        for a, b in self.edges(face):
            owners = self.owners[(a,b)]
            owners.remove(index)
            if not owners:
                del self.owners[(a,b)]
                self.neighbors[a].discard(b)
                self.neighbors[b].discard(a)

    def within(self, vertex_ids):
        ids = set(map(int, vertex_ids))
        return {(min(a,b),max(a,b)) for a in ids
                for b in self.neighbors.get(a, ()) if b in ids}
