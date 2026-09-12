"""Indexed winding repair: one shared-edge lookup, then linear graph traversal."""
import networkx as nx
import numpy as np


def fix_winding_indexed(mesh):
    """Match ordered breadth-first repair without per-face edge regrouping.

    Coordinates, face order and vertex membership never change. Non-orientable
    cycles remain visible to the caller's topology validation.
    """
    if mesh.is_winding_consistent:
        return
    faces = np.asarray(mesh.faces, dtype=np.int64)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if not len(adjacency):
        return
    shared = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
    directions = []
    for column in (0, 1):
        triangles = faces[adjacency[:, column]]
        directions.append(np.any(
            (triangles == shared[:, 0, None])
            & (np.roll(triangles, -1, axis=1) == shared[:, 1, None]), axis=1
        ))
    parity = directions[0] == directions[1]
    # Keep the original component and neighbor traversal order exactly.
    # Only shared-edge parity lookup is changed, eliminating tiny regroup calls.
    graph = nx.Graph()
    graph.add_weighted_edges_from(zip(adjacency[:, 0], adjacency[:, 1], parity))
    flips = np.zeros(len(faces), dtype=np.int8)
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        root = next(iter(subgraph.nodes()))
        for parent, child in nx.bfs_edges(subgraph, root):
            flips[child] = int(flips[parent]) ^ int(graph[parent][child]['weight'])
    changed = flips == 1
    if np.any(changed):
        repaired = faces.copy()
        repaired[changed] = repaired[changed, ::-1]
        mesh.faces = repaired
