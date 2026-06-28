import pytest
import numpy as np

from packed_data_structures.graph.overlay import GraphOverlay
from packed_data_structures.overlays.registry import SchemaRegistry
from packed_data_structures.database import PackedArrayDB


def test_graph_overlay_initialization():
    graph = GraphOverlay(index_dtype=np.uint32)
    nodes = graph.add_node_layer("nodes")
    edges = graph.add_edge_layer("edges", source=nodes, target=nodes)

    assert "nodes" in graph.layers
    assert "edges" in graph.layers
    assert edges.src.name == "src"
    assert edges.tgt.name == "tgt"


def test_graph_overlay_add_entries_and_get_touching():
    graph = GraphOverlay(index_dtype=np.uint32)
    nodes = graph.add_node_layer("nodes")
    edges = graph.add_edge_layer("edges", source=nodes, target=nodes)

    registry = SchemaRegistry()
    graph.register_schema(registry)

    db = PackedArrayDB(*registry.build())
    graph.bind(db)

    with db.transaction():
        # Add 3 nodes
        n0 = nodes.add_entry()
        n1 = nodes.add_entry()
        n2 = nodes.add_entry()

        # Add edges: n0 -> n1, n1 -> n2, n0 -> n2
        e0 = edges.add_entry(src=n0, tgt=n1)
        e1 = edges.add_entry(src=n1, tgt=n2)
        e2 = edges.add_entry(src=n0, tgt=n2)

    # Check touching edges for n0 (source for e0 and e2, target for none)
    touching_n0_out = edges.get_touching(n0, kind="outgoing")
    assert set(touching_n0_out) == {e0, e2}

    touching_n0_in = edges.get_touching(n0, kind="incoming")
    assert set(touching_n0_in) == set()

    # Check touching edges for n1 (source for e1, target for e0)
    touching_n1 = edges.get_touching(n1, kind="both")
    assert set(touching_n1) == {e0, e1}

    # Check touching edges for n2 (target for e1, e2)
    touching_n2 = edges.get_touching(n2, kind="incoming")
    assert set(touching_n2) == {e1, e2}
