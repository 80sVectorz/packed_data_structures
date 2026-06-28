import pytest
import numpy as np
from packed_data_structures.schemas import (
    TableSchema,
    IndexSpec,
    DataColSchema,
    ForeignKeySchema,
    AdjacencyListConf,
)
from packed_data_structures.database import PackedArrayDB
from packed_data_structures.graph.overlay import GraphOverlay


@pytest.fixture
def base_schemas():
    """Returns a tuple of (table_a_schema, table_b_schema) for basic testing."""
    idx_spec = IndexSpec.from_dtype(np.uint32)

    col_weight = DataColSchema("weight", np.float32, default=1.0)
    col_position = DataColSchema("position", np.float32, shape=(3,), default=0.0)

    table_a_schema = TableSchema(
        name="table_a",
        index_spec=idx_spec,
        cols=[col_weight, col_position]
    )

    fk_to_a = ForeignKeySchema(
        name="link_to_a",
        target_table=table_a_schema,
        adjacency_conf=AdjacencyListConf(track_counts=True)
    )

    table_b_schema = TableSchema(
        name="table_b",
        index_spec=idx_spec,
        cols=[fk_to_a]
    )

    return table_a_schema, table_b_schema


@pytest.fixture
def empty_db(base_schemas):
    """Returns an empty PackedArrayDB initialized with the base schemas."""
    table_a_schema, table_b_schema = base_schemas
    return PackedArrayDB(table_a_schema, table_b_schema)


@pytest.fixture
def populated_db(empty_db, base_schemas):
    """Returns a DB with some rows and linked foreign keys."""
    table_a_schema, table_b_schema = base_schemas
    table_a = empty_db.get_table(table_a_schema)
    table_b = empty_db.get_table(table_b_schema)

    col_weight = next(c for c in table_a_schema.cols if c.name == "weight")
    col_position = next(c for c in table_a_schema.cols if c.name == "position")
    fk_to_a = next(c for c in table_b_schema.cols if c.name == "link_to_a")

    with empty_db.transaction():
        staged_a = table_a.add_entries(
            records={
                col_weight: [1.0, 2.0, 3.0],
                col_position: [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
            },
            records_shape="col_major"
        )
        
        # Add entries to table_b pointing to the entries in table_a
        table_b.add_entries(
            records={
                fk_to_a: [staged_a[0], staged_a[1], staged_a[0]]
            },
            records_shape="col_major"
        )

    return empty_db


@pytest.fixture
def graph_db():
    """Returns an empty GraphOverlay instance."""
    graph = GraphOverlay(index_dtype=np.uint32)
    nodes = graph.add_node_layer("nodes")
    graph.add_edge_layer("edges", source=nodes, target=nodes)
    return graph
