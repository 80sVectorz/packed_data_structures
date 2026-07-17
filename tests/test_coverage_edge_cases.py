import numpy as np
from packed_data_structures.database import PackedArrayDB
from packed_data_structures.schemas import (
    TableSchema,
    IndexSpec,
    DataColSchema,
    ForeignKeySchema,
    AdjacencyListConf,
)


def test_add_node_and_edge_same_tx():
    # Test adding target and source in same transaction (hits lines 413, 520, 606 bounds checking)
    idx_spec = IndexSpec.from_dtype(np.uint32)
    col_val = DataColSchema("val", np.int32)
    schema_a = TableSchema(name="A", index_spec=idx_spec, cols=[col_val])

    fk = ForeignKeySchema(
        name="fk", target_table=schema_a, adjacency_conf=AdjacencyListConf()
    )
    schema_b = TableSchema(name="B", index_spec=idx_spec, cols=[fk])

    db = PackedArrayDB(schema_a, schema_b)
    table_a = db.get_table(schema_a)
    table_b = db.get_table(schema_b)

    with db.transaction():
        # Add to A
        table_a.add_entries(records={col_val: [10, 20]}, records_shape="col_major")
        # Add to B pointing to newly added A
        table_b.add_entries(records={fk: [0, 1]}, records_shape="col_major")

    assert len(table_a) == 2
    assert len(table_b) == 2
    assert table_a[fk.adj_head][0] == 0
    assert table_a[fk.adj_head][1] == 1


def test_update_and_relocate_same_tx():
    # Test updating a row's FK and relocating it in the same transaction
    idx_spec = IndexSpec.from_dtype(np.uint32)
    col_val = DataColSchema("val", np.int32)
    schema_a = TableSchema(name="A", index_spec=idx_spec, cols=[col_val])
    fk = ForeignKeySchema(
        name="fk",
        target_table=schema_a,
        adjacency_conf=AdjacencyListConf(track_counts=True, counts_dtype=np.uint32),
    )
    schema_b = TableSchema(name="B", index_spec=idx_spec, cols=[fk])

    db = PackedArrayDB(schema_a, schema_b)
    table_a = db.get_table(schema_a)
    table_b = db.get_table(schema_b)

    with db.transaction():
        table_a.add_entries(records={"val": [10, 20]}, records_shape="col_major")
        table_b.add_entries(records={fk: [0, 0]}, records_shape="col_major")

    assert len(table_b) == 2
    assert table_a[fk.adj_count][0] == 2

    # We have B[0] -> A[0], B[1] -> A[0]
    # Delete B[0], causing B[1] to be relocated to 0 via swap-and-pop.
    # In the SAME transaction, update B[1]'s target to A[1].
    with db.transaction():
        table_b.del_entry(0)
        table_b.update_entries(updates={fk: {1: 1}})

    assert len(table_b) == 1
    # B[1] moved to 0, and points to A[1]
    assert table_b[fk][0] == 1
    # A[1] should have head pointing to 0!
    assert table_a[fk.adj_head][1] == 0
