import pytest
from unittest.mock import patch
import packed_data_structures.transaction_context as tc
import packed_data_structures.edit_helpers as eh
import numpy as np
from packed_data_structures.database import PackedArrayDB
from packed_data_structures.schemas import (
    TableSchema,
    IndexSpec,
    DataColSchema,
    ForeignKeySchema,
    AdjacencyListConf,
    PolymorphicForeignKeySchema,
)

@pytest.fixture(params=["jit", "no_jit"], autouse=True)
def jit_mode(request):
    if request.param == "no_jit":
        patches = []
        for name in dir(tc):
            obj = getattr(tc, name)
            if callable(obj) and hasattr(obj, "py_func"):
                patches.append(patch.object(tc, name, obj.py_func))
        for p in patches:
            p.start()
        yield "no_jit"
        for p in patches:
            p.stop()
    else:
        yield "jit"


def test_combined_complex_schema():
    index_spec = IndexSpec(dtype=np.uint32, max_value=0xFFFFFFFE, missing=0xFFFFFFFF)

    # Table A
    col_a_val = DataColSchema(name="val_a", dtype=np.int32)
    schema_a = TableSchema(name="TableA", cols=[col_a_val], index_spec=index_spec)

    # Table B
    col_b_val = DataColSchema(name="val_b", dtype=np.float32)
    schema_b = TableSchema(name="TableB", cols=[col_b_val], index_spec=index_spec)

    # Table C (The combination table)
    col_c_val = DataColSchema(name="val_c", dtype=np.uint8)

    # Regular FK to A
    fk_to_a = ForeignKeySchema(
        name="fk_to_a",
        target_table=schema_a,
        adjacency_conf=AdjacencyListConf(track_counts=True, counts_dtype=np.uint32),
    )

    # Polymorphic FK to A or B
    pm_fk = PolymorphicForeignKeySchema(
        name="pm_fk",
        target_tables=[schema_a, schema_b],
        adjacency_conf=AdjacencyListConf(track_counts=True, counts_dtype=np.uint32),
    )

    schema_c = TableSchema(
        name="TableC", cols=[col_c_val, fk_to_a, pm_fk], index_spec=index_spec
    )

    # Create DB
    db = PackedArrayDB(schema_a, schema_b, schema_c)

    table_a = db.get_table(schema_a)
    table_b = db.get_table(schema_b)
    table_c = db.get_table(schema_c)

    # Start by inserting some A and B rows
    with db.transaction():
        staged_a = table_a.add_entries(
            records={col_a_val: [100, 200, 300]},
            records_shape="col_major",
        )
        staged_b = table_b.add_entries(
            records={col_b_val: [1.5, 2.5]},
            records_shape="col_major",
        )

    assert len(table_a) == 3
    assert len(table_b) == 2

    type_a = pm_fk.type_id_mapping[schema_a]
    type_b = pm_fk.type_id_mapping[schema_b]

    # Insert C rows linking to A and B using both FK and PM-FK
    with db.transaction():
        table_c.add_entries(
            records={
                col_c_val: [10, 20, 30, 40],
                fk_to_a: [staged_a[0], staged_a[0], staged_a[1], staged_a[2]],
                pm_fk: [staged_a[0], staged_b[1], staged_a[1], staged_b[0]],
                pm_fk.type_id_col: [type_a, type_b, type_a, type_b],
            },
            records_shape="col_major",
        )

    assert len(table_c) == 4

    # Verification
    # A[0] has C[0] and C[1] from fk_to_a, and C[0] from pm_fk
    assert table_a[fk_to_a.adj_count][0] == 2
    assert table_a[pm_fk.adj_count_columns[type_a]][0] == 1

    # B[1] has C[1] from pm_fk
    assert table_b[pm_fk.adj_count_columns[type_b]][1] == 1

    # Update C[0]'s fk_to_a from A[0] to A[1]
    # Update C[1]'s pm_fk from B[1] to A[2]
    # Delete A[1]
    with db.transaction():
        table_c.update_entries(
            updates={fk_to_a: {0: 1}, pm_fk: {1: 2}, pm_fk.type_id_col: {1: type_a}},
            shape="col_major",
        )

    with db.transaction():
        table_a.del_entries([1])

    # A[1] was deleted.
    # C[0] had fk_to_a = A[1], so it is cascade deleted.
    # C[2] had fk_to_a = A[1] and pm_fk = A[1], so it is cascade deleted.
    # C[1] and C[3] remain.
    # Swap-and-pop for C: C[3] moves to 0. C[1] stays at 1.
    assert len(table_a) == 2
    assert len(table_b) == 2
    assert len(table_c) == 2

    # New C[0] is old C[3]
    assert table_c[col_c_val][0] == 40
    # Old C[3] pointed to old A[2] via fk. Old A[2] moved to 1.
    assert table_c[fk_to_a][0] == 1
    # Old C[3] pointed to B[0] via pm_fk
    assert table_c[pm_fk][0] == 0
    assert table_c[pm_fk.type_id_col][0] == type_b

    # New C[1] is old C[1]
    assert table_c[col_c_val][1] == 20
    # Old C[1] pointed to A[0] via fk
    assert table_c[fk_to_a][1] == 0
    # Old C[1] was updated to point to old A[2] via pm_fk. Old A[2] moved to 1.
    assert table_c[pm_fk][1] == 1
    assert table_c[pm_fk.type_id_col][1] == type_a
