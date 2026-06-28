import pytest
import numpy as np

from packed_data_structures.table import PackedArrayTable
from packed_data_structures.transaction_context import TransactionContext


def test_transaction_context_additions(empty_db, base_schemas):
    table_a_schema, table_b_schema = base_schemas
    table_a = empty_db.get_table(table_a_schema)

    col_weight = next(c for c in table_a_schema.cols if c.name == "weight")
    col_position = next(c for c in table_a_schema.cols if c.name == "position")

    with empty_db.transaction():
        table_a.add_entries(
            records={col_weight: [1.0, 2.0], col_position: [[0, 0, 0], [1, 1, 1]]},
            records_shape="col_major",
        )

    assert len(table_a) == 2
    assert table_a[0][0] == 1.0
    assert table_a[1][0] == 2.0


def test_transaction_context_updates(populated_db, base_schemas):
    table_a_schema, _ = base_schemas
    table_a = populated_db.get_table(table_a_schema)

    col_weight = next(c for c in table_a_schema.cols if c.name == "weight")

    with populated_db.transaction():
        table_a.update_entries({col_weight: {1: 99.0}})

    assert table_a[1][0] == 99.0
    assert table_a[0][0] == 1.0


def test_transaction_context_deletions_swap_and_pop(populated_db, base_schemas):
    table_a_schema, _ = base_schemas
    table_a = populated_db.get_table(table_a_schema)

    assert len(table_a) == 3

    with populated_db.transaction():
        table_a.del_entry(0)

    assert len(table_a) == 2
    assert table_a[0][0] == 3.0


def test_transaction_context_foreign_key_topology_patching(populated_db, base_schemas):
    table_a_schema, table_b_schema = base_schemas
    table_a = populated_db.get_table(table_a_schema)
    table_b = populated_db.get_table(table_b_schema)

    fk_to_a = next(c for c in table_b_schema.cols if c.name == "link_to_a")

    with populated_db.transaction():
        # Deleting index 1 of A will cascade delete index 1 of B.
        # Then A[2] moves to A[1] and B[2] moves to B[1].
        table_a.del_entry(1)

    assert len(table_a) == 2
    assert len(table_b) == 2

    # B[0] originally linked to A[0], should still be A[0] (0)
    assert table_b[0][0] == 0
    # B[1] (which was B[2]) originally linked to A[0], should still be A[0] (0)
    assert table_b[1][0] == 0
