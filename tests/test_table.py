import pytest
import numpy as np

from packed_data_structures.table import SchemaAccessor, ForeignKeySchemaAccessor


def test_table_initialization(empty_db, base_schemas):
    table_a_schema, table_b_schema = base_schemas
    table_a = empty_db.get_table(table_a_schema)

    assert table_a.name == "table_a"
    assert (
        len(table_a.column_ids) == 4 * 2
    )  # 2 normal + 2 adj injected columns, times 2 for strings
    assert len(table_a.arrays) == 4


def test_table_indexing_schema_accessor(empty_db, base_schemas):
    table_a_schema, table_b_schema = base_schemas
    table_a = empty_db.get_table(table_a_schema)

    col_weight = next(c for c in table_a_schema.cols if c.name == "weight")

    accessor = table_a[col_weight]
    assert isinstance(accessor, SchemaAccessor)
    assert accessor.table is table_a


def test_table_indexing_foreign_key_accessor(empty_db, base_schemas):
    table_a_schema, table_b_schema = base_schemas
    table_b = empty_db.get_table(table_b_schema)

    fk_to_a = next(c for c in table_b_schema.cols if c.name == "link_to_a")

    accessor = table_b[fk_to_a]
    assert isinstance(accessor, ForeignKeySchemaAccessor)
    assert accessor.target_table.schema.name == "table_a"


def test_table_read_rows(populated_db, base_schemas):
    table_a_schema, _ = base_schemas
    table_a = populated_db.get_table(table_a_schema)

    row_0 = table_a[0]
    assert len(row_0) == 4
    assert row_0[0] == 1.0  # weight
    assert np.array_equal(row_0[1], [0, 0, 0])  # position

    rows = table_a[0:2]
    assert len(rows) == 2
    assert rows[1][0] == 2.0


def test_normalize_records_row_major(empty_db, base_schemas):
    table_a_schema, _ = base_schemas
    table_a = empty_db.get_table(table_a_schema)

    col_weight = next(c for c in table_a_schema.cols if c.name == "weight")
    col_pos = next(c for c in table_a_schema.cols if c.name == "position")

    records = list(
        table_a.normalize_records_row_major({col_weight: 5.0, col_pos: [1, 2, 3]})
    )

    assert len(records) == 1
    assert records[0][0] == 5.0
    assert np.array_equal(records[0][1], [1, 2, 3])


def test_normalize_records_col_major(empty_db, base_schemas):
    table_a_schema, _ = base_schemas
    table_a = empty_db.get_table(table_a_schema)

    col_weight = next(c for c in table_a_schema.cols if c.name == "weight")

    records = table_a.normalize_records_col_major({col_weight: [10.0, 20.0]})

    assert len(records) == 4  # 4 columns
    assert np.array_equal(records[0], [10.0, 20.0])
    # The default for missing columns is filled
    assert records[1].shape == (2, 3)  # position filled with defaults


def test_normalize_records_col_major_exceptions(empty_db, base_schemas):
    from packed_data_structures.schemas import DataColSchema

    table_a_schema, _ = base_schemas
    table_a = empty_db.get_table(table_a_schema)

    col_weight = next(c for c in table_a_schema.cols if c.name == "weight")
    col_pos = next(c for c in table_a_schema.cols if c.name == "position")

    # Mock an invalid schema key
    dummy_col = DataColSchema("dummy", np.float32)

    # 1. KeyError from dict
    with pytest.raises(
        KeyError, match="Table 'table_a' does not contain the included column 'dummy'"
    ):
        table_a.normalize_records_col_major({dummy_col: [1.0]})

    # 2. KeyError from sequence of tuples
    with pytest.raises(
        KeyError, match="Table 'table_a' does not contain the included column 'dummy'"
    ):
        table_a.normalize_records_col_major([(dummy_col, [1.0])])

    # 3. ValueError from misaligned columns
    with pytest.raises(ValueError, match="misaligned columns"):
        table_a.normalize_records_col_major(
            {
                col_weight: [10.0, 20.0, 30.0],
                col_pos: [[1, 2, 3], [4, 5, 6]],  # 3 items vs 2 items
            }
        )
