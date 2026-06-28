import pytest
import numpy as np

from packed_data_structures.schemas import TableSchema, IndexSpec, DataColSchema
from packed_data_structures.database import PackedArrayDB
from packed_data_structures.table import PackedArrayTable


def test_packed_array_db_initialization():
    idx_spec = IndexSpec.from_dtype(np.uint32)
    col = DataColSchema("val", np.int32)
    schema = TableSchema("my_table", idx_spec, [col])

    db = PackedArrayDB(schema)

    assert "my_table" in db.table_ids
    assert len(db.table_schemas) == 1
    assert isinstance(db.tables[0], PackedArrayTable)


def test_packed_array_db_get_table():
    idx_spec = IndexSpec.from_dtype(np.uint32)
    schema = TableSchema("my_table", idx_spec, [])

    db = PackedArrayDB(schema)

    table = db.get_table("my_table")
    assert table.name == "my_table"

    table2 = db.get_table(schema)
    assert table is table2

    with pytest.raises(KeyError):
        db.get_table("missing_table")


def test_packed_array_db_transaction_context():
    idx_spec = IndexSpec.from_dtype(np.uint32)
    schema = TableSchema("my_table", idx_spec, [])
    db = PackedArrayDB(schema)

    assert db._transaction_ctx is None

    ctx = db.transaction()
    assert ctx is not None
    assert db._transaction_ctx is ctx

    # same context returned within the block
    ctx2 = db.transaction()
    assert ctx is ctx2
