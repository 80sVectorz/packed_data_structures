import numpy as np
from packed_data_structures.database import PackedArrayDB
from packed_data_structures.schemas import TableSchema, IndexSpec
from packed_data_structures.schemas.ascii_string_col import AsciiStringColSchema


def test_ascii_string_col_basic():
    # Setup
    my_ascii_col = AsciiStringColSchema("my_ascii_col", max_length=10)
    idx_spec = IndexSpec.from_dtype(np.uint32)
    table_schema = TableSchema("test_table", index_spec=idx_spec, cols=[my_ascii_col])

    db = PackedArrayDB(table_schema)
    table = db.get_table(table_schema)

    # Add strings via transaction
    with db.transaction():
        table.add_entries(
            # Notice we can pass Python strings and NumPy will cast to bytes automatically
            records={my_ascii_col: ["hello", b"world", "toolongstring"]},
            records_shape="col_major",
        )

    # Access through accessor
    accessor = table[my_ascii_col]

    # Verify typing and access. Note that they come out as bytes!
    assert accessor[0] == b"hello"
    assert accessor[1] == b"world"

    # "toolongstring" is 13 chars, so it should be truncated to 10 chars by numpy "S10"
    assert accessor[2] == b"toolongstr"

    # Test slice access
    view_slice = accessor[0:2]
    assert len(view_slice) == 2
    assert view_slice[0] == b"hello"
    assert view_slice[1] == b"world"

    # Test np.ndarray view
    raw_view = accessor.view
    assert raw_view.dtype == np.dtype("S10")
    assert raw_view[0] == b"hello"

    # Test setting through the underlying array
    accessor.arr[0] = b"changed"
    assert accessor[0] == b"changed"
