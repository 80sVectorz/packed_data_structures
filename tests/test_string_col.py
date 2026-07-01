import pytest
import numpy as np
from packed_data_structures.database import PackedArrayDB
from packed_data_structures.schemas import TableSchema, IndexSpec
from packed_data_structures.schemas.string_col import StringColSchema

def test_string_col_basic():
    # Setup
    my_str_col = StringColSchema("my_str_col", max_length=10)
    idx_spec = IndexSpec.from_dtype(np.uint32)
    table_schema = TableSchema("test_table", index_spec=idx_spec, cols=[my_str_col])
    
    db = PackedArrayDB(table_schema)
    table = db.get_table(table_schema)
    
    # Add strings via transaction
    with db.transaction():
        table.add_entries(
            records={my_str_col: ["hello", "world", "toolongstring"]}, 
            records_shape="col_major"
        )

    # Access through accessor
    accessor = table[my_str_col]
    
    # Verify typing and access
    assert accessor[0] == "hello"
    assert accessor[1] == "world"
    
    # "toolongstring" is 13 chars, so it should be truncated to 10 chars by numpy "U10"
    assert accessor[2] == "toolongstr"
    
    # Test slice access
    view_slice = accessor[0:2]
    assert len(view_slice) == 2
    assert view_slice[0] == "hello"
    assert view_slice[1] == "world"
    
    # Test np.ndarray view
    raw_view = accessor.view
    assert raw_view.dtype == np.dtype('U10')
    assert raw_view[0] == "hello"
    
    # Test setting through the underlying array
    accessor.arr[0] = "changed"
    assert accessor[0] == "changed"
