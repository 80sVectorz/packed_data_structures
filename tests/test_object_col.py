import numpy as np
from typing import Any

from packed_data_structures.database import PackedArrayDB
from packed_data_structures.schemas import TableSchema, IndexSpec
from packed_data_structures.schemas.object_col import ObjectColSchema


class MyCustomObject:
    def __init__(self, value: int):
        self.value = value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MyCustomObject):
            return False
        return self.value == other.value


def test_object_col_basic():
    # Setup
    my_obj_col = ObjectColSchema[MyCustomObject]("my_obj_col")
    idx_spec = IndexSpec.from_dtype(np.uint32)
    table_schema = TableSchema("test_table", index_spec=idx_spec, cols=[my_obj_col])

    db = PackedArrayDB(table_schema)
    table = db.get_table(table_schema)

    # Add objects via transaction
    obj1 = MyCustomObject(10)
    obj2 = MyCustomObject(20)

    with db.transaction():
        table.add_entries(records={my_obj_col: [obj1, obj2]}, records_shape="col_major")

    # Access through accessor
    accessor = table[my_obj_col]

    # Verify typing and access
    assert accessor[0] == obj1
    assert accessor[1] == obj2

    # Test slice access
    view_slice = accessor[0:2]
    assert len(view_slice) == 2
    assert view_slice[0] == obj1
    assert view_slice[1] == obj2

    # Test np.ndarray view
    raw_view = accessor.view
    assert raw_view.dtype == np.object_
    assert raw_view[0] == obj1

    # Test setting through the underlying array
    obj3 = MyCustomObject(30)
    accessor.arr[0] = obj3
    assert accessor[0] == obj3


def test_object_col_none():
    my_obj_col = ObjectColSchema[str | None]("my_str_col")
    idx_spec = IndexSpec.from_dtype(np.uint32)
    table_schema = TableSchema("test_table", index_spec=idx_spec, cols=[my_obj_col])

    db = PackedArrayDB(table_schema)
    table = db.get_table(table_schema)

    with db.transaction():
        table.add_entries(
            records={my_obj_col: ["test", None]}, records_shape="col_major"
        )

    accessor = table[my_obj_col]

    assert accessor[0] == "test"
    assert accessor[1] is None
