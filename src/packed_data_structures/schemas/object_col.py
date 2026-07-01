from __future__ import annotations
from typing import cast
from dataclasses import dataclass

import numpy as np

from packed_data_structures.arrays import PackedObjectArray
from .col_schema_like import ColSchemaLike


@dataclass(eq=False)
class ObjectColSchema[T, *T_shape](ColSchemaLike[np.object_]):
    """Schema for a data column that uses numpy C-level object pointer arrays.

    Attributes:
        name: The string identifier for the column.
    """

    name: str

    def init_array(self) -> PackedObjectArray[T]:
        return cast(
            PackedObjectArray[T], PackedObjectArray(self.parent_table.pre_allocate)
        )

    def __hash__(self) -> int:
        return hash(id(self))
