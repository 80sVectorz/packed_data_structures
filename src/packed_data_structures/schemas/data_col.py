from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from packed_data_structures.arrays import PackedArray
from .col_schema_like import ColSchemaLike


@dataclass(eq=False)
class DataColSchema[T: np.generic, *T_shape](ColSchemaLike[T]):
    """Schema for a standard data column containing raw values.

    Defines the data type, default values, and shape of elements within
    the column. Maps directly to a PackedArray buffer at runtime.

    Attributes:
        name: The string identifier for the column.
        dtype: The numpy data type of the column's elements.
        default: The default value used to fill empty or newly allocated slots.
        shape: The shape of individual elements. An empty tuple indicates scalar values.
    """

    name: str
    dtype: type[T]
    default: Any | tuple[Any, ...] = 0
    shape: tuple[*T_shape] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.shape) != 0 and not isinstance(self.default, tuple):
            self.default = tuple(np.full(self.shape, self.default))

    def init_array(self) -> PackedArray[T]:
        return PackedArray(
            self.parent_table.pre_allocate,
            self.dtype,
            self.default,
            element_shape=self.shape,
        )

    def __hash__(self) -> int:
        return hash(id(self))
