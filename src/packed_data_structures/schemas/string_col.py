from __future__ import annotations
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from packed_data_structures.arrays import PackedStringArray
from .col_schema_like import ColSchemaLike


@dataclass(eq=False)
class StringColSchema(ColSchemaLike[np.str_]):
    """Schema for a fixed-length Unicode string column.

    Attributes:
        name: The string identifier for the column.
        max_length: The maximum allowed length for the string.
        default: The default string value.
    """

    name: str
    max_length: int
    default: str = ""
    dtype: np.dtype[np.str_] = field(init=False)

    def __post_init__(self):
        self.dtype = cast(np.dtype[np.str_], np.dtype((np.str_, self.max_length)))

    def init_array(self) -> PackedStringArray:
        return PackedStringArray(
            self.parent_table.pre_allocate,
            self.dtype,
            self.default,
        )

    def __hash__(self) -> int:
        return hash(id(self))
