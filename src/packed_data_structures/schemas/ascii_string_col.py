from __future__ import annotations
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from packed_data_structures.arrays import PackedAsciiStringArray
from .col_schema_like import ColSchemaLike


@dataclass(eq=False)
class AsciiStringColSchema(ColSchemaLike[np.bytes_]):
    """Schema for a fixed-length ASCII byte-string column.

    Attributes:
        name: The string identifier for the column.
        max_length: The maximum allowed length for the string (in bytes).
        default: The default byte-string value.
    """

    name: str
    max_length: int
    default: bytes | str = b""
    dtype: np.dtype[np.bytes_] = field(init=False)

    def __post_init__(self):
        self.dtype = cast(np.dtype[np.bytes_], np.dtype((np.bytes_, self.max_length)))

    def init_array(self) -> PackedAsciiStringArray:
        return PackedAsciiStringArray(
            self.parent_table.pre_allocate,
            self.dtype,
            self.default,
        )

    def __hash__(self) -> int:
        return hash(id(self))
