import numpy as np
from .packed_array import PackedArray
from typing import Any, cast


class PackedStringArray(PackedArray[np.str_]):
    """A specialized PackedArray for fixed-length Unicode strings."""

    def __init__(
        self,
        initial_capacity: int | tuple[int, ...],
        dtype: np.dtype[np.str_],
        default: str = "",
    ) -> None:
        super().__init__(initial_capacity, cast(type[np.str_], dtype), default)

    def __getitem__(self, key: Any) -> str:
        return cast(str, super().__getitem__(key))

    def append(self, element: str) -> None:
        super().append(element)  # ty:ignore[invalid-argument-type]
