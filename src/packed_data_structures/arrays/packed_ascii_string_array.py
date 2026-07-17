import numpy as np
from .packed_array import PackedArray
from typing import Any, cast


class PackedAsciiStringArray(PackedArray[np.bytes_]):
    """A specialized PackedArray for fixed-length ASCII byte-strings."""

    def __init__(
        self,
        initial_capacity: int | tuple[int, ...],
        dtype: np.dtype[np.bytes_],
        default: bytes | str = b"",
    ) -> None:
        super().__init__(initial_capacity, cast(type[np.bytes_], dtype), default)

    def __getitem__(self, key: Any) -> bytes:
        return cast(bytes, super().__getitem__(key))

    def append(self, element: bytes | str) -> None:
        super().append(element)  # ty:ignore[invalid-argument-type]
