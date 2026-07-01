import numpy as np
from .packed_array import PackedArray
from typing import Any, cast


class PackedObjectArray[T](PackedArray[np.object_]):
    """A specialized PackedArray for arbitrary Python objects."""

    def __init__(
        self,
        initial_capacity: int = 0,
    ) -> None:
        super().__init__(initial_capacity, np.object_, None)

    def __getitem__(self, key: Any) -> T:
        # Cast the np.object_ back to the user's requested type
        return cast(T, super().__getitem__(key))

    def append(self, element: T) -> None:  # ty:ignore[invalid-method-override]
        super().append(element)  # ty:ignore[invalid-argument-type]
