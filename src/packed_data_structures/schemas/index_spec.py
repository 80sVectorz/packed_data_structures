from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class IndexSpec[T: np.integer[Any]]:
    """Specification for integer indices used to address rows.

    Defines the underlying numpy data type, the sentinel value indicating
    a missing or null link, and the maximum valid row index.

    Attributes:
        dtype: The numpy integer data type for the index.
        missing: The sentinel integer value representing a missing index.
        max_value: The maximum valid integer value for an index.
    """

    dtype: type[T]
    missing: int
    max_value: int

    @classmethod
    def from_dtype(cls, dtype: type[T]) -> IndexSpec:
        """Create an IndexSpec from a numpy data type.

        By standard convention, the maximum representable value of the given
        integer type is reserved as the missing/sentinel value.

        Args:
            dtype: A numpy integer data type.

        Returns:
            A new IndexSpec instance.

        Raises:
            TypeError: If the provided dtype is not an integer type.
        """
        if not np.issubdtype(dtype, np.integer):
            raise TypeError(f"Index dtype must be integer, got {dtype}")

        info = np.iinfo(dtype)
        # Standard convention: Max value is the missing value sentinel
        return cls(dtype=dtype, missing=info.max, max_value=info.max - 1)

    def new_array(self, size: int) -> np.ndarray[Any, np.dtype[T]]:
        """Helper to allocate raw numpy arrays with correct initialization."""
        arr = np.empty(size, dtype=self.dtype)
        arr[:] = self.missing
        return arr
