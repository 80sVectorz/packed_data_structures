from __future__ import annotations

from collections.abc import Sequence, Iterator
from typing import overload, Any

import numpy as np


class IdRange(Sequence[np.generic]):
    """A virtual range of IDs supporting advanced array-like indexing.

    Unlike Python's built-in `range`, `IdRange` supports NumPy-style advanced
    indexing (boolean masks, lists of indices) while avoiding the memory cost
    of reifying an integer array up front. It strictly enforces a specific`index_dtype`.
    """

    _range: range
    index_dtype: np.dtype

    __slots__ = ("_range", "index_dtype")

    def __init__(self, r: range, index_dtype: np.dtype | type | str):
        self._range = r
        self.index_dtype = np.dtype(index_dtype)

    @property
    def start(self) -> int:
        return self._range.start

    @property
    def stop(self) -> int:
        return self._range.stop

    @property
    def step(self) -> int:
        return self._range.step

    def __len__(self) -> int:
        return len(self._range)

    def __iter__(self) -> Iterator[np.generic]:
        scalar_type = self.index_dtype.type
        for val in self._range:
            yield scalar_type(val)

    def __contains__(self, item: Any) -> bool:
        # Standard range __contains__ is fast O(1)
        return item in self._range

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IdRange):
            return (
                self._range == other._range and self.index_dtype == other.index_dtype
            )
        if isinstance(other, range):
            return self._range == other
        return False

    @overload
    def __getitem__(self, index: int) -> np.generic: ...

    @overload
    def __getitem__(
        self, index: slice | Sequence[int] | Sequence[bool] | np.ndarray
    ) -> np.ndarray: ...

    def __getitem__(
        self, index: int | slice | Sequence[int] | Sequence[bool] | np.ndarray
    ) -> np.generic | np.ndarray:
        if isinstance(index, int):
            val = self._range[index]
            return self.index_dtype.type(val)

        if isinstance(index, slice):
            # If the user provides a slice, we return a dense array instead of another range,
            # mirroring what an actual NumPy array would do.
            arr = np.arange(self.start, self.stop, self.step, dtype=self.index_dtype)
            return arr[index]

        # For lists of ints, boolean masks, or arrays:
        arr = np.arange(self.start, self.stop, self.step, dtype=self.index_dtype)
        idx_array = np.asarray(index)
        return arr[idx_array]

    def reify(self) -> np.ndarray:
        """Explicitly reify this virtual range into a dense NumPy array."""
        return np.arange(self.start, self.stop, self.step, dtype=self.index_dtype)

    def __repr__(self) -> str:
        return f"IdRange({self.start}, {self.stop}, {self.step}, dtype={self.index_dtype.name})"
