from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence, Iterator, Any

import numpy as np

from packed_data_structures.remap_oracle import (
    RemapOracle,
    oracle_resolve,
    oracle_resolve_array,
)


class BaseTracker(ABC):
    """Base class for explicit ID reification trackers.

    Trackers are used to map original or staged IDs to their final physical
    locations after a transaction has committed.
    """

    index_dtype: np.dtype
    storage_method: Literal["auto", "hard", "soft"]
    _is_committed: bool
    _reified_data: Any

    def __init__(
        self,
        index_dtype: np.dtype,
        storage_method: Literal["auto", "hard", "soft"] = "hard",
    ):
        self.index_dtype = np.dtype(index_dtype)
        self.storage_method = storage_method
        self._is_committed = False
        self._reified_data = None

    def _check_committed(self) -> None:
        if not self._is_committed:
            raise RuntimeError("Transaction not committed. Cannot reify or query.")

    def _fill(self, oracle: RemapOracle) -> None:
        """Fills the tracker using the provided oracle post-commit."""
        self._is_committed = True
        self._perform_fill(oracle)

    @abstractmethod
    def _perform_fill(self, oracle: RemapOracle) -> None: ...

    @abstractmethod
    def reify(self) -> np.ndarray | np.generic:
        """Returns the fully mapped IDs as a new, explicit copy.

        Returns:
            np.ndarray | np.generic: The fully mapped IDs in a new array or scalar.
        """
        ...

    @abstractmethod
    def query(self, original_id: int) -> np.generic:
        """Queries the final physical ID of a single specific original ID.

        Args:
            original_id (int): The original ID to lookup.

        Returns:
            np.generic: The physical ID mapped to the table's index dtype.
        """
        ...

    @abstractmethod
    def query_array(self, original_ids: np.ndarray) -> np.ndarray:
        """Queries the final physical IDs of a bulk list of original IDs.

        Args:
            original_ids (np.ndarray): The original IDs to lookup.

        Returns:
            np.ndarray: The physical IDs mapped to the table's index dtype.
        """
        ...

    @abstractmethod
    def __getitem__(self, index: int | slice) -> np.generic | np.ndarray: ...

    @abstractmethod
    def __iter__(self) -> Iterator[np.generic]: ...


class SingleTracker(BaseTracker):
    """A tracker optimized for resolving a single specific ID."""

    original_id: int

    def __init__(
        self,
        original_id: int,
        index_dtype: np.dtype,
        storage_method: Literal["auto", "hard", "soft"] = "hard",
    ):
        super().__init__(index_dtype, storage_method)
        self.original_id = original_id

    def _perform_fill(self, oracle: RemapOracle) -> None:
        val = oracle_resolve(self.original_id, oracle)
        self._reified_data = np.array(val, dtype=self.index_dtype)[()]

    def reify(self) -> np.generic:
        self._check_committed()
        return self._reified_data

    def query(self, original_id: int) -> np.generic:
        self._check_committed()
        if original_id != self.original_id:
            raise KeyError(f"Original ID {original_id} not tracked.")
        return self._reified_data

    def query_array(self, original_ids: np.ndarray) -> np.ndarray:
        self._check_committed()
        mask = original_ids == self.original_id
        if not np.all(mask):
            raise KeyError("Some Original IDs not tracked.")
        res = np.empty(len(original_ids), dtype=self.index_dtype)
        res[:] = self._reified_data
        return res

    def __getitem__(self, index: int | slice) -> np.generic | np.ndarray:
        self._check_committed()
        if isinstance(index, slice):
            return np.array([self._reified_data], dtype=self.index_dtype)[index]
        if index not in (0, -1):
            raise IndexError("SingleTracker index out of bounds")
        return self._reified_data

    def __iter__(self) -> Iterator[np.generic]:
        self._check_committed()
        yield self._reified_data


class ArrayTracker(BaseTracker):
    """A tracker optimized for resolving sequences and unstructured arrays of IDs."""

    original_ids: np.ndarray

    def __init__(
        self,
        original_ids: Sequence[int] | np.ndarray,
        index_dtype: np.dtype,
        storage_method: Literal["auto", "hard", "soft"] = "hard",
    ):
        super().__init__(index_dtype, storage_method)
        self.original_ids = np.asarray(original_ids, dtype=self.index_dtype)

    def _perform_fill(self, oracle: RemapOracle) -> None:
        view = oracle_resolve_array(self.original_ids, oracle)
        # oracle_resolve_array allocates a new array. "soft" and "hard" are identical.
        method = "hard" if self.storage_method == "auto" else self.storage_method
        self._reified_data = view.copy() if method == "hard" else view

    def reify(self) -> np.ndarray:
        self._check_committed()
        return self._reified_data.copy()

    def query(self, original_id: int) -> np.generic:
        self._check_committed()
        idx = np.where(self.original_ids == original_id)[0]
        if len(idx) == 0:
            raise KeyError(f"Original ID {original_id} not tracked.")
        return self._reified_data[idx[0]]

    def query_array(self, original_ids: np.ndarray) -> np.ndarray:
        self._check_committed()
        res = []
        for oid in original_ids:
            idx = np.where(self.original_ids == oid)[0]
            if len(idx) == 0:
                raise KeyError(f"Original ID {oid} not tracked.")
            res.append(self._reified_data[idx[0]])
        return np.array(res, dtype=self.index_dtype)

    def __getitem__(self, index: int | slice) -> np.generic | np.ndarray:
        self._check_committed()
        return self._reified_data[index]

    def __iter__(self) -> Iterator[np.generic]:
        self._check_committed()
        for i in range(len(self._reified_data)):
            yield self._reified_data[i]


class RangeTracker(BaseTracker):
    """A tracker mathematically optimized for Python range objects.

    Particularly efficient for O(1) tracking of newly staged IDs.
    """

    original_ids: range

    def __init__(
        self,
        original_ids: range,
        index_dtype: np.dtype,
        storage_method: Literal["auto", "hard", "soft"] = "hard",
    ):
        super().__init__(index_dtype, storage_method)
        self.original_ids = original_ids

    def _perform_fill(self, oracle: RemapOracle) -> None:
        # Fast path: Tracking exact contiguous additions.
        if (
            self.original_ids.step == 1
            and self.original_ids.start >= oracle.staged_indices_start
        ):
            offset_start = self.original_ids.start - oracle.staged_indices_start
            offset_stop = self.original_ids.stop - oracle.staged_indices_start

            view = oracle.addition_destinations[offset_start:offset_stop]
            method = self.storage_method
            if method == "auto":
                if len(view) < len(oracle.addition_destinations) * 0.5:
                    method = "hard"
                else:
                    method = "soft"

            self._reified_data = view.copy() if method == "hard" else view
        else:
            arr = np.fromiter(self.original_ids, dtype=self.index_dtype)
            view = oracle_resolve_array(arr, oracle)
            method = "hard" if self.storage_method == "auto" else self.storage_method
            self._reified_data = view.copy() if method == "hard" else view

    def reify(self) -> np.ndarray:
        self._check_committed()
        return self._reified_data.copy()

    def query(self, original_id: int) -> np.generic:
        self._check_committed()
        if original_id not in self.original_ids:
            raise KeyError(f"Original ID {original_id} not tracked.")
        idx = (original_id - self.original_ids.start) // self.original_ids.step
        return self._reified_data[idx]

    def query_array(self, original_ids: np.ndarray) -> np.ndarray:
        self._check_committed()
        if self.original_ids.step == 1:
            indices = original_ids - self.original_ids.start
            mask = (indices >= 0) & (indices < len(self.original_ids))
            if not np.all(mask):
                raise KeyError("Some Original IDs not tracked.")
            return self._reified_data[indices]
        else:
            indices = (original_ids - self.original_ids.start) // self.original_ids.step
            valid = (
                (indices >= 0)
                & (indices < len(self.original_ids))
                & ((original_ids - self.original_ids.start) % self.original_ids.step == 0)
            )
            if not np.all(valid):
                raise KeyError("Some Original IDs not tracked.")
            return self._reified_data[indices]

    def __getitem__(self, index: int | slice) -> np.generic | np.ndarray:
        self._check_committed()
        return self._reified_data[index]

    def __iter__(self) -> Iterator[np.generic]:
        self._check_committed()
        for i in range(len(self._reified_data)):
            yield self._reified_data[i]
