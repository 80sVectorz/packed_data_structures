from __future__ import annotations

from abc import ABC, abstractmethod
from time import time_ns
import numpy as np
from typing import Any, Literal, Protocol, override, runtime_checkable
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import functools
import numba as nb

UFuncMethod = Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"]


@runtime_checkable
class DirtyTimestampProvider(Protocol):
    """Protocol for anything that tracks a modification time."""

    @property
    def last_dirty_timestamp(self) -> int: ...


class ProvidesDirtyTimestamp(ABC):
    """Mixin that aggregates multiple dirty timestamp sources via caching.

    Subclasses must implement `_collect_dirty_sources`.
    Subclasses should call `_invalidate_dirty_cache` when their structure changes.
    """

    __slots__ = ("_dirty_sources_cache",)

    def __init__(self, *args, **kwargs):
        self._dirty_sources_cache: tuple[DirtyTimestampProvider, ...] | None = None
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _collect_dirty_sources(self) -> tuple[DirtyTimestampProvider, ...]:
        """Return a tuple of all sub-components that track dirtiness."""
        pass

    def _invalidate_dirty_cache(self) -> None:
        """Clear the cache. Call this when adding/removing fields or layers."""
        self._dirty_sources_cache = None

    @property
    def last_dirty_timestamp(self) -> int:
        """Returns the latest timestamp from all collected sources."""
        # 1. Lazy Load
        if self._dirty_sources_cache is None:
            self._dirty_sources_cache = self._collect_dirty_sources()

        # 2. Aggregation (Manual max loop for speed)
        max_ts = 0
        for s in self._dirty_sources_cache:
            ts = s.last_dirty_timestamp
            if ts > max_ts:
                max_ts = ts
        return max_ts


@dataclass(slots=True)
class TimestampRef(DirtyTimestampProvider):
    value: int

    def update(self) -> TimestampRef:
        self.value = time_ns()
        return self

    @property
    @override
    def last_dirty_timestamp(self) -> int:
        return self.value


class DirtyTrackingArray[T_s: tuple[int, ...], T_dt: np.generic](
    np.ndarray, DirtyTimestampProvider
):
    """A numpy array that updates a shared timestamp on every modification.

    This class auto-updates a shared timestamp reference whenever contents
    are modified via __setitem__ or in-place ufuncs.
    """

    __slots__ = ("timestamp_ref",)
    timestamp_ref: TimestampRef
    __array_priority__ = 1000

    @property
    @override
    def last_dirty_timestamp(self) -> int:
        return self.timestamp_ref.value

    def __new__(
        cls,
        input_array: np.ndarray[T_s, np.dtype[T_dt]],
        timestamp_ref: TimestampRef | None = None,
    ) -> DirtyTrackingArray[T_s, T_dt]:
        """Creates a new DirtyTrackingArray instance.

        Args:
            input_array: The input array to wrap or cast.
            timestamp_ref: Optional, pre-existing TimestampRef object.
        """
        obj = np.asarray(input_array).view(cls)
        if timestamp_ref is not None:
            obj.timestamp_ref = timestamp_ref  # type: ignore
        else:
            if isinstance(input_array, DirtyTrackingArray):
                obj.timestamp_ref = input_array.timestamp_ref.update()  # type: ignore
            else:
                obj.timestamp_ref = TimestampRef(time_ns())  # type: ignore
        return obj  # type: ignore

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, DirtyTrackingArray):
            try:
                self.timestamp_ref = obj.timestamp_ref
            except AttributeError:
                self.timestamp_ref = TimestampRef(time_ns())

    def __setitem__(self, key: Any, value: Any) -> None:
        if self.timestamp_ref is not None:
            self.timestamp_ref.value = time_ns()
        super().__setitem__(key, value)

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: UFuncMethod, *inputs: Any, **kwargs: Any
    ) -> Any:
        """Delegate ufuncs while preserving timestamp semantics.

        - If outputs include any DirtyTrackingArray, update their timestamp.
        - Allow broadcasts into DirtyTrackingArray outputs by coercing outputs to base
          ndarrays for the op, then wrapping/updating tracking after.
        """
        outputs = kwargs.get("out", ())
        has_out = bool(outputs)
        # Normalize outputs to a tuple of base ndarrays so numpy can broadcast freely
        if has_out:
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            coerced = []
            for o in outputs:
                coerced.append(np.asarray(o))
            kwargs["out"] = tuple(coerced)

        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if result is NotImplemented:
            clean_inputs = tuple(np.asarray(x) for x in inputs)

            if method == "__call__":
                result = ufunc(*clean_inputs, **kwargs)
            else:
                result = getattr(ufunc, method)(*clean_inputs, **kwargs)

        # Update timestamps on DirtyTrackingArray outputs
        if has_out:
            for o in outputs:
                if isinstance(o, DirtyTrackingArray) and o.timestamp_ref is not None:
                    o.timestamp_ref.update()
            # When numpy returns a tuple for multiple outputs, mimic numpy's contract
            return outputs[0] if len(outputs) == 1 else outputs

        return result


def tracked_njit(mutates: Callable[..., Sequence[Any]], **njit_kwargs):
    """Wraps a Numba function to automatically update timestamps of modified arrays.

    Args:
        mutates: A function (or lambda) that accepts the same arguments as the
                 decorated function and returns a sequence of arrays that
                 are predicted to be modified.
        **njit_kwargs: Arguments passed directly to nb.njit (e.g., cache=True).
    """

    def decorator(func):
        # Compile the actual function with Numba
        jitted_func = nb.njit(func, **njit_kwargs)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Run the prediction to find arrays that WILL be dirtied
            dirty_candidates = mutates(*args, **kwargs)

            # 2. Mark them as dirty (updates the shared TimestampRef)
            for arr in dirty_candidates:
                if isinstance(arr, DirtyTrackingArray):
                    arr.timestamp_ref.update()

            # 3. Execute the optimized Numba code
            return jitted_func(*args, **kwargs)

        return wrapper

    return decorator
