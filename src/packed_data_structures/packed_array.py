from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast, override
import numpy as np
from numpy.typing import DTypeLike

from packed_data_structures.edit_helpers import plan_bulk_edit

from .dirty_tracking import DirtyTrackingArray, DirtyTimestampProvider


class PackedArray[T: np.generic](
    np.lib.mixins.NDArrayOperatorsMixin, DirtyTimestampProvider
):
    """Base packed array class.

    Acts as an interface between user-side functions and the actual data.
    It handles things like resizing and maintaining contiguity.
    Uses a swap based removal approach.
    """

    resize_factor: int | float
    dtype: T
    empty_fill: int | float | np.generic = 0
    __array_priority__ = 1000

    _data: PackedArrayBuffer[T]
    _cached_vsize: int = -1
    _cached_view: DirtyTrackingArray[Any, T] | None = None

    def __init__(
        self,
        pre_allocated_capacity: int | tuple[int, ...],
        dtype: DTypeLike,
        empty_fill: Any = 0,
        resize_factor: int | float = 2,
        element_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize the PackedArray with specific capacity and data type.

        Args:
            pre_allocated_capacity (int | tuple[int, ...]): The initial capacity of the array.
                If a tuple is provided, the first dimension is treated as the
                capacity, and the remaining dimensions are treated as the
                `element_shape`.
            dtype (DTypeLike): The numpy data type for the underlying storage.
            empty_fill (Any, optional): The value used to fill empty or newly
                allocated slots. Defaults to 0.
            resize_factor (int | float, optional): The multiplier used when
                resizing the array. Defaults to 2.
            element_shape (tuple[int, ...] | None, optional): The shape of
                individual elements. Must be None if `true_size` is provided
                as a tuple. Defaults to None.

        Raises:
            ValueError: If `element_shape` is provided both as an argument and
                embedded within the `true_size` tuple.
            ValueError: If `true_size` is a tuple with no dimensions.
            ValueError: If the resulting size is negative.
        """
        if isinstance(pre_allocated_capacity, tuple):
            if element_shape is not None:
                raise ValueError(
                    "Provide element_shape either via pre_allocated_capacity tuple or element_shape argument, not both."
                )
            if not pre_allocated_capacity:
                raise ValueError(
                    "pre_allocated_capacity tuple must have at least one dimension."
                )
            pre_allocated_capacity, *rest = pre_allocated_capacity
            element_shape = tuple(rest)
        else:
            pre_allocated_capacity = int(pre_allocated_capacity)
            element_shape = () if element_shape is None else tuple(element_shape)

        if pre_allocated_capacity < 0:
            raise ValueError("pre_allocated_capacity must be non-negative.")

        self.dtype = cast(T, dtype)
        self.resize_factor = resize_factor
        self.empty_fill = empty_fill
        self.element_shape = element_shape
        self._data = PackedArrayBuffer(pre_allocated_capacity, dtype, element_shape)
        self._data.arr[:] = empty_fill

    @property
    def view(self) -> DirtyTrackingArray[Any, T]:
        if self._cached_view is None or self._cached_vsize != self._data.virtual_size:
            vsize = self._data.virtual_size
            new_view = self._data.arr[:vsize]
            self._cached_view = cast(DirtyTrackingArray[Any, T], new_view)
            self._cached_vsize = vsize
        return self._cached_view

    @property
    @override
    def last_dirty_timestamp(self) -> int:
        return self.view.last_dirty_timestamp

    def _invalidate(self):
        """Mark the current view as stale."""
        self._cached_view = None
        self._cached_vsize = -1

    def __len__(self) -> int:
        return self._data.virtual_size

    def __iter__(self):
        return iter(self.view)

    def __getitem__(self, key: Any) -> Any:
        return self.view[key]

    def __setitem__(self, key, value):
        self.view[key] = value

    def __array__[T_dtype: np.generic](
        self, dtype: T_dtype | None = None
    ) -> np.ndarray[Any, np.dtype[T_dtype]]:
        v = self.view
        return v.astype(dtype, copy=False) if dtype is not None else v

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        def unwrap(x):
            return x.view if isinstance(x, PackedArray) else x

        inputs = tuple(unwrap(x) for x in inputs)
        if "out" in kwargs and kwargs["out"] is not None:
            out = kwargs["out"]
            if isinstance(out, tuple):
                kwargs["out"] = tuple(unwrap(o) for o in out)
            else:
                kwargs["out"] = unwrap(out)
        return getattr(ufunc, method)(*inputs, **kwargs)

    _ALLOWED_NUMERIC_FUNCS = {
        np.sum,
        np.mean,
        np.std,
        np.var,
        np.min,
        np.max,
        np.argmin,
        np.argmax,
        np.all,
        np.any,
        np.prod,
        np.cumsum,
        np.cumprod,
        np.clip,
        np.round,
        np.around,
        np.sort,
        np.dot,
    }

    def __array_function__(self, func, types, args, kwargs):
        if not any(issubclass(t, PackedArray) for t in types):
            return NotImplemented
        if func not in self._ALLOWED_NUMERIC_FUNCS:
            return NotImplemented

        def unwrap(x):
            return x.view if isinstance(x, PackedArray) else x

        args = tuple(unwrap(a) for a in args)
        kwargs = {k: unwrap(v) for k, v in (kwargs or {}).items()}
        return func(*args, **kwargs)

    def append(self, element: T) -> None:
        t_size = self._data.true_size
        v_size = self._data.virtual_size
        if v_size >= t_size:
            if self.resize_factor != 0:
                new_t_size = max(
                    round(t_size * self.resize_factor),
                    t_size + 1,
                    v_size + 1,
                )
                self._data.resize(new_t_size)
                self._data.arr[v_size:] = self.empty_fill
            else:
                raise Exception("Append failed: max capacity reached")
        self._data.virtual_size += 1
        self._data.arr[v_size] = element
        self._invalidate()

    def ensure_size(self, size: int, virtual_shrink: bool = False):
        t_size = self._data.true_size
        v_size = self._data.virtual_size

        if size < v_size:
            if virtual_shrink:
                self._data.virtual_size = size
                self._invalidate()
            return

        if size > t_size:
            if self.resize_factor != 0:
                new_t_size = t_size
                while new_t_size < size:
                    next_size = round(new_t_size * self.resize_factor)
                    if next_size <= new_t_size:
                        next_size = new_t_size + 1
                    new_t_size = next_size
                self._data.resize(new_t_size)
                self._data.arr[size:] = self.empty_fill
            else:
                raise Exception("Pad to size failed: max capacity reached")

        self._data.arr[v_size:size] = self.empty_fill
        self._data.virtual_size = size
        self._invalidate()

    def remove(self, index: int) -> T:
        v_size = self._data.virtual_size
        if index >= v_size:
            raise IndexError(
                f"Remove failed. index out of range: {index} > {v_size - 1}"
            )
        if index < -v_size:
            raise IndexError(f"Remove failed. index out of range: {index} < {-v_size}")

        if index < 0:
            index += v_size

        val = self._data.arr[index]
        self._data.arr[index] = self.empty_fill

        if index != v_size - 1:
            self._swap(v_size - 1, index)

        self._data.virtual_size -= 1
        self._invalidate()

        return val

    def _swap(self, idx_a: int, idx_b: int):
        arr = self._data.arr
        arr[idx_a], arr[idx_b] = arr[idx_b], arr[idx_a]

    def swap(self, idx_a: int, idx_b: int):
        v_size = self._data.virtual_size
        if (c_idx := max(idx_a, idx_b)) >= v_size:
            raise IndexError(f"Swap failed. index out of range: {c_idx} > {v_size - 1}")
        if (c_idx := min(idx_a, idx_b)) < -v_size:
            raise IndexError(f"Swap failed. index out of range: {c_idx} < {-v_size}")

        if idx_a < 0:
            idx_a += v_size
        if idx_b < 0:
            idx_b += v_size

        self._swap(idx_a, idx_b)

    def bulk_edit(
        self,
        *,
        additions: list[T] | None = None,
        removals: list[int] | np.ndarray[Any, np.dtype[np.integer]] | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Applies multiple append and remove operations in one optimized procedure.

        Uses first-come-first-serve based addition and removal pair merging.
        Meaning that new values replace removal targets to avoid unnecessary relocations.

        Args:
            additions (list[T] | None, optional): List of entries to append.
            removals (list[int] | None, optional): List of indices to remove.

        Returns:
            Returns a list of appended value destinations, and a tuple of relocated indices.
            The relocated indices are split into a list of original indices.
            And a list of the indices they've been moved to.
        """
        current_size = len(self)
        n_additions = len(additions) if additions is not None else 0
        new_size, addition_dests, relocations = plan_bulk_edit(
            current_size, n_additions, removals or []
        )

        vw = self.view

        if relocations[0]:
            vw[relocations[1]] = vw[relocations[0]]

        self._data.virtual_size = new_size
        vw = self.view

        if n_additions:
            vw[addition_dests] = additions

        return addition_dests, relocations

    def apply_bulk_edit_plan(
        self,
        new_size: int,
        additions: tuple[
            list[T] | np.ndarray[Any, np.dtype[T]], list[int] | np.ndarray
        ],
        relocations: tuple[list[int], list[int]] | tuple[np.ndarray, np.ndarray],
        skip_shrink: bool = False,
    ):
        """Applies an existing bulk edit plan.

        Look at `edit_helpers.plan_bulk_edit` for more info.

        Args:
            new_size: Size of the array after the edits.
            additions: Tuple of a list of values and destination indices.
            relocations: A tuple of relocate from indices as 2 separate lists.
            skip_shrink: If the function should skip shrinking the array.
        """
        addition_values, addition_dests = additions

        vw = self.view

        if relocations[0]:
            vw[relocations[1]] = vw[relocations[0]]

        self.ensure_size(new_size, not skip_shrink)
        vw = self.view

        if len(addition_dests):
            vw[addition_dests] = addition_values

    def __str__(self) -> str:
        return str(self.view)


@dataclass(slots=True)
class PackedArrayBuffer[T: np.generic]:
    true_size: int
    dtype: DTypeLike
    element_shape: tuple[int, ...] = field(default_factory=tuple)

    virtual_size: int = field(init=False)
    arr: DirtyTrackingArray[Any, T] = field(init=False)

    def __post_init__(self) -> None:
        self.virtual_size = 0
        raw_arr = np.empty((self.true_size, *self.element_shape), dtype=self.dtype)
        self.arr = DirtyTrackingArray(raw_arr)

    def resize(self, new_true_size: int, allow_shrink: bool = False):
        virtual_size = self.virtual_size

        if not allow_shrink and new_true_size < virtual_size:
            raise ValueError(
                f"""Resize failed. Requested size smaller than current virtual size:
                {new_true_size} < {virtual_size}"""
            )

        old_arr = self.arr

        new_arr = np.empty((new_true_size, *self.element_shape), dtype=old_arr.dtype)

        # preserve existing content
        if virtual_size:
            new_arr[:virtual_size] = old_arr[:virtual_size]

        # fill the rest with something safe; caller sets values later
        if new_true_size > virtual_size:
            new_arr[virtual_size:] = 0

        self.arr = DirtyTrackingArray(new_arr, old_arr.timestamp_ref)
        self.true_size = new_true_size
