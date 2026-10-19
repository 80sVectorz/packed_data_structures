# Dirty Tracking Internals & Numba JIT Integration (`dirty_tracking.py`)

The dirty tracking system tracks mutations across database columns in $O(1)$ time, propagating change timestamps up the hierarchy to coordinate cache invalidations and user interface redraws.

---

## 1. Dirty Timestamp Providers

### `DirtyTimestampProvider` Protocol
An interface that describes any object tracking modification time.
* **Property:** `last_dirty_timestamp -> int` (returns nanosecond-precision epoch time).

### `ProvidesDirtyTimestamp` Mixin
An abstract class that aggregates child timestamps. It caches sources to avoid redundant traversals.
* **Internal Cache:** `_dirty_sources_cache` stores child providers.
* **Cache Invalidation:** Subclasses call `_invalidate_dirty_cache()` when schemas change (e.g., adding/removing columns).
* **Timestamp Evaluation:** The `last_dirty_timestamp` property performs a fast loop to return the maximum timestamp among all cached child providers.
* **Abstract Method:** Subclasses must implement `_collect_dirty_sources() -> tuple[DirtyTimestampProvider, ...]` to define child hierarchies.
  * `PackedArrayDB` returns `self.tables`.
  * `PackedArrayTable` returns `self.arrays`.

---

## 2. Array Mutability Tracking

### `TimestampRef`
A mutable wrapper class holding a single integer timestamp value:
* `update() -> TimestampRef`: Sets the internal value to `time_ns()` and returns itself.

### `DirtyTrackingArray`
A subclass of `numpy.ndarray` that acts as a `DirtyTimestampProvider`.
* **Array Priority:** Sets `__array_priority__ = 1000` to intercept ufuncs when mixed with standard NumPy arrays.
* **Subscript Interception:** Overrides `__setitem__` to update the associated `TimestampRef` to `time_ns()` before writing.
* **Universal Functions:** Overrides `__array_ufunc__` to intercept operations. If the output target (`out`) is a `DirtyTrackingArray`, PDS updates its timestamp ref.

---

## 3. Numba JIT Integration (`tracked_njit`)

When compiling NumPy array operations using Numba (`@njit`), Numba operates directly on raw memory pointer levels. Because Numba bypasses Python's `__setitem__` and ufunc wrappers, mutations made inside JIT-compiled kernels will not trigger dirty tracking updates.

To resolve this, PDS provides the `tracked_njit` decorator.

### Decorator Signature
```python
from packed_data_structures.arrays.dirty_tracking import tracked_njit


@tracked_njit(mutates=lambda arg1, arg2: (arg1, arg2), cache=True)
def jit_kernel(array_a, array_b):
    # Performance-critical compiled mutations
    pass
```

### Execution Lifecycle
When a function decorated with `tracked_njit` is called:
1. **Prediction Phase:** The `mutates` callable (typically a lambda) runs with the function's arguments to identify which tracking arrays are targeted for modification.
2. **Timestamp Update:** For each predicted array, if it is a `DirtyTrackingArray`, the wrapper updates its `TimestampRef` (`arr.timestamp_ref.update()`).
3. **Execution Phase:** The pre-compiled Numba JIT function runs at native speed.

---

## 4. Usage Patterns

### A. Checking Global vs. Local Modification States
```python
# Check if anything in the entire database changed
if db.last_dirty_timestamp > last_saved:
    save_database()

# Check if only the position column has changed
if table[col_position].arr.last_dirty_timestamp > last_render:
    update_vispy_buffers()
```

### B. Defining a JIT Mutation Kernel
```python
import numba as nb
import numpy as np
from packed_data_structures.arrays.dirty_tracking import tracked_njit


@tracked_njit(mutates=lambda weights, delta: (weights,), cache=True)
def apply_delta_kernel(weights: np.ndarray, delta: float):
    # Numba JIT loops mutates the raw memory buffer directly
    for i in range(len(weights)):
        weights[i] += delta
```
