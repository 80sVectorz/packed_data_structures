# Bonus: Dirty Tracking

When building applications that have high-frequency simulation loops (like physics engines or complex graph layout engines) running alongside user interfaces, you often face a synchronization problem: how does the UI know when it needs to redraw, and how does it know *what* data changed without doing a costly frame-by-frame comparison?

`packed_data_structures` solves this automatically with an internal **Dirty Tracking** system.

## The Timestamp Provider

Every structural level in the package (from the `OverlaidDB`, down to individual `TableSchema` instances, down to the actual column arrays) implements the `DirtyTimestampProvider` protocol. This means every object exposes a `.last_dirty_timestamp` property, representing the nanosecond-precision timestamp of its last modification.

When you ask a table for its timestamp, it returns the maximum timestamp of all its underlying columns. 

## The DirtyTrackingArray

The magic happens at the lowest level, where the raw data arrays are wrapped in a subclass of `numpy.ndarray` called the `DirtyTrackingArray`.

When you interact with a column's `.view` (e.g., `table[col_weight].view`), you are actually interacting with a `DirtyTrackingArray`. This subclass intercepts any operation that mutates the array data (such as `__setitem__` assignments or in-place ufuncs like `+=`) and instantly updates its internal `TimestampRef`.

Because these timestamps propagate upwards automatically, external systems can perform highly efficient checks:

```python
# Save the timestamp during the last render frame
last_rendered_time = db.last_dirty_timestamp

# ... simulation runs ...

# Check if anything in the entire database changed in O(1) time
if db.last_dirty_timestamp > last_rendered_time:
    # We only redraw if a change actually occurred
    trigger_redraw()
    last_rendered_time = db.last_dirty_timestamp
```

You can do this at any granularity. If your UI only cares about updates to the `Position` column, you can track `table[col_position].arr.last_dirty_timestamp` specifically. This allows disparate systems—like an asynchronous physics engine and a 60 FPS Qt interface—to perfectly synchronize state updates with practically zero overhead.
