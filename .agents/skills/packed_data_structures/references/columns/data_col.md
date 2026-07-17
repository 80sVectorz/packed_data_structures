# Data Columns (`DataColSchema`)

The `DataColSchema` describes a column that stores primitive, fixed-size numeric and boolean data. It allocates memory block-by-block using standard NumPy dtypes.

## Physical Memory & Performance
* **Contiguous Cache Allocation:** All data in a `DataColSchema` column is stored in a single, contiguous array block, which matches CPU cache line execution models.
* **Vectorization:** Accessing column views permits instant vectorized arithmetic operations (e.g., `*`, `+`, index slicing) across all elements simultaneously.
* **Numba JIT Compatibility:** The raw NumPy views returned by the accessor are fully compatible with Numba JIT-compiled kernels for maximum traversal speeds.

---

## Schema Definition
```python
from packed_data_structures.schemas import DataColSchema
import numpy as np

# Scalar float32 column
col_weight = DataColSchema("weight", np.float32, default=1.0)

# 3D vector float32 column
col_position = DataColSchema("position", np.float32, shape=(3,), default=0.0)

# Boolean column
col_active = DataColSchema("active", np.bool_, default=True)
```

---

## Accessing and Mutating Data
Indexing a table with a `DataColSchema` returns a standard `SchemaAccessor`.
* **Properties:**
  * `view`: Returns the raw NumPy array `np.ndarray` of active elements.
  * `arr`: Returns the underlying `PackedArray` memory container.
* **Indexing:**
  * Subscripting the accessor directly (e.g., `table[col][idx]`) delegates to the view, equivalent to `table[col].view[idx]`.

### Example Usage
```python
# Retrieve the live table instance
table = db.get_table(table_schema)

# Zero-overhead vectorized update (no transaction required for cell mutation)
table[col_weight].view[:] *= 2.0

# Write a single vector value
table[col_position].view[5] = [10.0, 0.0, 5.0]

# Subscripting delegate shortcut
position_5 = table[col_position][5]
```
