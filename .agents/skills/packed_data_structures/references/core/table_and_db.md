# Tables & Databases (`TableSchema`, `IndexSpec`, `PackedArrayDB`, `PackedArrayTable`)

PDS separates data schema layout from physical array allocations. 

---

## 1. Index Specifications (`IndexSpec`)
Every table must specify its row-indexing datatype using `IndexSpec`. 
* The index datatype dictates the maximum capacity of the table (e.g., `np.uint16` has a maximum size of 65,534 elements).
* **Missing Sentinel:** The maximum representable value of the specified index type is reserved as the "null" or missing indicator (accessible via `schema.index_spec.missing`).
* **Creation:**
  ```python
  from packed_data_structures.schemas import IndexSpec
  import numpy as np

  # Reserve uint32 index specification (sentinel is 4,294,967,295)
  idx_spec = IndexSpec.from_dtype(np.uint32)
  ```

---

## 2. Table Blueprints (`TableSchema`)
A `TableSchema` logical blueprint describes the columns and indexing specification.
```python
from packed_data_structures.schemas import TableSchema

schema_nodes = TableSchema(
    name="nodes", index_spec=idx_spec, cols=[col_weight, col_position]
)
```

---

## 3. Database Registry (`PackedArrayDB`)
The database registry takes schemas and allocates the physical tables and NumPy buffers.
```python
from packed_data_structures.database import PackedArrayDB

# Clean database initialization
db = PackedArrayDB(schema_nodes, schema_edges)

# Retrieve the live database table instance using the schema blueprint key
table_nodes = db.get_table(schema_nodes)
```

---

## 4. Live Table Interactions (`PackedArrayTable`)
The `PackedArrayTable` manages active data buffers and executes queries.

* **Logical Size:** `len(table)` returns the count of active rows currently stored in memory.
* **Single Row Lookup:** Subscripting with an integer (e.g., `table[idx]`) returns a `tuple[Any, ...]` representing cell values for all columns in schema registration order.
* **Slice Lookup:** Subscripting with a slice (e.g., `table[start:end]`) returns a `Sequence[tuple[Any, ...]]` of row tuples.
* **Direct Column Access:** Passing a column schema key returns the column's specialized accessor:
  * `table[col_schema].view` accesses raw NumPy arrays.
  ```python
  # Get coordinate array
  coords = table_nodes[col_position].view
  
  # Perform JIT-compatible vectorized coordinate translation
  coords[:] += [1.0, 0.0, 0.0]
  ```

### View Caching & Resizing
* **Internal View Caching:** The underlying `PackedArray` container caches the NumPy view. Accessing `table[col].view` retrieves the cached view with zero overhead.
* **Cache Invalidation:** When a table grows (such as during insertions or capacity changes), the underlying NumPy array may be resized and relocated in memory. PDS automatically invalidates the internal view cache. The next time `table[col].view` is accessed, it dynamically re-fetches the new memory slice.
* **Lifetime Rule:** Storing a column `.view` reference in a local variable across operations that grow the table is incorrect. The stored view reference will point to stale or deallocated memory. Always access `table[col].view` dynamically when performing mutations after size changes.

