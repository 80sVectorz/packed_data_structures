# Object Columns (`ObjectColSchema`)

The `ObjectColSchema` describes a column designed to store arbitrary Python objects (such as dictionaries, custom class instances, or variable-length lists) inside a NumPy object array.

## Memory Structure & Performance Characteristics
* **Pointer Arrays:** Object columns store arrays of 64-bit memory pointers referencing Python objects allocated on the system heap.
* **Cache Locality Trade-off:** Because the data itself is scattered on the Python heap, accessing these objects breaks CPU cache line locality. It does not benefit from contiguous memory execution.
* **JIT Incompatibility:** Object arrays and their contents cannot be used inside JIT-compiled Numba kernels.
* **Usage Guideline:** Object columns should only be used for non-performance-critical metadata, dynamic configurations, or structural properties that cannot be mapped into contiguous numeric data columns.

---

## Schema Definition
```python
from packed_data_structures.schemas import ObjectColSchema

# Define an object column that defaults to None
col_props = ObjectColSchema("properties", default=None)
```

---

## Accessing and Mutating Data
Indexing a table with an `ObjectColSchema` returns an `ObjectColSchemaAccessor`.
* **Properties:**
  * `view`: Returns the raw object view `np.ndarray[tuple[int], np.dtype[np.object_]]`.
  * `arr`: Returns the underlying `PackedObjectArray` container.
* **Indexing:**
  * Direct subscripting (e.g., `table[col][idx]`) delegates to the view, returning the Python object directly.

### Example Usage
```python
# Write Python objects directly to cells (no transaction block required)
table[col_props].view[5] = {"physics_enabled": True, "mass": 1.2}

# Read objects
metadata = table[col_props][5]
if metadata is not None:
    print(metadata["mass"])
```
