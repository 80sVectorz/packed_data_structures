# String Columns (`StringColSchema` & `AsciiStringColSchema`)

The `StringColSchema` and `AsciiStringColSchema` describe columns that store fixed-width string and character data. 

## Memory Structure & Performance Characteristics
* **Fixed-Width Allocation:** Unlike dynamic Python strings, strings in PDS are allocated with a fixed maximum length (`max_length`). Under the hood, they utilize NumPy's native unicode (`np.str_`) or fixed-length byte string representations.
* **Vectorized Slicing:** Because string widths are uniform, they are packed contiguously in memory and support zero-overhead slicing.
* **String Types:**
  * `StringColSchema`: Utilizes standard Python unicode characters. Suitable for user-facing labels and general text containing non-ASCII symbols.
  * `AsciiStringColSchema`: Utilizes native ASCII byte arrays. More memory efficient when only ASCII characters (e.g. unique identifiers, system keys) are stored.

---

## Schema Definition
```python
from packed_data_structures.schemas import StringColSchema, AsciiStringColSchema

# Unicode string with a maximum length of 128 characters
col_label = StringColSchema("label", max_length=128)

# ASCII-only string with a maximum length of 32 characters
col_sys_key = AsciiStringColSchema("sys_key", max_length=32)
```

---

## Accessing and Mutating Data
Indexing a table with a string column schema returns a `StringColSchemaAccessor` or `AsciiStringColSchemaAccessor`.
* **Properties:**
  * `view`: Returns the underlying NumPy string/byte view.
* **Indexing:**
  * Direct subscripting (e.g., `table[col][idx]`) delegates to the view.

### Example Usage
```python
# Write string values directly to cells (no transaction block required)
table[col_label].view[5] = "concept_node"
table[col_sys_key].view[5] = b"sys_id_8"  # ASCII columns accept bytes/ASCII strings

# Read string values
label = table[col_label][5]
print(label) # Outputs: "concept_node"
```
