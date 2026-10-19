# Transactions & Staged Updates (`db.transaction()`)

All structural alterations (insertions, deletions, and foreign key modifications) must execute within a database transaction block.

---

## 1. Transaction Context Manager
Transactions are initialized via the database instance and executed as a context manager:
```python
with db.transaction() as ctx:
    # Queue edits and relations modifications here
    pass
# Commits automatically on successful exit; rolls back if an exception is raised
```

---

## 2. Row Insertions (Staged Indices as Ephemeral IDs)
Row indices in PDS should be treated as ephemeral identifiers rather than permanent array offsets. Because physical deletions scramble row locations, an index is only a valid coordinate for the current database state sequence.

* **Staged Indices (Virtual IDs):** Indices returned by insertion methods inside an active transaction are virtual, temporary IDs. They do not correspond to physical array offsets and cannot be used to slice or subscript `.view` arrays during the transaction block. Their sole purpose is to serve as links when establishing relational foreign keys in the same transaction.
* **Row-Major Insertions:**
  ```python
  # Add a single entry, returning a staged virtual ID
  staged_id = table.add_entry({col_weight: 1.5, col_position: [0.0, 1.0, 2.0]})
  ```
* **Column-Major Bulk Insertions (Recommended for performance):**
  ```python
  # Add multiple rows, returning a range of staged virtual IDs
  staged_range = table.add_entries(
      records={
          col_weight: [2.0, 3.0, 4.0],
          col_position: [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
      },
      records_shape="col_major",
  )
  ```


---

## 3. Row Deletions
Deletions are staged during the transaction and processed as a bulk O(1) swap-and-pop edit at commit.
* **By Index:**
  ```python
  # Delete a single row
  table.del_entry(5)

  # Delete multiple rows
  table.del_entries([0, 10, 15])
  ```
* **By Boolean Mask:**
  ```python
  # Retrieve weight view and build mask
  weights = table[col_weight].view
  mask = weights < 0.5
  
  # Stages all matching elements for deletion
  table.del_entries(mask)
  ```

---

## 4. Modifying Foreign Keys
To preserve internal adjacency chains, **never write to the raw foreign key view array directly**. All updates must be declared within a transaction using the `update_entries` method.
```python
with db.transaction():
    # Remap foreign keys using update dictionary mapping {row_idx: target_idx}
    table_source.update_entries(updates={fk_to_target: {5: 22}}, shape="col_major")
```
* **Polymorphic Foreign Keys:** Both the target ID and target type column should be updated.
  ```python
  with db.transaction():
      table_junction.update_entries(
          updates={pm_fk: {5: 12}, pm_fk.type_id_col: {5: 1}}, shape="col_major"
      )
  ```

---

## 5. Record Structures & Performance Trade-offs

PDS supports multiple data shapes for both additions and updates. Selecting the right format allows you to balance readability against object creation overhead and garbage collection pressure in high-frequency execution loops.

### A. Additions (`add_entries` / `add_entry`)
When inserting rows, records can be structured as dictionaries or sequences:

1. **Schema Dictionaries (Convenient):**
   - **Format:** `dict[ColSchemaLike | str, Value | Sequence[Value]]`
   - **Trade-off:** Allocating Python dictionaries for every call incurs minor object creation and hashing overhead. Suitable for single insertions or low-frequency setups.
2. **Schema Sequences (High-Performance):**
   - **Format:** `Sequence[tuple[ColSchemaLike, Value | Sequence[Value]]]`
   - **Trade-off:** Bypasses dictionary instantiation entirely. Highly efficient when building bulk insertions inside loops.
   - **Example:**
     ```python
     table.add_entries(
         records=[(col_weight, [1.0, 2.0]), (col_position, [[0, 0, 0], [1, 1, 1]])],
         records_shape="col_major",
     )
     ```

### B. Updates (`update_entries`)
Updates are passed in column-major layout. The values for a column can be structured in two ways:

1. **Sparse Dictionary Mapping (Convenient):**
   - **Format:** `dict[row_idx: new_value]`
   - **Trade-off:** High allocation overhead. Creating a dictionary mapping for thousands of individual cell edits generates substantial garbage collection pressure.
   - **Example:** `updates={col_weight: {5: 2.0, 12: 3.5}}`
2. **Flat Parallel Sequences (High-Performance):**
   - **Format:** `tuple[Sequence[int] | np.ndarray, Sequence[Any] | np.ndarray]`
   - **Trade-off:** Direct array mapping. Passing a tuple of `(indices, values)` allows you to use preallocated NumPy arrays, completely avoiding Python dictionary instantiation.
   - **Example:**
     ```python
     # Flat arrays of indices to edit and their corresponding new values
     indices = np.array([5, 12], dtype=np.uint32)
     values = np.array([2.0, 3.5], dtype=np.float32)

     table.update_entries(updates={col_weight: (indices, values)}, shape="col_major")
     ```
     You can also use a sequence of tuples for the top-level structure to avoid dictionary instantiation entirely:
     ```python
     table.update_entries(updates=[(col_weight, (indices, values))], shape="col_major")
     ```

