# Foreign Keys (`ForeignKeySchema`)

The `ForeignKeySchema` describes a relational constraint that maps rows in a source table to target rows in a target table.

## Physical Adjacency & Relationships
A foreign key column is not just a passive reference. When a `ForeignKeySchema` is registered, the schema system implicitly registers adjacency columns in the database to manage an internal doubly-linked adjacency list:
* **`adj_head`** (in target table): Points to the first referring row in the source table.
* **`adj_next`** (in source table): Points to the next referring row sharing the same target.
* **`adj_prev`** (in source table): Points to the previous referring row.
* **`adj_count`** (in target table, optional): Tracks the number of active incoming links.

---

## Schema Definition
```python
from packed_data_structures.schemas import TableSchema, ForeignKeySchema, FksOnDeleteStyle, AdjacencyListConf

# Target table schema must be defined first
target_table_schema = TableSchema(name="target_table", ...)

# Define foreign key on source table schema
fk_to_target = ForeignKeySchema(
    name="target_ref",
    target_table=target_table_schema,
    on_delete=FksOnDeleteStyle.CASCADE,
    adjacency_conf=AdjacencyListConf(track_counts=True)
)
```

### Delete Styles (`FksOnDeleteStyle`)
* `CASCADE`: Deleting a target row automatically schedules deletions of all referencing source rows.
* `SET_NULL`: Deleting a target row nullifies the foreign key pointers of referencing rows.
* `RESTRICT`: Deleting a target row raises a `DeleteClaimedByStrictFkException` if references exist, blocking the deletion.

---

## Accessing and Traversing Relationships
Indexing a table with a `ForeignKeySchema` returns a `ForeignKeySchemaAccessor`.
* **Properties:**
  * `target_table`: Retrieves the live target `PackedArrayTable` instance.
  * `view`: Accesses the raw integer array backing the foreign key column.
* **Adjacency Traversals (Querying incoming links):**
  * `get_referencing_indices(head_indices)`: Traverses the internal doubly-linked lists backwards in $O(K)$ time (where $K$ is the number of incident edges). Returns a tuple of tuples containing the referring row indices for each requested target row index.

### Example Usage
```python
# Retrieve the accessor
fk_accessor = table_source[fk_to_target]

# Query target table
target_table = fk_accessor.target_table

# Find all source rows referencing target row 12 in O(K) time
referring_rows = fk_accessor.get_referencing_indices([12])[0]
```

---

## Updating Foreign Keys
To preserve list integrity, **never write to the raw foreign key view array directly**. All updates must be declared within a transaction using the `update_entries` method.

### Example Usage
```python
with db.transaction():
    # Safely move row 5 to reference target row 22
    table_source.update_entries(updates={fk_to_target: {5: 22}}, shape="col_major")
```
