# Polymorphic Foreign Keys (`PolymorphicForeignKeySchema`)

The `PolymorphicForeignKeySchema` describes a dynamic relationship that can reference a target row in one of several target tables. 

## Adjacency & Typology Pointers
To support dynamic targets, a polymorphic foreign key tracks two pieces of information:
1. **Type Identifier:** An integer column representing which target table is referenced (type_id).
2. **Row Index:** The index of the row inside that target table.

Adjacency list pointers (`adj_head`, `adj_next`, `adj_prev`) are injected across all possible target schemas. Backward lookups require specifying both the target table's type ID and the target row indices.

---

## Schema Definition
```python
from packed_data_structures.schemas import TableSchema, PolymorphicForeignKeySchema

schema_nodes = TableSchema(name="nodes", ...)
schema_edges = TableSchema(name="edges", ...)

# Define a polymorphic key that can target either a node or an edge
pm_fk_member = PolymorphicForeignKeySchema(
    name="member_ref",
    target_tables=(schema_nodes, schema_edges)
)
```
In this example:
* Target type `0` represents the `nodes` table.
* Target type `1` represents the `edges` table.

---

## Accessing and Traversing Dynamic Relationships
Indexing a table with a `PolymorphicForeignKeySchema` returns a `PolymorphicForeignKeySchemaAccessor`.
* **Properties:**
  * `view`: Accesses the raw integer array backing the target row indices.
  * `type_id_col`: Exposes the column schema representing the target type IDs.
* **Adjacency Traversals (Querying incoming links):**
  * `get_referencing_indices(type_id, head_indices)`: Traverses the dynamic adjacency lists backwards in $O(K)$ time. Returns a tuple of tuples containing the referring row indices for the specified targets.

### Example Usage
```python
# Retrieve the accessor
pm_accessor = table_junction[pm_fk_member]

# Find all junction rows referencing node index 8 (type_id 0)
referring_rows = pm_accessor.get_referencing_indices(type_id=0, head_indices=[8])[0]
```

---

## Updating Polymorphic Foreign Keys
All polymorphic target updates must happen inside a transaction context.
```python
with db.transaction():
    table_junction.update_entries(
        updates={
            pm_fk_member: {5: 12},  # Point to row 12
            pm_fk_member.type_id_col: {5: 1},  # Target type 1 (edges table)
        },
        shape="col_major",
    )
```
