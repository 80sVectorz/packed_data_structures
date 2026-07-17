# Implicitly Registered Columns (`InjectedLogicColSchema`)

The `InjectedLogicColSchema` describes columns that are implicitly generated and registered by the schema engine to support relationship structures (like adjacency lists and polymorphic types).

---

## 1. Subclass Roles

### Adjacency List Columns
These columns manage the relational pointers that link source and target rows:
* **`AdjHeadIdxColSchema`** (in target table): Points to the head of the referencing doubly-linked list.
* **`AdjNextIdxColSchema`** (in source table): Points to the next referring row in the list.
* **`AdjPrevIdxColSchema`** (in source table): Points to the previous referring row in the list.
* **`AdjCountColSchema`** (in target table, optional): Tracks the count of active incoming links.

### Polymorphic Columns
* **`PmFkTypeIdColSchema`** (in source table): An integer column storing type identifiers that map dynamic polymorphic references to their target tables.

---

## 2. Properties & Access Constraints

### Read-Only Access
To prevent structural corruption, implicitly registered columns cannot be manually mutated. 
* **`allows_updates = False`:** Passing an injected logic column schema to `update_entries` raises a `PermissionError`.
* **Direct array view mutations** (`table[injected_col].view[idx] = val`) should be avoided, as PDS manages these pointers during the transaction commit phase.

### Pointer to Creator Relationship
* **`parent_col`:** Accessing the `parent_col` property of an injected column returns the user-declared relationship schema (e.g., `ForeignKeySchema` or `PolymorphicForeignKeySchema`) that triggered its creation.

---

## 3. Usage & Introspection

When iterating over column schemas inside a `TableSchema`, identify implicitly registered columns by checking their types:

```python
from packed_data_structures.schemas.injected_logic_col import InjectedLogicColSchema

# Print only columns defined by the user
for col in table_schema.cols:
    if not isinstance(col, InjectedLogicColSchema):
        print(f"User Column: {col.name}")
```
