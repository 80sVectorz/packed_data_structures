---
name: packed_data_structures
description: Rules, API usage patterns, and recipes for the packed_data_structures (PDS) library.
---

# Skill: packed_data_structures (PDS)

Use this skill when reading or writing code that describes database schemas, initializes tables, or performs operations using the `packed_data_structures` library.

## Core Concepts & Architectural Split

The `packed_data_structures` library separates the description of data structures from the memory management and allocation. 

### 1. Blueprints vs. Containers
- **Schemas:** `TableSchema`, `DataColSchema`, and `ForeignKeySchema` are programmatic blueprints and type identifiers. They do not allocate memory or store data themselves.
- **Tables & Databases:** `PackedArrayDB` instantiates tables (`PackedArrayTable`) and allocates underlying contiguous arrays based on the schemas. All reads, writes, and slice lookups must occur via the table instances.

### 2. Transaction Isolation
Structural changes (inserting/deleting rows, updating foreign keys) are staged within a transaction context (`db.transaction()`) and are only reified (physically executed) when the transaction context exits successfully. Standard cell values (non-foreign-key views) are updated immediately outside transactions.

---

## Skill References Directory

Detailed guides, API signatures, and self-contained examples are split into modular files:

### Columns & Relationships
- [Data Columns (data_col.md)](file:///c:/Users/krijn/Documents/Programming/PythonPackages/packed_data_structures/.agents/skills/packed_data_structures/references/columns/data_col.md): Primitive type contiguous arrays for high-performance and JIT-compilation.
- [Foreign Key Columns (foreign_key.md)](file:///c:/Users/krijn/Documents/Programming/PythonPackages/packed_data_structures/.agents/skills/packed_data_structures/references/columns/foreign_key.md): Direct 1-to-N relationships and adjacency list tracking.
- [Polymorphic Foreign Keys (polymorphic_foreign_key.md)](file:///c:/Users/krijn/Documents/Programming/PythonPackages/packed_data_structures/.agents/skills/packed_data_structures/references/columns/polymorphic_foreign_key.md): Relational linking pointing to multiple target tables.
- [Object Columns (object_col.md)](file:///c:/Users/krijn/Documents/Programming/PythonPackages/packed_data_structures/.agents/skills/packed_data_structures/references/columns/object_col.md): Storing heap-allocated arbitrary Python objects (with cache locality and JIT performance trade-offs).
- [String Columns (string_cols.md)](file:///c:/Users/krijn/Documents/Programming/PythonPackages/packed_data_structures/.agents/skills/packed_data_structures/references/columns/string_cols.md): Storing fixed-length string and ASCII text data.
- [Implicit Columns (injected_logic_col.md)](file:///c:/Users/krijn/Documents/Programming/PythonPackages/packed_data_structures/.agents/skills/packed_data_structures/references/columns/injected_logic_col.md): Implicitly generated, read-only relationship columns.

### Core Architecture & Lifecycle
- [Tables & Databases (table_and_db.md)](file:///c:/Users/krijn/Documents/Programming/PythonPackages/packed_data_structures/.agents/skills/packed_data_structures/references/core/table_and_db.md): Accessing tables, indexing rows, slicing, and checking table size.
- [Transactions & Staged Updates (transactions.md)](file:///c:/Users/krijn/Documents/Programming/PythonPackages/packed_data_structures/.agents/skills/packed_data_structures/references/core/transactions.md): Staged additions/deletions, relational remapping, and bulk update signatures.
- [Trackers & Reification (trackers.md)](file:///c:/Users/krijn/Documents/Programming/PythonPackages/packed_data_structures/.agents/skills/packed_data_structures/references/core/trackers.md): Managing index remapping post-commit via Single, Range, and Array trackers.
- [Memory Layout & Mechanics (memory_and_remap.md)](file:///c:/Users/krijn/Documents/Programming/PythonPackages/packed_data_structures/.agents/skills/packed_data_structures/references/core/memory_and_remap.md): Swap-and-pop details and relocations planning.
- [Dirty Tracking & JIT (dirty_tracking.md)](file:///c:/Users/krijn/Documents/Programming/PythonPackages/packed_data_structures/.agents/skills/packed_data_structures/references/core/dirty_tracking.md): Modification aggregation, DirtyTrackingArray, and the tracked_njit decorator.

