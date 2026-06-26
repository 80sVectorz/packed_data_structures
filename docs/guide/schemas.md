# The Schemas System

The `packed_data_structures` package uses a programmatic schema system to define the shape and relationships of your data. Instead of relying on string-based keys or dynamic dictionaries, you construct singletons that represent your tables and columns. These objects act as typed identifiers that can be tracked by static code analysis tools, improving the safety and maintainability of your code.

## TableSchema

A `TableSchema` represents a collection of identically sized arrays (columns). It manages the allocation of rows and tracks the data types used for indexing.

```python
import numpy as np
from packed_data_structures.schemas import TableSchema, IndexSpec

# Create a schema for our table, specifying that we will address rows using 32-bit unsigned integers
table_a_schema = TableSchema(
    name="table_a", 
    index_spec=IndexSpec.from_dtype(np.uint32)
)
```

By defining an `IndexSpec`, the schema establishes a universal language for how rows in this table are addressed. The maximum representable value of the data type (e.g., `4,294,967,295` for `uint32`) is automatically reserved as the "missing" or "null" sentinel value.

## DataColSchema

Data columns define the actual properties stored within a table. A `DataColSchema` encapsulates the name, NumPy data type, default values, and shape (for vector data) of a property.

To build our table, we can provide the columns directly to the table schema during initialization, or register them manually later:

```python
from packed_data_structures.schemas import DataColSchema

# A scalar float32 column
col_weight = DataColSchema(name="weight", dtype=np.float32, default=1.0)

# A vector float32 column (e.g., a 3D vector)
col_position = DataColSchema(name="position", dtype=np.float32, shape=(3,), default=0.0)

# 1. Clean Initialization (Recommended)
table_a_schema = TableSchema(
    name="table_a", 
    index_spec=IndexSpec.from_dtype(np.uint32),
    cols=[col_weight, col_position]
)

# 2. Manual Registration
# table_a_schema.register(col_weight)
# table_a_schema.register(col_position)
```

### Type Hinting and Autocomplete

One of the primary reasons `DataColSchema` is implemented as an object singleton rather than a string is to provide strong type checking. The schema classes accept a generic type parameter `[T]` which allows type checkers (like Pyright or MyPy) to understand exactly what type of NumPy array is returned when you query the database.

For example, `DataColSchema[np.float32]` guarantees that the underlying array view will be typed as `np.ndarray[Any, np.dtype[np.float32]]`. This eliminates the need for runtime type casting and ensures robust IDE autocomplete when interacting with your data.

## ForeignKeySchema

Relational constraints are modeled using the `ForeignKeySchema`. A foreign key links a row in a source table to a row in a target table.

```python
from packed_data_structures.schemas import ForeignKeySchema, FksOnDeleteStyle, AdjacencyListConf

table_b_schema = TableSchema(name="table_b", index_spec=IndexSpec.from_dtype(np.uint32))

fk_to_a = ForeignKeySchema(
    name="link_to_a",
    target_table=table_a_schema,
    on_delete=FksOnDeleteStyle.CASCADE,
    adjacency_conf=AdjacencyListConf(track_counts=True)
)

table_b_schema.register(fk_to_a)
```

### Adjacency List Pointers

A key feature of the `ForeignKeySchema` is that it is not just a passive reference. When a foreign key is registered, it actively alters both the source and target table schemas by automatically injecting synthetic columns to manage an internal doubly-linked adjacency list.

For the `fk_to_a` schema above, the system silently injects:

- `adj_head`: A column in `table_a` pointing to the first connected row in `table_b`.

- `adj_next`: A column in `table_b` pointing to the next connected row in `table_b` that shares the same target in `table_a`.

- `adj_prev`: A column in `table_b` pointing to the previous connected row.

- `adj_count`: A column in `table_a` (enabled via `AdjacencyListConf`) that tracks the number of incoming links.


These topology pointers are essential for efficiently maintaining relational integrity when editing the database, particularly when resolving foreign keys during swap-and-pop deletions.

## The Database

It is important to remember that all of the schema classes discussed so far are simply **blueprints**. They define the structure, data types, and relationships of your data, but they do not allocate any arrays or store any data themselves.

To bring your schemas to life, you must initialize them within a `PackedArrayDB` (or an overlay).

You can provide the schemas directly when creating the database for clean initialization, or manually register them later:

```python
from packed_data_structures.database import PackedArrayDB

# 1. Clean Initialization (Recommended)
# The database allocates the memory immediately based on the provided schemas
db = PackedArrayDB(table_a_schema, table_b_schema)

# 2. Manual Initialization
# db = PackedArrayDB()
# db.init_table(table_a_schema)
# db.init_table(table_b_schema)

# Retrieve the live table instance to interact with the data
table_a = db.get_table(table_a_schema)
```

The database acts as the central registry, holding all the physical `PackedArrayTable` instances and providing the `TransactionContext` needed to safely modify them.
