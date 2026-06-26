# Public API & CRUD Operations

With your schemas defined and a database initialized, you can begin interacting with your tables. The `PackedArrayTable` provides the primary public interface for all Create, Read, Update, and Delete (CRUD) operations.

Behind the scenes, every structural modification to a table is strictly brokered by the **Transaction Context**, a central and mandatory concept that ensures memory and topology consistency.

## The Transaction Context

To ensure that the structure of your data remains consistent—especially when multiple arrays need to be resized simultaneously or when complex foreign key topologies must be rewritten—all structural changes must occur within a transaction block.

```python
from packed_data_structures.overlays.database import OverlaidDB
from packed_data_structures.schemas import TableSchema

# Assuming `table_a_schema` is defined
db = OverlaidDB()
db.init_table(table_a_schema)

table_a = db.get_table(table_a_schema)

with db.transaction():
    # Structural changes (additions, deletions, foreign key updates) go here
    pass
```

Transactions gather all your requested operations and apply them as a single atomic batch when the block exits.

## Adding Rows

You can add single entries or batches of entries to a table using the `add_entry` and `add_entries` methods. These methods return the new row indices immediately, allowing you to use them in the same transaction.

```python
with db.transaction():
    # Add a single row using a dictionary mapping Column Schemas to values
    new_idx = table_a.add_entry({
        col_weight: 1.5,
        col_position: [10.0, 0.0, 5.0]
    })
    
    # Add multiple rows in column-major format for bulk performance
    new_indices = table_a.add_entries(
        records={
            col_weight: [2.0, 3.0, 4.0],
            col_position: [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        },
        records_shape="col_major"
    )
```

## Reading and Mutating Data

One of the defining features of `packed_data_structures` is that reading and updating raw data columns does *not* require special methods. Because columns are just NumPy arrays, you can index into them directly using the `.view` property.

```python
# Access the raw, zero-overhead NumPy array backing the weight column
weight_array = table_a[col_weight].view

# Read data using standard NumPy indexing
first_weight = weight_array[0]

# Mutate data directly
weight_array[1] = 5.5

# Perform vectorized math on the entire column instantly
weight_array[:] *= 2.0
```

Raw data updates happen immediately and do not need to be wrapped in a transaction block.

## Updating Foreign Keys

> [!WARNING]
> While it is perfectly fine to modify standard data columns directly via `.view`, **Foreign Key updates must be performed using the official update methods**.

Unlike standard data, foreign keys dictate the relational topology (like the adjacency lists injected by the schema). If you overwrite a foreign key directly in the NumPy array, the underlying linked lists will break. 

To safely remap a foreign key, use the `update_entries` method within a transaction context:

```python
with db.transaction():
    # Safely move row index 5 to point to a new target
    table_b.update_entries(
        updates={fk_to_a: {5: new_target_idx}}, 
        shape="col_major"
    )
```
The transaction context will take care of unlinking the old adjacency pointers and establishing the new ones.

## Deleting Rows

Deletions are queued via the `del_entry` and `del_entries` methods. 

```python
with db.transaction():
    # Delete a single row
    table_a.del_entry(5)
    
    # Delete multiple rows using an iterable
    table_a.del_entries([0, 1, 2])
    
    # Delete using a NumPy boolean mask
    mask = table_a[col_weight].view < 0.5
    table_a.del_entries(mask)
```

During the transaction commit phase, any rows marked for deletion are swapped out using an O(1) swap-and-pop technique. We will discuss the architectural mechanics of this in the next chapter.
