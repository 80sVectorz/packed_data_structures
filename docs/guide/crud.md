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

You can add single entries or batches of entries to a table using the `add_entry` and `add_entries` methods. 

A powerful feature of the `TransactionContext` is that these methods return **staged indices**. Rather than immediately resolving the final physical row index, the system gives you a virtual index that is perfectly valid for the duration of the transaction block.

This allows you to stage complex edits—like creating a new row and immediately referencing it as a foreign key in another new row—before any physical memory shifts or swap-and-pop deletions occur. When the transaction finishes, all staged indices and their references are automatically mapped to their final physical locations.

```python
with db.transaction():
    # Add a single row using a dictionary mapping Column Schemas to values
    staged_idx = table_a.add_entry({
        col_weight: 1.5,
        col_position: [10.0, 0.0, 5.0]
    })
    
    # We can use the staged index to link new entries together during the same transaction
    table_b.add_entry({
        fk_to_a: staged_idx
    })
```

### Bulk Operations

For high performance, the system is designed to handle multiple operations at once using bulk inputs. Instead of looping through data and calling `add_entry` one by one, you should use `add_entries` and pass data in a column-major dictionary. 

```python
with db.transaction():
    # Add multiple rows simultaneously in column-major format
    # Note: add_entries returns a memory-efficient Python `range` object,
    # NOT a fully reified list or NumPy array!
    staged_indices_range = table_a.add_entries(
        records={
            col_weight: [2.0, 3.0, 4.0],
            col_position: [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        },
        records_shape="col_major"
    )
```
Bulk operations allow the underlying NumPy arrays to ingest all data simultaneously without the overhead of repeated Python function calls. Furthermore, returning a simple `range` of staged indices avoids allocating massive temporary lists in memory.

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

### Reading Entire Rows

If you need to retrieve all column values for a specific row index at once, you can index the table directly with an integer or slice. This returns a tuple of the row's values (ordered exactly as the columns were registered in the schema).

```python
# Get a tuple representing all data for row 5
row_5_data = table_a[5]

# Get a sequence of tuples for the first 10 rows
first_10_rows = table_a[0:10]
```

## Updating Foreign Keys

> **IMPORTANT**
> While it is perfectly fine to modify standard data columns directly via `.view`, Foreign Key updates must be performed using the official update methods.

Unlike standard data, foreign keys dictate the relational topology (like the adjacency lists injected by the schema). If you overwrite a foreign key directly in the NumPy array, the underlying linked lists will break. 

To safely remap a foreign key, use the `update_entries` method within a transaction context:

```python
with db.transaction():
    # Safely update the foreign key of row 5 to point to a new target
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
