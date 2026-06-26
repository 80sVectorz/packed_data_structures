# Technical Architecture

To fully utilize the `packed_data_structures` package, it helps to understand how data is physically arranged in memory.

## Structure of Arrays (SoA)

Instead of the traditional Object-Oriented approach where a "row" is represented as a single object containing multiple fields (Array of Structures), this package uses a **Structure of Arrays (SoA)** memory layout. 

In the SoA model, a table is simply a logical grouping of independent columns. Each column is backed by a single, contiguous 1-dimensional NumPy array. When you define a table with `Weight` and `Position` columns, you are creating one flat array of weights and another flat array of positions. A "row" is an abstract concept that merely represents sharing the same index across these independent arrays.

### Cache Locality

The SoA layout provides significant performance advantages when executing vectorized operations. Modern CPU caches load memory in contiguous blocks. When you write a NumPy operation like `table[col_weight].view *= 2.0`, the CPU can iterate linearly through a densely packed array of floats without loading irrelevant data (like `position` vectors) into the cache. 

## The Swap-and-Pop Mechanic

Because vectorized operations rely on contiguous memory to be fast, the arrays backing the columns must remain densely packed. If a row is deleted, leaving an empty "hole" in the middle of the arrays would break contiguous execution and introduce branching logic.

To maintain density, the package employs a **Swap-and-Pop** deletion mechanic:

1. When row `i` is deleted, the data from the *last* active row in the table is moved into row `i`.
2. The logical size of the table is decremented by 1 (the "pop").

This allows deletions to execute in $O(1)$ time without requiring a costly $O(N)$ memory shift of all subsequent rows.

### Trade-offs and Foreign Key Remapping

The primary trade-off of the Swap-and-Pop mechanic is that **row order is not preserved**. When row 10 is deleted, the data that used to be at row 99 might suddenly be moved to row 10. 

Because row indices are used as Foreign Keys by other tables, moving a row invalidates any foreign keys pointing to it. This is why the `TransactionContext` is mandatory for structural changes:

1. When a deletion is queued, the transaction computes which rows will be relocated (swapped) to fill the holes.
2. An internal "Remap Oracle" tracks where every row will end up.
3. Before the transaction commits, it iterates through all registered foreign keys across all tables and silently updates them to point to the new physical indices of the swapped rows.

By buffering these operations, the `TransactionContext` maintains referential integrity across the entire database, ensuring your topology remains correct despite physical rows shifting around in memory.
