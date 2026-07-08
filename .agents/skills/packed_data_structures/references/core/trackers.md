# Trackers & Reification (`ctx.create_tracker()`)

Row indices in PDS represent ephemeral identifiers rather than permanent positional slots. During transactions, structural edits (insertions and deletions) scramble physical array locations. Trackers serve as the mapping bridge, translating ephemeral virtual IDs (staged indices) and original target indices into their new, stable physical coordinates post-commit.

---

## 1. Creating Trackers
Trackers are instantiated inside the transaction block via the transaction context object. The system automatically selects the correct tracker type based on the shape of your input IDs.
* **Tracker Types:**
  * `SingleTracker`: Returned when tracking a single integer ID.
  * `RangeTracker`: Returned when tracking a Python `range` object (e.g. from `add_entries`).
  * `ArrayTracker`: Returned when tracking lists, sequences, or NumPy arrays of indices.
* **Storage Options (`storage_method`):**
  * `"hard"` (Default): Copies mapped values out of the transaction context. Safely releases transaction/oracle memory for garbage collection.
  * `"soft"`: Creates a zero-copy memory view into the internal remapping oracle. Highly efficient for massive datasets, but prevents the oracle from being garbage collected as long as the tracker is alive.
  * `"auto"`: Dynamically shifts between hard and soft based on the size of the tracked ID sequence.

---

## 2. Resolving Final Indices (`reify`)
Once the transaction context exits (indicating the transaction committed successfully), call the `.reify()` method on your tracker to resolve physical locations.
* **Sentinel Values:** If an entry was deleted during the transaction, its reified index will resolve to the table's missing index sentinel (`schema.index_spec.missing`).

### Example Usage
```python
# Open transaction
with db.transaction() as ctx:
    # 1. Stage additions, returning staged range
    staged_range = table.add_entries(records, records_shape="col_major")
    
    # 2. Stage deletion that shifts physical slots
    table.del_entries([2, 8])
    
    # 3. Register a tracker for the newly added rows
    tracker = ctx.create_tracker(table.schema, staged_range)

# Post-commit:
# reify() returns a NumPy array containing the final, stable physical indices
physical_indices = tracker.reify()

# Verify that the first added element survived deletion
missing_sentinel = table.schema.index_spec.missing
if physical_indices[0] != missing_sentinel:
    final_idx = int(physical_indices[0])
    print(f"Added item is physically located at row {final_idx}")
```
