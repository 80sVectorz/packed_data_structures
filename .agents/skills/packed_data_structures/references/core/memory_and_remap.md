# Memory Layout & Mechanics (SoA, Swap-and-Pop)

The performance design of PDS relies on a Structure of Arrays (SoA) layout and $O(1)$ swap-and-pop deletions.

---

## 1. Structure of Arrays (SoA)
Instead of representing entries as individual Python objects containing multiple properties (Array of Structures / AoS), a PDS table groups independent data columns. Each column is backed by a contiguous NumPy array (`numpy.ndarray`).
* **Cache Locality:** When executing operations across a single column (e.g., translation, physics integration), the CPU sequentially streams contiguous blocks of memory into its cache without loading irrelevant properties.
* **JIT Acceleration:** Contiguous arrays can be directly compiled into Numba JIT-kernels, bypassing Python interpreter overhead entirely.

---

## 2. Swap-and-Pop Deletions & Planning
To keep arrays densely packed and avoid costly $O(N)$ memory shifts during deletions, PDS uses a swap-and-pop strategy:
* **The Mechanic:** Deleting a row leaves a "hole". PDS relocates the element at the tail of the array to fill the hole, then decrements the logical size of the table by 1.
* **Relocation Planning:** During a transaction commit, `plan_bulk_edit` evaluates all additions and deletions to plan the physical modifications:
  1. **Replacement:** Newly added entries are written directly into holes created by deletions. This cancels out swaps, avoiding relocations.
  2. **Relocation:** If there are more deletions than additions, elements from the tail of the array are moved to fill the remaining holes.
  3. **Shrink/Grow:** The logical size (`table._len`) is adjusted.

---

## 3. Array Resizing & View Cache Lifecycles
* **Dynamic Resizing:** When a table's size exceeds its preallocated capacity, the database allocates a new, larger memory buffer (scaled by `resize_factor`) and copies the existing data. This relocates the physical memory address of the contiguous array.
* **Cache Invalidation:** To prevent referencing stale memory addresses, the internal `PackedArray` container caches the view object and invalidates it automatically when resizing or updates occur (`self._invalidate()`).
* **Accessor Resolution:** Accessing the `.view` property on the column accessor checks if the cache is stale and dynamically resolves the active view. Caching view references locally across operations that grow the table is incorrect, as the local reference will point to deallocated or old memory. Always access `.view` dynamically to ensure resolution of the active array buffer.
