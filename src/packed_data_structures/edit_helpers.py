from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import numba as nb

from packed_data_structures.nb_hash_set import (
    build_int_set_same_dtype,
    int_set_contains_same_dtype,
)


type T_IndexArray = np.ndarray[tuple[int], np.dtype[np.integer]]


def plan_bulk_edit(
    current_size: int, n_additions: int, removals: Sequence[int] | T_IndexArray | None
) -> tuple[int, T_IndexArray, tuple[T_IndexArray, T_IndexArray], T_IndexArray]:
    """Calculates the most efficient destinations for additions and relocations for removals.

    Uses first-come-first-served based addition and removal pair merging.
    Meaning that new values replace removal targets to avoid unnecessary swaps.

    Args:
        current_size: Current size of the array.
        n_additions: Number of editions.
        removals: List of indices to remove.

    Returns:
        Returns the new array size, the addition destinations,
        a list of relocation from & to mappings as a tuple of 2 lists,
        and the filtered normalized removal indices.
        (new_size, addition_destinations, (relocations_from, relocations_to), normalized_removal_indices)
    """
    if removals is None:
        removals = []

    # Normalize Removals
    if len(removals) == 0:
        removals_arr = np.empty(0)
    else:
        removals_arr = np.asanyarray(removals)
        # Handle negatives
        neg_mask = removals_arr < 0
        removals_arr[neg_mask] += current_size
        # Filter valid and unique
        valid_mask = (removals_arr >= 0) & (removals_arr < current_size)
        removals_arr = np.unique(removals_arr[valid_mask])

    n_removals = len(removals_arr)

    # Match Additions to Holes (Replacements)
    n_replacements = min(n_removals, n_additions)

    # The first N new items go directly into the first N holes
    addition_dests = list(removals_arr[:n_replacements])

    # Calculate relocations (Existing tail items moving to remaining holes)
    relocs_from: list[int] | np.ndarray = []
    relocs_to: list[int] | np.ndarray = []

    old_size = current_size
    new_size = old_size + n_additions - n_removals

    # Remaining removals that weren't filled by additions
    remaining_removals = removals_arr[n_replacements:]

    if new_size > old_size:
        # We are growing: extra additions go to the end
        addition_dests.extend(range(old_size, new_size))

    elif new_size < old_size:
        # We are shrinking: items at the tail must move to fill the remaining holes

        # Identify the boundary: removals < new_size are holes, >= new_size are voids
        split_idx = np.searchsorted(remaining_removals, new_size)

        # Holes that need filling
        relocs_to = remaining_removals[:split_idx]

        # Voids in the tail (we don't move these, they are just dropped)
        voids = remaining_removals[split_idx:]
        void_ptr = len(voids) - 1

        # Scan backwards from the old tail to find valid items to move
        for i in range(old_size - 1, new_size - 1, -1):
            if void_ptr >= 0 and i == voids[void_ptr]:
                void_ptr -= 1
                continue
            relocs_from.append(i)

    relocs_from = np.asarray(relocs_from)
    relocs_to = np.asarray(relocs_to)

    return new_size, np.asarray(addition_dests), (relocs_from, relocs_to), removals_arr  # type: ignore


@nb.njit(cache=True, inline="always")
def find_surviving_neighbor(
    start_node: int,
    adjacency: np.ndarray,
    unlinked_indices: np.ndarray,
    missing_val: int,
) -> int:
    """Oracle-guided lookahead to find the next valid node in a chain."""
    curr = start_node
    while curr != missing_val:
        # Check if the next node survives
        idx = np.searchsorted(unlinked_indices, curr)
        if idx < len(unlinked_indices) and unlinked_indices[idx] == curr:
            curr = adjacency[curr]  # Skip dead node
        else:
            return curr  # Found a survivor
    return missing_val


@nb.njit(cache=True)
def patch_unlinked_adjacency_and_heads(
    missing_val_src: int,
    missing_val_tgt: int,
    unlinked_indices: np.ndarray,
    head_ignore_indices: np.ndarray,
    parent_fk: np.ndarray,
    adj_next: np.ndarray,
    adj_prev: np.ndarray,
    target_adj_head: np.ndarray,
    out_next_idx: np.ndarray,
    out_next_val: np.ndarray,
    n_cursor: int,
    out_prev_idx: np.ndarray,
    out_prev_val: np.ndarray,
    p_cursor: int,
    out_head_idx: np.ndarray,
    out_head_val: np.ndarray,
    h_cursor: int,
) -> tuple[int, int, int]:
    """Computes sparse patches to repair adjacency lists after unlink/delete.

    The function collapses contiguous runs of unlinked nodes and patches only the
    surviving boundary pointers:
      prev_survivor.next = next_survivor
      next_survivor.prev = prev_survivor

    If an unlinked run starts at the head (prev == missing), it also patches:
      head[parent] = next_survivor

    Args:
        missing_val_src: Sentinel missing value of source/parent column (dtype max).
        missing_val_tgt: Sentinel missing value of target/child column (dtype max).
        unlinked_indices: Sorted or unsorted row indices to unlink. Must not contain
            the sentinel missing value.
        head_ignore_indices: Indices of rows who are guaranteed to have their head updated.
        parent_fk: Parent foreign key values for each row in the source table.
        adj_next: Source adjacency "next" array indexed by row.
        adj_prev: Source adjacency "prev" array indexed by row.
        target_adj_head: Target adjacency "head" array indexed by parent id.
        out_next_idx: Output indices for next patches (write into adj_next).
        out_next_val: Output values for next patches.
        n_cursor: Initial cursor for next patches.
        out_prev_idx: Output indices for prev patches (write into adj_prev).
        out_prev_val: Output values for prev patches.
        p_cursor: Initial cursor for prev patches.
        out_head_idx: Output indices for head patches (write into target_adj_head).
        out_head_val: Output values for head patches.
        h_cursor: Initial cursor for head patches.

    Returns:
        Updated cursors (n_cursor, p_cursor, h_cursor).
    """
    dead_set, dead_mask = build_int_set_same_dtype(unlinked_indices, missing_val_src)
    head_ignore_set, head_ignore_mask = build_int_set_same_dtype(
        head_ignore_indices, missing_val_tgt
    )

    for row in unlinked_indices:
        row = int(row)
        prev = int(adj_prev[row])

        # Only process run starts.
        if prev != missing_val_src and int_set_contains_same_dtype(
            dead_set, dead_mask, prev, missing_val_src
        ):
            continue

        # Find first surviving node after the deleted run.
        nxt = adj_next[row]
        while nxt != missing_val_src and int_set_contains_same_dtype(
            dead_set, dead_mask, nxt, missing_val_src
        ):
            nxt = adj_next[nxt]

        # Patch prev.next -> nxt
        if prev != missing_val_src and adj_next[prev] != nxt:
            out_next_idx[n_cursor] = prev
            out_next_val[n_cursor] = nxt
            n_cursor += 1

        # Patch nxt.prev -> prev
        if nxt != missing_val_src and adj_prev[nxt] != prev:
            out_prev_idx[p_cursor] = nxt
            out_prev_val[p_cursor] = prev
            p_cursor += 1

        # Head fix if the run started at the head.
        if prev == missing_val_src:
            parent = parent_fk[row]
            if (
                parent != missing_val_tgt
                and target_adj_head[parent] == row
                and target_adj_head[parent] != nxt
                and not int_set_contains_same_dtype(
                    head_ignore_set,
                    head_ignore_mask,
                    parent,
                    missing_val_tgt,
                )
            ):
                out_head_idx[h_cursor] = parent
                out_head_val[h_cursor] = nxt
                h_cursor += 1

    return n_cursor, p_cursor, h_cursor


@nb.njit(cache=True)
def patch_relocated_adjacency_and_heads(
    missing_val_src: int,
    missing_val_tgt: int,
    head_ignore_indices: np.ndarray,
    moves_from: np.ndarray,  # old row ids (source table)
    moves_to: np.ndarray,  # new row ids (source table)
    src_fk: np.ndarray,  # source fk values (targets)
    adj_next: np.ndarray,
    adj_prev: np.ndarray,
    target_adj_head: np.ndarray,
    out_next_idx: np.ndarray,
    out_next_val: np.ndarray,
    n_cursor: int,
    out_prev_idx: np.ndarray,
    out_prev_val: np.ndarray,
    p_cursor: int,
    out_head_idx: np.ndarray,
    out_head_val: np.ndarray,
    h_cursor: int,
) -> tuple[int, int, int]:
    """Generate sparse pointer fixes caused by relocations.

    For each relocated row r_old -> r_new, patch:
      adj_next[prev_old] = r_new  (if prev exists and pointed to r_old)
      adj_prev[next_old] = r_new  (if next exists and pointed to r_old)
      head[tgt] = r_new        (if head[tgt] == r_old)
    """
    head_ignore_set, head_ignore_mask = build_int_set_same_dtype(
        head_ignore_indices, missing_val_tgt
    )

    for k in range(len(moves_from)):
        r_old = moves_from[k]
        # We must use r_old in the output values because they are subject
        # to `oracle_resolve` in the calling function, which maps Old -> New.
        # If we wrote r_new (moves_to[k]) here, the oracle would treat it as an
        # old index. If r_new coincides with a deleted index (which it usually
        # does when filling holes), the oracle would resolve it to missing.
        # r_new = moves_to[k]

        prev_old = adj_prev[r_old]
        if prev_old != missing_val_src and adj_next[prev_old] == r_old:
            out_next_idx[n_cursor] = prev_old
            out_next_val[n_cursor] = r_old
            n_cursor += 1

        next_old = adj_next[r_old]
        if next_old != missing_val_src and adj_prev[next_old] == r_old:
            out_prev_idx[p_cursor] = next_old
            out_prev_val[p_cursor] = r_old
            p_cursor += 1

        tgt_row = src_fk[r_old]
        if (
            tgt_row != missing_val_tgt
            and target_adj_head[tgt_row] == r_old
            and not int_set_contains_same_dtype(
                head_ignore_set, head_ignore_mask, tgt_row, missing_val_tgt
            )
        ):
            out_head_idx[h_cursor] = tgt_row
            out_head_val[h_cursor] = r_old
            h_cursor += 1

    return n_cursor, p_cursor, h_cursor


@nb.njit(cache=True)
def fix_keys_broken_by_moved_targets(
    src_missing: int,
    moves_to: np.ndarray,  # (M,) new tgt ids in B
    head: np.ndarray,  # B.head[tgt] -> A row id (src id)
    src_fk: np.ndarray,  # A.fk[src] -> tgt id
    adj_next: np.ndarray,  # A.adj_next
    adj_prev: np.ndarray,  # A.adj_prev
) -> None:
    """Fix references to relocated targets (B) using adjacency head array.

    Must be run after relocation has already happened.

    Args:
        src_missing: Missing sentinel for source ids (A index dtype max).
        moves_to: New target ids.
        head: Adjacency heads indexed by target id, values are source ids.
        src_fk: FK values in source table A.
        adj_next: Next adjacency points in source table A.
        adj_prev: Prev adjacency points in source table A.

    Returns:
        None. Mutates head and src_fk.
    """
    for tgt in moves_to:
        h = head[tgt]
        if h == src_missing:
            continue

        # Make sure list-head prev is missing (head invariant).
        adj_prev[h] = src_missing

        # Rewrite FK values along the chain.
        src_row = h
        while src_row != src_missing:
            src_fk[src_row] = tgt
            src_row = adj_next[src_row]


@nb.njit(cache=True)
def interleave_new_rows(
    n_new: int,
    staged_start: int,
    missing_val_src: int,
    missing_val_tgt: int,
    targets: np.ndarray,
    current_heads: np.ndarray,
    adj_next: np.ndarray,
    unlinked_indices: np.ndarray,
    # Scratchpad write targets
    out_prev_idx: np.ndarray,
    out_prev_val: np.ndarray,
    p_cursor: int,
    out_head_idx: np.ndarray,
    out_head_val: np.ndarray,
    h_cursor: int,
) -> tuple[
    np.ndarray,  # new_nexts (len n_new)
    np.ndarray,  # new_prevs (len n_new)
    int,  # p_cursor
    int,  # h_cursor
]:
    """Interleaves new rows into per-target adjacency lists (physical-index mode).

    Groups additions by target and builds contiguous adjacency chains for the new rows.
    Each chain is then spliced in front of the existing head for that target.

    This function writes sparse patches into the provided scratchpad buffers:
      - head[target] = first_new
      - old_head.prev = last_new (only if old_head exists)

    It also returns next/prev arrays for the newly added rows, suitable for
    patching the addition buffers of the source table.

    Args:
        n_new: Number of new additions.
        staged_start: First staged ID.
        missing_val_src: Sentinel missing value of source/parent column (dtype max).
        missing_val_tgt: Sentinel missing value of target/child column (dtype max).
        targets: Target indices (FK targets) for each new row.
        current_heads: Current head pointers indexed by source id.
        adj_next: The adjacency next pointers.
        unlinked_indices: Sorted array of indices that have been unlinked.
        out_prev_idx: Scratchpad indices for prev-pointer patches (applied to source adj_prev).
        out_prev_val: Scratchpad values for prev-pointer patches.
        p_cursor: Initial cursor for prev patches.
        out_head_idx: Scratchpad indices for head-pointer patches (applied to target adj_head).
        out_head_val: Scratchpad values for head-pointer patches.
        h_cursor: Initial cursor for head patches.

    Returns:
        (new_nexts, new_prevs, p_cursor, h_cursor)
    """
    new_nexts = np.empty(n_new, dtype=out_prev_val.dtype)
    new_prevs = np.full(n_new, missing_val_src, dtype=out_prev_val.dtype)

    order = np.argsort(targets, kind="mergesort")

    i = 0
    while i < n_new:
        t = targets[order[i]]
        start = i
        i += 1
        while i < n_new and targets[order[i]] == t:
            i += 1
        end = i

        # Null-target additions: isolate.
        if t == missing_val_tgt:
            for k in range(start, end):
                idx = order[k]
                new_nexts[idx] = missing_val_src
                new_prevs[idx] = missing_val_src
            continue

        if t < len(current_heads):
            raw_head = current_heads[t]
        else:
            raw_head = missing_val_src

        old_head = find_surviving_neighbor(
            int(raw_head), adj_next, unlinked_indices, missing_val_src
        )

        first_idx = order[start]
        last_idx = order[end - 1]

        # head[target] = first_new
        out_head_idx[h_cursor] = t
        out_head_val[h_cursor] = staged_start + first_idx
        h_cursor += 1

        # Link the new chain internally.
        for k in range(start, end - 1):
            a = order[k]
            b = order[k + 1]
            new_nexts[a] = staged_start + b
            new_prevs[b] = staged_start + a

        # Tail points to old head.
        new_nexts[last_idx] = old_head

        # old_head.prev = last_new
        if old_head != missing_val_src:
            out_prev_idx[p_cursor] = old_head
            out_prev_val[p_cursor] = staged_start + last_idx
            p_cursor += 1

    return new_nexts, new_prevs, p_cursor, h_cursor


@nb.njit(cache=True)
def interleave_existing_rows(
    row_indices: np.ndarray,
    targets: np.ndarray,
    current_heads: np.ndarray,
    adj_next: np.ndarray,
    unlinked_indices: np.ndarray,
    missing_val_src: int,
    missing_val_tgt: int,
    # Scratchpad outputs
    out_next_idx: np.ndarray,
    out_next_val: np.ndarray,
    n_cursor: int,
    out_prev_idx: np.ndarray,
    out_prev_val: np.ndarray,
    p_cursor: int,
    out_head_idx: np.ndarray,
    out_head_val: np.ndarray,
    h_cursor: int,
) -> tuple[int, int, int]:
    """Interleaves existing rows into per-target adjacency lists.

    Used when a Foreign Key is updated.
    """
    # Sort by target to group additions
    order = np.argsort(targets, kind="mergesort")
    n_updates = len(row_indices)

    i = 0
    while i < n_updates:
        t = targets[order[i]]
        start = i
        i += 1
        while i < n_updates and targets[order[i]] == t:
            i += 1
        end = i

        if t == missing_val_tgt:
            # If moving to NULL, just clear the next/prev pointers of the row
            for k in range(start, end):
                idx = row_indices[order[k]]
                out_next_idx[n_cursor] = idx
                out_next_val[n_cursor] = missing_val_src
                n_cursor += 1

                out_prev_idx[p_cursor] = idx
                out_prev_val[p_cursor] = missing_val_src
                p_cursor += 1
            continue

        if t < len(current_heads):
            raw_head = current_heads[t]
        else:
            raw_head = missing_val_src

        old_head = find_surviving_neighbor(
            int(raw_head), adj_next, unlinked_indices, missing_val_src
        )

        first_row = row_indices[order[start]]
        last_row = row_indices[order[end - 1]]

        # Update Head: head[target] = first_row
        out_head_idx[h_cursor] = t
        out_head_val[h_cursor] = first_row
        h_cursor += 1

        # Link first_row.prev = missing (it's the new head)
        out_prev_idx[p_cursor] = first_row
        out_prev_val[p_cursor] = missing_val_src
        p_cursor += 1

        # Link the chain internally
        for k in range(start, end - 1):
            curr_row = row_indices[order[k]]
            next_row = row_indices[order[k + 1]]

            # curr.next = next_row
            out_next_idx[n_cursor] = curr_row
            out_next_val[n_cursor] = next_row
            n_cursor += 1

            # next_row.prev = curr
            out_prev_idx[p_cursor] = next_row
            out_prev_val[p_cursor] = curr_row
            p_cursor += 1

        # Tail points to old head
        out_next_idx[n_cursor] = last_row
        out_next_val[n_cursor] = old_head
        n_cursor += 1

        # old_head.prev = last_row
        if old_head != missing_val_src:
            out_prev_idx[p_cursor] = old_head
            out_prev_val[p_cursor] = last_row
            p_cursor += 1

    return n_cursor, p_cursor, h_cursor
