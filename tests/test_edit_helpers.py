import pytest
import numpy as np

from packed_data_structures.edit_helpers import (
    plan_bulk_edit,
    find_surviving_neighbor,
    patch_unlinked_adjacency_and_heads,
    patch_relocated_adjacency_and_heads,
    fix_keys_broken_by_moved_targets,
    interleave_new_rows,
    interleave_existing_rows,
)
from .testing_utils import get_func_variants


def test_plan_bulk_edit():
    # Null removals
    new_size, add_dests, relocs, rem_arr = plan_bulk_edit(
        current_size=10, n_additions=2, removals=None
    )
    assert new_size == 12
    assert np.array_equal(add_dests, [10, 11])
    assert len(relocs[0]) == 0 and len(relocs[1]) == 0
    assert len(rem_arr) == 0

    # Negative removals
    new_size, add_dests, relocs, rem_arr = plan_bulk_edit(
        current_size=10, n_additions=0, removals=[-1, -2]
    )
    assert new_size == 8
    assert len(add_dests) == 0
    assert np.array_equal(rem_arr, [8, 9])
    
    # Growing array (additions > removals)
    new_size, add_dests, relocs, rem_arr = plan_bulk_edit(
        current_size=10, n_additions=3, removals=[2]
    )
    assert new_size == 12
    assert np.array_equal(rem_arr, [2])
    assert np.array_equal(add_dests, [2, 10, 11])
    assert len(relocs[0]) == 0 and len(relocs[1]) == 0

    # Shrinking array (additions < removals)
    new_size, add_dests, relocs, rem_arr = plan_bulk_edit(
        current_size=10, n_additions=1, removals=[2, 4, 8]
    )
    assert new_size == 8
    assert np.array_equal(add_dests, [2])
    assert np.array_equal(relocs[0], [9])
    assert np.array_equal(relocs[1], [4])

    # Perfect replacement (additions == removals)
    new_size, add_dests, relocs, rem_arr = plan_bulk_edit(
        current_size=10, n_additions=2, removals=[3, 7]
    )
    assert new_size == 10
    assert np.array_equal(add_dests, [3, 7])
    assert len(relocs[0]) == 0 and len(relocs[1]) == 0


@pytest.mark.parametrize(
    "name, find_surviving_fn", get_func_variants(find_surviving_neighbor)
)
def test_find_surviving_neighbor(name, find_surviving_fn):
    adjacency = np.array([1, 2, 3, 4, 999], dtype=np.int32)
    missing = 999
    
    # Head survives
    unlinked = np.array([2], dtype=np.int32)
    assert find_surviving_fn(0, adjacency, unlinked, missing) == 0
    
    # Head is dead, next survives
    unlinked = np.array([0], dtype=np.int32)
    assert find_surviving_fn(0, adjacency, unlinked, missing) == 1
    
    # Entire chain is dead
    unlinked = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    assert find_surviving_fn(0, adjacency, unlinked, missing) == 999
    
    # Head dead, next dead, third survives
    unlinked = np.array([0, 1], dtype=np.int32)
    assert find_surviving_fn(0, adjacency, unlinked, missing) == 2


@pytest.mark.parametrize(
    "name, patch_unlinked_fn", get_func_variants(patch_unlinked_adjacency_and_heads)
)
def test_patch_unlinked_adjacency_and_heads(name, patch_unlinked_fn):
    missing_src = 999
    missing_tgt = 999
    
    unlinked = np.array([1, 2, 4], dtype=np.int32)
    head_ignore = np.array([], dtype=np.int32)
    
    parent_fk = np.array([10, 10, 10, 10, 10, 10], dtype=np.int32)
    
    # Original adjacency:
    # 0 <-> 1 <-> 2 <-> 3 <-> 4 <-> 5
    # Delete 1, 2, 4
    adj_next = np.array([1, 2, 3, 4, 5, missing_src], dtype=np.int32)
    adj_prev = np.array([missing_src, 0, 1, 2, 3, 4], dtype=np.int32)
    
    target_adj_head = np.full(20, missing_src, dtype=np.int32)
    target_adj_head[10] = 0
    
    out_next_idx = np.zeros(10, dtype=np.int32)
    out_next_val = np.zeros(10, dtype=np.int32)
    out_prev_idx = np.zeros(10, dtype=np.int32)
    out_prev_val = np.zeros(10, dtype=np.int32)
    out_head_idx = np.zeros(10, dtype=np.int32)
    out_head_val = np.zeros(10, dtype=np.int32)
    
    nc, pc, hc = patch_unlinked_fn(
        missing_src, missing_tgt, unlinked, head_ignore, parent_fk,
        adj_next, adj_prev, target_adj_head,
        out_next_idx, out_next_val, 0,
        out_prev_idx, out_prev_val, 0,
        out_head_idx, out_head_val, 0
    )
    
    assert nc == 2
    assert pc == 2
    assert hc == 0
    
    assert out_next_idx[0] == 0 and out_next_val[0] == 3
    assert out_next_idx[1] == 3 and out_next_val[1] == 5
    
    assert out_prev_idx[0] == 3 and out_prev_val[0] == 0
    assert out_prev_idx[1] == 5 and out_prev_val[1] == 3

    # Test Unlink Head
    unlinked = np.array([0], dtype=np.int32)
    nc, pc, hc = patch_unlinked_fn(
        missing_src, missing_tgt, unlinked, head_ignore, parent_fk,
        adj_next, adj_prev, target_adj_head,
        out_next_idx, out_next_val, 0,
        out_prev_idx, out_prev_val, 0,
        out_head_idx, out_head_val, 0
    )
    assert nc == 0
    assert pc == 1
    assert hc == 1
    assert out_prev_idx[0] == 1 and out_prev_val[0] == missing_src
    assert out_head_idx[0] == 10 and out_head_val[0] == 1


@pytest.mark.parametrize(
    "name, patch_reloc_fn", get_func_variants(patch_relocated_adjacency_and_heads)
)
def test_patch_relocated_adjacency_and_heads(name, patch_reloc_fn):
    missing_src = 999
    missing_tgt = 999
    head_ignore = np.array([], dtype=np.int32)
    
    adj_next = np.array([1, 2, missing_src], dtype=np.int32)
    adj_prev = np.array([missing_src, 0, 1], dtype=np.int32)
    src_fk = np.array([10, 10, 10], dtype=np.int32)
    
    target_adj_head = np.full(20, missing_src, dtype=np.int32)
    target_adj_head[10] = 0
    
    moves_from = np.array([1], dtype=np.int32)
    moves_to = np.array([5], dtype=np.int32)
    
    out_next_idx = np.zeros(10, dtype=np.int32)
    out_next_val = np.zeros(10, dtype=np.int32)
    out_prev_idx = np.zeros(10, dtype=np.int32)
    out_prev_val = np.zeros(10, dtype=np.int32)
    out_head_idx = np.zeros(10, dtype=np.int32)
    out_head_val = np.zeros(10, dtype=np.int32)
    
    nc, pc, hc = patch_reloc_fn(
        missing_src, missing_tgt, head_ignore, moves_from, moves_to, src_fk,
        adj_next, adj_prev, target_adj_head,
        out_next_idx, out_next_val, 0,
        out_prev_idx, out_prev_val, 0,
        out_head_idx, out_head_val, 0
    )
    
    assert nc == 1
    assert pc == 1
    assert hc == 0
    assert out_next_idx[0] == 0 and out_next_val[0] == 1
    assert out_prev_idx[0] == 2 and out_prev_val[0] == 1
    
    # Relocate Head
    moves_from = np.array([0], dtype=np.int32)
    moves_to = np.array([4], dtype=np.int32)
    nc, pc, hc = patch_reloc_fn(
        missing_src, missing_tgt, head_ignore, moves_from, moves_to, src_fk,
        adj_next, adj_prev, target_adj_head,
        out_next_idx, out_next_val, 0,
        out_prev_idx, out_prev_val, 0,
        out_head_idx, out_head_val, 0
    )
    assert nc == 0
    assert pc == 1
    assert hc == 1
    assert out_prev_idx[0] == 1 and out_prev_val[0] == 0
    assert out_head_idx[0] == 10 and out_head_val[0] == 0


@pytest.mark.parametrize(
    "name, fix_keys_fn", get_func_variants(fix_keys_broken_by_moved_targets)
)
def test_fix_keys_broken_by_moved_targets(name, fix_keys_fn):
    src_missing = 999
    moves_to = np.array([5], dtype=np.int32)
    
    head = np.full(10, src_missing, dtype=np.int32)
    head[5] = 0
    
    src_fk = np.array([3, 3, 3], dtype=np.int32)
    adj_next = np.array([1, 2, src_missing], dtype=np.int32)
    adj_prev = np.array([99, 0, 1], dtype=np.int32)
    
    fix_keys_fn(src_missing, moves_to, head, src_fk, adj_next, adj_prev)
    
    assert adj_prev[0] == src_missing
    assert np.array_equal(src_fk, [5, 5, 5])


@pytest.mark.parametrize(
    "name, interleave_new_fn", get_func_variants(interleave_new_rows)
)
def test_interleave_new_rows(name, interleave_new_fn):
    n_new = 3
    staged_start = 10
    missing_src = 999
    missing_tgt = 999
    
    targets = np.array([5, 5, missing_tgt], dtype=np.int32)
    current_heads = np.full(10, missing_src, dtype=np.int32)
    current_heads[5] = 2
    
    adj_next = np.array([0, 1, missing_src], dtype=np.int32)
    unlinked = np.array([], dtype=np.int32)
    
    out_prev_idx = np.zeros(10, dtype=np.int32)
    out_prev_val = np.zeros(10, dtype=np.int32)
    out_head_idx = np.zeros(10, dtype=np.int32)
    out_head_val = np.zeros(10, dtype=np.int32)
    
    new_nexts, new_prevs, pc, hc = interleave_new_fn(
        n_new, staged_start, missing_src, missing_tgt, targets,
        current_heads, adj_next, unlinked,
        out_prev_idx, out_prev_val, 0,
        out_head_idx, out_head_val, 0
    )
    
    assert new_nexts[0] == 11
    assert new_prevs[0] == missing_src
    assert new_nexts[1] == 2
    assert new_prevs[1] == 10
    
    assert new_nexts[2] == missing_src
    assert new_prevs[2] == missing_src
    
    assert hc == 1
    assert out_head_idx[0] == 5 and out_head_val[0] == 10
    assert pc == 1
    assert out_prev_idx[0] == 2 and out_prev_val[0] == 11


@pytest.mark.parametrize(
    "name, interleave_existing_fn", get_func_variants(interleave_existing_rows)
)
def test_interleave_existing_rows(name, interleave_existing_fn):
    missing_src = 999
    missing_tgt = 999
    
    row_indices = np.array([2, 4, 6], dtype=np.int32)
    targets = np.array([5, 5, missing_tgt], dtype=np.int32)
    
    current_heads = np.full(10, missing_src, dtype=np.int32)
    current_heads[5] = 8
    
    adj_next = np.array([0]*10, dtype=np.int32)
    adj_next[8] = missing_src
    unlinked = np.array([], dtype=np.int32)
    
    out_next_idx = np.zeros(10, dtype=np.int32)
    out_next_val = np.zeros(10, dtype=np.int32)
    out_prev_idx = np.zeros(10, dtype=np.int32)
    out_prev_val = np.zeros(10, dtype=np.int32)
    out_head_idx = np.zeros(10, dtype=np.int32)
    out_head_val = np.zeros(10, dtype=np.int32)
    
    nc, pc, hc = interleave_existing_fn(
        row_indices, targets, current_heads, adj_next, unlinked,
        missing_src, missing_tgt,
        out_next_idx, out_next_val, 0,
        out_prev_idx, out_prev_val, 0,
        out_head_idx, out_head_val, 0
    )
    
    assert hc == 1
    assert out_head_idx[0] == 5 and out_head_val[0] == 2
    
    assert nc == 3
    assert out_next_idx[0] == 2 and out_next_val[0] == 4
    assert out_next_idx[1] == 4 and out_next_val[1] == 8
    assert out_next_idx[2] == 6 and out_next_val[2] == missing_src
    
    assert pc == 4
    assert out_prev_idx[0] == 2 and out_prev_val[0] == missing_src
    assert out_prev_idx[1] == 4 and out_prev_val[1] == 2
    assert out_prev_idx[2] == 8 and out_prev_val[2] == 4
    assert out_prev_idx[3] == 6 and out_prev_val[3] == missing_src
