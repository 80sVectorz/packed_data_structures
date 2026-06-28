import pytest
import numpy as np

from packed_data_structures.nb_utils import nb_array_mapping_get, nb_find_replace
from packed_data_structures.nb_hash_set import (
    build_int_set_same_dtype,
    int_set_contains_same_dtype,
    _ceil_pow2,
    _mix_u64,
)
from packed_data_structures.nb_adjacency_list_helpers import (
    nb_count_adj_elements,
    nb_get_adj_elements,
)
from .testing_utils import get_func_variants

@pytest.mark.parametrize(
    "name, nb_array_mapping_get_fn", get_func_variants(nb_array_mapping_get)
)
def test_nb_array_mapping_get(name, nb_array_mapping_get_fn):
    keys = np.array([10, 20, 30], dtype=np.int32)
    values = np.array([100, 200, 300], dtype=np.int32)

    assert nb_array_mapping_get_fn(keys, values, 20) == 1
    assert (
        nb_array_mapping_get_fn(keys, values, 40) == 3
    )  # returns len(keys) if not found


@pytest.mark.parametrize("name, nb_find_replace_fn", get_func_variants(nb_find_replace))
def test_nb_find_replace(name, nb_find_replace_fn):
    data = np.array([10, 50, 20, 10, 30], dtype=np.int32)
    keys = np.array([10, 20], dtype=np.int32)
    values = np.array([99, 88], dtype=np.int32)

    n_replaced = nb_find_replace_fn(data, keys, values)

    assert n_replaced == 3
    assert np.array_equal(data, [99, 50, 88, 99, 30])


@pytest.mark.parametrize("name, ceil_pow2_fn", get_func_variants(_ceil_pow2))
def test_ceil_pow2(name, ceil_pow2_fn):
    assert ceil_pow2_fn(3) == 4
    assert ceil_pow2_fn(4) == 4
    assert ceil_pow2_fn(5) == 8
    assert ceil_pow2_fn(16) == 16


@pytest.mark.parametrize("name1, build_fn", get_func_variants(build_int_set_same_dtype))
@pytest.mark.parametrize(
    "name2, contains_fn", get_func_variants(int_set_contains_same_dtype)
)
def test_hash_set_operations(name1, build_fn, name2, contains_fn):
    keys = np.array([10, 20, 30], dtype=np.int32)
    sentinel = np.int32(-1)

    table, mask = build_fn(keys, sentinel)

    assert contains_fn(table, mask, 10, sentinel)
    assert contains_fn(table, mask, 20, sentinel)
    assert contains_fn(table, mask, 30, sentinel)
    assert not contains_fn(table, mask, 40, sentinel)


@pytest.mark.parametrize("name, count_fn", get_func_variants(nb_count_adj_elements))
def test_nb_count_adj_elements(name, count_fn):
    head_arr = np.array([0, 3, -1], dtype=np.int32)
    next_arr = np.array([1, 2, -1, 4, -1], dtype=np.int32)
    missing_idx = -1

    head_indices = np.array([0, 1, 2], dtype=np.int32)
    out = np.zeros(3, dtype=np.int32)

    count_fn(head_indices, head_arr, next_arr, missing_idx, out=out)

    assert out[0] == 3
    assert out[1] == 2
    assert out[2] == 0


@pytest.mark.parametrize("name, get_adj_fn", get_func_variants(nb_get_adj_elements))
def test_nb_get_adj_elements(name, get_adj_fn):
    head_arr = np.array([0, 3, -1], dtype=np.int32)
    next_arr = np.array([1, 2, -1, 4, -1], dtype=np.int32)

    head_indices = np.array([0, 1, 2], dtype=np.int32)
    counts = np.array([3, 2, 0], dtype=np.int32)

    out = (
        np.zeros(3, dtype=np.int32),
        np.zeros(2, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
    )

    get_adj_fn(head_indices, counts, head_arr, next_arr, out=out)

    assert np.array_equal(out[0], [0, 1, 2])
    assert np.array_equal(out[1], [3, 4])
    assert np.array_equal(out[2], [])
