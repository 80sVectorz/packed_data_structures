from typing import Any
import numpy as np
import numba as nb


@nb.njit
def nb_array_mapping_get(keys: np.ndarray, values: np.ndarray, key: Any) -> int:
    for i in range(len(keys)):
        k = keys[i]
        if k == key:
            return i
    return len(keys)


@nb.njit
def nb_find_replace(data: np.ndarray, keys: np.ndarray, values: np.ndarray) -> int:
    n_replaced = 0

    n_replace_mappings = len(keys)

    for i in nb.prange(len(data)):
        replace_val_idx = nb_array_mapping_get(keys, values, data[i])
        if replace_val_idx < n_replace_mappings:
            data[i] = values[replace_val_idx]
            n_replaced += 1

    return n_replaced
