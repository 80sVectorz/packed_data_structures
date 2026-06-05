import numpy as np
import numba as nb


@nb.njit
def nb_count_adj_elements(
    head_indices: tuple[int, ...] | np.ndarray,
    head_arr: np.ndarray,
    next_arr: np.ndarray,
    missing_idx: int,
    *,
    out: np.ndarray,
):
    for i, head_idx in enumerate(head_indices):
        cnt = 0
        nxt = head_arr[head_idx]
        while nxt != missing_idx:
            cnt += 1
            nxt = next_arr[nxt]
        out[i] = cnt


@nb.njit
def nb_get_adj_elements(
    head_indices: tuple[int, ...] | np.ndarray,
    counts: tuple[int] | np.ndarray,
    head_arr: np.ndarray,
    next_arr: np.ndarray,
    *,
    out: tuple[np.ndarray, ...],
):
    for i, head_idx in enumerate(head_indices):
        current_out = out[i]
        count = counts[i]
        if count == 0:
            continue

        nxt = head_arr[head_idx]
        current_out[0] = nxt

        for j in range(1, counts[i]):
            nxt = next_arr[nxt]
            current_out[j] = nxt
