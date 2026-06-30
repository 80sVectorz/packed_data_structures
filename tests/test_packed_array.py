import pytest
import numpy as np
from packed_data_structures.packed_array import PackedArray, PackedArrayBuffer


def test_buffer_initialization():
    buf = PackedArrayBuffer(capacity=10, dtype=np.float32)
    assert buf.capacity == 10
    assert buf.size == 0
    assert buf.arr.shape == (10,)
    assert buf.arr.dtype == np.float32


def test_buffer_initialization_with_shape():
    buf = PackedArrayBuffer(capacity=5, dtype=np.int32, element_shape=(3,))
    assert buf.capacity == 5
    assert buf.arr.shape == (5, 3)


def test_buffer_resize():
    buf = PackedArrayBuffer(capacity=5, dtype=np.float32)
    buf.size = 3
    buf.arr[:3] = [1.0, 2.0, 3.0]

    buf.resize(10)
    assert buf.capacity == 10
    assert buf.arr.shape == (10,)
    assert np.array_equal(buf.arr[:3], [1.0, 2.0, 3.0])


def test_buffer_resize_shrink_fails_if_not_allowed():
    buf = PackedArrayBuffer(capacity=5, dtype=np.float32)
    buf.size = 4
    with pytest.raises(ValueError):
        buf.resize(3, allow_shrink=False)


def test_packed_array_initialization():
    pa = PackedArray(pre_allocated_capacity=10, dtype=np.int32, empty_fill=-1)
    assert len(pa) == 0
    assert pa._data.capacity == 10
    assert pa.empty_fill == -1


def test_packed_array_initialization_tuple():
    pa = PackedArray(pre_allocated_capacity=(10, 3), dtype=np.float32)
    assert pa.element_shape == (3,)
    assert pa._data.capacity == 10


def test_packed_array_append():
    pa = PackedArray(pre_allocated_capacity=2, dtype=np.int32)
    pa.append(10)
    pa.append(20)
    assert len(pa) == 2
    assert np.array_equal(pa.view, [10, 20])


def test_packed_array_append_resize():
    pa = PackedArray(pre_allocated_capacity=2, dtype=np.int32, resize_factor=2)
    pa.append(10)
    pa.append(20)
    pa.append(30)
    assert len(pa) == 3
    assert pa._data.capacity >= 3
    assert np.array_equal(pa.view, [10, 20, 30])


def test_packed_array_append_no_resize_fails():
    pa = PackedArray(pre_allocated_capacity=2, dtype=np.int32, resize_factor=0)
    pa.append(10)
    pa.append(20)
    with pytest.raises(Exception, match="max capacity reached"):
        pa.append(30)


def test_packed_array_ensure_size():
    pa = PackedArray(pre_allocated_capacity=5, dtype=np.int32, empty_fill=0)
    pa.ensure_size(3)
    assert len(pa) == 3
    assert np.array_equal(pa.view, [0, 0, 0])

    # test shrink
    pa.ensure_size(1, shrink_size=True)
    assert len(pa) == 1


def test_packed_array_remove_swap_and_pop():
    pa = PackedArray(pre_allocated_capacity=5, dtype=np.int32)
    for v in [10, 20, 30, 40, 50]:
        pa.append(v)

    val = pa.remove(1)  # removes 20, swaps 50 into its place
    assert val == 20
    assert len(pa) == 4
    assert pa[1] == 50
    assert np.array_equal(pa.view, [10, 50, 30, 40])

    val = pa.remove(-1)  # removes 40
    assert val == 40
    assert len(pa) == 3
    assert np.array_equal(pa.view, [10, 50, 30])


def test_packed_array_remove_out_of_bounds():
    pa = PackedArray(pre_allocated_capacity=5, dtype=np.int32)
    pa.append(10)
    with pytest.raises(IndexError):
        pa.remove(1)
    with pytest.raises(IndexError):
        pa.remove(-2)


def test_packed_array_swap():
    pa = PackedArray(pre_allocated_capacity=5, dtype=np.int32)
    for v in [10, 20, 30]:
        pa.append(v)

    pa.swap(0, 2)
    assert np.array_equal(pa.view, [30, 20, 10])


def test_packed_array_numpy_interface():
    pa = PackedArray(pre_allocated_capacity=5, dtype=np.int32)
    for v in [1, 2, 3]:
        pa.append(v)

    assert np.sum(pa) == 6
    pa[:] *= 2
    assert np.array_equal(pa.view, [2, 4, 6])

    # test ufunc with another packed array
    pa2 = PackedArray(pre_allocated_capacity=5, dtype=np.int32)
    for v in [10, 20, 30]:
        pa2.append(v)

    result = pa + pa2
    assert np.array_equal(result, [12, 24, 36])
