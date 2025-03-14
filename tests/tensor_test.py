import pytest
import numpy as np
import lumine as lm


@pytest.fixture
def generate_tensor():
    """Fixture to generate tensors for given shapes and dtype."""

    def _generate(shape, dtype="int32", min_val=0, max_val=100):
        array = np.random.randint(min_val, max_val + 1, size=shape, dtype=dtype)
        return lm.tensor(array.tolist()), array

    return _generate


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4), (5, 5, 5)])
def test_shape(generate_tensor, shape):
    _tensor, np_array = generate_tensor(shape)
    assert _tensor.shape == np_array.shape


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4), (5, 5, 5)])
def test_subtensor(generate_tensor, shape):
    _tensor, np_array = generate_tensor(shape)
    assert _tensor[0].shape == np_array[0].shape


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4), (5, 5, 5)])
def test_tolist(generate_tensor, shape):
    _tensor, np_array = generate_tensor(shape)
    assert _tensor.tolist() == np_array.tolist()


@pytest.mark.parametrize("shape", [(5, 5, 5)])
def test_add(generate_tensor, shape):
    _tensor1, np_array1 = generate_tensor(shape)
    _tensor2, np_array2 = generate_tensor(shape)
    assert (_tensor1 + _tensor2).tolist() == (np_array1 + np_array2).tolist()


@pytest.mark.parametrize("shape", [(5, 5, 5), (100, 99, 1, 30), (200, 1, 99)])
def test_sub(generate_tensor, shape):
    _tensor1, np_array1 = generate_tensor(shape)
    _tensor2, np_array2 = generate_tensor(shape)
    assert (_tensor1 - _tensor2).tolist() == (np_array1 - np_array2).tolist()


@pytest.mark.parametrize(
    "shapes",
    [
        ([[3, 3], [3]]),
        ([[2, 1], [1, 3]]),
        ([[2, 3], [1, 3]]),
        ([[3, 1], [3]]),
        ([[1, 3, 3], [3, 1]]),
        ([[4, 1, 3], [1, 5, 1]]),
    ],
)
def test_broadcast_sum(generate_tensor, shapes):
    _tensor1, np_array1 = generate_tensor(shapes[0])
    _tensor2, np_array2 = generate_tensor(shapes[1])
    assert (_tensor1 + _tensor2).tolist() == (_tensor2 + _tensor1).tolist()
    assert (_tensor1 + _tensor2).tolist() == (np_array1 + np_array2).tolist()


@pytest.mark.parametrize(
    "shapes",
    [
        ([[3, 3], [3]]),
        ([[2, 1], [1, 3]]),
        ([[2, 3], [1, 3]]),
        ([[3, 1], [3]]),
        ([[1, 3, 3], [3, 1]]),
        ([[4, 1, 3], [1, 5, 1]]),
    ],
)
def test_broadcast_sub(generate_tensor, shapes):
    _tensor1, np_array1 = generate_tensor(shapes[0])
    _tensor2, np_array2 = generate_tensor(shapes[1])
    assert (_tensor1 - _tensor2).tolist() == (np_array1 - np_array2).tolist()
    assert (_tensor2 - _tensor1).tolist() == (np_array2 - np_array1).tolist()


@pytest.mark.parametrize(
    "shapes",
    [
        ([[3, 3], [3]]),
        ([[2, 1], [1, 3]]),
        ([[2, 3], [1, 3]]),
        ([[3, 1], [3]]),
        ([[1, 3, 3], [3, 1]]),
        ([[4, 1, 3], [1, 5, 1]]),
    ],
)
def test_broadcast_mul(generate_tensor, shapes):
    _tensor1, np_array1 = generate_tensor(shapes[0])
    _tensor2, np_array2 = generate_tensor(shapes[1])
    assert (_tensor1 * _tensor2).tolist() == (_tensor2 * _tensor1).tolist()
    assert (_tensor1 * _tensor2).tolist() == (np_array1 * np_array2).tolist()


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4), (5, 5, 5)])
def test_ones_zeros_ones_like_zeros_like(generate_tensor, shape):
    _tensor, np_array = generate_tensor(shape)
    assert lm.ones(shape).tolist() == np.ones(shape).tolist()
    assert lm.zeros(shape).tolist() == np.zeros(shape).tolist()
    assert lm.ones_like(_tensor).tolist() == np.ones(shape).tolist()
    assert lm.zeros_like(_tensor).tolist() == np.zeros(shape).tolist()


@pytest.mark.parametrize(
    "a_shape, b_shape, error",
    [
        ((2, 3), (3, 4), True),  # Unbatched valid case
        ((3, 2), (2, 3), True),  # Another unbatched valid case
        (
            (4, 3),
            (2, 3),
            False,
        ),  # Unbatched incompatible shapes (should raise an error)
        ((2, 3, 4), (2, 4, 5), True),  # Batched valid case
        ((3, 2, 3), (3, 3, 2), True),  # Another batched valid case
        (
            (2, 3, 4),
            (3, 4, 5),
            False,
        ),  # Incompatible batch sizes (should raise an error)
        ((1, 2, 3, 4), (1, 2, 4, 5), True),  # Higher-dimensional batch, valid
        ((2, 1, 3, 4), (1, 2, 4, 5), True),  # Broadcasting batch, valid
        ((2, 3, 4), (4, 5), True),  # Implicit broadcasting, valid
        ((2, 3, 4), (3, 4, 5), False),  # Incompatible batch size, should raise an error
        (
            (1, 2, 3, 4),
            (2, 3, 4, 5),
            False,
        ),  # Incompatible batch size, should raise an error
    ],
)
def test_matmul(generate_tensor, a_shape, b_shape, error):
    a_tensor, a_np = generate_tensor(a_shape)
    b_tensor, b_np = generate_tensor(b_shape)

    try:
        result = lm.matmul(a_tensor, b_tensor)
        expected = np.matmul(a_np, b_np)
        assert result.tolist() == expected.tolist()
    except ValueError:
        if not error:
            pass
        else:
            raise
