import pytest
import numpy as np
from lumine import tensor

@pytest.fixture
def generate_tensor():
    """Fixture to generate tensors for given shapes and dtype."""
    def _generate(shape, dtype="int32", min_val=0, max_val=100):
        array = np.random.randint(min_val, max_val + 1, size=shape, dtype=dtype)
        return tensor(array.tolist()), array
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
    #assert (_tensor1 - _tensor2).tolist() == (_tensor2 - _tensor1).tolist()
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

