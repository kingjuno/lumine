import pytest
from lumine import Tensor
from test_utils import generate_nd_list


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4), (5, 5, 5)])
def test_shape(shape):
    tensor = Tensor(generate_nd_list(shape, min_val=0, max_val=100))
    assert tensor.shape == shape


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4), (5, 5, 5)])
def test_subtensor(shape):
    tensor = Tensor(generate_nd_list(shape, min_val=0, max_val=100))
    subshape = shape[1:]
    subtensor = tensor[0]
    assert subtensor.shape == subshape
