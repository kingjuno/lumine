import pytest
from lumine import tensor
from test_utils import generate_nd_list


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4), (5, 5, 5)])
def test_shape(shape):
    _tensor = tensor(generate_nd_list(shape, min_val=0, max_val=100))
    assert _tensor.shape == shape


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4), (5, 5, 5)])
def test_subtensor(shape):
    _tensor = tensor(generate_nd_list(shape, min_val=0, max_val=100))
    subshape = shape[1:]
    subtensor = _tensor[0]
    assert subtensor.shape == subshape

@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4), (5, 5, 5)])
def test_tolist(shape):
    _list = generate_nd_list(shape, min_val=0, max_val=100)
    _tensor = tensor(_list)
    assert _tensor.tolist() == _list

@pytest.mark.parametrize("shape", [(5, 5, 5)])
def test_add(shape):
    # use numpy to tolist and equate the result
    import numpy as np
    _list1 = generate_nd_list(shape, min_val=0, max_val=100)
    _list2 = generate_nd_list(shape, min_val=0, max_val=100)
    _tensor1 = tensor(_list1, "int32")
    _tensor2 = tensor(_list2, "int32")
    _tensor = (_tensor1 + _tensor2).tolist()
    assert (np.array(_list1)+np.array(_list2)).tolist()==_tensor

@pytest.mark.parametrize("shape", [(5, 5, 5),(100,99,1,30),(200,1,99)])
def test_sub(shape):
    # use numpy to tolist and equate the result
    import numpy as np
    _list1 = generate_nd_list(shape, min_val=0, max_val=100)
    _list2 = generate_nd_list(shape, min_val=0, max_val=100)
    _tensor1 = tensor(_list1, "int32")
    _tensor2 = tensor(_list2, "int32")
    _tensor = (_tensor1 - _tensor2).tolist()
    assert (np.array(_list1)-np.array(_list2)).tolist()==_tensor
