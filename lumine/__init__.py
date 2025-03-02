import ctypes
import os

VALID_DTYPES = {"float32", "int32"}
VALID_DEVICES = {"cpu", "gpu"}


class _TensorLib:
    _build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
    _lib_path = os.path.join(_build_dir, "lumine.so")
    if not os.path.exists(_lib_path):
        raise RuntimeError(f"Library not found at: {_lib_path}")
    _lib = ctypes.CDLL(_lib_path)

    @classmethod
    def _signature(cls):
        cls._lib.create_tensor.argtypes = [
            ctypes.c_void_p,  # data ptr
            ctypes.POINTER(ctypes.c_int),  # shape
            ctypes.c_int,  # dim
            ctypes.c_char_p,  # Device
            ctypes.c_char_p,  # Dtype
        ]

        cls._lib.create_tensor.restype = ctypes.c_void_p
        cls._lib.print_tensor.argtypes = [ctypes.c_void_p]
        cls._lib.print_tensor.restype = ctypes.c_char_p
        cls._lib.get_item.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        cls._lib.get_item.restype = ctypes.c_void_p
        cls._lib.get_shape.argtypes = [ctypes.c_void_p]
        cls._lib.get_shape.restype = ctypes.POINTER(ctypes.c_int)
        cls._lib.astype.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        cls._lib.astype.restype = ctypes.c_void_p
        cls._lib.get_data_ptr.argtypes = [ctypes.c_void_p]
        cls._lib.get_data_ptr.restype = ctypes.c_void_p
        cls._lib.tensor_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls._lib.tensor_add.restype = ctypes.c_void_p
        cls._lib.tensor_sub.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls._lib.tensor_sub.restype = ctypes.c_void_p
        cls._lib.tensor_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls._lib.tensor_mul.restype = ctypes.c_void_p

        cls._lib.reshape.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        cls._lib.reshape.restype = ctypes.c_void_p
        cls._lib.get_last_error.argtypes = []
        cls._lib.get_last_error.restype = ctypes.c_char_p
        cls._lib.ones.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        cls._lib.ones.restype = ctypes.c_void_p
        cls._lib.zeros.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        cls._lib.zeros.restype = ctypes.c_void_p

    @classmethod
    def get_library(cls):
        return cls._lib


_TensorLib._signature()
_lib = _TensorLib.get_library()

from .tensor import tensor


def check_error(tensor):
    if not tensor:
        error_msg = _lib.get_last_error()
        if error_msg:
            raise RuntimeError(error_msg.decode("utf-8"))
        else:
            raise RuntimeError("Unknown error! Please report this issue.")

def ones(shape, dtype="float32", device="cpu"):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    else:
        shape = tuple(shape)

    if len(shape) == 0:
        raise ValueError("Shape cannot be empty.")

    if dtype not in VALID_DTYPES:
        raise ValueError(f"Invalid dtype: {dtype}. Supported: {VALID_DTYPES}")
    if device not in VALID_DEVICES:
        raise ValueError(f"Invalid device: {device}. Supported: {VALID_DEVICES}")

    device = device.encode("utf-8") if isinstance(device, str) else device
    dtype = dtype.encode("utf-8") if isinstance(dtype, str) else dtype

    shape_array = (ctypes.c_int * len(shape))(*shape)
    tensor_ptr = _lib.ones(shape_array, len(shape), device, dtype)
    return tensor(_tensor=tensor_ptr, dtype=dtype, device=device, ndim=len(shape))


def zeros(shape, dtype="float32", device="cpu"):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    else:
        shape = tuple(shape)

    if len(shape) == 0:
        raise ValueError("Shape cannot be empty.")

    if dtype not in VALID_DTYPES:
        raise ValueError(f"Invalid dtype: {dtype}. Supported: {VALID_DTYPES}")
    if device not in VALID_DEVICES:
        raise ValueError(f"Invalid device: {device}. Supported: {VALID_DEVICES}")

    device = device.encode("utf-8") if isinstance(device, str) else device
    dtype = dtype.encode("utf-8") if isinstance(dtype, str) else dtype

    shape_array = (ctypes.c_int * len(shape))(*shape)
    tensor_ptr = _lib.zeros(shape_array, len(shape), device, dtype)
    return tensor(_tensor=tensor_ptr, dtype=dtype, device=device, ndim=len(shape))


def ones_like(tensor):
    return ones(
        tensor.shape,
        dtype=tensor.dtype.decode("utf-8"),
        device=tensor.device.decode("utf-8"),
    )


def zeros_like(tensor):
    return zeros(
        tensor.shape,
        dtype=tensor.dtype.decode("utf-8"),
        device=tensor.device.decode("utf-8"),
    )
