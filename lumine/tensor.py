import ctypes
import os

# TODO
# 1. Fix Tensor print by returning from cpp funtion
# 2. slicing
# 3. Free C Tensor Memory
# 4. Cache str representation and free memory

VALID_DTYPES = {"float32", "int32"}
VALID_DEVICES = {"cpu", "gpu"}


class _TensorLib:
    _build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
    _lib_path = os.path.join(_build_dir, "lumine.so")
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
        cls._lib.get_data_ptr.argtypes =[ctypes.c_void_p]
        cls._lib.get_data_ptr.restype =ctypes.c_void_p

    @classmethod
    def get_library(cls):
        return cls._lib


class Tensor:
    """
    A Python wrapper for a C-based Tensor implementation.
    """

    _lib = None

    @classmethod
    def _initialize_library(cls):
        """Initialize _lib only once."""
        if cls._lib is None:
            _TensorLib._signature()
            cls._lib = _TensorLib.get_library()

    def __init__(
        self,
        data=None,
        dtype="float32",
        device="cpu",
        _tensor: ctypes.c_void_p = None,
        ndim=None,
    ):
        self._initialize_library()

        if _tensor and data:
            raise ValueError("Can't initialize tensor and data at the same time")

        if _tensor is not None:
            self._tensor = _tensor
            self.device = device
            self.dtype = dtype
            self.ndim = ndim
            self._garbage_shape = True
            return

        if data is None:
            raise ValueError("Data cannot be None when creating a new tensor.")

        _data, self._shape = self._flatten(data)
        if not _data or not self._shape:
            raise ValueError("Tensor and shape cannot be empty.")

        self.device = device.encode("utf-8") if isinstance(device, str) else device
        self.dtype = dtype.encode("utf-8") if isinstance(dtype, str) else dtype
       
        self.ndim = len(self._shape)

        if dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid dtype: {dtype}. Supported: {VALID_DTYPES}")
        if device not in VALID_DEVICES:
            raise ValueError(f"Invalid device: {device}. Supported: {VALID_DEVICES}")

        if dtype =="float32":
            _data_array = (ctypes.c_float * len(_data))(*_data)
        if dtype =="int32":
            _data_array = (ctypes.c_int32* len(_data))(*_data)
        else:
            _data_array = (ctypes.c_float * len(_data))(*_data)

        _shape_array = (ctypes.c_int * len(self._shape))(*self._shape)

        self._tensor = self._lib.create_tensor(
            _data_array, _shape_array, self.ndim, self.device, self.dtype
        )
        if not self._tensor:
            raise RuntimeError("Failed to create Tensor.")

        self._garbage_shape = False  # to avoid recomputing shape

    def __str__(self) -> None:
        """
        Print tensor info.
        """
        arr = ctypes.cast(
            self._lib.print_tensor(self._tensor), ctypes.c_char_p
        ).value.decode("utf-8")
        return arr

    def __repr__(self):
        arr = ctypes.cast(
            self._lib.print_tensor(self._tensor), ctypes.c_char_p
        ).value.decode("utf-8")
        return f"array({arr}, dtype: {self.dtype.decode()}, device: {self.device.decode()})"

    @staticmethod
    def _flatten(lst):
        if not isinstance(lst, list):
            raise ValueError("Input must be a list")

        shape = []
        flattened = []

        def _recursive_flatten(sublist, depth):
            if isinstance(sublist, list):
                if len(shape) <= depth:
                    shape.append(len(sublist))
                elif shape[depth] != len(sublist):
                    raise ValueError("Jagged arrays are not supported")
                for item in sublist:
                    _recursive_flatten(item, depth + 1)
            else:
                flattened.append(sublist)

        _recursive_flatten(lst, 0)
        return flattened, tuple(shape)

    def __getitem__(self, indices):
        if isinstance(indices, slice) or indices is Ellipsis:
            raise NotImplementedError("Slicing and Ellipsis are not supported yet.")

        if not isinstance(indices, tuple):
            indices = (indices,)

        if any(not isinstance(idx, int) for idx in indices):
            raise ValueError("Only integer indexing is currently supported.")

        index_array = (ctypes.c_int * self.ndim)(*indices)

        try:
            tensor = self._lib.get_item(self._tensor, index_array, len(indices))
        except Exception as e:
            raise RuntimeError(f"Failed to get tensor item: {e}")

        if len(indices) == self.ndim:
            scalar_ptr = ctypes.cast(
                tensor,
                ctypes.POINTER(
                    ctypes.c_float if self.dtype == b"float32" else ctypes.c_int32
                ),
            )
            
            return scalar_ptr.contents.value

        return Tensor(
            _tensor=tensor,
            dtype=self.dtype,
            device=self.device,
            ndim=self.ndim - len(indices),
        )
    def astype(self , target_dtype): 
        """
        cast the tensor to the new dtype
        
        """

        if target_dtype not in VALID_DTYPES: 
            raise ValueError(f"Invalid targer dtype: {target_dtype}")
        if self.dtype.decode() == target_dtype: 
            return self
        target_dtype = target_dtype.encode('utf-8')
        tensor_ptr = self._lib.astype(self._tensor, target_dtype)

        if not tensor_ptr:
            raise RuntimeError(f"Failed to cast tensor to {target_dtype.decode()}.")

        return Tensor(
            _tensor=tensor_ptr,
            dtype=target_dtype,
            device=self.device,
            ndim=self.ndim,
        )
    def tolist(self):
        """
        convert tensor to a python list
        """

        data_ptr = self._lib.get_data_ptr(self._tensor)
        if not data_ptr:
            raise RuntimeError("Failed to get tensor data.")
        
        num_element =1; 

        for dim in self.shape:
            num_element*=dim

        if self.dtype == b"float32":
            array_type = ctypes.POINTER(ctypes.c_float)
        elif self.dtype == b"int32":
            array_type = ctypes.POINTER(ctypes.c_int32)
        else:
            raise ValueError("Unsupported dtype.")

        c_array = ctypes.cast(data_ptr, array_type)
 
        flat_list = [c_array[i] for i in range(num_element)]
        
        #reccurive function to reshape the list
        def reshape(flat, shape):
            if len(shape)==1:
                return flat[:shape[0]]
            sub_size = int(len(flat)/ shape[0])

            return [reshape(flat[i* sub_size:(i+1)* sub_size], shape[1:])for i in range(shape[0])] 
        
        return reshape(flat_list, list(self.shape))
    @property
    def shape(self):
        if self._garbage_shape == False:
            return self._shape
        else:
            tensor_shape = self._lib.get_shape(self._tensor)
            shape = [tensor_shape[i] for i in range(self.ndim)]
            self._shape = tuple(shape)
            self._garbage_shape = False
            return self._shape
