#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstring>
#include "tensor.h"
#include "utils/stream.h"
#include "cpu/math.h"

thread_local std::string last_error;
inline std::string DTypeToString(DType dt) {
    switch (dt) {
    case DType::FLOAT32:
        return "float32";
    case DType::INT32:
        return "int32";
    default:
        return "unknown";
    }
}

template <typename T>
Tensor<T>::Tensor(T* data_ptr, int* shape, int ndim, std::string device, DType dtype_enum)
    : ndim(ndim), device(device), dtype_enum(dtype_enum) {
    if (data_ptr == nullptr) {
        throw std::runtime_error("Empty Array not supported!");
    }

    this->shape = new int[ndim];
    this->strides = new int[ndim];

    if (this->strides == nullptr || this->shape == nullptr) {
        throw std::runtime_error("Memory Allocation failed!");
    }

    memcpy(this->shape, shape, ndim * sizeof(int));
    int stride = 1;
    int _linear_size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        this->strides[i] = stride;
        stride *= shape[i];
        _linear_size *= shape[i];
    }

    this->_linear_size = _linear_size;
    this->data_ptr = new T[_linear_size];
    memcpy(this->data_ptr, data_ptr, _linear_size * sizeof(T));
}

template <typename T>
Tensor<T>::~Tensor() {
    delete[] shape;
    delete[] strides;
    delete[] data_ptr;
}

template <typename T>
const int* Tensor<T>::get_shape() const {
    return shape;
}

template <typename T>
const int* Tensor<T>::get_strides() const {
    return strides;
}

template <typename T>
DType Tensor<T>::get_dtype_enum() const {
    return dtype_enum;
}

template <typename T>
int Tensor<T>::get_ndim() const {
    return ndim;
}

template <typename T>
int Tensor<T>::get_linear_size() const {
    return _linear_size;
}

template <typename T>
void* Tensor<T>::get_data_ptr() const {
    return data_ptr;
}

template <typename T>
std::string Tensor<T>::get_device() const {
    return device;
}


template <typename T>
std::string Tensor<T>::print_recursive(const T* data, const int* shape, const int* strides, int ndim, int dim, int offset) const {
    OStream oss(6);
    if (dim == ndim - 1) {
        oss << "[";
        for (int i = 0; i < shape[dim]; i++) {
            oss << data[offset + i * strides[dim]];
            if (i < shape[dim] - 1) oss << ", ";
        }
        oss << "]";
    } else {
        oss << "[";
        for (int i = 0; i < shape[dim]; i++) {
            oss << print_recursive(data, shape, strides, ndim, dim + 1, offset + i * strides[dim]);
            if (i < shape[dim] - 1) oss << ", ";
        }
        oss << "]";
    }
    return oss.str();
}

template <typename T>
std::string Tensor<T>::print() const {
    return print_recursive(data_ptr, shape, strides, ndim);
}

template <typename T>
BaseTensor* Tensor<T>::astype(DType target_type) {
    if (target_type == DType::FLOAT32) {
        float* new_data = new float[_linear_size];
        for (int i = 0; i < _linear_size; i++) {
            new_data[i] = static_cast<float>(data_ptr[i]);
        }
        return new Tensor<float>(new_data, shape, ndim, device, DType::FLOAT32);
    }
    if (target_type == DType::INT32) {
        int* new_data = new int[_linear_size];
        for (int i = 0; i < _linear_size; i++) {
            new_data[i] = static_cast<int>(data_ptr[i]);
        }
        return new Tensor<int>(new_data, shape, ndim, device, DType::INT32);
    }
    return nullptr;
}

// Explicit instantiation for supported types
template class Tensor<float>;
template class Tensor<int>;

// External C functions
extern "C" {
    BaseTensor* create_tensor(void* data_ptr, int* shape, int ndim, const char* device, const char* dtype) {
        try {
            if (std::strcmp(dtype, "float32") == 0)
                return new Tensor<float>(static_cast<float*>(data_ptr), shape, ndim, device, DType::FLOAT32);
            else if (std::strcmp(dtype, "int32") == 0)
                return new Tensor<int>(static_cast<int*>(data_ptr), shape, ndim, device, DType::INT32);
            throw std::runtime_error("Invalid dtype!");
        }
        catch (const std::exception& e) {
            last_error = e.what();  // Store error message
            return nullptr;         // Return nullptr to indicate failure
        }

    }

    const char* print_tensor(BaseTensor* tensor) {
        std::string array = tensor->print();
        char* carray = new char[array.length() + 1];
        std::strcpy(carray, array.c_str());
        return carray;
    }

    BaseTensor* get_item(BaseTensor* tensor, int* indices, int ind_len) {
        try {
            int ndim = tensor->get_ndim();
            const int* shape = tensor->get_shape();
            const int* strides = tensor->get_strides();
            if (ind_len > ndim)
                throw std::runtime_error("Too many indices!");

            int linear_index = 0;
            for (int i = 0; i < ind_len; i++) {
                if (indices[i] < 0)
                    throw std::runtime_error("Negative Indexing not supported yet!");
                if (indices[i] >= shape[i])
                    throw std::runtime_error("Indices out of bound!");
                linear_index += indices[i] * strides[i];
            }

            void* data_ptr = tensor->get_data_ptr();
            DType dtype = tensor->get_dtype_enum();
            if (ndim == ind_len) {
                switch (dtype) {
                case DType::FLOAT32: {
                    float* scalar_value = new float(static_cast<float*>(data_ptr)[linear_index]);
                    return reinterpret_cast<BaseTensor*>(scalar_value);
                }
                case DType::INT32: {
                    int* scalar_value = new int(static_cast<int*>(data_ptr)[linear_index]);
                    return reinterpret_cast<BaseTensor*>(scalar_value);
                }
                default:
                    throw std::runtime_error("Unsupported dtype!");
                }
            } else {
                int new_dim = ndim - ind_len;
                int* new_shape = new int[ndim];
                memcpy(new_shape, shape + ind_len, new_dim * sizeof(int));
                switch (dtype) {
                case DType::FLOAT32:
                    return new Tensor<float>(
                               static_cast<float*>(data_ptr) + linear_index,
                               new_shape,
                               new_dim,
                               "cpu",
                               DType::FLOAT32);
                case DType::INT32:
                    return new Tensor<int>(
                               static_cast<int*>(data_ptr) + linear_index,
                               new_shape,
                               new_dim,
                               "cpu",
                               DType::INT32);
                default:
                    throw std::runtime_error("Unsupported dtype!");
                }
            }
        }
        catch (const std::exception& e) {
            last_error = e.what();
            return nullptr;
        }
    }

    int* get_shape(BaseTensor* tensor) {
        return const_cast<int*>(tensor->get_shape());
    }

    BaseTensor* astype(BaseTensor* tensor, const char* target_type) {
        try {
            if (std::strcmp(target_type, "float32") == 0)
                return tensor->astype(DType::FLOAT32);
            if (std::strcmp(target_type, "int32") == 0)
                return tensor->astype(DType::INT32);
            throw std::runtime_error("Unsupported dtype!");
        }
        catch (const std::exception& e) {
            last_error = e.what();
            return nullptr;
        }
    }

    void* get_data_ptr(BaseTensor* tensor) {
        return tensor->get_data_ptr();
    }

    // math functions
    BaseTensor* tensor_add(const BaseTensor* _this, const BaseTensor* _other) {
        try {
            DType dtype1 = _this->get_dtype_enum();
            DType dtype2 = _other->get_dtype_enum();

            if (dtype1 == DType::FLOAT32 || dtype2 == DType::FLOAT32) {
                Tensor<float>* t1 = dynamic_cast<Tensor<float>*>(const_cast<BaseTensor*>(_this));
                Tensor<float>* t2 = dynamic_cast<Tensor<float>*>(const_cast<BaseTensor*>(_other));
                if (!t1 || !t2) {
                    throw std::runtime_error("Type mismatch in tensor_add!");
                }
                float* data_ptr = new float[t1->get_linear_size()];
                int* shape = new int[t1->get_ndim()];
                memcpy(data_ptr, t1->get_data_ptr(), t1->get_linear_size() * sizeof(float));
                memcpy(shape, t1->get_shape(), t1->get_ndim() * sizeof(int));
                Tensor<float>* result = new Tensor<float>(data_ptr, shape, t1->get_ndim(), "cpu", DType::FLOAT32);
                cpu_tensor_add(*result, *t2);

                return result;
            } else if (dtype1 == DType::INT32 || dtype2 == DType::INT32) {
                Tensor<int>* t1 = dynamic_cast<Tensor<int>*>(const_cast<BaseTensor*>(_this));
                Tensor<int>* t2 = dynamic_cast<Tensor<int>*>(const_cast<BaseTensor*>(_other));
                if (!t1 || !t2) {
                    throw std::runtime_error("Type mismatch in tensor_add!");
                }
                int* data_ptr = new int[t1->get_linear_size()];
                int* shape = new int[t1->get_ndim()];
                memcpy(data_ptr, t1->get_data_ptr(), t1->get_linear_size() * sizeof(int));
                memcpy(shape, t1->get_shape(), t1->get_ndim() * sizeof(int));
                Tensor<int>* result = new Tensor<int>(data_ptr, shape, t1->get_ndim(), "cpu", DType::INT32);
                cpu_tensor_add(*result, *t2);

                return result;
            }
            else {
                throw std::runtime_error("Unsupported dtype!");
            }
        }
        catch (const std::exception& e) {
            last_error = e.what();
            return nullptr;
        }

    }

    BaseTensor* tensor_sub(const BaseTensor* _this, const BaseTensor* _other) {
        try {
            DType dtype1 = _this->get_dtype_enum();
            DType dtype2 = _other->get_dtype_enum();

            if(dtype1 == DType::FLOAT32 || dtype2 == DType::FLOAT32)
            {
                Tensor<float>* t1 = dynamic_cast<Tensor<float>*>(const_cast<BaseTensor*>(_this));
                Tensor<float>* t2 = dynamic_cast<Tensor<float>*>(const_cast<BaseTensor*>(_other));

                if (!t1 || !t2) {
                    throw std::runtime_error("Type mismatch in tensor_sub!");
                }
                float* data_ptr = new float[t1->get_linear_size()];
                int* shape = new int[t1->get_ndim()];

                memcpy(data_ptr, t1->get_data_ptr(), t1->get_linear_size()*sizeof(float));
                memcpy(shape, t1->get_shape(), t1->get_ndim()*sizeof(int));

                Tensor<float>* result = new Tensor<float>(data_ptr, shape, t1->get_ndim(),"cpu", DType::FLOAT32);
                cpu_tensor_sub(*result, *t2);
                return result;

            }
            else if(dtype1 == DType::INT32 || dtype2 == DType::INT32) {
                Tensor<int>* t1 = dynamic_cast<Tensor<int>*>(const_cast<BaseTensor*> (_this));
                Tensor<int>* t2 = dynamic_cast<Tensor<int>*>(const_cast<BaseTensor*> (_other));

                if (!t1 || !t2) {
                    throw std::runtime_error("Type mismatch in tensor_sub!");
                }
                int* data_ptr = new int[t1->get_linear_size()];
                int* shape = new int[t1->get_ndim()];

                memcpy(data_ptr, t1->get_data_ptr(), t1->get_linear_size()*sizeof(int));
                memcpy(shape, t1->get_shape(), t1->get_ndim()*sizeof(int));

                Tensor<int>* result = new Tensor<int>(data_ptr, shape, t1->get_ndim(),"cpu", DType::INT32);
                cpu_tensor_sub(*result, *t2);
                return result;
            }
            else {
                throw std::runtime_error("Unsupported dtype!");
            }
        } catch(const std::exception& e) {
            last_error = e.what();
            return nullptr;
        }
    }
    BaseTensor *reshape(BaseTensor *tensor, int *new_shape, int ndim) {
        try {
            int old_dim = tensor->get_ndim();
            int old_size = tensor->get_linear_size();
            int new_size = 1;
            for (int i = 0; i < ndim; i++) {
                new_size *= new_shape[i];
            }
            if (old_size != new_size) {
                throw std::runtime_error("Total size of new shape must be unchanged!");
            }
            void *data_ptr = tensor->get_data_ptr();
            DType dtype = tensor->get_dtype_enum();
            if (dtype == DType::FLOAT32) {
                float *new_data = new float[new_size];
                memcpy(new_data, static_cast<float *>(data_ptr), new_size * sizeof(float));
                return new Tensor<float>(new_data, new_shape, ndim, "cpu", DType::FLOAT32);
            } else if (dtype == DType::INT32) {
                int *new_data = new int[new_size];
                memcpy(new_data, static_cast<int *>(data_ptr), new_size * sizeof(int));
                return new Tensor<int>(new_data, new_shape, ndim, "cpu", DType::INT32);
            } else {
                throw std::runtime_error("Unsupported dtype!");
            }
        } catch(const std::exception& e) {
            last_error = e.what();
            return nullptr;
        }
    }

    const char* get_last_error() {
        return last_error.empty() ? nullptr : last_error.c_str();
    }

}