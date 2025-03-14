#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstring>
#include "tensor.h"
#include "utils/stream.h"
#include "cpu/math.h"
#include <cstdlib>

#if defined(_WIN32)  // Windows
#include <windows.h>

int get_cache_size() {
    DWORD buffer_size = 0;
    GetLogicalProcessorInformationEx(RelationCache, NULL, &buffer_size);
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)malloc(buffer_size);
    GetLogicalProcessorInformationEx(RelationCache, buffer, &buffer_size);

    int cache_size = buffer->Cache.CacheSize;
    free(buffer);
    return cache_size;
}

#elif defined(__linux__)
#include <fstream>

int get_cache_size() {
    std::ifstream file("/sys/devices/system/cpu/cpu0/cache/index0/size");
    if (!file) {
        return 32 * 1024;
    }
    int cache_size;
    file >> cache_size;
    return cache_size * 1024;
}

#elif defined(__APPLE__)
#include <sys/sysctl.h>

int get_cache_size() {
    size_t cache_size;
    size_t size = sizeof(cache_size);
    sysctlbyname("hw.l1dcachesize", &cache_size, &size, NULL, 0);
    return cache_size;
}

#else  // Fallback
int get_cache_size() {
    return 32 * 1024;
}
#endif

thread_local std::string last_error;
const int BLOCK_SIZE = get_cache_size();

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

int compute_broadcast_shape(const int* shape1, const int* shape2, int ndim1, int ndim2, int* broadcast_shape) {
    int max_dim = std::max(ndim1, ndim2);
    bool can_broadcast = true;

    for(int i = 0; i < max_dim; i++) {
        int dim1 = i < ndim1 ? shape1[ndim1 - i - 1] : 1;
        int dim2 = i < ndim2 ? shape2[ndim2 - i - 1] : 1;

        if(dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            can_broadcast = false;
        }
        broadcast_shape[max_dim - i - 1] = std::max(dim1, dim2);
    }
    return can_broadcast ? max_dim : -1;
}

template <typename T>
Tensor<T>* tensor_add_impl(Tensor<T>* t1, Tensor<T>* t2) {
    if (!t1 || !t2) {
        throw std::runtime_error("Type mismatch in tensor_add!");
    }

    int ndim = t1->get_ndim();
    const int* shape1 = t1->get_shape();
    int ndim2 = t2->get_ndim();
    const int* shape2 = t2->get_shape();

    if (ndim == ndim2 && std::equal(shape1, shape1 + ndim, shape2)) {
        T* data_ptr = new T[t1->get_linear_size()];
        int* shape = new int[t1->get_ndim()];
        memcpy(data_ptr, t1->get_data_ptr(), t1->get_linear_size() * sizeof(T));
        memcpy(shape, t1->get_shape(), t1->get_ndim() * sizeof(int));

        Tensor<T>* result = new Tensor<T>(data_ptr, shape, t1->get_ndim(), "cpu", (std::is_same<T, float>::value) ? DType::FLOAT32 : DType::INT32);
        cpu_tensor_add(*result, *t2);
        return result;
    }

    int* broadcast_shape = new int[std::max(ndim, ndim2)];
    int broadcast_ndim = compute_broadcast_shape(shape1, shape2, ndim, ndim2, broadcast_shape);

    if (broadcast_ndim > 0) {
        int linear_size = 1;
        for (int i = 0; i < broadcast_ndim; i++) {
            linear_size *= broadcast_shape[i];
        }

        T* data_ptr = new T[linear_size];
        Tensor<T>* result = new Tensor<T>(data_ptr, broadcast_shape, broadcast_ndim, "cpu", (std::is_same<T, float>::value) ? DType::FLOAT32 : DType::INT32);

        cpu_tensor_add_broadcast(*result, *t1, *t2, broadcast_shape, linear_size);
        return result;
    }

    delete[] broadcast_shape;
    throw std::runtime_error("Shape mismatch for tensor addition!");
}


template <typename T>
Tensor<T>* tensor_sub_impl(Tensor<T>* t1, Tensor<T>* t2) {
    if (!t1 || !t2) {
        throw std::runtime_error("Type mismatch in tensor_sub!");
    }

    int ndim = t1->get_ndim();
    const int* shape1 = t1->get_shape();
    int ndim2 = t2->get_ndim();
    const int* shape2 = t2->get_shape();

    if (ndim == ndim2 && std::equal(shape1, shape1 + ndim, shape2)) {
        T* data_ptr = new T[t1->get_linear_size()];
        int* shape = new int[t1->get_ndim()];
        memcpy(data_ptr, t1->get_data_ptr(), t1->get_linear_size() * sizeof(T));
        memcpy(shape, t1->get_shape(), t1->get_ndim() * sizeof(int));

        Tensor<T>* result = new Tensor<T>(data_ptr, shape, t1->get_ndim(), "cpu", (std::is_same<T, float>::value) ? DType::FLOAT32 : DType::INT32);
        cpu_tensor_sub(*result, *t2);
        return result;
    }

    int* broadcast_shape = new int[std::max(ndim, ndim2)];
    int broadcast_ndim = compute_broadcast_shape(shape1, shape2, ndim, ndim2, broadcast_shape);

    if (broadcast_ndim > 0) {
        int linear_size = 1;
        for (int i = 0; i < broadcast_ndim; i++) {
            linear_size *= broadcast_shape[i];
        }

        T* data_ptr = new T[linear_size];
        Tensor<T>* result = new Tensor<T>(data_ptr, broadcast_shape, broadcast_ndim, "cpu", (std::is_same<T, float>::value) ? DType::FLOAT32 : DType::INT32);

        cpu_tensor_sub_broadcast(*result, *t1, *t2, broadcast_shape, linear_size);
        return result;
    }

    delete[] broadcast_shape;
    throw std::runtime_error("Shape mismatch for tensor subraction!");
}

template <typename T>
Tensor<T>* tensor_mul_impl(Tensor<T>* t1, Tensor<T>* t2) {
    if (!t1 || !t2) {
        throw std::runtime_error("Type mismatch in tensor_mul!");
    }

    int ndim = t1->get_ndim();
    const int* shape1 = t1->get_shape();
    int ndim2 = t2->get_ndim();
    const int* shape2 = t2->get_shape();

    if (ndim == ndim2 && std::equal(shape1, shape1 + ndim, shape2)) {
        T* data_ptr = new T[t1->get_linear_size()];
        int* shape = new int[t1->get_ndim()];
        memcpy(data_ptr, t1->get_data_ptr(), t1->get_linear_size() * sizeof(T));
        memcpy(shape, t1->get_shape(), t1->get_ndim() * sizeof(int));

        Tensor<T>* result = new Tensor<T>(data_ptr, shape, t1->get_ndim(), "cpu", (std::is_same<T, float>::value) ? DType::FLOAT32 : DType::INT32);
        cpu_tensor_mul(*result, *t2);
        return result;
    }

    int* broadcast_shape = new int[std::max(ndim, ndim2)];
    int broadcast_ndim = compute_broadcast_shape(shape1, shape2, ndim, ndim2, broadcast_shape);

    if (broadcast_ndim > 0) {
        int linear_size = 1;
        for (int i = 0; i < broadcast_ndim; i++) {
            linear_size *= broadcast_shape[i];
        }

        T* data_ptr = new T[linear_size];
        Tensor<T>* result = new Tensor<T>(data_ptr, broadcast_shape, broadcast_ndim, "cpu", (std::is_same<T, float>::value) ? DType::FLOAT32 : DType::INT32);

        cpu_tensor_mul_broadcast(*result, *t1, *t2, broadcast_shape, linear_size);
        return result;
    }

    delete[] broadcast_shape;
    throw std::runtime_error("Shape mismatch for tensor multiplication!");
}

template <typename T>
Tensor<T>* tensor_matmul_impl(Tensor<T>& A, Tensor<T>& B) {
    // cache efficient implementation ig?
    // room for optimization still there
    // TODO: check for possible optimization or better implementation
    // whatever that makes this faster, the better
    const int* shapeA = A.get_shape();
    const int* shapeB = B.get_shape();
    int dimsA = A.get_ndim();
    int dimsB = B.get_ndim();

    if (dimsA < 2 || dimsB < 2) {
        throw std::runtime_error("Both tensors must have at least 2 dimensions for matrix multiplication.");
    }

    int batch_dims = std::max(dimsA, dimsB) - 2;
    int batch_shape[batch_dims];

    for (int i = 0; i < batch_dims; i++) {
        int dimA = (i < dimsA - 2) ? shapeA[i] : 1;
        int dimB = (i < dimsB - 2) ? shapeB[i] : 1;

        if (dimA != dimB && dimA != 1 && dimB != 1) {
            throw std::runtime_error("Batch dimensions are not broadcastable.");
        }
        batch_shape[i] = std::max(dimA, dimB);
    }

    int N = shapeA[dimsA - 2];
    int K = shapeA[dimsA - 1];
    int M = shapeB[dimsB - 1];

    if (K != shapeB[dimsB - 2]) {
        throw std::runtime_error("Matrix multiplication shape mismatch.");
    }

    int batch_size = 1;
    for (int i = 0; i < batch_dims; i++) {
        batch_size *= batch_shape[i];
    }

    T* result = new T[batch_size * N * M]();

    T* A_data = static_cast<T*>(A.get_data_ptr());
    T* B_data = static_cast<T*>(B.get_data_ptr());

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        int batch_A = 0, batch_B = 0;
        int temp_b = b;
        for (int i = batch_dims - 1; i >= 0; i--) {
            int idx = temp_b % batch_shape[i];
            temp_b /= batch_shape[i];

            batch_A = (shapeA[i] == batch_shape[i]) ? batch_A * shapeA[i] + idx : batch_A;
            batch_B = (shapeB[i] == batch_shape[i]) ? batch_B * shapeB[i] + idx : batch_B;
        }

        for (int i = 0; i < N; i += BLOCK_SIZE) {
            for (int j = 0; j < M; j += BLOCK_SIZE) {
                for (int k = 0; k < K; k += BLOCK_SIZE) {
                    for (int ii = i; ii < std::min(i + BLOCK_SIZE, N); ++ii) {
                        for (int kk = k; kk < std::min(k + BLOCK_SIZE, K); ++kk) {
                            T temp = A_data[batch_A * (N * K) + ii * K + kk];

                            for (int jj = j; jj < std::min(j + BLOCK_SIZE, M); ++jj) {
                                result[b * (N * M) + ii * M + jj] += temp * B_data[batch_B * (K * M) + kk * M + jj];
                            }
                        }
                    }
                }
            }
        }
    }

    int result_shape[batch_dims + 2];
    for (int i = 0; i < batch_dims; i++) {
        result_shape[i] = batch_shape[i];
    }
    result_shape[batch_dims] = N;
    result_shape[batch_dims + 1] = M;

    return new Tensor<T>(result, result_shape, batch_dims + 2, "cpu",
                         std::is_same<T, float>::value ? DType::FLOAT32 : DType::INT32);
}


// Explicit instantiation for supported types
template class Tensor<float>;
template class Tensor<int>;

// External C functions
extern "C" {
    BaseTensor* create_tensor(void* data_ptr, int* shape, int ndim, const char* device, const char* dtype) {
        // TODO
        // return signals for error are required to distingush between error types (for eg: runtime, value etc)
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
                return tensor_add_impl(t1, t2);
            }
            else if (dtype1 == DType::INT32 || dtype2 == DType::INT32) {
                Tensor<int>* t1 = dynamic_cast<Tensor<int>*>(const_cast<BaseTensor*>(_this));
                Tensor<int>* t2 = dynamic_cast<Tensor<int>*>(const_cast<BaseTensor*>(_other));
                return tensor_add_impl(t1, t2);
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

            if (dtype1 == DType::FLOAT32 || dtype2 == DType::FLOAT32) {
                Tensor<float>* t1 = dynamic_cast<Tensor<float>*>(const_cast<BaseTensor*>(_this));
                Tensor<float>* t2 = dynamic_cast<Tensor<float>*>(const_cast<BaseTensor*>(_other));
                return tensor_sub_impl(t1, t2);
            }
            else if (dtype1 == DType::INT32 || dtype2 == DType::INT32) {
                Tensor<int>* t1 = dynamic_cast<Tensor<int>*>(const_cast<BaseTensor*>(_this));
                Tensor<int>* t2 = dynamic_cast<Tensor<int>*>(const_cast<BaseTensor*>(_other));
                return tensor_sub_impl(t1, t2);
            }
            else {
                throw std::runtime_error("Unsupported dtype!");
            }
        } catch (const std::exception& e) {
            last_error = e.what();
            return nullptr;
        }
    }

    BaseTensor *tensor_mul(const BaseTensor* _this, const BaseTensor* _other)
    {
        try {
            DType dtype1 = _this->get_dtype_enum();
            DType dtype2 = _other->get_dtype_enum();

            if (dtype1 == DType::FLOAT32 || dtype2 == DType::FLOAT32) {
                Tensor<float>* t1 = dynamic_cast<Tensor<float>*>(const_cast<BaseTensor*>(_this));
                Tensor<float>* t2 = dynamic_cast<Tensor<float>*>(const_cast<BaseTensor*>(_other));
                return tensor_mul_impl(t1, t2);
            }
            else if (dtype1 == DType::INT32 || dtype2 == DType::INT32) {
                Tensor<int>* t1 = dynamic_cast<Tensor<int>*>(const_cast<BaseTensor*>(_this));
                Tensor<int>* t2 = dynamic_cast<Tensor<int>*>(const_cast<BaseTensor*>(_other));
                return tensor_mul_impl(t1, t2);
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

    BaseTensor *ones(int *shape, int ndim, const char *device, const char *dtype) {
        try {
            int _linear_size = 1;
            for (int i = 0; i < ndim; i++) {
                _linear_size *= shape[i];
            }
            if (std::strcmp(dtype, "float32") == 0) {
                float *data = new float[_linear_size];
                std::fill(data, data + _linear_size, 1.0f);
                return new Tensor<float>(data, shape, ndim, device, DType::FLOAT32);
            } else if (std::strcmp(dtype, "int32") == 0) {
                int *data = new int[_linear_size];
                std::fill(data, data + _linear_size, 1);
                return new Tensor<int>(data, shape, ndim, device, DType::INT32);
            } else {
                throw std::runtime_error("Unsupported dtype!");
            }
        } catch(const std::exception& e) {
            last_error = e.what();
            return nullptr;
        }
    }

    BaseTensor *zeros(int *shape, int ndim, const char *device, const char *dtype) {
        try {
            int _linear_size = 1;
            for (int i = 0; i < ndim; i++) {
                _linear_size *= shape[i];
            }
            if (std::strcmp(dtype, "float32") == 0) {
                float *data = new float[_linear_size];
                std::fill(data, data + _linear_size, 0.0f);
                return new Tensor<float>(data, shape, ndim, device, DType::FLOAT32);
            } else if (std::strcmp(dtype, "int32") == 0) {
                int *data = new int[_linear_size];
                std::fill(data, data + _linear_size, 0);
                return new Tensor<int>(data, shape, ndim, device, DType::INT32);
            } else {
                throw std::runtime_error("Unsupported dtype!");
            }
        } catch(const std::exception& e) {
            last_error = e.what();
            return nullptr;
        }
    }

    BaseTensor *tensor_matmul(const BaseTensor *t1, const BaseTensor *t2) {
        try {
            DType dtype1 = t1->get_dtype_enum();
            DType dtype2 = t2->get_dtype_enum();
            if (dtype1 == DType::FLOAT32 || dtype2 == DType::FLOAT32) {
                Tensor<float> *tensor1 = dynamic_cast<Tensor<float> *>(const_cast<BaseTensor *>(t1));
                Tensor<float> *tensor2 = dynamic_cast<Tensor<float> *>(const_cast<BaseTensor *>(t2));
                return tensor_matmul_impl(*tensor1, *tensor2);
            } else if (dtype1 == DType::INT32 || dtype2 == DType::INT32) {
                Tensor<int> *tensor1 = dynamic_cast<Tensor<int> *>(const_cast<BaseTensor *>(t1));
                Tensor<int> *tensor2 = dynamic_cast<Tensor<int> *>(const_cast<BaseTensor *>(t2));
                return tensor_matmul_impl(*tensor1, *tensor2);
            } else {
                throw std::runtime_error("Unsupported dtype!");
            }
        } catch(const std::exception& e) {
            last_error = e.what();
            return nullptr;
        }
    }
}