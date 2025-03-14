#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstring>
#include "tensor.h"
#include "utils/stream.h"
#include "cpu/math.h"
#include <cstdlib>
#include <immintrin.h>
#include <cstring>
#include <algorithm>

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
    const int* shapeA = A.get_shape();
    const int* shapeB = B.get_shape();
    int dimsA = A.get_ndim();
    int dimsB = B.get_ndim();

    if (dimsA < 2 || dimsB < 2) {
        throw std::runtime_error("Both tensors must have at least 2 dimensions.");
    }

    int N = shapeA[dimsA - 2];
    int K = shapeA[dimsA - 1];
    int M = shapeB[dimsB - 1];

    if (K != shapeB[dimsB - 2]) {
        throw std::runtime_error("Matrix multiplication shape mismatch.");
    }

    int result_ndim = std::max(dimsA, dimsB);
    int batch_dims = result_ndim - 2;
    std::vector<int> batch_shape(batch_dims > 0 ? batch_dims : 1);
    int batch_size = 1;

    for (int i = 0; i < batch_dims; i++) {
        int dimA_idx = i - (batch_dims - (dimsA - 2));
        int dimB_idx = i - (batch_dims - (dimsB - 2));
        int dimA = (dimA_idx >= 0) ? shapeA[dimA_idx] : 1;
        int dimB = (dimB_idx >= 0) ? shapeB[dimB_idx] : 1;

        if (dimA != dimB && dimA != 1 && dimB != 1) {
            throw std::runtime_error("Batch dimensions not broadcastable.");
        }
        batch_shape[i] = std::max(dimA, dimB);
        batch_size *= batch_shape[i];
    }

    std::vector<int> result_shape(result_ndim);
    for (int i = 0; i < batch_dims; i++) {
        result_shape[i] = batch_shape[i];
    }
    result_shape[batch_dims] = N;
    result_shape[batch_dims + 1] = M;

    T* result_data = new T[batch_size * N * M]();
    T* A_data = static_cast<T*>(A.get_data_ptr());
    T* B_data = static_cast<T*>(B.get_data_ptr());

    std::vector<int> A_strides(dimsA);
    std::vector<int> B_strides(dimsB);
    A_strides[dimsA - 1] = 1;
    B_strides[dimsB - 1] = 1;
    for (int i = dimsA - 2; i >= 0; i--) {
        A_strides[i] = A_strides[i + 1] * shapeA[i + 1];
    }
    for (int i = dimsB - 2; i >= 0; i--) {
        B_strides[i] = B_strides[i + 1] * shapeB[i + 1];
    }

    // Cache blocking parameters (tuned for L1/L2 cache)
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_K = 32;

    // Parallelize over batches
    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < batch_size; ++b) {
        std::vector<int> batch_indices(batch_dims);
        int temp_b = b;
        for (int i = batch_dims - 1; i >= 0; i--) {
            batch_indices[i] = temp_b % batch_shape[i];
            temp_b /= batch_shape[i];
        }

        int A_batch_idx = 0, B_batch_idx = 0;
        for (int i = 0; i < batch_dims; i++) {
            int dimA_idx = i - (batch_dims - (dimsA - 2));
            int dimB_idx = i - (batch_dims - (dimsB - 2));
            if (dimA_idx >= 0) {
                int A_idx = (shapeA[dimA_idx] == 1) ? 0 : batch_indices[i];
                A_batch_idx += A_idx * A_strides[dimA_idx];
            }
            if (dimB_idx >= 0) {
                int B_idx = (shapeB[dimB_idx] == 1) ? 0 : batch_indices[i];
                B_batch_idx += B_idx * B_strides[dimB_idx];
            }
        }

        T* C_batch = result_data + b * N * M;
        T* B_trans = new T[K * M];
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < M; j++) {
                B_trans[j * K + i] = B_data[B_batch_idx + i * B_strides[dimsB - 2] + j];
            }
        }

        // Blocked matrix multiplication with AVX for float
        for (int i = 0; i < N; i += BLOCK_N) {
            for (int j = 0; j < M; j += BLOCK_M) {
                for (int k = 0; k < K; k += BLOCK_K) {
                    int i_max = std::min(i + BLOCK_N, N);
                    int j_max = std::min(j + BLOCK_M, M);
                    int k_max = std::min(k + BLOCK_K, K);

                    if constexpr (std::is_same_v<T, float>) {
                        // AVX optimization for float
                        for (int ii = i; ii < i_max; ii++) {
                            for (int jj = j; jj < j_max; jj += 8) {
                                __m256 c_vec = _mm256_loadu_ps(C_batch + ii * M + jj);
                                for (int kk = k; kk < k_max; kk++) {
                                    __m256 a_vec = _mm256_set1_ps(A_data[A_batch_idx + ii * A_strides[dimsA - 2] + kk]);
                                    __m256 b_vec = _mm256_loadu_ps(B_trans + jj * K + kk);
                                    c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                                }
                                _mm256_storeu_ps(C_batch + ii * M + jj, c_vec);
                            }
                        }
                    } else {
                        for (int ii = i; ii < i_max; ii++) {
                            for (int jj = j; jj < j_max; jj++) {
                                T sum = 0;
                                for (int kk = k; kk < k_max; kk++) {
                                    sum += A_data[A_batch_idx + ii * A_strides[dimsA - 2] + kk] *
                                           B_trans[jj * K + kk];
                                }
                                C_batch[ii * M + jj] += sum;
                            }
                        }
                    }
                }
            }
        }
        delete[] B_trans;
    }
    return new Tensor<T>(result_data, result_shape.data(), result_ndim, "cpu",
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