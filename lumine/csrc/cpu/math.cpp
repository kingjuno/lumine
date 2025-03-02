// math.cpp
#include "math.h"
#include <stdexcept>

template <typename T>
void cpu_tensor_add(Tensor<T>& out, Tensor<T>& t2) {
    if (out.get_device() != t2.get_device() || out.get_device() != "cpu") {
        throw std::runtime_error("Device mismatch or unsupported device for addition!");
    }
    if (out.get_ndim() != t2.get_ndim()) {
        throw std::runtime_error("Dimension mismatch for tensor addition!");
    }
    const int* shape1 = out.get_shape();
    const int* shape2 = t2.get_shape();
    for (int i = 0; i < out.get_ndim(); i++) {
        if (shape1[i] != shape2[i]) {
            throw std::runtime_error("Shape mismatch for tensor addition!");
        }
    }
    int size = out.get_linear_size();
    T* out_data = static_cast<T*>(out.get_data_ptr());
    T* t2_data = static_cast<T*>(t2.get_data_ptr());
    for (int i = 0; i < size; i++) {
        out_data[i] += t2_data[i];
    }
}

template <typename T>
void cpu_tensor_sub(Tensor<T>& out, Tensor<T>& t2)
{
    if(out.get_device()!= t2.get_device() || out.get_device() != "cpu")
    {
        throw std::runtime_error("Device mismatch or unsupported device for subraction!");
    }
    if (out.get_ndim() != t2.get_ndim()) {
        throw std::runtime_error("Dimension mismatch for tensor subraction!");
    }
    const int* shape1 = out.get_shape();
    const int* shape2 = t2.get_shape();

    for(int i = 0; i<out.get_ndim(); i++)
    {
        if(shape1[i] != shape2[i])
        {
            throw std::runtime_error("Shape mismatch for tensor subraction");
        }
    }

    const int size = out.get_linear_size();
    T* out_data = static_cast<T*>(out.get_data_ptr());
    T* t2_data = static_cast<T*>(t2.get_data_ptr());
    for(int i = 0; i<size; i++)
    {
        out_data[i] -=t2_data[i];
    }
}

template <typename T>
void cpu_tensor_mul(Tensor<T>& out, Tensor<T>& t2)
{
    if(out.get_device()!= t2.get_device() || out.get_device() != "cpu")
    {
        throw std::runtime_error("Device mismatch or unsupported device for multiplication!");
    }
    if (out.get_ndim() != t2.get_ndim()) {
        throw std::runtime_error("Dimension mismatch for tensor multiplication!");
    }
    const int* shape1 = out.get_shape();
    const int* shape2 = t2.get_shape();

    for(int i = 0; i<out.get_ndim(); i++)
    {
        if(shape1[i] != shape2[i])
        {
            throw std::runtime_error("Shape mismatch for tensor multiplication");
        }
    }

    const int size = out.get_linear_size();
    T* out_data = static_cast<T*>(out.get_data_ptr());
    T* t2_data = static_cast<T*>(t2.get_data_ptr());
    for(int i = 0; i<size; i++)
    {
        out_data[i] *=t2_data[i];
    }
    
}

template <typename T>
void cpu_tensor_add_broadcast(Tensor<T>& out, const Tensor<T>& t1, const Tensor<T>& t2, int *broadcast_shape, int broadcast_size) {
    if (out.get_device() != t2.get_device() || out.get_device() != "cpu") {
        throw std::runtime_error("Device mismatch or unsupported device for addition!");
    }

    int max_dim = std::max(t1.get_ndim(), t2.get_ndim());
    int *strides1 = new int[max_dim]();
    int *strides2 = new int[max_dim]();
    if(strides1 == nullptr || strides2 == nullptr)
    {
        throw std::runtime_error("Memory Allocation failed!");
    }
    int stride1 = 1, stride2 = 1;
    for (int i = max_dim - 1; i >= 0; i--) {
        int dim1 = (i >= max_dim - t1.get_ndim()) ? t1.get_shape()[i - (max_dim - t1.get_ndim())] : 1;
        int dim2 = (i >= max_dim - t2.get_ndim()) ? t2.get_shape()[i - (max_dim - t2.get_ndim())] : 1;
        strides1[i] = (dim1 == 1) ? 0 : stride1;
        strides2[i] = (dim2 == 1) ? 0 : stride2;
        stride1 *= dim1;
        stride2 *= dim2;
    }

    T* out_data = static_cast<T*>(out.get_data_ptr());
    T* t1_data = static_cast<T*>(t1.get_data_ptr());
    T* t2_data = static_cast<T*>(t2.get_data_ptr());

    // Perform element-wise addition with broadcasting
    for (int i = 0; i < broadcast_size; i++) {
        int index1 = 0, index2 = 0;
        int linear_index = i;

        for (int j = max_dim - 1; j >= 0; j--) {
            int pos = linear_index % broadcast_shape[j];
            linear_index /= broadcast_shape[j];
            if (strides1[j] > 0) index1 += pos * strides1[j];
            if (strides2[j] > 0) index2 += pos * strides2[j];
        }
        out_data[i] = t1_data[index1] + t2_data[index2];
    }

    // Free strides
    free(strides1);
    free(strides2);
}


template <typename T>
void cpu_tensor_sub_broadcast(Tensor<T>& out, const Tensor<T>& t1, const Tensor<T>& t2, int *broadcast_shape, int broadcast_size) {
    if (out.get_device() != t2.get_device() || out.get_device() != "cpu") {
        throw std::runtime_error("Device mismatch or unsupported device for subraction!");
    }

    int max_dim = std::max(t1.get_ndim(), t2.get_ndim());
    int *strides1 = new int[max_dim]();
    int *strides2 = new int[max_dim]();
    if(strides1 == nullptr || strides2 == nullptr)
    {
        throw std::runtime_error("Memory Allocation failed!");
    }
    int stride1 = 1, stride2 = 1;
    for (int i = max_dim - 1; i >= 0; i--) {
        int dim1 = (i >= max_dim - t1.get_ndim()) ? t1.get_shape()[i - (max_dim - t1.get_ndim())] : 1;
        int dim2 = (i >= max_dim - t2.get_ndim()) ? t2.get_shape()[i - (max_dim - t2.get_ndim())] : 1;
        strides1[i] = (dim1 == 1) ? 0 : stride1;
        strides2[i] = (dim2 == 1) ? 0 : stride2;
        stride1 *= dim1;
        stride2 *= dim2;
    }

    T* out_data = static_cast<T*>(out.get_data_ptr());
    T* t1_data = static_cast<T*>(t1.get_data_ptr());
    T* t2_data = static_cast<T*>(t2.get_data_ptr());

    // Perform element-wise addition with broadcasting
    for (int i = 0; i < broadcast_size; i++) {
        int index1 = 0, index2 = 0;
        int linear_index = i;

        for (int j = max_dim - 1; j >= 0; j--) {
            int pos = linear_index % broadcast_shape[j];
            linear_index /= broadcast_shape[j];
            if (strides1[j] > 0) index1 += pos * strides1[j];
            if (strides2[j] > 0) index2 += pos * strides2[j];
        }
        out_data[i] = t1_data[index1] - t2_data[index2];
    }

    // Free strides
    free(strides1);
    free(strides2);
}


template <typename T>
void cpu_tensor_mul_broadcast(Tensor<T>& out, const Tensor<T>& t1, const Tensor<T>& t2, int *broadcast_shape, int broadcast_size) {
    if (out.get_device() != t2.get_device() || out.get_device() != "cpu") {
        throw std::runtime_error("Device mismatch or unsupported device for multiplication");
    }

    int max_dim = std::max(t1.get_ndim(), t2.get_ndim());
    int *strides1 = new int[max_dim]();
    int *strides2 = new int[max_dim]();
    if(strides1 == nullptr || strides2 == nullptr)
    {
        throw std::runtime_error("Memory Allocation failed!");
    }
    int stride1 = 1, stride2 = 1;
    for (int i = max_dim - 1; i >= 0; i--) {
        int dim1 = (i >= max_dim - t1.get_ndim()) ? t1.get_shape()[i - (max_dim - t1.get_ndim())] : 1;
        int dim2 = (i >= max_dim - t2.get_ndim()) ? t2.get_shape()[i - (max_dim - t2.get_ndim())] : 1;
        strides1[i] = (dim1 == 1) ? 0 : stride1;
        strides2[i] = (dim2 == 1) ? 0 : stride2;
        stride1 *= dim1;
        stride2 *= dim2;
    }

    T* out_data = static_cast<T*>(out.get_data_ptr());
    T* t1_data = static_cast<T*>(t1.get_data_ptr());
    T* t2_data = static_cast<T*>(t2.get_data_ptr());

    // Perform element-wise addition with broadcasting
    for (int i = 0; i < broadcast_size; i++) {
        int index1 = 0, index2 = 0;
        int linear_index = i;

        for (int j = max_dim - 1; j >= 0; j--) {
            int pos = linear_index % broadcast_shape[j];
            linear_index /= broadcast_shape[j];
            if (strides1[j] > 0) index1 += pos * strides1[j];
            if (strides2[j] > 0) index2 += pos * strides2[j];
        }
        out_data[i] = t1_data[index1] * t2_data[index2];
    }

    // Free strides
    free(strides1);
    free(strides2);
}

// Explicit instantiations
template void cpu_tensor_add<float>(Tensor<float>& out,  Tensor<float>& t2);
template void cpu_tensor_add<int>(Tensor<int>& out,  Tensor<int>& t2);
template void cpu_tensor_sub<float>(Tensor<float>& out,  Tensor<float>& t2);
template void cpu_tensor_sub<int>(Tensor<int>& out,  Tensor<int>& t2);
template void cpu_tensor_mul<float>(Tensor<float>& out,  Tensor<float>& t2);
template void cpu_tensor_mul<int>(Tensor<int>& out,  Tensor<int>& t2);
template void cpu_tensor_add_broadcast<float>(Tensor<float>& out, const Tensor<float>& t1, const Tensor<float>& t2, int *broadcast_shape, int broadcast_ndim);
template void cpu_tensor_add_broadcast<int>(Tensor<int>& out,  const Tensor<int>& t1, const Tensor<int>& t2, int *broadcast_shape, int broadcast_ndim);
template void cpu_tensor_sub_broadcast<float>(Tensor<float>& out, const Tensor<float>& t1, const Tensor<float>& t2, int *broadcast_shape, int broadcast_ndim);
template void cpu_tensor_sub_broadcast<int>(Tensor<int>& out,  const Tensor<int>& t1, const Tensor<int>& t2, int *broadcast_shape, int broadcast_ndim);
template void cpu_tensor_mul_broadcast<float>(Tensor<float>& out, const Tensor<float>& t1, const Tensor<float>& t2, int *broadcast_shape, int broadcast_ndim);
template void cpu_tensor_mul_broadcast<int>(Tensor<int>& out,  const Tensor<int>& t1, const Tensor<int>& t2, int *broadcast_shape, int broadcast_ndim);
