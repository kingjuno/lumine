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
// Explicit instantiations
template void cpu_tensor_add<float>(Tensor<float>& out,  Tensor<float>& t2);
template void cpu_tensor_add<int>(Tensor<int>& out,  Tensor<int>& t2);


template void cpu_tensor_sub<float>(Tensor<float>& out,  Tensor<float>& t2);
template void cpu_tensor_sub<int>(Tensor<int>& out,  Tensor<int>& t2);

template void cpu_tensor_mul<float>(Tensor<float>& out,  Tensor<float>& t2);
template void cpu_tensor_mul<int>(Tensor<int>& out,  Tensor<int>& t2);