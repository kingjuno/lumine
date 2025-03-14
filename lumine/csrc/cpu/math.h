#ifndef CPU_H
#define CPU_H

#include "../tensor.h"

template <typename T>
void cpu_tensor_add(Tensor<T>& out, Tensor<T>& t2);
template <typename T>
void cpu_tensor_sub(Tensor<T>& out, Tensor<T>& t2);
template <typename T>
void cpu_tensor_mul(Tensor<T>& out, Tensor<T>& t2);
template <typename T>
void cpu_tensor_add_broadcast(Tensor<T>& out, const Tensor<T>& t1, const Tensor<T>& t2, int *broadcast_shape, int broadcast_ndim);
template <typename T>
void cpu_tensor_sub_broadcast(Tensor<T>& out, const Tensor<T>& t1, const Tensor<T>& t2, int *broadcast_shape, int broadcast_ndim);
template <typename T>
void cpu_tensor_mul_broadcast(Tensor<T>& out, const Tensor<T>& t1, const Tensor<T>& t2, int *broadcast_shape, int broadcast_ndim);

#endif // CPU_H