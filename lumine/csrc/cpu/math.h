#ifndef CPU_H
#define CPU_H

#include "../tensor.h"

template <typename T>
void cpu_tensor_add(Tensor<T>& out, Tensor<T>& t2);

#endif // CPU_H