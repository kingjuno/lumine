// tensor_ext.h

#ifndef TENSOR_EXT_H
#define TENSOR_EXT_H

#ifdef __cplusplus
extern "C" {
#endif

// Declare an opaque type.
typedef struct Tensor Tensor;

// Create a new tensor.
//   data   : pointer to the float data
//   shape  : pointer to an array of ints representing the shape
//   ndim   : number of dimensions
//   device : string representing the device (e.g., "cpu")
Tensor *tensor_new(float *data, int *shape, int ndim, char *device);

// Print tensor information.
void tensor_print(Tensor *t);

// Free tensor memory.
void tensor_free(Tensor *t);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_EXT_H
