#include <iostream>
#include <string>
#include <cstring>
#include <stdexcept>
#include<vector>
#include <sstream>
#include <cstring>
#include <cstring>  // For std::strcpy
#include <string>   // For std::string

// TODO
// 1. Tensor class              Progressing
//  1.1 constructors            Done
//  1.2 destructors             Done
// 2. Storing elements          Done
// 3. Printing utils            Progressing
//  3.1 Base                    Done
//  3.2 readability             Not Done
//  3.3 Return as string        Not Done
// 4. fetching subarrays        Done
// 5. support null array
// 6. Negative indexing
// 7. Device
//   7.1 get_subtensor device (cpu->gpu)
// 8. Casting types

//  Data Type Enum
enum class DType
{
    FLOAT32,
    INT32,
};

inline std::string DTypeToString(DType dt)
{
    switch (dt)
    {
    case DType::FLOAT32:
        return "float32";
    case DType::INT32:
        return "int32";
    default:
        return "unknown";
    }
};

// Base class for Tensor (Type Erasure)
class BaseTensor
{
public:
    virtual std::string print() const = 0; // Existing virtual function

    // Virtual functions to access shape and ndim
    virtual const int *get_shape() const = 0;
    virtual const int *get_strides() const = 0;
    virtual DType get_dtype_enum() const = 0;
    virtual int get_ndim() const = 0;
    virtual void *get_data_ptr() const = 0; // Generic data pointer
    // virtual void *astype(DType type) const = 0;
    virtual ~BaseTensor() = default;
};

template <typename T>
class Tensor : public BaseTensor
{
private:
    T *data_ptr;
    int ndim;
    int *shape;
    int *strides;
    int _linear_size;
    std::string device;
    DType dtype_enum;

public:
    Tensor(T *data_ptr, int *shape, int ndim, std::string device, DType dtype_enum)
        : ndim(ndim), device(device), dtype_enum(dtype_enum)
    {
        if (data_ptr == nullptr)
        {
            throw std::runtime_error("Empty Array not supported!");
        }

        this->shape = new int[ndim];
        this->strides = new int[ndim];

        if (this->strides == nullptr || this->shape == nullptr)
        {
            throw std::runtime_error("Memory Allocation failed!");
        }

        memcpy(this->shape, shape, ndim * sizeof(int));
        int stride = 1;
        int _linear_size = 1;
        for (int i = ndim - 1; i >= 0; i--)
        {
            this->strides[i] = stride;
            stride *= shape[i];
            _linear_size *= shape[i];
        }

        this->_linear_size = _linear_size;
        this->data_ptr = new T[_linear_size];
        memcpy(this->data_ptr, data_ptr, _linear_size * sizeof(T));
    }

    // Override virtual functions from BaseTensor
    const int *get_shape() const override {
        return shape;
    }
    const int *get_strides() const override {
        return strides;
    }
    DType get_dtype_enum() const override {
        return dtype_enum;
    }
    int get_ndim() const override {
        return ndim;
    }
    void *get_data_ptr() const override {
        return data_ptr;
    }

    std::string print_recursive(const T* data, const int* shape, const int* strides, int ndim, int dim = 0, int offset = 0) const {
        std::ostringstream oss;
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

    std::string print() const override {
        return print_recursive(data_ptr, shape, strides, ndim);
    }



    ~Tensor() {
        delete[] shape;
        delete[] strides;
        delete[] data_ptr;
    }
};

extern "C"
{

    BaseTensor *create_tensor(void *data_ptr, int *shape, int ndim, const char *device, const char *dtype)
    {
        if (std::strcmp(dtype, "float32") == 0)
            return new Tensor<float>(static_cast<float *>(data_ptr), shape, ndim, device, DType::FLOAT32);
        else if (std::strcmp(dtype, "int32") == 0)
            return new Tensor<int>(static_cast<int *>(data_ptr), shape, ndim, device, DType::INT32);
        else
            throw std::runtime_error("Invalid dtype!");
        return nullptr;
    }

    const char* print_tensor(BaseTensor *tensor)
    {
        std::string array = tensor->print();  // Get the string from tensor

        // Allocate memory for the C-string
        char *carray = new char[array.length() + 1];

        // Copy the content safely
        std::strcpy(carray, array.c_str());
        return carray;  // Caller must free this memory!
    }

    BaseTensor *get_item(BaseTensor *tensor, int *indices, int ind_len)
    {
        int ndim = tensor->get_ndim();
        const int *shape = tensor->get_shape();
        const int *strides = tensor->get_strides();
        if (ind_len > ndim)
            throw std::runtime_error("Too many indices!");

        int linear_index = 0;
        for (int i = 0; i < ind_len; i++)
        {
            if (indices[i] < 0)
                throw std::runtime_error("Negative Indexing not supported yet!");
            if (indices[i] >= shape[i])
                throw std::runtime_error("Indices out of bound!");
            linear_index += indices[i] * strides[i];
        }

        void *data_ptr = tensor->get_data_ptr();
        DType dtype = tensor->get_dtype_enum();
        if (ndim == ind_len)
        {
            switch (dtype)
            {
            case DType::FLOAT32: {
                float *scalar_value = new float(static_cast<float *>(data_ptr)[linear_index]);
                return reinterpret_cast<BaseTensor *>(scalar_value);
            }
            case DType::INT32: {
                int *scalar_value = new int(static_cast<int *>(data_ptr)[linear_index]);
                return reinterpret_cast<BaseTensor *>(scalar_value);
            }
            default:
                throw std::runtime_error("Unsupported dtype!");
            }
        }
        else
        {
            // deals sub tensors
            int new_dim = ndim - ind_len;
            int *new_shape = new int[ndim];
            memcpy(new_shape, shape + ind_len, new_dim * sizeof(int));
            switch (dtype)
            {
            case DType::FLOAT32:
                return new Tensor<float>(
                           static_cast<float *>(data_ptr) + linear_index,
                           new_shape,
                           new_dim,
                           "cpu",
                           DType::FLOAT32);
            case DType::INT32:
                return new Tensor<int>(
                           static_cast<int *>(data_ptr) + linear_index,
                           new_shape,
                           new_dim,
                           "cpu",
                           DType::INT32);
            default:
                throw std::runtime_error("Unsupported dtype!");
            }
        }
    }

    int *get_shape(BaseTensor *tensor)
    {
        return const_cast<int *>(tensor->get_shape());
    }
}
