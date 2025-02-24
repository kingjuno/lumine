#ifndef TENSOR_H
#define TENSOR_H

#include <string>
#include <stdexcept>

// Data Type Enum
enum class DType {
    FLOAT32,
    INT32,
};

inline std::string DTypeToString(DType dt);

// Base class for Tensor (Type Erasure)
class BaseTensor {
public:
    virtual std::string print() const = 0;
    virtual BaseTensor* astype(DType target_type) = 0;
    virtual const int* get_shape() const = 0;
    virtual const int* get_strides() const = 0;
    virtual DType get_dtype_enum() const = 0;
    virtual int get_ndim() const = 0;
    virtual void* get_data_ptr() const = 0;
    virtual int get_linear_size() const = 0;
    virtual std::string get_device() const = 0;
    virtual ~BaseTensor() = default;
};

template <typename T>
class Tensor : public BaseTensor {
private:
    T* data_ptr;
    int ndim;
    int* shape;
    int* strides;
    int _linear_size;
    std::string device;
    DType dtype_enum;

public:
    Tensor(T* data_ptr, int* shape, int ndim, std::string device, DType dtype_enum);
    ~Tensor();

    // Override virtual functions
    const int* get_shape() const override;
    const int* get_strides() const override;
    DType get_dtype_enum() const override;
    int get_ndim() const override;
    void* get_data_ptr() const override;
    int get_linear_size() const override;
    std::string get_device() const;
    std::string print() const override;
    BaseTensor* astype(DType target_type) override;

    // Helper method
    std::string print_recursive(const T* data, const int* shape, const int* strides, int ndim, int dim = 0, int offset = 0) const;
};

// External C interface
extern "C" {
    BaseTensor* create_tensor(void* data_ptr, int* shape, int ndim, const char* device, const char* dtype);
    const char* print_tensor(BaseTensor* tensor);
    BaseTensor* get_item(BaseTensor* tensor, int* indices, int ind_len);
    int* get_shape(BaseTensor* tensor);
    BaseTensor* astype(BaseTensor* tensor, const char* target_type);
    void* get_data_ptr(BaseTensor* tensor);
}

#endif // TENSOR_H