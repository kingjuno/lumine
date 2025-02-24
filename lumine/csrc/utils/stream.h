#ifndef STREAM_H
#define STREAM_H

#include <sstream>
#include <iomanip>
#include <string>

class OStream {
private:
    std::ostringstream oss;
    static const int default_max_precision = 6;
    int max_precision;

    // Helper function to trim trailing zeros
    std::string trimTrailingZeros(const std::string& str) const;

public:
    OStream(int max_prec = default_max_precision);

    template<typename T>
    OStream& operator<<(const T& value) {
        oss << value;
        return *this;
    }

    // Specialization for float
    OStream& operator<<(float value);

    // Specialization for double
    OStream& operator<<(double value);

    std::string str() const;
    void clear();

    int getMaxPrecision() const;
    void setMaxPrecision(int prec);
};

#endif // STREAM_H