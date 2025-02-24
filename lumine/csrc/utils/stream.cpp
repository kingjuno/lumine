#include "stream.h"
#include <algorithm>
#include <iostream>

OStream::OStream(int max_prec) : max_precision(max_prec) {}

std::string OStream::trimTrailingZeros(const std::string& str) const {
    size_t lastNonZero = str.find_last_not_of('0');
    if (lastNonZero != std::string::npos) {
        // If the last non-zero character is a decimal point, keep one zero
        if (str[lastNonZero] == '.') {
            return str.substr(0, lastNonZero + 2);
        }
        return str.substr(0, lastNonZero + 1);
    }
    return str;
}

OStream& OStream::operator<<(float value) {
    std::ostringstream temp;
    temp << std::fixed << std::setprecision(max_precision) << value;
    std::string str = trimTrailingZeros(temp.str());
    oss << str;
    return *this;
}

OStream& OStream::operator<<(double value) {
    std::ostringstream temp;
    temp << std::fixed << std::setprecision(max_precision) << value;
    std::string str = trimTrailingZeros(temp.str());
    oss << str;
    return *this;
}

std::string OStream::str() const {
    return oss.str();
}

void OStream::clear() {
    oss.str("");
    oss.clear();
}

int OStream::getMaxPrecision() const {
    return max_precision;
}

void OStream::setMaxPrecision(int prec) {
    max_precision = prec;
}