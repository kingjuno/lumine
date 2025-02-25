# Lumine [WIP]

A small project offering a Python-wrapped C++ library for tensor operations.

## Features
- Both CPU and C++ code for tensor creation, shape queries, slicing, and basic arithmetic.
- Python interface built using CFFI-like strategies via ctypes.

## Installation
1. Clone the repo:
   ```sh
   git clone https://github.com/kingjuno/lumine.git
   cd lumine
   ```
2. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Build with CMake:
   ```sh
   cmake -S cmake -B build
   cmake --build build
   ```
4. Install the Python package:
   ```sh
   pip install -e .
   ```

## Usage
After installation, import the Python API:
```py
from lumine import tensor

arr = tensor([[1, 2], [3, 4]])
print(arr)  # Displays the 2D array
```

## Contributing
Contributions are welcome. Submit pull requests or file issues.

## License
This project is licensed under the MIT License.
