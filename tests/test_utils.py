import random


def generate_nd_list(shape, min_val=0, max_val=100, dtype=int):
    if len(shape) == 1:
        if dtype == int:
            return [random.randint(min_val, max_val) for _ in range(shape[0])]
        elif dtype == float:
            return [random.uniform(min_val, max_val) for _ in range(shape[0])]
        else:
            raise ValueError("dtype must be either int or float")

    return [
        generate_nd_list(shape[1:], min_val, max_val, dtype) for _ in range(shape[0])
    ]
