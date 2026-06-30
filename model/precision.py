"""
Helpers for the high-precision numeric objects used in the simulator.

The model works with mpmath matrices because some simulations need more decimal
precision than float64 can provide. Numpy arrays are still used for inputs and
for visualization libraries, so this file keeps those conversions explicit.
"""
from typing import Iterable

import numpy as np
from mpmath import exp, fsum, matrix, mp, mpf


def set_precision(dps: int) -> None:
    """Set the number of decimal digits used by mpmath."""
    mp.dps = dps


set_precision(50)


def mp_zeros(n: int) -> matrix:
    """Create a zero column vector."""
    return matrix(n, 1)


def mp_zeros_matrix(rows: int, cols: int) -> matrix:
    """Create a zero matrix."""
    return matrix(rows, cols)


def mp_eye(n: int) -> matrix:
    """Create an identity matrix."""
    identity = matrix(n, n)
    for i in range(n):
        identity[i, i] = mpf(1)
    return identity


def mp_column(values: Iterable) -> matrix:
    """Convert a Python iterable into a column vector of mpmath values."""
    values_list = list(values)
    column = matrix(len(values_list), 1)
    for i, value in enumerate(values_list):
        column[i, 0] = mpf(value)
    return column


def mp_from_numpy(arr: np.ndarray) -> matrix:
    """Convert a 1-D or 2-D numpy array into an mpmath matrix."""
    if arr.ndim == 1:
        column = matrix(arr.shape[0], 1)
        for i in range(arr.shape[0]):
            column[i, 0] = mpf(float(arr[i]))
        return column

    if arr.ndim == 2:
        result = matrix(arr.shape[0], arr.shape[1])
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                result[i, j] = mpf(float(arr[i, j]))
        return result

    raise ValueError(f"mp_from_numpy expects 1-D or 2-D arrays, got ndim={arr.ndim}")


def mp_sum(values) -> mpf:
    """Sum values without falling back to float64 arithmetic."""
    if isinstance(values, matrix):
        return fsum(values[i, j] for i in range(values.rows) for j in range(values.cols))
    return fsum(values)


def mp_copy(value: matrix) -> matrix:
    """Copy an mpmath matrix before storing it in a history list."""
    return matrix(value)


def to_float64_matrix(value: matrix) -> np.ndarray:
    """Convert to float64 for libraries that do not work with mpmath matrices."""
    result = np.zeros((value.rows, value.cols), dtype=np.float64)
    for i in range(value.rows):
        for j in range(value.cols):
            result[i, j] = float(value[i, j])
    return result


__all__ = [
    "mp",
    "mpf",
    "matrix",
    "exp",
    "set_precision",
    "mp_zeros",
    "mp_zeros_matrix",
    "mp_eye",
    "mp_column",
    "mp_from_numpy",
    "mp_sum",
    "mp_copy",
    "to_float64_matrix",
]
