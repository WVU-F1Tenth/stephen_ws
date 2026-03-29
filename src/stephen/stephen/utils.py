import numpy
from numba import njit

@njit
def threshold_index_cumulative(ar, start_index, threshold) -> int:
    """
    Returns first index from start index where
    (sum of array[start_index:index+1] > threshold) on circular array.
    If none found returns 0.
    """
    sum = 0
    offset = 0
    index = start_index
    while offset < ar.size and sum <= threshold:
        offset += 1
        index = (start_index + offset) % ar.size
        sum += ar[index]
    if offset == ar.size:
        return 0
    else:
        return index

@njit
def threshold_index(ar, start_index, threshold) -> int:
    """
    Returns first index from start index where (array[index] > threshold) on circular array.
    If none found returns 0.
    """
    offset = 0
    index = start_index
    while offset < ar.size and ar[index] <= threshold:
        offset += 1
        index = (start_index + offset) % ar.size
    if offset == ar.size:
        return 0
    else:
        return index