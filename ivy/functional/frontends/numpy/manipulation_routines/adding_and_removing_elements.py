import ivy
import math


def delete(arr, obj, axis=None):
    arr = ivy.asarray(arr)  # Convert to ivy array
    if arr.ndim == 0:
        arr = ivy.reshape(arr, (1,))  # Convert single digit to array
    if axis is None:
        arr.reshape((math.prod(arr.shape)))  # Flatten
        axis = 0
    cutsizes = []
    if isinstance(obj, slice):
        start, stop, step = obj.indices(arr.shape[axis])
        xr = range(start, stop, step)
        num_to_remove = len(xr)
        if step < 0:
            step = -step
            start = xr[-1]
            stop = xr[0] + 1
        # First cut is up to start
        cutsizes.append(start)
        for _ in range(num_to_remove - 1):
            # Middle cuts
            cutsizes.append(1)
            cutsizes.append(step - 1)
        # Final item
        cutsizes.append(1)
        cutsizes.append(arr.shape[axis] - stop)
    else:
        # Sort and remove duplicates
        if isinstance(obj, int):
            obj = set([obj])
        else:
            obj = sorted(set(obj))
        prev = 0
        for x in obj:
            cutsizes.append(x - prev)
            prev = x + 1
            cutsizes.append(1)
        cutsizes.append(arr.shape[axis] - prev)
    split_array = ivy.split(arr, num_or_size_splits=cutsizes, axis=axis)
    # Remove every other array in list
    del split_array[1::2]
    # Stick them back together
    return ivy.concat(split_array, axis=axis)


def insert(arr, obj, values, *, axis):
    return
