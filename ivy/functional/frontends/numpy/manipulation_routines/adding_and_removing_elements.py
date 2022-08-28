import ivy
import math


def delete(arr, obj, axis=None):
    arr.asarray(out=arr)
    if axis is None:
        arr.reshape((math.prod(arr.shape)))
        axis = 0

    if isinstance(obj, slice):
        obj = range(arr.shape[axis][obj])
    obj.asarray(out=obj)
    includeList = [x for x in range(arr.shape[axis]) if x not in obj]
    return [ivy.gather(arr, i, axis=axis) for i in includeList]
