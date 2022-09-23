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


def insert(arr, obj, values, axis=None):
    arr = ivy.asarray(arr)  # Convert to ivy array

    ndim = arr.ndim
    if arr.ndim == 0:
        arr = ivy.reshape(arr, (1,))  # Convert single digit to array
    if axis is None:
        arr.reshape((math.prod(arr.shape)))  # Flatten
        axis = 0

    slobj = [slice(None)] * ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)

    if isinstance(obj, slice):
        # turn it into a range object
        start, stop, step = obj.indices(N)
        indices = ivy.arange(start, stop=stop, step=step)
    else:
        # need to copy obj, because indices will be changed in-place
        indices = ivy.array(obj)
        if indices.dtype == bool:
            indices = indices.astype(int)
    if indices.size == 1:
        index = indices.item()
        if index < 0:
            index += N

        # There are some object array corner cases here, but we cannot avoid
        # that:
        values = ivy.array(values, copy=False, dtype=arr.dtype)
        while values.ndim < arr.ndim:
            values = ivy.expand_dims(values, axis=0)
        if indices.ndim == 0:
            # broadcasting is very different here, since a[:,0,:] = ... behaves
            # very different from a[:,[0],:] = ...! This changes values so that
            # it works likes the second case. (here a[:,0:1,:])

            # Moves the axis at axis to the front
            values = (
                values.expand_dims(axis=0).swapaxes(0, axis + 1).squeeze(axis=axis + 1)
            )
        numnew = values.shape[axis]
        newshape[axis] += numnew
        new = ivy.empty(newshape, dtype=arr.dtype)

        slobj[axis] = slice(None, index)
        new[tuple(slobj)] = arr[tuple(slobj)]
        slobj[axis] = slice(index, index + numnew)
        new[tuple(slobj)] = values
        slobj[axis] = slice(index + numnew, None)
        slobj2 = [slice(None)] * ndim
        slobj2[axis] = slice(index, None)
        new[tuple(slobj)] = arr[tuple(slobj2)]

        return new
    elif indices.size == 0:
        indices = indices.astype(int)

    indices[indices < 0] += N

    numnew = len(indices)
    order = indices.argsort()
    indices[order] += ivy.arange(numnew)

    newshape[axis] += numnew
    old_mask = ivy.ones(newshape[axis], dtype=bool)
    old_mask[indices] = False

    new = ivy.empty(newshape, dtype=arr.dtype)
    slobj2 = [slice(None)] * ndim
    slobj[axis] = indices
    slobj2[axis] = old_mask
    new[tuple(slobj)] = values
    new[tuple(slobj2)] = arr

    return new


def append(arr, values, axis=None):
    if not axis:
        axis = 0
        arr = ivy.reshape(arr, (1,))
        values = ivy.reshape(values, (1,))
    return ivy.concat(arr, values, axis=axis)


def resize(a, new_shape):
    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    a = ivy.reshape(a, (1,))

    new_size = 1
    for dim in new_shape:
        new_size *= dim

    if a.size == 0 or new_size == 0:
        return ivy.zeros(new_shape, dtype=a.dtype)

    repeats = -(-new_size // a.size)  # ceiling
    a = ivy.concat((a,) * repeats)[:new_size]

    return ivy.reshape(a, new_shape)


def trim_zeros(filt, trim="fb"):
    trim = trim.lower()
    front = 0
    back = len(filt)
    if "f" in trim:
        for i in filt:
            if filt[i] == 0:
                front += 1
            else:
                break
    if "b" in trim:
        for i in filt[::-1]:
            if filt[i] == 0:
                back -= 1
            else:
                break
    return filt[front:back]


def unique(
    ar, return_index=False, return_inverse=False, return_counts=False, axis=None
):
    if axis is None:
        ar = ivy.reshape(ar, (1,))
    else:
        ar = ar.expand_dims(axis=0).swapaxes(0, axis + 1).squeeze(axis=axis + 1)

    output = ivy.unique_all(ar)

    return_list = [output.values]
    if return_index:
        return_list.append(output.indices)
    if return_inverse:
        return_list.append(output.inverse_indices)
    if return_counts:
        return_list.append(output.counts)
