import numpy as np
from hypothesis import given, strategies as st

from ivy_tests.test_ivy.test_functional.test_core.test_manipulation import (
    _arrays_idx_n_dtypes,
)

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# delete
@st.composite
def _delete_helper(draw):
    shape = draw(helpers.get_shape(min_num_dims=1))

    dtype_and_arr = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=shape,
        )
    )

    ndim = len(shape)
    if ndim == 0:
        axis = draw(st.none())
        obj = draw(
            st.one_of(
                st.slices(0),
                st.lists(st.integers(0, 0), min_size=1).map(np.asarray),
                st.integers(min_value=0, max_value=0),
            )
        )
    else:
        axis = draw(st.none() | st.integers(-len(shape), len(shape) - 1))
        axis_len = shape[0 if axis is None else axis]
        obj = draw(
            st.one_of(
                st.slices(axis_len),
                st.lists(
                    st.integers(min_value=-axis_len, max_value=axis_len - 1), min_size=1
                ).map(np.asarray),
                st.integers(min_value=-axis_len, max_value=axis_len - 1),
            )
        )
    if type(obj) is np.ndarray:
        dtype_and_obj = ([obj.dtype], obj)
    else:
        dtype_and_obj = ([], obj)
    return (dtype_and_arr, dtype_and_obj, axis)


@handle_cmd_line_args
@given(
    dtype_and_inputs=_delete_helper(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.delete"
    ),
)
def test_numpy_delete(
    dtype_and_inputs,
    as_variable,
    num_positional_args,
    native_array,
):
    (arr_dtype, arr), (obj_dtype, obj), axis = dtype_and_inputs
    if type(obj) == np.ndarray:
        arr_dtype.append(obj.dtype)
    helpers.test_frontend_function(
        input_dtypes=arr_dtype + obj_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="delete",
        arr=arr,
        obj=obj,
        axis=axis,
    )


# append
@handle_cmd_line_args
@given(
    dtype_and_inputs=_arrays_idx_n_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.append"
    ),
)
def test_numpy_append(as_variable, num_positional_args, native_array, dtype_and_inputs):
    arr_list, arr_dtype, axis = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=tuple(arr_dtype),
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="append",
        arr=arr_list[0],
        values=arr_list[1],
        axis=axis,
    )
