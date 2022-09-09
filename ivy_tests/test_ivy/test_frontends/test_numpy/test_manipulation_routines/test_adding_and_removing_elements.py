import numpy as np
from hypothesis import given, strategies as st

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
    return (dtype_and_arr, axis, obj)


@handle_cmd_line_args
@given(
    dtype_and_arr_axis_obj=_delete_helper(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.delete"
    ),
)
def test_numpy_delete(
    dtype_and_arr_axis_obj,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    (arr_dtype, arr), axis, obj = dtype_and_arr_axis_obj
    input_dtypes = [arr_dtype]
    if type(obj) == np.ndarray:
        input_dtypes.append(obj.dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="delete",
        arr=np.array(arr, dtype=arr_dtype),
        obj=obj,
        axis=axis,
    )
