import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import basic_indices

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# delete
@handle_cmd_line_args
@given(
    dtype_and_arr=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        shape=st.shared(helpers.get_shape(), key="arr_shape"),
    ),
    obj=basic_indices(shape=st.shared(helpers.get_shape(), key="arr_shape")),
    axis=st.one_of(
        st.none(),
        st.integers(
            min_value=0,
            max_value=len(st.shared(helpers.get_shape(), key="arr_shape")) - 1,
        ),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.delete"
    ),
)
def test_numpy_delete(
    dtype_and_arr,
    obj,
    axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    arr_dtype, arr = dtype_and_arr
    helpers.test_frontend_function(
        input_dtypes=[arr_dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="delete",
        arr=np.array(arr, dtype=arr_dtype),
        obj=np.array(obj),
        axis=axis,
    )
