#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_reshape_0_elementwise_add_1_0_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  blocks__0___se_reduce_offset = (float* )(*(void **)(void_args[0]));
  const float*  conv2d_1__tmp_0 = (float* )(*(void **)(void_args[1]));
  float*  var_8 = (float* )(*(void **)(void_args[2]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space4_fn_reshape_0_elementwise_add_1_0_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      for (int32_t flat_i = 0; flat_i < 8; flat_i += 1) {
        var_8[flat_i] = (conv2d_1__tmp_0[flat_i] + blocks__0___se_reduce_offset[flat_i]);
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
