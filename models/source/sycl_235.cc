#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_fill_constant_0_max_1_broadcast_to_2_divide_3_2_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  p_norm_0__tmp_0 = (float* )(*(void **)(void_args[0]));
  const float*  linear_0__tmp_1 = (float* )(*(void **)(void_args[1]));
  float*  var_7 = (float* )(*(void **)(void_args[2]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space235_fn_fill_constant_0_max_1_broadcast_to_2_divide_3_2_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      for (int32_t flat_i = 0; flat_i < 128; flat_i += 1) {
        var_7[flat_i] = (linear_0__tmp_1[flat_i] / sycl::max(p_norm_0__tmp_0[0], 9.99999996e-13f));
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
