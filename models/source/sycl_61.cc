#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_reshape_1_elementwise_add_2_scale_3_exp_4_scale_5_fill_constant_0_divide_6_broadcast_to_7_elementwise_mul_8_7_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  blocks__7___se_expand_offset = (float* )(*(void **)(void_args[0]));
  const float*  conv2d_30__tmp_0 = (float* )(*(void **)(void_args[1]));
  const float*  swish_22__tmp_0 = (float* )(*(void **)(void_args[2]));
  float*  var_22 = (float* )(*(void **)(void_args[3]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space61_fn_reshape_1_elementwise_add_2_scale_3_exp_4_scale_5_fill_constant_0_divide_6_broadcast_to_7_elementwise_mul_8_7_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      for (int32_t flat_i = 0; flat_i < 94080; flat_i += 1) {
        var_22[flat_i] = (swish_22__tmp_0[flat_i] / (1.00000000f + cinn_sycl_exp_fp32((-1.00000000f * (conv2d_30__tmp_0[((flat_i % 94080) / 196)] + blocks__7___se_expand_offset[((flat_i % 94080) / 196)])))));
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
