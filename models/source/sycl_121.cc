#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_reshape_0_reshape_1_reshape_2_reshape_3_broadcast_to_5_fill_constant_10_scale_7_pow_11_broadcast_to_12_subtract_6_broadcast_to_14_elementwise_mul_13_broadcast_to_16_elementwise_mul_15_elementwise_add_17_18_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  blocks__15___bn0_mean = (float* )(*(void **)(void_args[0]));
  const float*  blocks__15___bn0_offset = (float* )(*(void **)(void_args[1]));
  const float*  blocks__15___bn0_scale = (float* )(*(void **)(void_args[2]));
  const float*  blocks__15___bn0_variance = (float* )(*(void **)(void_args[3]));
  const float*  conv2d_60__tmp_0 = (float* )(*(void **)(void_args[4]));
  float*  var_49 = (float* )(*(void **)(void_args[5]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space121_fn_reshape_0_reshape_1_reshape_2_reshape_3_broadcast_to_5_fill_constant_10_scale_7_pow_11_broadcast_to_12_subtract_6_broadcast_to_14_elementwise_mul_13_broadcast_to_16_elementwise_mul_15_elementwise_add_17_18_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      for (int32_t flat_i = 0; flat_i < 56448; flat_i += 1) {
        var_49[flat_i] = ((blocks__15___bn0_scale[(flat_i / 49)] * (cinn_sycl_rsqrt_fp32((blocks__15___bn0_variance[((flat_i % 56448) / 49)] + 0.00100000005f)) * conv2d_60__tmp_0[flat_i])) + ((-1.00000000f * (blocks__15___bn0_scale[(flat_i / 49)] * (cinn_sycl_rsqrt_fp32((blocks__15___bn0_variance[((flat_i % 56448) / 49)] + 0.00100000005f)) * blocks__15___bn0_mean[(flat_i / 49)]))) + blocks__15___bn0_offset[(flat_i / 49)]));
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
