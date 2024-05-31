#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_reshape_0_reshape_1_reshape_2_reshape_3_broadcast_to_7_slice_4_fill_constant_11_scale_6_pow_12_subtract_8_broadcast_to_13_elementwise_mul_14_broadcast_to_15_broadcast_to_17_elementwise_mul_16_elementwise_add_18_19_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  bn0_mean = (float* )(*(void **)(void_args[0]));
  const float*  bn0_offset = (float* )(*(void **)(void_args[1]));
  const float*  bn0_scale = (float* )(*(void **)(void_args[2]));
  const float*  bn0_variance = (float* )(*(void **)(void_args[3]));
  const float*  conv2d_0__tmp_0 = (float* )(*(void **)(void_args[4]));
  float*  var_51 = (float* )(*(void **)(void_args[5]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space1_fn_reshape_0_reshape_1_reshape_2_reshape_3_broadcast_to_7_slice_4_fill_constant_11_scale_6_pow_12_subtract_8_broadcast_to_13_elementwise_mul_14_broadcast_to_15_broadcast_to_17_elementwise_mul_16_elementwise_add_18_19_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      for (int32_t flat_i = 0; flat_i < 401408; flat_i += 1) {
        var_51[flat_i] = ((bn0_scale[(flat_i / 12544)] * (cinn_sycl_rsqrt_fp32((bn0_variance[((flat_i % 401408) / 12544)] + 0.00100000005f)) * conv2d_0__tmp_0[(114 + ((flat_i % 112) + (((flat_i / 12544) * 12769) + (113 * ((flat_i % 12544) / 112)))))])) + ((-1.00000000f * (bn0_scale[(flat_i / 12544)] * (cinn_sycl_rsqrt_fp32((bn0_variance[((flat_i % 401408) / 12544)] + 0.00100000005f)) * bn0_mean[(flat_i / 12544)]))) + bn0_offset[(flat_i / 12544)]));
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
